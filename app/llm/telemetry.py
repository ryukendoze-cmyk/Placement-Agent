"""LLM token accounting, cost governance and provider circuit breaker (Milestone 22).

``LLMTelemetryTracker`` keeps per-provider counters (calls, prompt/completion
tokens, estimated USD cost, errors, fallbacks, latency) and a sliding-window
circuit breaker: 3 consecutive provider failures within a 60s window open the
circuit for a 30s cooldown during which all traffic is served by the offline
``MockLLM`` so a misbehaving network can never hang a request.

``get_llm`` (app.llm.base) stays untouched - this module adds
``get_instrumented_llm`` / :class:`TelemetryLLM` which wrap the raw provider.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Any

from .base import LLMClient, get_llm
from .mock import MockLLM

logger = logging.getLogger(__name__)

CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_WINDOW_SECONDS = 60.0
CIRCUIT_COOLDOWN_SECONDS = 30.0

# USD per 1M tokens (prompt, completion) - conservative public list prices.
PRICING_USD_PER_M: dict[str, tuple[float, float]] = {
    "mock": (0.0, 0.0),
    "bedrock": (0.05, 0.20),    # amazon.nova-lite
    "openai": (0.15, 0.60),     # gpt-4o-mini
}


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: ~4 chars/token, minimum 1 for non-empty."""
    return max(1, len(text) // 4) if text else 0


def _new_provider_stats() -> dict[str, Any]:
    return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "estimated_cost_usd": 0.0, "errors": 0,
            "fallbacks": 0, "total_latency_ms": 0.0, "avg_latency_ms": 0.0}


class LLMTelemetryTracker:
    def __init__(self, now_fn=time.time):
        self._now = now_fn
        self._lock = threading.RLock()
        self._providers: dict[str, dict[str, Any]] = {}
        self._failure_times: dict[str, deque[float]] = {}
        self._circuit_open_until: dict[str, float] = {}

    # -- recording ------------------------------------------------------------

    def record(self, provider: str, prompt: str, completion: str = "",
               latency_ms: float = 0.0, error: bool = False,
               usage: dict[str, int] | None = None) -> None:
        with self._lock:
            stats = self._providers.setdefault(provider, _new_provider_stats())
            now = self._now()
            if error:
                stats["calls"] += 1
                stats["errors"] += 1
                window = self._failure_times.setdefault(provider, deque())
                window.append(now)
                while window and now - window[0] > CIRCUIT_WINDOW_SECONDS:
                    window.popleft()
                if len(window) >= CIRCUIT_FAILURE_THRESHOLD:
                    self._circuit_open_until[provider] = now + CIRCUIT_COOLDOWN_SECONDS
                    window.clear()
                    logger.warning(
                        "LLM provider %s tripped circuit breaker (%d failures in "
                        "%.0fs); cooling down %.0fs on MockLLM.",
                        provider, CIRCUIT_FAILURE_THRESHOLD, CIRCUIT_WINDOW_SECONDS,
                        CIRCUIT_COOLDOWN_SECONDS)
                return
            if usage:
                prompt_tokens = int(usage.get("prompt_tokens") or 0) or estimate_tokens(prompt)
                completion_tokens = (int(usage.get("completion_tokens") or 0)
                                     or estimate_tokens(completion))
            else:
                prompt_tokens = estimate_tokens(prompt)
                completion_tokens = estimate_tokens(completion)
            stats["calls"] += 1
            stats["prompt_tokens"] += prompt_tokens
            stats["completion_tokens"] += completion_tokens
            stats["total_tokens"] += prompt_tokens + completion_tokens
            price_p, price_c = PRICING_USD_PER_M.get(provider, (0.0, 0.0))
            stats["estimated_cost_usd"] += (
                prompt_tokens * price_p + completion_tokens * price_c) / 1_000_000.0
            stats["total_latency_ms"] += float(latency_ms)
            stats["avg_latency_ms"] = stats["total_latency_ms"] / stats["calls"]
            self._failure_times.get(provider, deque()).clear()
            self._circuit_open_until.pop(provider, None)

    def record_fallback(self, provider: str, latency_ms: float = 0.0) -> None:
        with self._lock:
            stats = self._providers.setdefault(provider, _new_provider_stats())
            stats["fallbacks"] += 1

    # -- circuit breaker -------------------------------------------------------

    def should_use_fallback(self, provider: str) -> bool:
        with self._lock:
            until = self._circuit_open_until.get(provider)
            if until is None:
                return False
            if self._now() >= until:
                self._circuit_open_until.pop(provider, None)
                return False
            return True

    # -- reporting -------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        with self._lock:
            providers: dict[str, dict[str, Any]] = {}
            for name, stats in self._providers.items():
                entry = dict(stats)
                until = self._circuit_open_until.get(name)
                entry["circuit_open"] = bool(until is not None and self._now() < until)
                providers[name] = entry
            return {
                "providers": providers,
                "totals": {
                    "total_calls": sum(p["calls"] for p in providers.values()),
                    "total_tokens": sum(p["total_tokens"] for p in providers.values()),
                    "estimated_cost_usd": sum(p["estimated_cost_usd"]
                                              for p in providers.values()),
                    "total_errors": sum(p["errors"] for p in providers.values()),
                    "total_fallbacks": sum(p["fallbacks"] for p in providers.values()),
                },
            }


class TelemetryLLM:
    """LLMClient wrapper: records telemetry + enforces the circuit breaker."""

    def __init__(self, settings: Any, tracker: LLMTelemetryTracker | None = None):
        self.settings = settings
        self.provider = (getattr(settings, "llm_provider", "mock") or "mock").lower()
        self.tracker = tracker or LLMTelemetryTracker()
        self.primary: LLMClient = get_llm(settings)
        self.mock = MockLLM()

    def complete(self, system: str, user: str, json_mode: bool = False, **kwargs) -> str:
        if self.tracker.should_use_fallback(self.provider):
            self.tracker.record_fallback(self.provider)
            return self.mock.complete(system, user, json_mode=json_mode, **kwargs)
        t0 = time.perf_counter()
        try:
            out = self.primary.complete(system, user, json_mode=json_mode, **kwargs)
        except Exception:
            self.tracker.record(self.provider, system, "",
                                latency_ms=(time.perf_counter() - t0) * 1000.0,
                                error=True)
            raise
        self.tracker.record(self.provider, system, out,
                            latency_ms=(time.perf_counter() - t0) * 1000.0)
        return out

    def stream(self, system: str, user: str, json_mode: bool = False) -> Iterator[str]:
        if self.tracker.should_use_fallback(self.provider):
            self.tracker.record_fallback(self.provider)
            yield from self.mock.stream(system, user, json_mode=json_mode)
            return
        t0 = time.perf_counter()
        chunks: list[str] = []
        try:
            for chunk in self.primary.stream(system, user, json_mode=json_mode):
                chunks.append(chunk)
                yield chunk
        except Exception:
            self.tracker.record(self.provider, system, "".join(chunks),
                                latency_ms=(time.perf_counter() - t0) * 1000.0,
                                error=True)
            raise
        self.tracker.record(self.provider, system, "".join(chunks),
                            latency_ms=(time.perf_counter() - t0) * 1000.0)


def get_instrumented_llm(settings: Any,
                         tracker: LLMTelemetryTracker | None = None) -> TelemetryLLM:
    """Production LLM factory: raw provider wrapped with telemetry + breaker."""
    return TelemetryLLM(settings, tracker=tracker)
