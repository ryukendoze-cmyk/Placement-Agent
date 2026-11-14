"""Sliding-window API rate limiter (Milestone 15).

Thread-safe in-memory fixed-count sliding window keyed by ``(identity, bucket)``.
Disabled by default (``settings.rate_limit_enabled = False``) so the fast offline
unit suite never hits 429s; enable with ``RATE_LIMIT_ENABLED=true`` (production) or
flip the setting at runtime in a targeted rate-limit test.

Identity resolution prefers an authenticated principal (``X-Demo-Student-ID`` or
the ``ps06_session`` claim) and falls back to the client host, so one noisy client
cannot exhaust another's quota.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from ..config import settings

#: Default limits (bucket -> (max_requests, window_seconds)) for protected routes.
DEFAULT_LIMITS: dict[str, tuple[int, int]] = {
    "jobs_scrape": (10, 60),
    "agent_run": (30, 60),
    "copilot_ask": (60, 60),
}


class RateLimiter:
    """Thread-safe sliding-window counter keyed by ``(identity, bucket)``."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str, bucket: str, max_requests: int,
              window_seconds: int, now: float | None = None) -> tuple[bool, int]:
        """Record a hit for ``identity``/``bucket``.

        Returns ``(allowed, retry_after_seconds)``. ``retry_after`` is 0 when
        allowed, otherwise the seconds until the oldest hit leaves the window.
        Prunes expired hits under the lock so memory stays bounded.
        """
        if max_requests <= 0:
            return True, 0
        ts = time.time() if now is None else now
        window = max(1, int(window_seconds))
        key = (str(identity), str(bucket))
        with self._lock:
            hits = self._hits[key]
            cutoff = ts - window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) < int(max_requests):
                hits.append(ts)
                return True, 0
            retry_after = max(1, int(hits[0] + window - ts) + 1)
            return False, retry_after

    def reset(self) -> None:
        """Clear all counters (test hygiene)."""
        with self._lock:
            self._hits.clear()


#: Process-wide limiter instance (one per app process).
limiter = RateLimiter()


def _identity(request: Request) -> str:
    is_strict = False
    try:
        from ..config import settings as _settings
        from .rbac import decode_session
        is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
        claims = decode_session(request.cookies.get("ps06_session")) or {}
        sid = str(claims.get("student_id") or "").strip()
        if sid:
            return sid
        uid = str(claims.get("user_id") or "").strip()
        if uid:
            return uid
    except Exception:  # pragma: no cover - defensive
        pass
    if not is_strict:
        hdr = str(request.headers.get("x-demo-student-id", "") or "").strip()
        if hdr:
            return hdr
    xff = str(request.headers.get("x-forwarded-for", "") or "").strip()
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "anonymous"


def check_rate_limit(bucket: str, max_requests: int | None = None,
                     window_seconds: int | None = None):
    """FastAPI dependency enforcing a sliding-window limit on a route.

    No-ops when ``settings.rate_limit_enabled`` is False (default), so the offline
    unit suite is never throttled. Raises ``HTTPException(429)`` with a ``Retry-After``
    header once the window is exceeded.
    """
    d_max, d_win = DEFAULT_LIMITS.get(bucket, (60, 60))
    max_requests = d_max if max_requests is None else int(max_requests)
    window_seconds = d_win if window_seconds is None else int(window_seconds)

    def _dep(request: Request) -> None:
        if not getattr(settings, "rate_limit_enabled", False):
            return
        allowed, retry_after = limiter.allow(_identity(request), bucket,
                                             max_requests, window_seconds)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"rate limit exceeded for {bucket}; retry in {retry_after}s",
                headers={"Retry-After": str(retry_after)},
            )

    return _dep
