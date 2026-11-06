"""Live job scraping service utilizing python-jobspy for Indian tech job boards.

Also hosts :class:`CompositeJobScraper`, the Milestone 6 multi-provider aggregator
(ARCHITECTURE §12.2 B) that dispatches one search across every registered provider
with per-provider error isolation, then merges the results through a unified
cross-provider deduplication pass (canonical URL first, soft title+company second,
richer-description / verified-URL collision resolution).
"""

import logging
import math
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import pandas as pd
from jobspy import scrape_jobs

from app.scraper.providers import (
    AutomationJobScraperProvider,
    BaseJobScraperProvider,
    JobSpyProvider,
    ProxyAPIJobScraperProvider,
    RapidJobScraperProvider,
    StealthHarvesterProvider,
)

logger = logging.getLogger(__name__)

SUPPORTED_SITES = {"naukri", "linkedin", "indeed"}


def _sanitize_value(val: Any) -> Any:
    """Sanitize float NaN, pd.NA, and pd.isna values to clean Python primitives."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_value(v) for v in val]
    return val


def _sanitize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sanitize all dictionary records converted from a pandas DataFrame."""
    sanitized: list[dict[str, Any]] = []
    for rec in records:
        clean_rec = {k: _sanitize_value(v) for k, v in rec.items()}
        sanitized.append(clean_rec)
    return sanitized


class JobSpyScraper:
    """Resilient scraper wrapping python-jobspy with platform validation and error containment."""

    def fetch_jobs(
        self,
        search_term: str | list[str] = "Software Engineer",
        location: str = "Bengaluru",
        sites: list[str] | None = None,
        results_wanted: int = 20,
        hours_old: int = 168,
        country_indeed: str = "India",
        fetch_description: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch job postings from live platforms with resilient error handling and NaN sanitization.

        Args:
            search_term: Single keyword/role title or list of targeted query strings.
            location: City or location in India.
            sites: Target job boards (must be subset of 'naukri', 'linkedin', 'indeed').
            results_wanted: Maximum number of results requested.
            hours_old: Filter to jobs posted within this timeframe in hours.
            country_indeed: Target country for Indeed queries.
            fetch_description: Whether to fetch full job descriptions from detail pages.

        Returns:
            List of sanitized job dictionaries or an empty list if scraping fails.
        """
        if not sites:
            normalized_sites = ["naukri", "linkedin", "indeed"]
        else:
            normalized_sites = []
            for s in sites:
                cleaned = str(s).strip().lower()
                if cleaned not in SUPPORTED_SITES:
                    raise ValueError(
                        f"Unsupported site: {s!r}. Supported sites: {sorted(SUPPORTED_SITES)}"
                    )
                normalized_sites.append(cleaned)

        if isinstance(search_term, list):
            terms = [str(t).strip() for t in search_term if str(t).strip()]
            if not terms:
                terms = ["Software Engineer"]
        else:
            terms = [str(search_term).strip() or "Software Engineer"]

        if len(terms) == 1:
            try:
                df = scrape_jobs(
                    site_name=normalized_sites,
                    search_term=terms[0],
                    location=location,
                    results_wanted=results_wanted,
                    hours_old=hours_old,
                    country_indeed=country_indeed,
                    fetch_description=fetch_description,
                )
            except Exception as exc:
                logger.warning(
                    "JobSpy scraping failed for search_term=%r, location=%r, sites=%r: %s",
                    terms[0],
                    location,
                    normalized_sites,
                    exc,
                )
                return []

            if df is None or not hasattr(df, "empty") or df.empty:
                return []

            try:
                raw_records = df.to_dict(orient="records")
                return _sanitize_records(raw_records)
            except Exception as exc:
                logger.warning("Failed to serialize scraped DataFrame to records: %s", exc)
                return []

        # Multiple targeted keywords: scrape each keyword and merge deduplicated records
        per_term_results = max(5, round(results_wanted / len(terms)))
        combined_records: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        for term in terms:
            try:
                df = scrape_jobs(
                    site_name=normalized_sites,
                    search_term=term,
                    location=location,
                    results_wanted=per_term_results,
                    hours_old=hours_old,
                    country_indeed=country_indeed,
                    fetch_description=fetch_description,
                )
            except Exception as exc:
                logger.warning(
                    "JobSpy scraping failed for search_term=%r, location=%r, sites=%r: %s",
                    term,
                    location,
                    normalized_sites,
                    exc,
                )
                continue

            if df is None or not hasattr(df, "empty") or df.empty:
                continue

            try:
                recs = df.to_dict(orient="records")
                for r in recs:
                    url = str(r.get("job_url") or r.get("url") or "").strip()
                    title = str(r.get("title") or "").strip().lower()
                    company = str(r.get("company") or "").strip().lower()
                    dedup_key = url if url else f"{title}::{company}"
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        combined_records.append(r)
            except Exception as exc:
                logger.warning("Failed to serialize scraped DataFrame for term %r: %s", term, exc)

        return _sanitize_records(combined_records)


# ---------------------------------------------------------------------------
# Multi-provider aggregation (Milestone 6, ARCHITECTURE §12.2 B)
# ---------------------------------------------------------------------------

#: Query parameters stripped when canonicalizing URLs for cross-provider dedup.
_TRACKING_PARAM_PREFIXES: tuple[str, ...] = ("utm_",)
_TRACKING_PARAMS: frozenset[str] = frozenset(
    {
        "ref",
        "ref_",
        "referrer",
        "source",
        "src",
        "trk",
        "trackingid",
        "tracking_id",
        "gclid",
        "fbclid",
        "utm",
    }
)


def _clean_url(url: str) -> str:
    """Canonicalize a job URL for deduplication.

    Strips tracking query params (``utm_*``, ``ref``, ``gclid``, ...), lowercases
    the host, drops fragments, and removes a trailing slash from the path.
    """
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.rstrip("/")
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not any(k.lower().startswith(p) for p in _TRACKING_PARAM_PREFIXES)
        and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def _clean_key(title: str, company: str) -> tuple[str, str]:
    """Normalize ``(title, company)`` for soft dedup: lowercase, alphanumeric only."""
    return (
        re.sub(r"[^a-z0-9]", "", str(title or "").lower()),
        re.sub(r"[^a-z0-9]", "", str(company or "").lower()),
    )


def _is_richer(new_job: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Decide whether ``new_job`` should replace ``existing`` on a dedup collision.

    Prefers the entry with the richer (longer) description; ties are broken by
    preferring a verified non-empty ``job_url``.
    """
    new_desc = len(str(new_job.get("description") or ""))
    old_desc = len(str(existing.get("description") or ""))
    if new_desc != old_desc:
        return new_desc > old_desc
    new_url = str(new_job.get("job_url") or "").strip()
    old_url = str(existing.get("job_url") or "").strip()
    if bool(new_url) != bool(old_url):
        return bool(new_url)
    return False


class CompositeJobScraper:
    """Multi-provider aggregator dispatching one search across all registered scrapers.

    Aggregates raw postings from every selected provider with isolated exception
    handling (one provider's failure never blocks the others), then applies a
    unified cross-provider deduplication pass:

    1. Exact canonical ``job_url`` match (tracking params stripped).
    2. Soft ``(title, company)`` match after ``re.sub(r'[^a-z0-9]', '', ...)``.
    3. Collisions resolved in favour of the richer description and a verified
       non-empty ``job_url``.

    Args:
        providers: Provider instances to register; ``None`` registers the five
            default providers (jobspy, rapid, automation, proxy_api,
            stealth_harvester).
    """

    def __init__(self, providers: list[BaseJobScraperProvider] | None = None) -> None:
        if providers is None:
            providers = [
                JobSpyProvider(),
                RapidJobScraperProvider(),
                AutomationJobScraperProvider(),
                ProxyAPIJobScraperProvider(),
                StealthHarvesterProvider(),
            ]
        self.providers: dict[str, BaseJobScraperProvider] = {p.name: p for p in providers}

    def get_available_providers(self) -> list[dict[str, Any]]:
        """Return metadata (name, display_name, supported_sites) for all registered providers."""
        return [
            {"name": p.name, "display_name": p.display_name, "supported_sites": list(p.supported_sites)}
            for p in self.providers.values()
        ]

    def fetch_jobs(
        self,
        search_term: str | list[str],
        location: str = "India",
        results_wanted: int = 15,
        providers: list[str] | None = None,
        sites: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Dispatch a search across selected providers and return merged, deduplicated jobs.

        Args:
            search_term: Single query or list of targeted query strings.
            location: Location passed through to every provider.
            results_wanted: Per-provider result budget.
            providers: Provider names to run; ``None`` runs all registered providers.
            sites: Site/platform filter forwarded to each provider when supported.
            **kwargs: Extra provider-specific parameters (``hours_old``, ...).

        Returns:
            Aggregated, cross-provider-deduplicated list of raw job dicts. Any
            provider that raises is logged and skipped without affecting others.
        """
        if providers is None:
            target_providers = list(self.providers.values())
        else:
            wanted = {str(name).strip().lower() for name in providers}
            target_providers = [p for name, p in self.providers.items() if name in wanted]
            unknown = wanted - set(self.providers)
            if unknown:
                logger.warning("Unknown provider(s) requested: %s", sorted(unknown))

        all_jobs: list[dict[str, Any]] = []
        for p in target_providers:
            try:
                jobs = p.fetch_jobs(
                    search_term=search_term,
                    location=location,
                    results_wanted=results_wanted,
                    sites=sites,
                    **kwargs,
                )
                if jobs:
                    all_jobs.extend(jobs)
            except Exception as exc:  # noqa: BLE001 - isolate each provider
                logger.warning("Provider %s failed during fetch_jobs: %s", p.name, exc)

        return _deduplicate_jobs(all_jobs)

    def fetch_jobs_streaming(
        self,
        search_term: str | list[str],
        location: str = "India",
        results_wanted: int = 15,
        providers: list[str] | None = None,
        sites: list[str] | None = None,
        on_event: Any = None,
        portals: list[str] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute providers with per-portal lifecycle telemetry (ARCHITECTURE §16.2.B).

        Runs each target provider in sequence, emitting structured telemetry
        events through ``on_event(name, data)`` as the run progresses, then
        returns the aggregated, cross-provider-deduplicated job list (identical
        semantics to :meth:`fetch_jobs`).

        Event contract:

        - ``init`` → ``{"portals": [...], "total": int}`` — target unit list
          before any work starts.
        - ``portal_start`` → ``{"portal": str, "timestamp": ISO-8601,
          "status": "running"}``.
        - ``portal_progress`` → ``{"portal": str, "jobs_found": int,
          "elapsed_s": float}``.
        - ``portal_done`` → ``{"portal": str, "status": "ok", "jobs": int,
          "duration_s": float}``.
        - ``portal_error`` → ``{"portal": str, "status": "warning",
          "message": str, "duration_s": float}``.
        - ``complete`` → ``{"total_jobs": int, "new_ingested": 0,
          "elapsed_total_s": float}`` (``new_ingested`` is filled by the caller
          that performs the database ingest).

        Streaming units are portals when ``portals`` is supplied (or the
        stealth harvester is among the selected providers — its ``supported_sites``
        become the unit list), otherwise provider names. The ``portals`` value
        is forwarded to the stealth harvester as its portal scope.

        Zero-crash guarantee: an exception in any single portal/provider is
        recorded as a ``portal_error`` event and the stream continues.

        Args:
            search_term: Single query or list of targeted query strings.
            location: Location passed through to every provider.
            results_wanted: Per-provider result budget.
            providers: Provider names to run; ``None`` runs all registered.
            sites: Site/platform filter forwarded to providers that support it.
            on_event: Callback ``on_event(name: str, data: dict) -> None``. Any
                exception raised by the callback is swallowed (telemetry must
                never break the scrape).
            portals: Explicit portal scope for streaming telemetry (forwards
                to the stealth harvester's live run).
            **kwargs: Extra provider-specific parameters (``hours_old``,
                ``live``, ...).

        Returns:
            Aggregated, deduplicated list of raw job dicts.
        """
        if providers is None:
            target_providers = list(self.providers.values())
        else:
            wanted = {str(name).strip().lower() for name in providers}
            target_providers = [p for name, p in self.providers.items() if name in wanted]
            unknown = wanted - set(self.providers)
            if unknown:
                logger.warning("Unknown provider(s) requested: %s", sorted(unknown))

        def emit(name: str, data: dict[str, Any]) -> None:
            if on_event is None:
                return
            try:
                on_event(name, data)
            except Exception as exc:  # noqa: BLE001 - telemetry must never break the scrape
                logger.warning("streaming on_event(%s) failed: %s", name, exc)

        def now_iso() -> str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Streaming units: explicit portals, else the stealth harvester's 8
        # portals when it participates, else provider names.
        if portals is None and any(
            getattr(p, "name", "") == "stealth_harvester" for p in target_providers
        ):
            stealth = next(
                p for p in target_providers if getattr(p, "name", "") == "stealth_harvester"
            )
            unit_portals: list[str] | None = list(stealth.supported_sites)
        else:
            unit_portals = list(portals) if portals else None

        if unit_portals is not None:
            units = unit_portals
            kwargs["portals"] = unit_portals
        else:
            units = [str(getattr(p, "name", f"provider-{i}")) for i, p in enumerate(target_providers)]

        total_start = time.perf_counter()
        emit("init", {"portals": units, "total": len(units)})

        all_jobs: list[dict[str, Any]] = []
        if unit_portals is not None:
            for unit in units:
                target = [
                    p for p in target_providers
                    if unit in getattr(p, "supported_sites", []) or getattr(p, "name", "") == "stealth_harvester"
                ]
                if not target:
                    target = target_providers
                self._stream_unit(
                    unit, target, search_term, location, results_wanted, sites, kwargs, emit, all_jobs, is_portal=True
                )
        else:
            for unit, p in zip(units, target_providers):
                self._stream_unit(
                    unit, [p], search_term, location, results_wanted, sites, kwargs, emit, all_jobs, is_portal=False
                )

        deduped = _deduplicate_jobs(all_jobs)
        emit(
            "complete",
            {
                "total_jobs": len(deduped),
                "new_ingested": 0,
                "elapsed_total_s": round(time.perf_counter() - total_start, 3),
            },
        )
        return deduped

    @staticmethod
    def _stream_unit(
        unit: str,
        providers: list[BaseJobScraperProvider],
        search_term: str | list[str],
        location: str,
        results_wanted: int,
        sites: list[str] | None,
        kwargs: dict[str, Any],
        emit: Any,
        all_jobs: list[dict[str, Any]],
        is_portal: bool = False,
    ) -> None:
        """Run one streaming unit (portal or provider) with lifecycle telemetry.

        Emits ``portal_start`` -> ``portal_progress`` -> ``portal_done`` (or
        ``portal_error`` on failure); never raises.
        """
        unit_kwargs = dict(kwargs)
        is_portal_unit = is_portal or any(unit in getattr(p, "supported_sites", []) for p in providers)
        if is_portal_unit:
            # Portal unit: scope the provider strictly to this single portal.
            # Never overwrite with a multi-site caller filter.
            unit_kwargs["portals"] = [unit]
            unit_kwargs["sites"] = [unit]
        elif sites is not None:
            unit_kwargs["sites"] = sites

        call_kwargs = dict(unit_kwargs)

        started = time.perf_counter()
        emit(
            "portal_start",
            {
                "portal": unit,
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "status": "running",
            },
        )
        jobs: list[dict[str, Any]] = []
        try:
            for p in providers:
                found = p.fetch_jobs(
                    search_term=search_term,
                    location=location,
                    results_wanted=results_wanted,
                    **call_kwargs,
                )
                if found:
                    jobs.extend(found)
            elapsed = time.perf_counter() - started
            emit(
                "portal_progress",
                {"portal": unit, "jobs_found": len(jobs), "elapsed_s": round(elapsed, 3)},
            )
            emit(
                "portal_done",
                {"portal": unit, "status": "ok", "jobs": len(jobs), "duration_s": round(elapsed, 3)},
            )
        except Exception as exc:  # noqa: BLE001 - isolate each portal/provider
            elapsed = time.perf_counter() - started
            logger.warning("Streaming unit %s failed: %s", unit, exc)
            emit(
                "portal_error",
                {"portal": unit, "status": "warning", "message": str(exc), "duration_s": round(elapsed, 3)},
            )
        if jobs:
            all_jobs.extend(jobs)


def _deduplicate_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Unified cross-provider deduplication (canonical URL, then title+company).

    Keeps insertion order; when a duplicate is detected, resolves the collision by
    keeping the richer description and preferring a verified non-empty ``job_url``.
    """
    merged: list[dict[str, Any]] = []
    seen_urls: dict[str, dict[str, Any]] = {}
    seen_keys: dict[tuple[str, str], dict[str, Any]] = {}

    for job in jobs:
        if not isinstance(job, dict):
            continue
        url = _clean_url(str(job.get("job_url") or job.get("url") or ""))
        key = _clean_key(str(job.get("title") or ""), str(job.get("company") or ""))

        # Primary: exact canonical job_url match (tracking params stripped).
        existing = seen_urls.get(url) if url else None
        # Secondary: soft (title, company) match — same normalized identity
        # collapses across providers even when the direct links differ.
        if existing is None and key != ("", ""):
            existing = seen_keys.get(key)

        if existing is None:
            merged.append(job)
            if url:
                seen_urls[url] = job
            if key != ("", ""):
                seen_keys[key] = job
            continue

        if _is_richer(job, existing):
            merged[merged.index(existing)] = job
            if url:
                seen_urls[url] = job
            if key != ("", ""):
                seen_keys[key] = job

    return merged
