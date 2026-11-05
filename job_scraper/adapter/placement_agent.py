"""Drop-in scraper provider for ps06-placement-agent.

Copy this file (or the whole job_scraper/ package) into
ps06-placement-agent and register the provider. The class emits
normalized job dicts that match the project's SQLModel `Job` entity exactly
(same fields as data/jobs.json).

Standalone usage (no placement agent installed):
    provider = AdvancedStealthScraperProvider()
    jobs = provider.scrape_jobs_sync(["Python"], location="Bengaluru", days_old=7, limit=10)
"""
from __future__ import annotations

import asyncio
from typing import Any

# When copied into ps06-placement-agent this import resolves to the real base
# class. Standalone (this repo) we fall back to a minimal ABC so the module
# stays importable and testable.
try:  # pragma: no cover - depends on host project layout
    from app.scraper.providers import BaseJobScraperProvider  # type: ignore
except Exception:  # pragma: no cover
    from abc import ABC, abstractmethod

    class BaseJobScraperProvider(ABC):  # type: ignore
        name: str = "base"

        @abstractmethod
        async def scrape_jobs(
            self,
            keywords: list[str],
            location: str = "India",
            days_old: int = 7,
            limit: int = 20,
        ) -> list[dict[str, Any]]:
            ...


class AdvancedStealthScraperProvider(BaseJobScraperProvider):
    """Stealth-harvester provider covering 8 Indian job portals.

    - TLS-impersonated HTTP (curl_cffi) + Playwright stealth for JS/CF portals
    - Per-domain rate limiting, jittered retries, error containment
    - Canonical URL dedup + LPA salary normalization
    - Zero placeholders: only live posting URLs are emitted
    """

    name = "advanced_stealth_harvester"
    display_name = "Stealth Harvester (8 Indian Portals)"
    supported_sites = [
        "naukri",
        "linkedin",
        "indeed",
        "foundit",
        "internshala",
        "unstop",
        "instahyre",
        "hirist",
    ]

    def __init__(self, portals: list[str] | None = None) -> None:
        self.portals = portals

    def fetch_jobs(
        self,
        search_term: str | list[str],
        location: str = "India",
        results_wanted: int = 15,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Unified provider interface called by CompositeJobScraper."""
        keywords = [search_term] if isinstance(search_term, str) else list(search_term)
        days_old = int(kwargs.get("days_old", kwargs.get("hours_old", 168) // 24) or 7)
        return self.scrape_jobs_sync(
            keywords=keywords,
            location=location,
            days_old=days_old,
            limit=results_wanted,
            output_dir=kwargs.get("output_dir"),
        )

    async def scrape_jobs(
        self,
        keywords: list[str],
        location: str = "India",
        days_old: int = 7,
        limit: int = 20,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run the full stealth pipeline; return normalized Job-ready dicts."""
        from job_scraper.pipeline import run_pipeline

        out = await run_pipeline(
            keywords=keywords,
            location=location,
            days_old=days_old,
            limit=limit,
            portals=self.portals,
            output_dir=output_dir or "output",
        )
        return out["normalized"]

    def scrape_jobs_sync(
        self,
        keywords: list[str],
        location: str = "India",
        days_old: int = 7,
        limit: int = 20,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous wrapper.

        Safe to call from a running event loop (FastAPI/Celery): if a loop is
        already running on this thread, the coroutine is executed on a worker
        thread with its own loop instead of failing with RuntimeError.
        """
        coro = self.scrape_jobs(
            keywords,
            location=location,
            days_old=days_old,
            limit=limit,
            output_dir=output_dir,
        )
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return asyncio.run(coro)
