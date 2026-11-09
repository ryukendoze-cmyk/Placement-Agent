from __future__ import annotations

import asyncio
import html
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor, PAN_INDIAN

logger = logging.getLogger(__name__)

GREENHOUSE_API_BASE = "https://boards-api.greenhouse.io/v1/boards"

DEFAULT_GREENHOUSE_COMPANIES: list[dict[str, str]] = [
    {"company": "GitLab", "slug": "gitlab"},
    {"company": "Figma", "slug": "figma"},
    {"company": "Stripe", "slug": "stripe"},
    {"company": "Reddit", "slug": "reddit"},
    {"company": "Twilio", "slug": "twilio"},
    {"company": "Automattic", "slug": "automattic"},
]

RESTRICTED_NON_INDIA_PATTERNS = (
    "united states",
    "us only",
    "u.s. only",
    "canada",
    "europe",
    "uk",
    "germany",
    "france",
    "australia",
)

INDIA_KEYWORDS = (
    "india",
    "bengaluru",
    "bangalore",
    "blore",
    "hyderabad",
    "pune",
    "delhi",
    "gurgaon",
    "gurugram",
    "noida",
    "mumbai",
    "chennai",
    "kolkata",
    "ahmedabad",
)


def _load_greenhouse_companies() -> list[dict[str, str]]:
    paths = [
        Path("data/ats_companies.json"),
        Path(__file__).resolve().parents[2] / "data" / "ats_companies.json",
    ]
    for p in paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    gh_comps = [
                        {"company": c.get("company", c.get("slug")), "slug": c["slug"]}
                        for c in data
                        if isinstance(c, dict) and c.get("source") == "greenhouse" and c.get("slug")
                    ]
                    if gh_comps:
                        return gh_comps
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", p, exc)
    return DEFAULT_GREENHOUSE_COMPANIES


class GreenhouseExtractor(BaseExtractor):
    """Direct-ATS job harvester for companies using Greenhouse.

    Fetches public JSON career feeds straight from Greenhouse's Board API:
        GET https://boards-api.greenhouse.io/v1/boards/{slug}/jobs?content=true
    """

    portal_name: str = "greenhouse"
    site: str = "greenhouse"
    scraper_engine: str = "greenhouse_harvester"

    def __init__(self, session: Any = None, companies: Optional[list[dict[str, str]]] = None) -> None:
        super().__init__(session=session)
        self.companies = companies or _load_greenhouse_companies()

    async def _fetch_company_board(
        self, client: httpx.AsyncClient, company_info: dict[str, str]
    ) -> list[dict[str, Any]]:
        slug = company_info["slug"]
        url = f"{GREENHOUSE_API_BASE}/{slug}/jobs?content=true"
        try:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", [])
                return jobs if isinstance(jobs, list) else []
            return []
        except Exception as exc:
            logger.debug("Greenhouse fetch failed for %s: %s", slug, exc)
            return []

    def _matches_location(self, raw_loc: str, wanted_loc: str) -> bool:
        low_wanted = (wanted_loc or "").strip().lower()
        loc_low = (raw_loc or "").strip().lower()

        in_india = any(k in loc_low for k in INDIA_KEYWORDS)

        if not low_wanted or low_wanted in PAN_INDIAN:
            if in_india:
                return True
            if "remote" in loc_low:
                if any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS):
                    return False
                return True
            return False

        if low_wanted == "remote":
            return "remote" in loc_low and not any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS)

        variants = self.location_variants(low_wanted)
        return any(v in loc_low for v in variants)

    def _matches_keyword(self, raw_job: dict[str, Any], keyword: str) -> bool:
        if not keyword or not keyword.strip():
            return True
        tokens = [k.strip().lower() for k in keyword.split() if len(k.strip()) > 1]
        if not tokens:
            return True

        title = (raw_job.get("title") or "").lower()
        desc = (raw_job.get("content") or "").lower()
        search_corpus = f"{title} {desc}"
        return any(t in search_corpus for t in tokens)

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        collected: list[RawScrapedJob] = []
        sem = asyncio.Semaphore(10)

        async with httpx.AsyncClient(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/json",
            },
            follow_redirects=True,
        ) as client:

            async def _process_company(comp: dict[str, str]) -> list[RawScrapedJob]:
                async with sem:
                    raw_jobs = await self._fetch_company_board(client, comp)
                    results: list[RawScrapedJob] = []
                    for rj in raw_jobs:
                        pub_at = rj.get("first_published") or rj.get("updated_at")
                        if not self.is_fresh(pub_at, freshness_days):
                            continue

                        loc_obj = rj.get("location") or {}
                        loc_str = loc_obj.get("name") if isinstance(loc_obj, dict) else str(loc_obj or "India")
                        if not self._matches_location(loc_str, location):
                            continue

                        if not self._matches_keyword(rj, keyword):
                            continue

                        job_url = rj.get("absolute_url")
                        if not job_url:
                            continue

                        raw_content = rj.get("content") or ""
                        decoded_content = html.unescape(raw_content)
                        clean_desc = self.html_to_text(decoded_content)

                        departments = [d.get("name") for d in (rj.get("departments") or []) if isinstance(d, dict) and d.get("name")]

                        job = self.make_job(
                            title=rj.get("title") or "Software Engineer",
                            company=comp["company"],
                            location=loc_str,
                            description=clean_desc,
                            job_url=job_url,
                            date_posted=pub_at,
                            is_remote="remote" in (loc_str or "").lower(),
                            tags=departments,
                            site=self.site,
                        )
                        if job:
                            results.append(job)
                    return results

            tasks = [_process_company(c) for c in self.companies]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for b in batch_results:
                if isinstance(b, list):
                    collected.extend(b)
                if len(collected) >= limit:
                    break

        return collected[:limit]
