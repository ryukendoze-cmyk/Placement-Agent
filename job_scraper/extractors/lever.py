from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor, PAN_INDIAN

logger = logging.getLogger(__name__)

LEVER_API_BASE = "https://api.lever.co/v0/postings"

DEFAULT_LEVER_COMPANIES: list[dict[str, str]] = [
    {"company": "Zeta", "slug": "zeta"},
    {"company": "Ro", "slug": "ro"},
    {"company": "Wealthfront", "slug": "wealthfront"},
    {"company": "AngelList (Wellfound)", "slug": "angellist"},
    {"company": "Tala", "slug": "tala"},
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


def _load_lever_companies() -> list[dict[str, str]]:
    paths = [
        Path("data/ats_companies.json"),
        Path(__file__).resolve().parents[2] / "data" / "ats_companies.json",
    ]
    for p in paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    lever_comps = [
                        {"company": c.get("company", c.get("slug")), "slug": c["slug"]}
                        for c in data
                        if isinstance(c, dict) and c.get("source") == "lever" and c.get("slug")
                    ]
                    if lever_comps:
                        return lever_comps
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", p, exc)
    return DEFAULT_LEVER_COMPANIES


class LeverExtractor(BaseExtractor):
    """Direct-ATS job harvester for companies using Lever.

    Fetches public JSON career feeds straight from Lever's Postings API:
        GET https://api.lever.co/v0/postings/{slug}?mode=json
    """

    portal_name: str = "lever"
    site: str = "lever"
    scraper_engine: str = "lever_harvester"

    def __init__(self, session: Any = None, companies: Optional[list[dict[str, str]]] = None) -> None:
        super().__init__(session=session)
        self.companies = companies or _load_lever_companies()

    async def _fetch_company_board(
        self, client: httpx.AsyncClient, company_info: dict[str, str]
    ) -> list[dict[str, Any]]:
        slug = company_info["slug"]
        url = f"{LEVER_API_BASE}/{slug}?mode=json"
        try:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else []
            return []
        except Exception as exc:
            logger.debug("Lever fetch failed for %s: %s", slug, exc)
            return []

    def _matches_location(self, raw_loc: str, is_remote: bool, wanted_loc: str) -> bool:
        low_wanted = (wanted_loc or "").strip().lower()
        loc_low = (raw_loc or "").strip().lower()

        in_india = any(k in loc_low for k in INDIA_KEYWORDS)

        if not low_wanted or low_wanted in PAN_INDIAN:
            if in_india:
                return True
            if is_remote or "remote" in loc_low:
                if any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS):
                    return False
                return True
            return False

        if low_wanted == "remote":
            return (is_remote or "remote" in loc_low) and not any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS)

        variants = self.location_variants(low_wanted)
        return any(v in loc_low for v in variants)

    def _matches_keyword(self, raw_job: dict[str, Any], keyword: str) -> bool:
        if not keyword or not keyword.strip():
            return True
        tokens = [k.strip().lower() for k in keyword.split() if len(k.strip()) > 1]
        if not tokens:
            return True

        title = (raw_job.get("text") or "").lower()
        desc = (raw_job.get("descriptionPlain") or "").lower()
        cats = raw_job.get("categories") or {}
        dept = (cats.get("department") or "").lower()
        team = (cats.get("team") or "").lower()

        search_corpus = f"{title} {desc} {dept} {team}"
        return any(t in search_corpus for t in tokens)

    def _parse_compensation(self, raw_job: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
        sr = raw_job.get("salaryRange")
        if not isinstance(sr, dict):
            return None, None, "lpa"

        min_val = sr.get("min")
        max_val = sr.get("max")
        currency = (sr.get("currency") or "INR").upper()

        if min_val is None and max_val is None:
            return None, None, "lpa"

        lo = float(min_val) if min_val is not None else None
        hi = float(max_val) if max_val is not None else None

        if currency == "INR":
            if lo and lo >= 100_000:
                lo = round(lo / 100_000, 2)
            if hi and hi >= 100_000:
                hi = round(hi / 100_000, 2)
            return lo, hi, "lpa"

        return lo, hi, "annual"

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
                        created_at = rj.get("createdAt")
                        iso_date = self.ms_to_iso(created_at)
                        if not self.is_fresh(iso_date, freshness_days):
                            continue

                        cats = rj.get("categories") or {}
                        loc_str = cats.get("location") or "India"
                        is_remote = rj.get("workplaceType") == "remote" or "remote" in loc_str.lower()
                        if not self._matches_location(loc_str, is_remote, location):
                            continue

                        if not self._matches_keyword(rj, keyword):
                            continue

                        job_url = rj.get("hostedUrl") or rj.get("applyUrl")
                        if not job_url:
                            continue

                        desc = rj.get("descriptionPlain") or ""
                        min_amt, max_amt, interval = self._parse_compensation(rj)

                        tags = []
                        if cats.get("department"):
                            tags.append(cats["department"])
                        if cats.get("team"):
                            tags.append(cats["team"])
                        if cats.get("commitment"):
                            tags.append(cats["commitment"])

                        job = self.make_job(
                            title=rj.get("text") or "Software Engineer",
                            company=comp["company"],
                            location=loc_str,
                            description=desc,
                            job_url=job_url,
                            date_posted=iso_date,
                            min_amount=min_amt,
                            max_amount=max_amt,
                            interval=interval,
                            is_remote=is_remote,
                            tags=tags,
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
