from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import httpx

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import PAN_INDIAN, BaseExtractor

logger = logging.getLogger(__name__)

ASHBY_API_BASE = "https://api.ashbyhq.com/posting-api/job-board"

# Curated fallback list of prominent tech companies on Ashby
DEFAULT_ASHBY_COMPANIES: list[dict[str, str]] = [
    {"company": "Ashby", "slug": "Ashby"},
    {"company": "OpenAI", "slug": "openai"},
    {"company": "Notion", "slug": "Notion"},
    {"company": "Ramp", "slug": "ramp"},
    {"company": "Linear", "slug": "linear"},
    {"company": "PostHog", "slug": "posthog"},
    {"company": "Cursor", "slug": "cursor"},
    {"company": "Vanta", "slug": "vanta"},
    {"company": "Snowflake", "slug": "snowflake"},
    {"company": "Deel", "slug": "deel"},
    {"company": "Retool", "slug": "retool"},
    {"company": "Sentry", "slug": "sentry"},
    {"company": "Replit", "slug": "replit"},
    {"company": "Temporal", "slug": "temporal"},
    {"company": "Superhuman", "slug": "superhuman"},
]

RESTRICTED_NON_INDIA_PATTERNS = (
    "- us",
    "- usa",
    "- united states",
    "- canada",
    "- european union",
    "- eu",
    "- uk",
    "- emea",
    "- namer",
    "- latam",
    "us only",
    "u.s. only",
    "united states only",
    "north america only",
    "canada only",
    "uk only",
    "europe only",
    "emea only",
    "latam only",
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


def _load_ashby_companies() -> list[dict[str, str]]:
    """Load Ashby companies from data/ats_companies.json or fallback list."""
    paths = [
        Path("data/ats_companies.json"),
        Path(__file__).resolve().parents[2] / "data" / "ats_companies.json",
    ]
    for p in paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    ashby_comps = [
                        {"company": c.get("company", c.get("slug")), "slug": c["slug"]}
                        for c in data
                        if isinstance(c, dict) and c.get("source") == "ashby" and c.get("slug")
                    ]
                    if ashby_comps:
                        return ashby_comps
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", p, exc)
    return DEFAULT_ASHBY_COMPANIES


class AshbyExtractor(BaseExtractor):
    """Direct-ATS job harvester for companies using Ashby.

    Fetches public JSON career feeds straight from Ashby's Posting API:
        GET https://api.ashbyhq.com/posting-api/job-board/{slug}?includeCompensation=true

    Applies in-memory targeted filtering for:
      - Candidate keyword matching (title, department, description, tags)
      - Location matching (Indian tech hubs + global remote roles)
      - Freshness filtering
      - Base compensation extraction & normalization
    """

    portal_name: str = "ashby"
    site: str = "ashby"
    scraper_engine: str = "ashby_harvester"

    def __init__(self, session: Any = None, companies: Optional[list[dict[str, str]]] = None) -> None:
        super().__init__(session=session)
        self.companies = companies or _load_ashby_companies()

    async def _fetch_company_board(
        self, client: httpx.AsyncClient, company_info: dict[str, str]
    ) -> list[dict[str, Any]]:
        slug = company_info["slug"]
        url = f"{ASHBY_API_BASE}/{slug}?includeCompensation=true"
        try:
            resp = await client.get(url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                jobs = data.get("jobs", [])
                return jobs if isinstance(jobs, list) else []
            logger.debug("Ashby company %s returned status %d", slug, resp.status_code)
            return []
        except Exception as exc:
            logger.debug("Ashby fetch failed for %s: %s", slug, exc)
            return []

    def _matches_location(self, raw_loc: str, is_remote: bool, wanted_loc: str) -> bool:
        low_wanted = (wanted_loc or "").strip().lower()
        loc_low = (raw_loc or "").strip().lower()

        # Check explicit India presence first
        in_india = any(k in loc_low for k in INDIA_KEYWORDS)

        if not low_wanted or low_wanted in PAN_INDIAN:
            if in_india:
                return True
            if is_remote or "remote" in loc_low:
                # Reject remote roles restricted to non-India regions
                if any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS):
                    return False
                # Accept if global/worldwide/anywhere/apac or simply Remote without foreign country tag
                is_global = any(g in loc_low for g in ("global", "worldwide", "anywhere", "apac", "asia"))
                return is_global or not any(c in loc_low for c in ("us", "canada", "europe", "france", "germany", "australia"))
            return False

        # City-specific or custom location search (e.g. "Bengaluru")
        if low_wanted == "remote":
            return (is_remote or "remote" in loc_low) and not any(r in loc_low for r in RESTRICTED_NON_INDIA_PATTERNS)

        variants = self.location_variants(low_wanted)
        if any(v in loc_low for v in variants):
            return True

        if is_remote and (any(g in loc_low for g in ("global", "worldwide", "anywhere")) or loc_low == "remote"):
            return True

        return False

    def _matches_keyword(self, raw_job: dict[str, Any], keyword: str) -> bool:
        if not keyword or not keyword.strip():
            return True
        kw_tokens = [k.strip().lower() for k in keyword.split() if len(k.strip()) > 1]
        if not kw_tokens:
            return True

        title = (raw_job.get("title") or "").lower()
        dept = (raw_job.get("department") or "").lower()
        team = (raw_job.get("team") or "").lower()
        desc = (raw_job.get("descriptionPlain") or "").lower()

        search_corpus = f"{title} {dept} {team} {desc}"
        return any(token in search_corpus for token in kw_tokens)

    def _parse_compensation(self, raw_job: dict[str, Any]) -> tuple[Optional[float], Optional[float], str]:
        """Extract min/max compensation and normalize interval to lpa if available."""
        comp = raw_job.get("compensation") or {}
        components = comp.get("summaryComponents") or []
        salary_comp = next((c for c in components if c.get("compensationType") == "Salary"), None)

        if not salary_comp and comp.get("compensationTiers"):
            for tier in comp["compensationTiers"]:
                for c in tier.get("components", []):
                    if c.get("compensationType") == "Salary":
                        salary_comp = c
                        break
                if salary_comp:
                    break

        if not salary_comp:
            return None, None, "lpa"

        min_val = salary_comp.get("minValue")
        max_val = salary_comp.get("maxValue")
        currency = (salary_comp.get("currencyCode") or "INR").upper()

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
        elif currency == "USD":
            # 1 USD ≈ 86.5 INR. Convert USD annual salary to INR LPA:
            # e.g. $207,000 USD * 86.5 / 100,000 = ~179 LPA
            USD_TO_INR = 86.5
            if lo:
                raw_lo = lo * 1000 if lo < 1000 else lo
                lo = round(raw_lo * USD_TO_INR / 100_000, 2)
            if hi:
                raw_hi = hi * 1000 if hi < 1000 else hi
                hi = round(raw_hi * USD_TO_INR / 100_000, 2)
            return lo, hi, "lpa"
        elif currency in ("EUR", "GBP"):
            FX = 92.0 if currency == "EUR" else 108.0
            if lo:
                raw_lo = lo * 1000 if lo < 1000 else lo
                lo = round(raw_lo * FX / 100_000, 2)
            if hi:
                raw_hi = hi * 1000 if hi < 1000 else hi
                hi = round(raw_hi * FX / 100_000, 2)
            return lo, hi, "lpa"

        return lo, hi, "annual"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        """Fetch and filter open jobs across Ashby company boards."""
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
                        if rj.get("isListed") is False:
                            continue

                        pub_at = rj.get("publishedAt")
                        if not self.is_fresh(pub_at, freshness_days):
                            continue

                        loc_str = rj.get("location") or "India"
                        is_remote = bool(rj.get("isRemote"))
                        if not self._matches_location(loc_str, is_remote, location):
                            continue

                        if not self._matches_keyword(rj, keyword):
                            continue

                        job_url = rj.get("jobUrl") or rj.get("applyUrl")
                        if not job_url:
                            continue

                        desc_plain = rj.get("descriptionPlain")
                        if not desc_plain and rj.get("descriptionHtml"):
                            desc_plain = self.html_to_text(rj["descriptionHtml"])
                        desc = desc_plain or ""

                        min_amt, max_amt, interval = self._parse_compensation(rj)

                        tags = []
                        if rj.get("department"):
                            tags.append(rj["department"])
                        if rj.get("team"):
                            tags.append(rj["team"])
                        if rj.get("employmentType"):
                            tags.append(rj["employmentType"])

                        job = self.make_job(
                            title=rj.get("title") or "Software Engineer",
                            company=comp["company"],
                            location=loc_str,
                            description=desc,
                            job_url=job_url,
                            date_posted=pub_at,
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

            # Concurrently process companies
            tasks = [_process_company(c) for c in self.companies]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for b in batch_results:
                if isinstance(b, list):
                    collected.extend(b)
                if len(collected) >= limit:
                    break

        return collected[:limit]
