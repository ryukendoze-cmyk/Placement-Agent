from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional
from urllib.parse import quote_plus

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

SEARCH_API = "https://www.instahyre.com/api/v1/job_search"
SEARCH_HOME = "https://www.instahyre.com/search-jobs/"
DETAIL_PACE_SECONDS = 4.0

# Instahyre page noise to cut after the job description.
_STOP_MARKERS = (
    "similar jobs", "apply now", "salary & benefits", "job description",
    "company overview", "interview process", "about the company",
)


def _skills(obj: dict[str, Any]) -> list[str]:
    kws = obj.get("keywords") or []
    out = [k for k in kws if isinstance(k, str)]
    return list(dict.fromkeys(out))[:20]


def parse_instahyre_job(obj: dict[str, Any]) -> Optional[RawScrapedJob]:
    title = (obj.get("title") or "").strip()
    company = ((obj.get("employer") or {}).get("company_name") or "").strip()
    url = obj.get("public_url") or ""
    if not url.startswith("http"):
        return None
    locations = obj.get("locations")
    if isinstance(locations, list):
        loc = ", ".join(str(x) for x in locations if x) or "India"
    else:
        loc = str(locations or "India").strip() or "India"

    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=loc,
        description="",  # filled from SSR detail page in search()
        job_url=url,
        date_posted=obj.get("updated_at") or obj.get("created_at"),
        is_remote="remote" in loc.lower(),
        tags=_skills(obj),
        site="instahyre",
    )


def parse_instahyre_payload(objects: list[dict[str, Any]]) -> list[RawScrapedJob]:
    jobs = []
    for obj in objects or []:
        try:
            job = parse_instahyre_job(obj)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("instahyre parse error: %s", exc)
    return jobs


def extract_instahyre_description(html: str) -> str:
    """Pull the SSR #job-description block and trim footer noise."""
    if not html:
        return ""
    m = re.search(r'<div[^>]*id="job-description"[^>]*>(.*?)</div>\s*<', html, re.S)
    block = m.group(1) if m else html
    text = BaseExtractor.html_to_text(block)
    lowered = text.lower()
    cut = len(text)
    for marker in _STOP_MARKERS:
        idx = lowered.find(marker, 200)  # skip the first occurrences in headings
        if idx != -1 and idx < cut:
            cut = idx
    text = text[:cut].strip()
    return text


class InstahyreExtractor(BaseExtractor):
    """Instahyre startup job API + SSR description pages."""

    portal_name = "instahyre"
    site = "instahyre"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("InstahyreExtractor requires a StealthSession")

        # Warm the session cookie jar on the search homepage first.
        try:
            await self.session.get(SEARCH_HOME, timeout=30)
        except Exception as exc:
            logger.debug("instahyre warmup failed: %s", exc)

        results: list[RawScrapedJob] = []
        offset = 0
        page_size = 35
        while len(results) < limit:
            city = location
            if city.lower() in {"india", "pan india", "pan-india", "remote", ""}:
                city = "Bangalore"  # API requires a concrete location; filter client-side
            url = (
                f"{SEARCH_API}?company_size=0&job_type=0&offset={offset}"
                f"&jobLocations={quote_plus(city)}&skills={quote_plus(keyword)}"
            )
            resp = await self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning("instahyre offset %d HTTP %d", offset, resp.status_code)
                break
            try:
                payload = resp.json()
            except Exception as exc:
                logger.warning("instahyre offset %d bad JSON: %s", offset, exc)
                break
            objects = payload.get("objects") or payload.get("data") or []
            page_jobs = parse_instahyre_payload(objects)
            if not page_jobs:
                break
            results.extend(page_jobs)
            offset += len(objects)
            if len(objects) < page_size:
                break

        filtered = [j for j in results if self.location_match(j.location, location)]
        if location.lower() == "remote":
            remote = [j for j in filtered if j.is_remote]
            if remote:
                filtered = remote
        filtered = filtered[:limit]

        # Enrich with full descriptions from the SSR job pages (paced).
        for job in filtered:
            try:
                await asyncio.sleep(DETAIL_PACE_SECONDS)
                dresp = await self.session.get(
                    job.job_url, headers={"Referer": SEARCH_HOME}, timeout=40
                )
                if dresp.status_code == 200:
                    desc = extract_instahyre_description(dresp.text)
                    if desc:
                        job.description = desc
            except Exception as exc:
                logger.debug("instahyre detail error for %s: %s", job.job_url, exc)

        return filtered
