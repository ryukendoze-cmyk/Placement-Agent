from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from urllib.parse import quote_plus

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

SEARCH_API = "https://gladiator.hirist.tech/job/search"
DETAIL_API = "https://gladiator.hirist.tech/job/detail"
DETAIL_PACE_SECONDS = 4.0


def _locations(job: dict[str, Any]) -> str:
    locs = job.get("locations") or []
    names = [l.get("name") for l in locs if isinstance(l, dict) and l.get("name")]
    if names:
        return ", ".join(dict.fromkeys(names))
    loc = job.get("location")
    if isinstance(loc, str) and loc.strip():
        return loc.strip()
    return "India"


def _skills(job: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for t in job.get("tags") or []:
        if isinstance(t, dict) and t.get("name"):
            out.append(str(t["name"]))
        elif isinstance(t, str):
            out.append(t)
    return list(dict.fromkeys(out))[:20]


def parse_hirist_job(job: dict[str, Any]) -> Optional[RawScrapedJob]:
    title = (job.get("title") or "").strip()
    company = ((job.get("companyData") or {}).get("companyName") or "").strip()
    url = job.get("jobDetailUrl") or ""
    if not url.startswith("http"):
        return None

    min_sal = job.get("minSal") or 0
    max_sal = job.get("maxSal") or 0
    min_sal = float(min_sal) if min_sal else None
    max_sal = float(max_sal) if max_sal else None

    exp_min = job.get("min")
    exp_min = float(exp_min) if exp_min is not None else None

    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=_locations(job),
        description="",  # filled from detail API in search()
        job_url=url,
        date_posted=job.get("createdTime") or job.get("createdTimeMs"),
        min_amount=min_sal,
        max_amount=max_sal,
        interval="annual" if (min_sal or max_sal) else None,
        is_remote=bool(job.get("workFromHome")),
        experience_years=exp_min,
        tags=_skills(job),
        site="hirist",
    )


def parse_hirist_payload(data: list[dict[str, Any]]) -> list[RawScrapedJob]:
    jobs = []
    for item in data or []:
        try:
            job = parse_hirist_job(item)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("hirist parse error: %s", exc)
    return jobs


def parse_hirist_detail(detail: dict[str, Any]) -> str:
    data = detail.get("data") or detail
    return BaseExtractor.html_to_text(data.get("introText") or "")


class HiristExtractor(BaseExtractor):
    """Hirist startup job board (gladiator API)."""

    portal_name = "hirist"
    site = "hirist"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("HiristExtractor requires a StealthSession")

        results: list[RawScrapedJob] = []
        page = 0
        while len(results) < limit:
            url = (
                f"{SEARCH_API}?query={quote_plus(keyword)}&page={page}"
                f"&posting=0&industry=&size=20"
            )
            resp = await self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning("hirist page %d HTTP %d", page, resp.status_code)
                break
            try:
                payload = resp.json()
            except Exception as exc:
                logger.warning("hirist page %d bad JSON: %s", page, exc)
                break
            data = payload.get("data") or []
            if not data:
                break
            results.extend(parse_hirist_payload(data))
            if len(data) < 20:
                break
            page += 1

        filtered = [
            j
            for j in results
            if self.location_match(j.location, location)
            and self.is_fresh(j.date_posted, freshness_days)
        ]
        if location.lower() == "remote":
            remote = [j for j in filtered if j.is_remote]
            if remote:
                filtered = remote
        filtered = filtered[:limit]

        # Enrich with full descriptions from the detail API (paced).
        for job in filtered:
            try:
                code = (job.job_url.rstrip("/").split("/")[-1]).rsplit("-", 1)[-1]
                detail_url = f"{DETAIL_API}?jobcode={code}"
                dresp = await self.session.get(detail_url, timeout=30)
                if dresp.status_code == 200:
                    desc = parse_hirist_detail(dresp.json())
                    if desc:
                        job.description = desc
            except Exception as exc:
                logger.debug("hirist detail error for %s: %s", job.job_url, exc)
            await asyncio.sleep(DETAIL_PACE_SECONDS)

        return filtered
