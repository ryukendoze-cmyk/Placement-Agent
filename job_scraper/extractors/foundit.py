from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import quote_plus

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

RAVEN_API = "https://apiv3.monsterindia.com/raven/api/public/search/v1/jobs"
BASE_URL = "https://www.foundit.in"

# Raven freshness filter buckets (server-side). Map requested days to the
# smallest bucket that covers it.
_FRESHNESS_BUCKETS = (1, 3, 7, 15, 30)


def _freshness_bucket(days: int) -> Optional[str]:
    for b in _FRESHNESS_BUCKETS:
        if days <= b:
            return str(b)
    return str(_FRESHNESS_BUCKETS[-1])


def _job_location(job: dict[str, Any]) -> str:
    locs = job.get("locations") or []
    cities = [l.get("city") for l in locs if l.get("city")]
    if cities:
        return ", ".join(dict.fromkeys(cities))
    countries = {l.get("country") for l in locs if l.get("country")}
    if countries:
        return ", ".join(sorted(countries))
    return "India"


def _is_remote(job: dict[str, Any]) -> bool:
    for l in job.get("locations") or []:
        if (l.get("city") or "").lower() == "remote":
            return True
    for t in job.get("jobTypes") or []:
        if "remote" in str(t).lower() or "work from home" in str(t).lower():
            return True
    return False


def _skills(job: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for key in ("skills", "itSkills"):
        for s in job.get(key) or []:
            if isinstance(s, dict) and s.get("text"):
                out.append(str(s["text"]))
            elif isinstance(s, str):
                out.append(s)
    out = list(dict.fromkeys(out))[:18]
    # Syndicated postings carry a direct ATS apply URL (Workday/Oracle/LinkedIn).
    apply_url = job.get("applyUrl") or ""
    if isinstance(apply_url, str) and apply_url.startswith("http"):
        out.append(f"direct_apply:{apply_url}")
    return out[:20]


def parse_foundit_job(job: dict[str, Any]) -> Optional[RawScrapedJob]:
    """Parse one raven `data[]` item into a RawScrapedJob (or None)."""
    title = (job.get("title") or "").strip()
    company = ((job.get("company") or {}).get("name") or "").strip() or (
        job.get("companyName") or ""
    ).strip()
    jd_url = job.get("jdUrl") or ""
    if not jd_url:
        return None
    if jd_url.startswith("/"):
        jd_url = BASE_URL + jd_url
    elif not jd_url.startswith("http"):
        jd_url = BASE_URL + "/" + jd_url

    min_sal = (job.get("minimumSalary") or {}).get("absoluteValue") or 0
    max_sal = (job.get("maximumSalary") or {}).get("absoluteValue") or 0
    min_sal = float(min_sal) if min_sal else None
    max_sal = float(max_sal) if max_sal else None

    exp_min = (job.get("minimumExperience") or {}).get("years")
    exp_min = float(exp_min) if exp_min is not None else None

    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=_job_location(job),
        description=BaseExtractor.html_to_text(job.get("description") or ""),
        job_url=jd_url,
        date_posted=job.get("postedAt") or job.get("createdAt"),
        min_amount=min_sal,
        max_amount=max_sal,
        interval="annual" if (min_sal or max_sal) else None,
        is_remote=_is_remote(job),
        experience_years=exp_min,
        tags=_skills(job),
        site="foundit",
    )


def parse_foundit_payload(payload: dict[str, Any]) -> list[RawScrapedJob]:
    jobs = []
    for item in payload.get("data") or []:
        try:
            job = parse_foundit_job(item)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("foundit parse error: %s", exc)
    return jobs


class FounditExtractor(BaseExtractor):
    """Foundit India (ex-Monster) via the public Raven search API."""

    portal_name = "foundit"
    site = "foundit"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("FounditExtractor requires a StealthSession")

        params: dict[str, str] = {
            "query": keyword,
            "siteContext": "rexmonster",
            "limit": "10",
        }
        if freshness_days and freshness_days > 0:
            bucket = _freshness_bucket(freshness_days)
            if bucket:
                params["jobFreshness"] = bucket
        # Server-side city filter only for known single-city requests.
        city = (location or "").strip()
        if city and city.lower() not in {"india", "pan india", "pan-india", "remote"}:
            params["locations"] = city

        results: list[RawScrapedJob] = []
        page = 1
        headers = {
            "Accept": "application/json",
            "Origin": "https://www.foundit.in",
            "Referer": "https://www.foundit.in/",
        }
        while len(results) < limit:
            url = f"{RAVEN_API}?{self._qs(params)}&page={page}"
            resp = await self.session.get(url, headers=headers, timeout=30)
            if resp.status_code != 200:
                logger.warning("foundit page %d HTTP %d", page, resp.status_code)
                break
            try:
                payload = resp.json()
            except Exception as exc:
                logger.warning("foundit page %d bad JSON: %s", page, exc)
                break
            page_jobs = parse_foundit_payload(payload)
            if not page_jobs:
                break
            results.extend(page_jobs)
            paging = ((payload.get("meta") or {}).get("paging") or {})
            total = paging.get("total") or 0
            if total and len(results) >= min(total, limit):
                break
            if len(page_jobs) < 10:
                break
            page += 1

        # Location + freshness client-side filter (server filters are best-effort).
        filtered = [
            j
            for j in results
            if self.location_match(j.location, location)
            and self.is_fresh(j.date_posted, freshness_days if location.lower() in {"india", ""} else None)
        ]
        if location.lower() == "remote":
            remote = [j for j in filtered if j.is_remote]
            if remote:
                filtered = remote
        return filtered[:limit]

    @staticmethod
    def _qs(params: dict[str, str]) -> str:
        return "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
