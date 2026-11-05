from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import quote_plus

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

API = "https://unstop.com/api/public/opportunity/search-result"


def _salary(job_detail: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    min_sal = job_detail.get("min_salary") or 0
    max_sal = job_detail.get("max_salary") or 0
    pay_in = job_detail.get("pay_in")
    min_sal = float(min_sal) if min_sal else None
    max_sal = float(max_sal) if max_sal else None
    if not (min_sal or max_sal):
        return None, None, None
    if pay_in == "monthly":
        return min_sal, max_sal, "monthly"
    if pay_in in ("annually", "annual"):
        return min_sal, max_sal, "annual"
    # Unstop often omits pay_in for INR figures that are annual.
    if (max_sal or 0) >= 100000:
        return min_sal, max_sal, "annual"
    return min_sal, max_sal, "monthly"


def _locations(opp: dict[str, Any]) -> str:
    locs = opp.get("locations") or []
    cities = [l.get("city") for l in locs if isinstance(l, dict) and l.get("city")]
    if cities:
        return ", ".join(dict.fromkeys(cities))
    for l in locs:
        if isinstance(l, str) and l.strip():
            return l.strip()
    return "India"


def _skills(opp: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for s in opp.get("required_skills") or []:
        if isinstance(s, dict) and s.get("skill_name"):
            out.append(str(s["skill_name"]))
        elif isinstance(s, str):
            out.append(s)
    for t in opp.get("tags") or []:
        if isinstance(t, str):
            out.append(t)
    return list(dict.fromkeys(out))[:20]


def parse_unstop_item(opp: dict[str, Any]) -> Optional[RawScrapedJob]:
    title = (opp.get("title") or "").strip()
    company = ((opp.get("organisation") or {}).get("name") or "").strip()
    url = opp.get("seo_url") or ""
    if not url.startswith("http"):
        pub = opp.get("public_url") or ""
        url = f"https://unstop.com/{pub.lstrip('/')}" if pub else ""
    if not url:
        return None

    jd = opp.get("jobDetail") or {}
    min_sal, max_sal, interval = _salary(jd)
    exp_min = jd.get("min_experience")
    exp_min = float(exp_min) if exp_min is not None else None

    loc = _locations(opp)
    is_remote = loc.lower() == "remote" or (jd.get("type") == "online")

    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=loc,
        description=BaseExtractor.html_to_text(opp.get("details") or ""),
        job_url=url,
        date_posted=opp.get("approved_date"),
        min_amount=min_sal,
        max_amount=max_sal,
        interval=interval,
        is_remote=is_remote,
        experience_years=exp_min,
        tags=_skills(opp),
        site="unstop",
    )


def parse_unstop_payload(items: list[dict[str, Any]]) -> list[RawScrapedJob]:
    jobs = []
    for item in items or []:
        try:
            job = parse_unstop_item(item)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("unstop parse error: %s", exc)
    return jobs


class UnstopExtractor(BaseExtractor):
    """Unstop public opportunity API (jobs/early-career roles)."""

    portal_name = "unstop"
    site = "unstop"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("UnstopExtractor requires a StealthSession")

        results: list[RawScrapedJob] = []
        page = 1
        while len(results) < limit:
            url = (
                f"{API}?opportunity=jobs&page={page}&per_page=15"
                f"&oppstatus=open&searchTerm={quote_plus(keyword)}"
            )
            resp = await self.session.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning("unstop page %d HTTP %d", page, resp.status_code)
                break
            try:
                payload = resp.json()
            except Exception as exc:
                logger.warning("unstop page %d bad JSON: %s", page, exc)
                break
            items = ((payload.get("data") or {}).get("data")) or []
            page_jobs = parse_unstop_payload(items)
            if not page_jobs:
                break
            results.extend(page_jobs)
            if len(items) < 15:
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
        return filtered[:limit]
