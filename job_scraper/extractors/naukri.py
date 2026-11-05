from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from job_scraper.core.client import BrowserStealthContext
from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

HOMEPAGE = "https://www.naukri.com/"
BASE = "https://www.naukri.com"
SCROLL_PASSES = 3
SCROLL_WAIT_MS = 4000


def _salary(job: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    sd = job.get("salaryDetail") or {}
    if sd.get("hideSalary"):
        return None, None
    lo = sd.get("minimumSalary") or sd.get("minSalaryPerMonth") or 0
    hi = sd.get("maximumSalary") or sd.get("maxSalaryPerMonth") or 0
    lo = float(lo) if lo else None
    hi = float(hi) if hi else None
    if not lo and not hi:
        return None, None
    # Naukri salaryDetail minimumSalary is INR per annum when > 100k.
    if (hi or 0) >= 100000 or (lo or 0) >= 100000:
        return lo, hi  # caller passes interval="annual"
    return lo, hi  # monthly fallback handled by caller via interval


def _location_from_placeholders(job: dict[str, Any]) -> str:
    for p in job.get("placeholders") or []:
        if p.get("type") == "location" and p.get("label"):
            return str(p["label"])
    return "India"


def _tags(job: dict[str, Any]) -> list[str]:
    raw = job.get("tagsAndSkills") or ""
    if isinstance(raw, str):
        parts = [t.strip() for t in raw.split(",") if t.strip()]
        return parts[:20]
    if isinstance(raw, list):
        return [str(t) for t in raw][:20]
    return []


def parse_naukri_job(job: dict[str, Any]) -> Optional[RawScrapedJob]:
    title = (job.get("title") or "").strip()
    company = (job.get("companyName") or "").strip()
    jd_url = job.get("jdURL") or job.get("staticUrl") or ""
    if not jd_url:
        return None
    if jd_url.startswith("/"):
        jd_url = BASE + jd_url
    elif not jd_url.startswith("http"):
        jd_url = BASE + "/" + jd_url

    loc = _location_from_placeholders(job)
    is_remote = "remote" in loc.lower()

    lo, hi = _salary(job)
    interval = "annual" if (lo or hi) else None

    exp_min = job.get("minimumExperience")
    exp_min = float(exp_min) if exp_min is not None else None

    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=loc,
        description=BaseExtractor.html_to_text(job.get("jobDescription") or ""),
        job_url=jd_url,
        date_posted=job.get("createdDate") or job.get("footerPlaceholderLabel"),
        min_amount=lo,
        max_amount=hi,
        interval=interval,
        is_remote=is_remote,
        experience_years=exp_min,
        tags=_tags(job),
        site="naukri",
    )


def parse_naukri_payload(job_details: list[dict[str, Any]]) -> list[RawScrapedJob]:
    jobs = []
    for item in job_details or []:
        try:
            job = parse_naukri_job(item)
            if job:
                jobs.append(job)
        except Exception as exc:
            logger.debug("naukri parse error: %s", exc)
    return jobs


class NaukriExtractor(BaseExtractor):
    """Naukri via Playwright: warm cookies, keyword QSB, intercept jobapi JSON.

    Location is applied client-side on `placeholders[type=location]` because
    SEO location routes are behind Akamai 403s. Freshness is filtered on
    `createdDate` (the API `days` param is unreliable without cookies).
    """

    portal_name = "naukri"
    site = "naukri"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        captured: list[dict[str, Any]] = []

        async def on_response(resp):
            try:
                if "jobapi" in resp.url and "/search" in resp.url and resp.status == 200:
                    payload = json.loads(await resp.text())
                    details = payload.get("jobDetails") or []
                    for item in details:
                        if item not in captured:
                            captured.append(item)
            except Exception:
                pass

        async with BrowserStealthContext() as page:
            page.on("response", on_response)
            await page.goto(HOMEPAGE, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(4000)

            inputs = page.locator('input[type="text"]')
            await inputs.nth(0).fill(keyword)
            await inputs.nth(0).press("Enter")
            await page.wait_for_timeout(10000)

            for _ in range(SCROLL_PASSES):
                if len(captured) >= limit * 2:
                    break
                await page.mouse.wheel(0, 3000)
                await page.wait_for_timeout(SCROLL_WAIT_MS)

        results = parse_naukri_payload(captured)

        # De-dup by URL.
        seen: set[str] = set()
        unique: list[RawScrapedJob] = []
        for j in results:
            key = re.sub(r"\?.*$", "", j.job_url)
            if key not in seen:
                seen.add(key)
                unique.append(j)

        filtered = [
            j for j in unique
            if self.location_match(j.location, location)
            and self.is_fresh(j.date_posted, freshness_days)
        ]
        if location.lower() == "remote":
            remote = [j for j in filtered if j.is_remote]
            if remote:
                filtered = remote
        return filtered[:limit]
