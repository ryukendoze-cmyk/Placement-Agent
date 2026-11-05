from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote_plus, urlsplit, urlunsplit

from bs4 import BeautifulSoup

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

WARMUP = "https://www.linkedin.com/jobs/search"
GUEST_API = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search"
INDIA_GEO_ID = "102713980"
DETAIL_PACE_SECONDS = 6.0


def _clean_view_url(href: str) -> str:
    parts = urlsplit(href)
    # Keep only the path (slug+id); drop all tracking query params.
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def parse_linkedin_cards(html: str) -> list[dict[str, str]]:
    """Parse LinkedIn guest-search card HTML into lightweight card dicts."""
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    cards = []
    for card in soup.select("div.base-card"):
        link = card.select_one("a.base-card__full-link, a[href*='/jobs/view/']")
        if not link or not link.get("href"):
            continue
        url = _clean_view_url(link["href"])
        if not url.startswith("http"):
            continue
        title = card.select_one(".base-search-card__title")
        company = card.select_one(".base-search-card__subtitle")
        loc_el = card.select_one(".job-search-card__location") or card.select_one(
            ".base-search-card__metadata"
        )
        time_el = card.select_one("time[datetime]")
        if not title:
            continue
        cards.append(
            {
                "title": title.get_text(strip=True),
                "company": company.get_text(strip=True) if company else "",
                "location": loc_el.get_text(strip=True) if loc_el else "",
                "date": (time_el.get("datetime") if time_el else "") or "",
                "url": url,
            }
        )
    # De-dup by url, preserve order.
    seen: set[str] = set()
    unique = []
    for c in cards:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c)
    return unique


def extract_linkedin_detail(html: str) -> tuple[str, str, str]:
    """Return (description, min_exp_text, criteria_text) from a detail page."""
    if not html:
        return "", "", ""
    soup = BeautifulSoup(html, "lxml")
    desc_el = soup.select_one(".description__text")
    desc = desc_el.get_text("\n", strip=True) if desc_el else ""
    criteria = []
    for item in soup.select(".job-criteria-item"):
        label = item.select_one(".job-criteria-sub-header")
        value = item.select_one(".job-criteria-value")
        if label and value:
            criteria.append(f"{label.get_text(strip=True)}: {value.get_text(strip=True)}")
    return desc, "", "\n".join(criteria)


class LinkedInExtractor(BaseExtractor):
    """LinkedIn India guest job search + paced detail pages."""

    portal_name = "linkedin"
    site = "linkedin"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("LinkedInExtractor requires a StealthSession")

        location_param = "india" if location.lower() in {"india", "", "pan india", "pan-india"} else location
        f_tpr = f"&f_TPR=r{int(freshness_days) * 86400}" if freshness_days and freshness_days > 0 else ""

        results: list[RawScrapedJob] = []
        start = 0
        page_size = 10
        first_page = True
        while len(results) < limit:
            if first_page:
                # Warm up the guest session cookie jar on the search page.
                warm_url = (
                    f"{WARMUP}?keywords={quote_plus(keyword)}"
                    f"&location={quote_plus(location_param)}"
                )
                try:
                    await self.session.get(warm_url, timeout=40)
                except Exception as exc:
                    logger.debug("linkedin warmup failed: %s", exc)
                await asyncio.sleep(8)
                first_page = False
            url = (
                f"{GUEST_API}?keywords={quote_plus(keyword)}"
                f"&location={quote_plus(location_param)}"
                f"&geoId={INDIA_GEO_ID}&start={start}{f_tpr}"
            )
            resp = await self.session.get(
                url,
                headers={"Referer": f"{WARMUP}?keywords={quote_plus(keyword)}&location=india"},
                timeout=40,
            )
            if resp.status_code != 200:
                logger.warning("linkedin start=%d HTTP %d", start, resp.status_code)
                break
            cards = parse_linkedin_cards(resp.text)
            if not cards:
                break
            for card in cards:
                if len(results) >= limit:
                    break
                job = BaseExtractor.make_job(
                    title=card["title"],
                    company=card["company"],
                    location=card["location"] or "India",
                    description="",
                    job_url=card["url"],
                    date_posted=card["date"],
                    is_remote="remote" in (card["location"] or "").lower(),
                    tags=[],
                    site="linkedin",
                )
                if job:
                    results.append(job)
            if len(cards) < page_size:
                break
            start += page_size
            await asyncio.sleep(3)

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

        # Enrich with full descriptions (paced to avoid 429s).
        for job in filtered:
            try:
                await asyncio.sleep(DETAIL_PACE_SECONDS)
                dresp = await self.session.get(
                    job.job_url, headers={"Referer": "https://www.linkedin.com/jobs/"}, timeout=40
                )
                if dresp.status_code == 200:
                    desc, _, criteria = extract_linkedin_detail(dresp.text)
                    parts = [p for p in (desc, criteria) if p]
                    if parts:
                        job.description = "\n\n".join(parts)
            except Exception as exc:
                logger.debug("linkedin detail error for %s: %s", job.job_url, exc)

        ensure_descriptions(filtered)
        return filtered


def ensure_descriptions(jobs: list) -> None:
    """Guarantee a usable description even if detail fetches were blocked
    (429/timeout) — downstream ATS matching fails on empty JDs."""
    for job in jobs:
        if len(job.description.strip()) < 40:
            job.description = (
                f"{job.title} at {job.company}, based in {job.location}. "
                f"This listing was captured from LinkedIn's live jobs search "
                f"on {job.date_posted or 'an unknown date'}; the full job "
                f"description could not be retrieved during this run because "
                f"LinkedIn rate-limited the detail page. Open the job URL to "
                f"review the complete role description, responsibilities, and "
                f"seniority requirements before applying."
            )
