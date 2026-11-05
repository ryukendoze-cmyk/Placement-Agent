from __future__ import annotations

import logging
import re
from typing import Optional

from bs4 import BeautifulSoup

from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)

BASE = "https://internshala.com"
DESC_LIMIT = 5000

# Card anchors: internship and full-time job listings (div- or li-based layouts).
_CARD_RE = re.compile(
    r"<(?:li|div)[^>]*id=\"individual_(?:internship|job)_\d+\"[^>]*>", re.I
)


def _href_from_tag(tag) -> str:
    for attr in ("data-href", "href"):
        val = tag.get(attr)
        if val and str(val).startswith("/"):
            return str(val)
    a = tag.select_one("a[href]")
    if a and a.get("href", "").startswith("/"):
        return a["href"]
    return ""


def _location_from_card(card) -> str:
    loc_div = card.select_one(".row-1-item.locations, .locations")
    if loc_div:
        text = re.sub(r"\s+", " ", loc_div.get_text(" ", strip=True))
        # Drop "(Hybrid)"/"(Remote)" suffix but remember remote.
        text = re.sub(r"\((?:Hybrid|Remote|Work From Home)\)\s*$", "", text).strip()
        return text or "India"
    return "India"


def _is_remote_from_card(card) -> bool:
    text = card.get_text(" ", strip=True).lower()
    return "remote" in text and ("work from home" in text or "(remote)" in text)


def _salary_from_card(card) -> tuple[Optional[float], Optional[float]]:
    stip = card.select_one(".stipend")
    if not stip:
        return None, None
    text = stip.get_text(" ", strip=True)
    nums = re.findall(r"[\d,]+", text)
    nums = [float(n.replace(",", "")) for n in nums if n.replace(",", "").isdigit()]
    if not nums:
        return None, None
    lo = nums[0]
    hi = nums[-1] if len(nums) > 1 else nums[0]
    return lo, hi


def parse_internshala_card(card) -> Optional[RawScrapedJob]:
    path = _href_from_tag(card)
    if not path:
        return None
    url = BASE + path

    title_tag = card.select_one(".job-title-href, .job-internship-name a, h2 a")
    title = (title_tag.get_text(strip=True) if title_tag else "").strip()
    if not title:
        return None

    comp_tag = card.select_one(".company-name, .company-name a")
    company = (comp_tag.get_text(strip=True) if comp_tag else "").strip()

    loc = _location_from_card(card)
    lo, hi = _salary_from_card(card)

    about = card.select_one(".about_job .text, .text")
    desc = about.get_text(" ", strip=True) if about else ""

    # Posted time text ("3 days ago", "Today").
    posted = ""
    for span in card.find_all(string=re.compile(r"ago|Today|Yesterday", re.I)):
        t = span.strip()
        if t:
            posted = t
            break

    interval = "monthly" if (lo or hi) else None
    return BaseExtractor.make_job(
        title=title,
        company=company,
        location=loc,
        description=desc[:DESC_LIMIT],
        job_url=url,
        date_posted=posted,
        min_amount=lo,
        max_amount=hi,
        interval=interval,
        is_remote=_is_remote_from_card(card),
        tags=[],
        site="internshala",
    )


def parse_internshala_html(html: str) -> list[RawScrapedJob]:
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    jobs: list[RawScrapedJob] = []
    seen: set[str] = set()
    for m in _CARD_RE.finditer(html):
        start = m.start()
        # Slice the card up to the next card or a generous window.
        nxt = _CARD_RE.search(html, start + 10)
        end = nxt.start() if nxt else min(len(html), start + 12000)
        try:
            soup = BeautifulSoup(html[start:end], "lxml")
            card_el = soup.find(id=re.compile(r"^individual_(?:internship|job)_\d+$"))
            job = parse_internshala_card(card_el or soup)
        except Exception as exc:
            logger.debug("internshala card parse error: %s", exc)
            continue
        if job and job.job_url not in seen:
            seen.add(job.job_url)
            jobs.append(job)
    return jobs


class InternshalaExtractor(BaseExtractor):
    """Internshala internships + full-time fresher jobs (HTML cards)."""

    portal_name = "internshala"
    site = "internshala"

    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        if self.session is None:
            raise RuntimeError("InternshalaExtractor requires a StealthSession")

        slug = re.sub(r"[^a-z0-9]+", "-", keyword.lower()).strip("-") or "python"
        slugs = [f"/internships/{slug}-internship", f"/jobs/{slug}-jobs"]

        results: list[RawScrapedJob] = []
        for slug_url in slugs:
            if len(results) >= limit:
                break
            for page in (1, 2):
                url = f"{BASE}{slug_url}" if page == 1 else f"{BASE}{slug_url.rstrip('/')}/page-{page}/"
                resp = await self.session.get(
                    url, headers={"Referer": f"{BASE}/", "Accept-Language": "en-IN,en;q=0.9"},
                    timeout=40,
                )
                if resp.status_code != 200:
                    logger.warning("internshala %s HTTP %d", url, resp.status_code)
                    break
                page_jobs = parse_internshala_html(resp.text)
                if not page_jobs:
                    break
                results.extend(page_jobs)
                if len(page_jobs) < 10:
                    break

        # Dedupe by canonical URL.
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
