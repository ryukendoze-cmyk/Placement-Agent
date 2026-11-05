from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from bs4 import BeautifulSoup

from job_scraper.core.models import RawScrapedJob
from job_scraper.core.normalizer import parse_date_posted

logger = logging.getLogger(__name__)

DESC_MAX_LEN = 20_000

# Location aliases: canonical query term -> accepted variants found in job data.
LOCATION_ALIASES: dict[str, list[str]] = {
    "bengaluru": ["bengaluru", "bangalore", "blr"],
    "hyderabad": ["hyderabad", "secunderabad"],
    "pune": ["pune"],
    "mumbai": ["mumbai", "thane", "navi mumbai", "nashik"],
    "chennai": ["chennai", "madras", "coimbatore", "tamil nadu"],
    "delhi ncr": ["delhi", "gurgaon", "gurugram", "noida", "faridabad", "ghaziabad", "ncr"],
    "delhi": ["delhi", "gurgaon", "gurugram", "noida", "ncr"],
    "kolkata": ["kolkata", "calcutta"],
    "ahmedabad": ["ahmedabad", "gandhinagar", "surat"],
    "india": ["india", "pan india", "pan-india", "all india"],
    "remote": ["remote"],
}

PAN_INDIAN = {"india", "pan india", "pan-india", "all india", ""}


class BaseExtractor(ABC):
    """Common contract + helpers for all portal extractors.

    Subclasses implement `search()` which must:
      - return a list of RawScrapedJob (never raise for a single bad record)
      - apply freshness filtering when the portal exposes dates
      - truncate to `limit`
    """

    portal_name: str = "base"
    site: str = "base"
    scraper_engine: str = "stealth_harvester"

    def __init__(self, session: Any = None) -> None:
        # `session` is a StealthSession (or a test double) used by subclasses.
        self.session = session

    @abstractmethod
    async def search(
        self,
        keyword: str,
        location: str = "India",
        freshness_days: int = 7,
        limit: int = 50,
    ) -> list[RawScrapedJob]:
        """Yield raw scraped jobs from the portal."""

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def html_to_text(html: str, max_len: int = DESC_MAX_LEN) -> str:
        """Convert HTML job descriptions to readable plain text."""
        if not html:
            return ""
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [re.sub(r"[ \t\u00a0]+", " ", line).strip() for line in text.splitlines()]
        out = "\n".join(line for line in lines if line)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        if len(out) > max_len:
            out = out[:max_len].rstrip() + " …"
        return out

    @staticmethod
    def ms_to_iso(ms: Any) -> Optional[str]:
        if ms in (None, "", 0):
            return None
        try:
            num = float(ms)
        except (TypeError, ValueError):
            return None
        if num > 1e12:
            num /= 1000.0
        if num < 1e9:
            return None
        try:
            return datetime.fromtimestamp(num, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (OverflowError, OSError, ValueError):
            return None

    @staticmethod
    def parse_date(value: Any) -> Optional[datetime]:
        """Any portal date -> aware UTC datetime (or None)."""
        iso = parse_date_posted(value)
        if not iso:
            return None
        try:
            return datetime.fromisoformat(iso.replace("Z", "+00:00"))
        except ValueError:
            return None

    @classmethod
    def is_fresh(cls, value: Any, freshness_days: Optional[int]) -> bool:
        if not freshness_days or freshness_days <= 0:
            return True
        dt = cls.parse_date(value)
        if dt is None:
            # Unknown date: keep the record (portal may already pre-filter).
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(days=freshness_days)
        return dt >= cutoff

    @staticmethod
    def location_variants(location: str) -> list[str]:
        key = (location or "").strip().lower()
        if not key or key in PAN_INDIAN:
            return []
        return LOCATION_ALIASES.get(key, [key])

    @classmethod
    def location_match(cls, job_location: str, wanted: str) -> bool:
        """True when the job satisfies the requested location.

        Pan-India/empty requests match everything; 'Remote' matches remote
        listings (and anything, since remote jobs are often also city-based);
        city requests match any alias contained in the job location string.
        """
        key = (wanted or "").strip().lower()
        if not key or key in PAN_INDIAN:
            return True
        variants = cls.location_variants(key)
        if not variants:
            return True
        hay = (job_location or "").lower()
        if not hay:
            return True
        if key == "remote":
            # Remote listings always qualify; keep city listings too only if
            # nothing better is found (handled by caller ordering).
            return "remote" in hay or True
        return any(v in hay for v in variants)

    @classmethod
    def make_job(
        cls,
        *,
        title: str,
        company: str,
        location: str,
        description: str,
        job_url: str,
        date_posted: Any = None,
        min_amount: Optional[float] = None,
        max_amount: Optional[float] = None,
        interval: Optional[str] = None,
        is_remote: bool = False,
        experience_years: Optional[float] = None,
        tags: Optional[list[str]] = None,
        site: Optional[str] = None,
    ) -> Optional[RawScrapedJob]:
        """Build a validated RawScrapedJob; returns None if the record is unusable."""
        try:
            return RawScrapedJob(
                title=(title or "").strip()[:300],
                company=(company or "").strip()[:200] or "Unknown",
                location=(location or "India").strip()[:200] or "India",
                description=description or "",
                job_url=job_url,
                site=site or cls.site,
                scraper_engine=cls.scraper_engine,
                date_posted=parse_date_posted(date_posted) or "",
                min_amount=min_amount,
                max_amount=max_amount,
                interval=interval or "lpa",
                is_remote=is_remote,
                experience_years=experience_years,
                tags=list(tags or [])[:20],
            )
        except Exception as exc:
            logger.debug("dropping invalid %s record %s: %s", cls.portal_name, job_url, exc)
            return None
