from __future__ import annotations


class ScraperException(Exception):
    """Base exception for all scraper failures."""

    def __init__(self, message: str, *, portal: str = "", url: str = "") -> None:
        super().__init__(message)
        self.portal = portal
        self.url = url


class RateLimitBlockError(ScraperException):
    """HTTP 429 / rate-limited response; triggers exponential backoff."""


class BotDetectionError(ScraperException):
    """Cloudflare/Akamai challenge or captcha wall; triggers session rotation."""


class SelectorExtractionError(ScraperException):
    """DOM/API schema drift; logged, does not crash the runner."""


class ExtractionError(ScraperException):
    """A single job record failed to parse; skipped, pipeline continues."""
