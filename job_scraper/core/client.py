from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from job_scraper.core.anti_detect import (
    AsyncRateLimiter,
    BrowserIdentity,
    exponential_backoff,
    random_identity,
)
from job_scraper.core.exceptions import (
    BotDetectionError,
    RateLimitBlockError,
    ScraperException,
)

logger = logging.getLogger(__name__)

# Body fragments that indicate a bot/challenge wall (Cloudflare, Akamai, reCAPTCHA).
_CHALLENGE_MARKERS = (
    "just a moment",
    "cf-challenge",
    "challenge-running",
    "challenge-platform",
    "security check",
    "access denied",
    "reference #18",
    "recaptcha",
    "unusual activity",
    "captcha",
    "attention required",
    "verify you are human",
)

# Headers curl_cffi manages itself as part of the TLS-bound impersonation
# profile; overriding them from Python breaks the fingerprint.
_DO_NOT_OVERRIDE = {"accept-encoding", "connection"}

_CHALLENGE_STATUS = (403, 429, 503)


def _looks_like_challenge(status_code: int, text: str) -> bool:
    if status_code not in _CHALLENGE_STATUS:
        return False
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _CHALLENGE_MARKERS)


class StealthSession:
    """Resilient async HTTP client on top of curl_cffi's TLS-impersonating session.

    Features:
      - cohesive browser identity (UA + Sec-CH-UA + TLS target agree)
      - per-domain async rate limiting with jittered spacing
      - exponential backoff retries on 429/5xx/network errors
      - BotDetectionError on challenge walls, RateLimitBlockError on 429 exhaustion
    """

    def __init__(
        self,
        identity: Optional[BrowserIdentity] = None,
        impersonate: Optional[str] = None,
        min_interval: float = 1.5,
        max_retries: int = 3,
        backoff_base: float = 1.5,
        timeout: float = 30.0,
        session_factory: Optional[Callable[[], Any]] = None,
        proxy: Optional[str] = None,
    ) -> None:
        self.identity = identity or random_identity()
        self.impersonate = impersonate or self.identity.impersonate
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._session_factory = session_factory
        self._session: Any = None
        self._proxy = proxy
        self._rate_limiter = AsyncRateLimiter(min_interval=min_interval)

    async def __aenter__(self) -> "StealthSession":
        if self._session_factory is not None:
            self._session = self._session_factory()
        else:
            from curl_cffi.requests import AsyncSession as CurlAsyncSession

            kwargs: dict[str, Any] = {"impersonate": self.impersonate}
            if self._proxy:
                kwargs["proxies"] = {"http": self._proxy, "https": self._proxy}
            self._session = CurlAsyncSession(**kwargs)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._session is not None:
            try:
                await self._session.aclose()
            except Exception:  # pragma: no cover - best effort
                pass
        self._session = None

    def _base_headers(self, extra: Optional[dict[str, str]]) -> dict[str, str]:
        headers = self.identity.headers()
        for name in _DO_NOT_OVERRIDE:
            for key in list(headers):
                if key.lower() == name:
                    headers.pop(key, None)
        if extra:
            headers.update(extra)
        return headers

    async def get(self, url: str, *, headers: Optional[dict[str, str]] = None, **kwargs: Any):
        return await self._request("GET", url, headers=headers, **kwargs)

    async def head(self, url: str, *, headers: Optional[dict[str, str]] = None, **kwargs: Any):
        return await self._request("HEAD", url, headers=headers, **kwargs)

    async def post(
        self,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[Any] = None,
        **kwargs: Any,
    ):
        return await self._request("POST", url, headers=headers, json_body=json_body, **kwargs)

    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[dict[str, str]] = None,
        json_body: Optional[Any] = None,
        **kwargs: Any,
    ):
        if self._session is None:
            raise ScraperException("StealthSession used outside of async with", portal="", url=url)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            await self._rate_limiter.acquire(url)
            try:
                call_kwargs: dict[str, Any] = {
                    "headers": self._base_headers(headers),
                    "timeout": self.timeout,
                }
                if json_body is not None:
                    call_kwargs["json"] = json_body
                call_kwargs.update(kwargs)
                response = await self._session.request(method, url, **call_kwargs)
            except Exception as exc:  # network-level failure
                last_exc = exc
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, base=self.backoff_base)
                    logger.warning(
                        "network error on %s %s (attempt %d): %s; retrying in %.1fs",
                        method, url, attempt + 1, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise ScraperException(f"network error after retries: {exc}", portal="", url=url) from exc

            status = getattr(response, "status_code", 0)
            text = ""
            try:
                text = response.text or ""
            except Exception:
                text = ""

            if _looks_like_challenge(status, text):
                # Static WAF challenge walls never resolve via raw-HTTP retries;
                # fail fast so callers can escalate (stealth browser) immediately.
                logger.warning(
                    "bot challenge detected on %s (status %d); failing fast",
                    url, status,
                )
                raise BotDetectionError(
                    f"bot challenge on {url} (status {status})", url=url
                ) from None

            if status == 429:
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, base=self.backoff_base)
                    logger.warning("429 on %s; backing off %.1fs", url, delay)
                    await asyncio.sleep(delay)
                    continue
                raise RateLimitBlockError(f"rate limited on {url}", url=url) from None

            if 500 <= status < 600:
                if attempt < self.max_retries:
                    delay = exponential_backoff(attempt, base=self.backoff_base)
                    logger.warning("HTTP %d on %s; retrying in %.1fs", status, url, delay)
                    await asyncio.sleep(delay)
                    continue
                raise ScraperException(f"HTTP {status} after retries on {url}", url=url) from None

            return response

        raise ScraperException(f"request failed on {url}", url=url) from last_exc


class BrowserStealthContext:
    """Async context manager yielding a stealth-patched Playwright page.

    Usage:
        async with BrowserStealthContext() as page:
            await page.goto("https://www.naukri.com/")
    """

    def __init__(
        self,
        headless: bool = True,
        user_agent: Optional[str] = None,
        locale: str = "en-IN",
        timezone_id: str = "Asia/Kolkata",
        viewport: dict[str, int] | None = None,
        extra_launch_args: tuple[str, ...] = (),
    ) -> None:
        self.headless = headless
        self.user_agent = user_agent or random_identity().user_agent
        self.locale = locale
        self.timezone_id = timezone_id
        self.viewport = viewport or {"width": 1920, "height": 1080}
        self.extra_launch_args = extra_launch_args
        self._playwright_cm = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    @staticmethod
    def build_launch_args() -> list[str]:
        return [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--disable-dev-shm-usage",
        ]

    @staticmethod
    def build_context_options(
        user_agent: Optional[str] = None,
        locale: str = "en-IN",
        timezone_id: str = "Asia/Kolkata",
        viewport: Optional[dict[str, int]] = None,
    ) -> dict[str, Any]:
        return {
            "user_agent": user_agent or random_identity().user_agent,
            "locale": locale,
            "timezone_id": timezone_id,
            "viewport": viewport or {"width": 1920, "height": 1080},
            "device_scale_factor": 1,
            "extra_http_headers": {
                "Accept-Language": "en-IN,en-GB,en;q=0.9,hi;q=0.8",
            },
        }

    async def __aenter__(self) -> Any:
        from playwright.async_api import async_playwright
        from playwright_stealth import Stealth

        self._playwright_cm = Stealth().use_async(async_playwright())
        self._playwright = await self._playwright_cm.__aenter__()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[*self.build_launch_args(), *self.extra_launch_args],
        )
        self._context = await self._browser.new_context(
            **self.build_context_options(
                user_agent=self.user_agent,
                locale=self.locale,
                timezone_id=self.timezone_id,
                viewport=self.viewport,
            )
        )
        self._page = await self._context.new_page()
        return self._page

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for closer in (
            self._page,
            self._context,
            self._browser,
        ):
            if closer is not None:
                try:
                    await closer.close()
                except Exception:  # pragma: no cover - best effort
                    pass
        if self._playwright_cm is not None:
            try:
                await self._playwright_cm.__aexit__(exc_type, exc, tb)
            except Exception:  # pragma: no cover - best effort
                pass
        self._playwright_cm = self._playwright = self._browser = self._context = self._page = None
