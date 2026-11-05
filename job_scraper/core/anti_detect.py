from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

# Chrome/Edge desktop UA pool covering current major versions on Windows/macOS.
USER_AGENTS: list[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36",
]

# Cohesive fingerprint profiles: UA + TLS impersonation target + platform must
# agree. Pairing a Chrome UA with a Safari JA3/JA4 handshake (or vice versa)
# is a high-confidence bot signal for Akamai/Cloudflare.
PROFILES: list[dict[str, str]] = [
    {"ua": USER_AGENTS[0], "impersonate": "chrome124", "platform": "Windows"},
    {"ua": USER_AGENTS[1], "impersonate": "chrome131", "platform": "Windows"},
    {"ua": USER_AGENTS[2], "impersonate": "chrome133a", "platform": "Windows"},
    {"ua": USER_AGENTS[3], "impersonate": "chrome136", "platform": "Windows"},
    {"ua": USER_AGENTS[4], "impersonate": "chrome124", "platform": "Windows"},
    {"ua": USER_AGENTS[5], "impersonate": "chrome124", "platform": "macOS"},
    {"ua": USER_AGENTS[6], "impersonate": "chrome131", "platform": "macOS"},
    {"ua": USER_AGENTS[7], "impersonate": "chrome142", "platform": "Windows"},
    {"ua": USER_AGENTS[8], "impersonate": "chrome145", "platform": "Windows"},
    {"ua": USER_AGENTS[9], "impersonate": "chrome142", "platform": "macOS"},
]

# Retained for compatibility/tests; every entry is chrome-compatible with PROFILES.
IMPERSONATIONS: list[str] = sorted({p["impersonate"] for p in PROFILES})

ACCEPT_LANGUAGES = "en-IN,en-GB,en;q=0.9,hi;q=0.8"


def _parse_chrome_version(ua: str) -> tuple[str, str]:
    """Return (major, full) Chrome version from a UA string."""
    try:
        chrome_part = ua.split("Chrome/")[1]
        full = chrome_part.split(" ")[0]
        major = full.split(".")[0]
        return major, full
    except IndexError:
        return "124", "124.0.0.0"


def build_sec_ch_ua(ua: str) -> str:
    """Build a valid RFC 8941 Sec-CH-UA client hint header for the given UA.

    Each brand token already carries its own `;v="N"` parameter — do not append
    a trailing version parameter (invalid structured-field syntax; strict WAF
    parsers flag it).
    """
    major, _ = _parse_chrome_version(ua)
    is_edge = "Edg/" in ua
    if is_edge:
        brands = [
            f'"Not_A Brand";v="8"',
            f'"Chromium";v="{major}"',
            f'"Microsoft Edge";v="{major}"',
        ]
    else:
        brands = [
            f'"Not;A=Brand";v="99"',
            f'"Google Chrome";v="{major}"',
            f'"Chromium";v="{major}"',
        ]
    random.shuffle(brands)
    return ", ".join(brands)


@dataclass
class BrowserIdentity:
    """A coherent browser fingerprint: UA, client hints, and TLS target agree."""

    user_agent: str
    impersonate: str
    platform: str = "Windows"
    mobile: bool = False

    @property
    def sec_ch_ua(self) -> str:
        return build_sec_ch_ua(self.user_agent)

    @property
    def sec_ch_ua_platform(self) -> str:
        return f'"{self.platform}"'

    @property
    def sec_ch_ua_mobile(self) -> str:
        return "?1" if self.mobile else "?0"

    @property
    def accept_language(self) -> str:
        return ACCEPT_LANGUAGES

    def headers(self) -> dict[str, str]:
        """Full authentic header set for raw HTTP requests."""
        h = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": self.accept_language,
            "sec-ch-ua": self.sec_ch_ua,
            "sec-ch-ua-mobile": self.sec_ch_ua_mobile,
            "sec-ch-ua-platform": self.sec_ch_ua_platform,
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-User": "?1",
            "Sec-Fetch-Dest": "document",
            "Upgrade-Insecure-Requests": "1",
            "Connection": "keep-alive",
            "Cache-Control": "max-age=0",
        }
        return h

    def api_headers(self) -> dict[str, str]:
        """Headers for XHR/fetch-style API calls (navigation-only hints removed)."""
        h = self.headers()
        h["Sec-Fetch-Site"] = "same-origin"
        h["Sec-Fetch-Mode"] = "cors"
        h["Sec-Fetch-Dest"] = "empty"
        h["Accept"] = "application/json, text/plain, */*"
        h.pop("Upgrade-Insecure-Requests", None)
        # Sec-Fetch-User is only sent on user-initiated top-level navigations,
        # never on XHR/fetch API calls.
        h.pop("Sec-Fetch-User", None)
        return h


def random_identity(rng: Optional[random.Random] = None) -> BrowserIdentity:
    """Select an entire cohesive profile (UA + impersonation + platform)."""
    r = rng or random
    profile = r.choice(PROFILES)
    return BrowserIdentity(
        user_agent=profile["ua"],
        impersonate=profile["impersonate"],
        platform=profile["platform"],
    )


def jittered_delay(min_s: float = 1.5, max_s: float = 4.5, rng: Optional[random.Random] = None) -> float:
    r = rng or random
    return r.uniform(min_s, max_s)


def exponential_backoff(attempt: int, base: float = 1.5, cap: float = 45.0, rng: Optional[random.Random] = None) -> float:
    r = rng or random
    delay = min(cap, base * (2 ** attempt))
    return delay * r.uniform(0.75, 1.25)


@dataclass
class RateLimiter:
    """Enforces a minimum delay between requests to the same domain (sync)."""

    min_interval: float = 1.5
    _last: dict[str, float] = field(default_factory=dict)

    def wait_until_allowed(self, domain: str, now: float) -> float:
        """Return how long (seconds) to sleep before hitting `domain` at `now`."""
        last = self._last.get(domain)
        if last is None:
            return 0.0
        elapsed = now - last
        remaining = self.min_interval - elapsed
        return max(0.0, remaining)

    def mark(self, domain: str, now: float) -> None:
        self._last[domain] = now


def _domain_of(url: str) -> str:
    try:
        return urlsplit(url).netloc.lower() or url
    except ValueError:
        return url


class AsyncRateLimiter:
    """Async-safe per-domain rate limiter (lock-guarded; no burst fire)."""

    def __init__(self, min_interval: float = 1.5) -> None:
        self.min_interval = min_interval
        self._last: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, url: str) -> None:
        """Wait until the domain identified by `url` allows another request."""
        domain = _domain_of(url)
        async with self._lock:
            last = self._last.get(domain)
            now = asyncio.get_running_loop().time()
            wait = 0.0
            if last is not None:
                wait = max(0.0, self.min_interval - (now - last))
            self._last[domain] = now + wait
        if wait > 0:
            await asyncio.sleep(wait)

    def mark_now(self, url: str) -> None:
        """Record an immediate request timestamp for `url`'s domain."""
        domain = _domain_of(url)
        try:
            self._last[domain] = asyncio.get_running_loop().time()
        except RuntimeError:
            pass
