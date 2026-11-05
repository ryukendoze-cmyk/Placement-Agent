import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from job_scraper.core.anti_detect import (
    ACCEPT_LANGUAGES,
    IMPERSONATIONS,
    USER_AGENTS,
    RateLimiter,
    build_sec_ch_ua,
    exponential_backoff,
    jittered_delay,
    random_identity,
)


class TestUserAgentPool:
    def test_all_desktop_chrome_or_edge(self):
        for ua in USER_AGENTS:
            assert "Chrome/" in ua
            assert "Windows NT 10.0" in ua or "Macintosh" in ua
            assert "Mobile" not in ua

    def test_impersonations_available_in_curl_cffi(self):
        import typing

        from curl_cffi.requests import BrowserTypeLiteral

        available = set(typing.get_args(BrowserTypeLiteral))
        for imp in IMPERSONATIONS:
            assert imp in available, f"{imp} not supported by curl_cffi"


class TestSecChUa:
    def test_matches_ua_chrome_major(self):
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        hint = build_sec_ch_ua(ua)
        assert '"Chromium";v="131"' in hint
        assert '"Google Chrome";v="131"' in hint

    def test_is_valid_rfc8941_structured_field(self):
        """Each brand token carries its own ;v="N" param; no trailing version."""
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        hint = build_sec_ch_ua(ua)
        tokens = [t.strip() for t in hint.split(",")]
        assert len(tokens) == 3
        for tok in tokens:
            # Structure: "Brand";v="N" (brand name itself may contain a semicolon)
            assert ';v="' in tok
            assert tok.endswith('"')
        # No unattached trailing "; v=..." parameter (invalid structured-field syntax).
        assert "; v=" not in hint

    def test_edge_branding(self):
        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36 Edg/136.0.0.0"
        hint = build_sec_ch_ua(ua)
        assert "Microsoft Edge" in hint
        assert '"Microsoft Edge";v="136"' in hint


class TestBrowserIdentity:
    def test_headers_include_india_accept_language(self):
        ident = random_identity()
        h = ident.headers()
        assert h["Accept-Language"] == ACCEPT_LANGUAGES
        assert "en-IN" in h["Accept-Language"]

    def test_client_hint_headers_present_and_consistent(self):
        ident = random_identity()
        h = ident.headers()
        assert "sec-ch-ua" in h
        assert h["sec-ch-ua-platform"] in ('"Windows"', '"macOS"')
        assert h["sec-ch-ua-mobile"] == "?0"
        major = ident.user_agent.split("Chrome/")[1].split(".")[0]
        assert f';v="{major}"' in h["sec-ch-ua"]
        assert "; v=" not in h["sec-ch-ua"]

    def test_api_headers_drop_nav_only_hints(self):
        ident = random_identity()
        h = ident.api_headers()
        assert h["Sec-Fetch-Mode"] == "cors"
        assert "Upgrade-Insecure-Requests" not in h
        assert "Sec-Fetch-User" not in h

    def test_random_identity_profile_is_cohesive(self):
        """UA major must match the TLS impersonation target's major."""
        for _ in range(40):
            ident = random_identity()
            ua_major = ident.user_agent.split("Chrome/")[1].split(".")[0]
            imp_major = "".join(ch for ch in ident.impersonate if ch.isdigit())
            assert ua_major == imp_major[: len(ua_major)] or ua_major.startswith(imp_major[:2])
            assert not ident.impersonate.startswith("safari")

    def test_identity_is_cohesive(self):
        ident = random_identity()
        assert isinstance(ident.impersonate, str) and ident.impersonate
        assert ident.platform == ("macOS" if "Macintosh" in ident.user_agent else "Windows")


class TestJitterAndBackoff:
    def test_jitter_within_bounds(self):
        for _ in range(50):
            d = jittered_delay(1.5, 4.5)
            assert 1.5 <= d <= 4.5

    def test_backoff_grows_exponentially_and_caps(self):
        d1 = exponential_backoff(0, base=1.5, cap=45.0)
        d3 = exponential_backoff(3, base=1.5, cap=45.0)
        d10 = exponential_backoff(10, base=1.5, cap=45.0)
        assert d1 < d3
        assert d10 <= 45.0 * 1.25


class TestRateLimiter:
    def test_first_request_no_wait(self):
        rl = RateLimiter(min_interval=2.0)
        assert rl.wait_until_allowed("naukri.com", now=100.0) == 0.0

    def test_enforces_min_interval(self):
        rl = RateLimiter(min_interval=2.0)
        rl.mark("naukri.com", now=100.0)
        assert rl.wait_until_allowed("naukri.com", now=101.0) == 1.0
        assert rl.wait_until_allowed("naukri.com", now=102.5) == 0.0

    def test_domains_independent(self):
        rl = RateLimiter(min_interval=2.0)
        rl.mark("a.com", now=100.0)
        assert rl.wait_until_allowed("b.com", now=100.1) == 0.0


class TestAsyncRateLimiter:
    async def test_first_acquire_no_sleep(self):
        from job_scraper.core.anti_detect import AsyncRateLimiter

        rl = AsyncRateLimiter(min_interval=0.05)
        await rl.acquire("https://example.org/a")

    async def test_second_acquire_waits(self):
        import time

        from job_scraper.core.anti_detect import AsyncRateLimiter

        rl = AsyncRateLimiter(min_interval=0.3)
        await rl.acquire("https://example.org/a")
        t0 = time.monotonic()
        await rl.acquire("https://example.org/a")
        assert time.monotonic() - t0 >= 0.25

    async def test_domains_independent(self):
        import time

        from job_scraper.core.anti_detect import AsyncRateLimiter

        rl = AsyncRateLimiter(min_interval=5.0)
        await rl.acquire("https://a.example/x")
        t0 = time.monotonic()
        await rl.acquire("https://b.example/y")
        assert time.monotonic() - t0 < 1.0
