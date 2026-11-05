import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from job_scraper.core.anti_detect import BrowserIdentity
from job_scraper.core.exceptions import (
    BotDetectionError,
    RateLimitBlockError,
    ScraperException,
)


class FakeResponse:
    def __init__(self, status_code: int, text: str = "", url: str = ""):
        self.status_code = status_code
        self.text = text
        self.url = url
        self.headers = {}

    def json(self):
        import json

        return json.loads(self.text)


class FakeSession:
    """Scriptable stand-in for curl_cffi's AsyncSession."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    async def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        if self.scripted:
            item = self.scripted.pop(0)
        else:
            item = FakeResponse(200, "ok")
        if isinstance(item, Exception):
            raise item
        item.url = item.url or url
        return item

    async def aclose(self):
        pass


def make_session(identity, scripted, **kw):
    from job_scraper.core.client import StealthSession

    fake = FakeSession(scripted)
    sess = StealthSession(
        identity=identity,
        session_factory=lambda: fake,
        **kw,
    )
    return sess, fake


IDENT = BrowserIdentity(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    impersonate="chrome131",
    platform="Windows",
)


class TestStealthSessionHeaders:
    async def test_sends_cohesive_identity_headers(self):
        sess, fake = make_session(IDENT, [FakeResponse(200, "ok")], min_interval=0.01)
        async with sess as s:
            resp = await s.get("https://www.example-jobs.in/x")
        assert resp.status_code == 200
        kwargs = fake.calls[0]["kwargs"]
        headers = {k.lower(): v for k, v in kwargs.get("headers", {}).items()}
        assert headers["user-agent"] == IDENT.user_agent
        assert "sec-ch-ua" in headers
        assert '"Chromium";v="131"' in headers["sec-ch-ua"]
        assert headers["accept-language"].startswith("en-IN")
        # Accept-Encoding/Connection must not override curl_cffi's TLS-bound values.
        assert "accept-encoding" not in headers
        assert "connection" not in headers

    async def test_caller_headers_override_and_extend(self):
        sess, fake = make_session(IDENT, [FakeResponse(200, "ok")], min_interval=0.01)
        async with sess as s:
            await s.get(
                "https://www.example-jobs.in/api",
                headers={"Referer": "https://www.example-jobs.in/", "X-Custom": "1"},
            )
        headers = {k.lower(): v for k, v in fake.calls[0]["kwargs"].get("headers", {}).items()}
        assert headers["referer"] == "https://www.example-jobs.in/"
        assert headers["x-custom"] == "1"


class TestStealthSessionRateLimiting:
    async def test_same_domain_requests_are_spaced(self):
        sess, _ = make_session(IDENT, [FakeResponse(200), FakeResponse(200)], min_interval=0.25)
        async with sess as s:
            t0 = time.monotonic()
            await s.get("https://a.example.com/1")
            await s.get("https://a.example.com/2")
            elapsed = time.monotonic() - t0
        assert elapsed >= 0.2

    async def test_different_domains_not_spaced(self):
        sess, _ = make_session(IDENT, [FakeResponse(200), FakeResponse(200)], min_interval=5.0)
        async with sess as s:
            await s.get("https://a.example.com/1")
            t0 = time.monotonic()
            await s.get("https://b.example.com/1")
            elapsed = time.monotonic() - t0
        assert elapsed < 1.0


class TestStealthSessionRetries:
    async def test_retries_5xx_then_succeeds(self):
        sess, fake = make_session(
            IDENT,
            [FakeResponse(503, "oops"), FakeResponse(200, "recovered")],
            min_interval=0.01,
            max_retries=3,
            backoff_base=0.01,
        )
        async with sess as s:
            resp = await s.get("https://a.example.com/x")
        assert resp.status_code == 200
        assert len(fake.calls) == 2

    async def test_retries_429_then_succeeds(self):
        sess, fake = make_session(
            IDENT,
            [FakeResponse(429, "slow down"), FakeResponse(200, "ok now")],
            min_interval=0.01,
            max_retries=3,
            backoff_base=0.01,
        )
        async with sess as s:
            resp = await s.get("https://a.example.com/x")
        assert resp.status_code == 200
        assert len(fake.calls) == 2

    async def test_429_exhaustion_raises_rate_limit_block(self):
        sess, fake = make_session(
            IDENT,
            [FakeResponse(429)] * 4,
            min_interval=0.01,
            max_retries=3,
            backoff_base=0.01,
        )
        with pytest.raises(RateLimitBlockError):
            async with sess as s:
                await s.get("https://a.example.com/x")
        assert len(fake.calls) == 4  # 1 + max_retries

    async def test_5xx_exhaustion_raises_scraper_exception(self):
        sess, _ = make_session(
            IDENT,
            [FakeResponse(500)] * 4,
            min_interval=0.01,
            max_retries=3,
            backoff_base=0.01,
        )
        with pytest.raises(ScraperException):
            async with sess as s:
                await s.get("https://a.example.com/x")


class TestStealthSessionBotDetection:
    @pytest.mark.parametrize(
        "body",
        [
            "<title>Just a moment...</title><div id='challenge-running'>",
            "Access Denied\nReference #18.7f",
            '{"message":"recaptcha required"}',
            "Unusual Activity Detected. Please verify",
        ],
    )
    async def test_challenge_pages_raise_bot_detection(self, body):
        sess, fake = make_session(
            IDENT,
            [FakeResponse(403, body)] * 4,
            min_interval=0.01,
            max_retries=3,
            backoff_base=0.01,
        )
        with pytest.raises(BotDetectionError):
            async with sess as s:
                await s.get("https://a.example.com/x")
        # Fail-fast: a static WAF wall must NOT be hammered with retries.
        assert len(fake.calls) == 1

    async def test_plain_403_is_returned_not_raised(self):
        sess, _ = make_session(IDENT, [FakeResponse(403, "forbidden here")], min_interval=0.01)
        async with sess as s:
            resp = await s.get("https://a.example.com/x")
        assert resp.status_code == 403


class TestStealthSessionTransport:
    async def test_post_sends_json(self):
        sess, fake = make_session(IDENT, [FakeResponse(200, '{"ok": true}')], min_interval=0.01)
        async with sess as s:
            resp = await s.post("https://a.example.com/api", json_body={"q": "python"})
        assert resp.json()["ok"] is True
        call = fake.calls[0]
        assert call["method"] == "POST"
        assert call["kwargs"].get("json") == {"q": "python"}

    async def test_network_error_retried_then_raises(self):
        sess, _ = make_session(
            IDENT,
            [ConnectionError("reset"), ConnectionError("reset"), ConnectionError("reset")],
            min_interval=0.01,
            max_retries=2,
            backoff_base=0.01,
        )
        with pytest.raises(ScraperException):
            async with sess as s:
                await s.get("https://a.example.com/x")


class TestBrowserStealthContextConfig:
    def test_launch_args_hide_automation(self):
        from job_scraper.core.client import BrowserStealthContext

        args = BrowserStealthContext.build_launch_args()
        assert "--disable-blink-features=AutomationControlled" in args
        assert "--no-sandbox" in args

    def test_context_options_are_india_desktop(self):
        from job_scraper.core.client import BrowserStealthContext

        ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        opts = BrowserStealthContext.build_context_options(user_agent=ua)
        assert opts["locale"] == "en-IN"
        assert opts["timezone_id"] == "Asia/Kolkata"
        assert opts["viewport"] == {"width": 1920, "height": 1080}
        assert opts["user_agent"] == ua
        assert "Windows NT 10.0" in opts["user_agent"]

    def test_random_context_options_use_cohesive_profile(self):
        """Default (random) options must draw a UA that belongs to a
        cohesive profile — never a macOS UA paired with a Windows assumption."""
        from job_scraper.core.anti_detect import PROFILES
        from job_scraper.core.client import BrowserStealthContext

        known_uas = {p["ua"] for p in PROFILES}
        for _ in range(10):
            opts = BrowserStealthContext.build_context_options()
            assert opts["user_agent"] in known_uas
            ua = opts["user_agent"]
            if "Macintosh" in ua:
                assert "Mac OS X" in ua
            else:
                assert "Windows NT 10.0" in ua


class TestBrowserStealthContextLive:
    @pytest.mark.live
    async def test_stealth_page_masks_webdriver(self):
        from job_scraper.core.client import BrowserStealthContext

        async with BrowserStealthContext() as page:
            await page.goto("https://example.com", wait_until="domcontentloaded", timeout=60000)
            webdriver = await page.evaluate("() => navigator.webdriver")
            assert not webdriver
            ua = await page.evaluate("() => navigator.userAgent")
            assert "Chrome/" in ua


class TestStealthSessionLiveTLS:
    @pytest.mark.live
    async def test_tls_fingerprint_is_real_chrome(self):
        """Server-side TLS inspection must see a genuine Chrome JA3/JA4 + HTTP/2."""
        from job_scraper.core.client import StealthSession

        async with StealthSession(min_interval=0.5, max_retries=2) as s:
            resp = await s.get("https://tls.browserleaks.com/json")
        assert resp.status_code == 200
        data = resp.json()
        ja4 = data.get("ja4") or ""
        assert ja4.startswith("t13"), f"expected TLS1.3, got {ja4}"
        assert "h2" in ja4, "expected HTTP/2 ALPN"
        ja3 = data.get("ja3_hash") or ""
        assert len(ja3) == 32, "expected 32-hex JA3 hash"
        # The server-observed UA must be a modern desktop Chrome.
        ua = data.get("user_agent") or ""
        assert "Chrome/1" in ua and "Mobile" not in ua
