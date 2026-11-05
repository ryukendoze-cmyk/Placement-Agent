import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FIX = Path(__file__).resolve().parent / "fixtures"


def read_text(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8", errors="replace")


def read_json(name: str):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class TestBaseHelpers:
    def test_html_to_text_strips_scripts(self):
        from job_scraper.extractors.base import BaseExtractor

        html = "<p>Hello</p><script>var x = 1;</script><div>World</div>"
        text = BaseExtractor.html_to_text(html)
        assert "Hello" in text
        assert "World" in text
        assert "var x" not in text

    def test_ms_to_iso(self):
        from job_scraper.extractors.base import BaseExtractor

        assert BaseExtractor.ms_to_iso(1762073400000) == "2025-11-02T08:50:00Z"
        assert BaseExtractor.ms_to_iso(None) is None
        assert BaseExtractor.ms_to_iso(0) is None

    def test_location_match_pan_india(self):
        from job_scraper.extractors.base import BaseExtractor

        assert BaseExtractor.location_match("Bengaluru", "India")
        assert BaseExtractor.location_match("Bengaluru", "")

    def test_location_match_city_aliases(self):
        from job_scraper.extractors.base import BaseExtractor

        assert BaseExtractor.location_match("Bangalore, Karnataka", "Bengaluru")
        assert BaseExtractor.location_match("Bengaluru", "Bengaluru")
        assert BaseExtractor.location_match("Gurgaon", "Delhi NCR")
        assert not BaseExtractor.location_match("Chennai", "Bengaluru")

    def test_is_fresh(self):
        from job_scraper.extractors.base import BaseExtractor

        assert BaseExtractor.is_fresh("2026-11-01T00:00:00Z", 7)
        assert not BaseExtractor.is_fresh("2026-09-01T00:00:00Z", 7)
        assert BaseExtractor.is_fresh(None, 7)  # unknown date kept


class TestFounditParser:
    def test_parses_all_items(self):
        from job_scraper.extractors.foundit import parse_foundit_payload

        payload = read_json("foundit_search.json")
        jobs = parse_foundit_payload(payload)
        assert len(jobs) == 5
        first = jobs[0]
        assert first.title == "Senior Python Developer"
        assert first.company == "Tata Consultancy Services"
        assert first.job_url.startswith("https://www.foundit.in/job/")
        assert first.date_posted and first.date_posted.endswith("Z")
        assert first.experience_years == 7.0
        assert "SAP IS-U" in first.tags or "MDM" in first.tags

    def test_salary_hidden_defaults_none(self):
        from job_scraper.extractors.foundit import parse_foundit_payload

        payload = read_json("foundit_search.json")
        jobs = parse_foundit_payload(payload)
        for j in jobs:
            assert j.min_amount is None or j.min_amount > 0


class TestUnstopParser:
    def test_parses_all_items(self):
        from job_scraper.extractors.unstop import parse_unstop_payload

        items = read_json("unstop_search.json")
        jobs = parse_unstop_payload(items)
        assert len(jobs) == len(items)
        first = jobs[0]
        assert first.job_url.startswith("https://unstop.com/jobs/")
        assert first.company == "TrellisConsulting"
        assert first.date_posted == "2026-10-30T00:00:00Z"

    def test_annual_salary_detected(self):
        from job_scraper.extractors.unstop import parse_unstop_payload

        items = read_json("unstop_search.json")
        jobs = parse_unstop_payload(items)
        first = jobs[0]
        assert first.min_amount == 400000.0
        assert first.max_amount == 600000.0
        assert first.interval == "annual"

    def test_skills_mapped(self):
        from job_scraper.extractors.unstop import parse_unstop_payload

        items = read_json("unstop_search.json")
        jobs = parse_unstop_payload(items)
        assert "Python" in jobs[0].tags


class TestHiristParser:
    def test_parses_all_items(self):
        from job_scraper.extractors.hirist import parse_hirist_payload

        data = read_json("hirist_search.json")
        jobs = parse_hirist_payload(data)
        assert len(jobs) == 5
        first = jobs[0]
        assert first.title == "Python Developer - FastAPI"
        assert first.company == "Techmatters Technologies"
        assert first.job_url.startswith("https://www.hirist.tech/j/")
        assert first.experience_years == 4.0
        assert "Python" in first.tags and "FastAPI" in first.tags

    def test_detail_description(self):
        from job_scraper.extractors.hirist import parse_hirist_detail

        detail = read_json("hirist_detail.json")
        desc = parse_hirist_detail(detail)
        assert len(desc) > 200
        assert "Python" in desc


class TestInstahyreParser:
    def test_parses_all_items(self):
        from job_scraper.extractors.instahyre import parse_instahyre_payload

        objects = read_json("instahyre_search.json")
        jobs = parse_instahyre_payload(objects)
        assert len(jobs) == 5
        first = jobs[0]
        assert first.title == "Python Developer"
        assert first.company == "TopDoc"
        assert first.job_url.startswith("https://www.instahyre.com/")
        assert "Python" in first.tags

    def test_description_extraction_and_trim(self):
        from job_scraper.extractors.instahyre import extract_instahyre_description

        html = read_text("instahyre_detail.html")
        desc = extract_instahyre_description(html)
        assert len(desc) > 100
        assert "similar jobs" not in desc.lower()


class TestIndeedParser:
    def test_extracts_mosaic_and_parses_results(self):
        from job_scraper.extractors.indeed import parse_indeed_html

        html = read_text("indeed_serp.html")
        jobs = parse_indeed_html(html)
        assert len(jobs) == 15
        first = jobs[0]
        assert first.company == "Rodeo FX"
        assert first.title == "Python Software Developer"
        assert first.job_url.startswith("https://in.indeed.com/viewjob?jk=")
        assert first.location == "Bengaluru, Karnataka"
        assert first.date_posted == "2026-10-30T14:01:05Z"

    def test_mosaic_json_extracts(self):
        from job_scraper.extractors.indeed import extract_mosaic_json

        html = read_text("indeed_serp.html")
        payload = extract_mosaic_json(html)
        assert payload is not None
        results = payload["metaData"]["mosaicProviderJobCardsModel"]["results"]
        assert len(results) == 15
        assert all(r.get("jobkey") for r in results)

    async def test_unresolved_browser_challenge_raises_bot_detection(self):
        """Tier B that can't solve the Cloudflare wall must raise
        BotDetectionError (not silently return zero jobs)."""
        from job_scraper.core.exceptions import BotDetectionError
        from job_scraper.extractors.indeed import IndeedExtractor

        class FakePage:
            async def goto(self, *a, **k):
                pass

            async def wait_for_function(self, *a, **k):
                raise TimeoutError("mosaic never appeared")

            def frame_locator(self, *a, **k):
                raise RuntimeError("no challenge iframe")

            async def evaluate(self, *a, **k):
                return None

            async def content(self):
                return "<html><head><title>Just a moment...</title></head></html>"

            async def title(self):
                return "Just a moment..."

        class FakeBrowserCtx:
            async def __aenter__(self):
                return FakePage()

            async def __aexit__(self, *a):
                return None

        ext = IndeedExtractor(session=object())
        ext._browser = FakeBrowserCtx()
        with pytest.raises(BotDetectionError):
            await ext._fetch_page("https://in.indeed.com/jobs?q=python", True)



class TestInternshalaParser:
    def test_parses_internship_cards(self):
        from job_scraper.extractors.internshala import parse_internshala_html

        html = read_text("internshala_internships.html")
        jobs = parse_internshala_html(html)
        assert len(jobs) >= 30
        first = jobs[0]
        assert first.title == "Python Development"
        assert first.company == "UV Farms Private Limited"
        assert first.job_url.startswith("https://internshala.com/internship/detail/")
        assert first.location == "Nagercoil"

    def test_parses_job_cards(self):
        from job_scraper.extractors.internshala import parse_internshala_html

        html = read_text("internshala_jobs.html")
        jobs = parse_internshala_html(html)
        assert len(jobs) >= 20
        assert all(j.job_url.startswith("https://internshala.com/") for j in jobs)

    def test_stipend_parsed_monthly(self):
        from job_scraper.extractors.internshala import parse_internshala_html

        html = read_text("internshala_internships.html")
        jobs = parse_internshala_html(html)
        with_sal = [j for j in jobs if j.min_amount is not None]
        assert with_sal, "expected at least one card with stipend"
        assert with_sal[0].interval == "monthly"
        assert with_sal[0].min_amount <= with_sal[0].max_amount


class TestLinkedInParser:
    def test_parses_cards(self):
        from job_scraper.extractors.linkedin import parse_linkedin_cards

        html = read_text("linkedin_cards.html")
        cards = parse_linkedin_cards(html)
        assert len(cards) == 10
        first = cards[0]
        assert first["url"].startswith("https://in.linkedin.com/jobs/view/")
        assert "?" not in first["url"]
        assert first["date"].startswith("2026-")
        assert first["title"]

    def test_detail_description(self):
        from job_scraper.extractors.linkedin import extract_linkedin_detail

        html = read_text("linkedin_detail.html")
        desc, _, criteria = extract_linkedin_detail(html)
        assert len(desc) > 500
        assert "job-criteria" not in desc

    def test_ensure_descriptions_fills_blocked_details(self):
        """Jobs whose detail pages were 429'd get a substantial fallback
        description (empty JDs break downstream ATS matching)."""
        from job_scraper.core.models import RawScrapedJob
        from job_scraper.extractors.linkedin import ensure_descriptions

        good = RawScrapedJob(
            title="Backend", company="A", location="Pune",
            description="A perfectly long description that is well over forty characters long today.",
            job_url="https://in.linkedin.com/jobs/view/x", site="linkedin",
        )
        blocked = RawScrapedJob(
            title="Python Developer", company="B", location="Remote",
            description="", job_url="https://in.linkedin.com/jobs/view/y", site="linkedin",
        )
        ensure_descriptions([good, blocked])
        assert "perfectly long" in good.description
        assert len(blocked.description) > 200
        assert "Python Developer" in blocked.description
        assert "rate-limited" in blocked.description



class TestNaukriParser:
    def test_parses_job_details(self):
        from job_scraper.extractors.naukri import parse_naukri_payload

        fixture = read_json("naukri_search.json")
        jobs = parse_naukri_payload(fixture["jobs"])
        assert len(jobs) == 20
        first = jobs[0]
        assert first.title == "IN Senior Associate Python Developer"
        assert first.company == "PwC Service Delivery Center"
        assert first.job_url.startswith("https://www.naukri.com/job-listings-")
        assert first.experience_years == 9.0
        assert "Automation" in first.tags
        assert len(first.description) > 500
        assert first.location == "Bengaluru"
