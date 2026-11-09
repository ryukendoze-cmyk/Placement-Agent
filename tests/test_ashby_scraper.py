from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")
NOW_MS = int(NOW.timestamp() * 1000)

from app.scraper.normalizer import normalize_scraped_job
from app.scraper.providers import AshbyATSProvider, BaseJobScraperProvider
from app.scraper.service import CompositeJobScraper
from app.taxonomy.taxonomy import load_taxonomy
from job_scraper.core.models import RawScrapedJob
from job_scraper.extractors.ashby import AshbyExtractor
from job_scraper.extractors.greenhouse import GreenhouseExtractor
from job_scraper.extractors.lever import LeverExtractor


MOCK_ASHBY_RESPONSE = {
    "jobs": [
        {
            "id": "ashby-1",
            "title": "Software Engineer, Backend",
            "location": "Bengaluru, Karnataka, India",
            "isRemote": False,
            "publishedAt": NOW_ISO,
            "jobUrl": "https://jobs.ashbyhq.com/acme/ashby-1",
            "applyUrl": "https://jobs.ashbyhq.com/acme/ashby-1/application",
            "descriptionPlain": "We are seeking a Backend Engineer with Python, FastAPI, and PostgreSQL skills.",
            "descriptionHtml": "<p>We are seeking a Backend Engineer with Python, FastAPI, and PostgreSQL skills.</p>",
            "department": "Engineering",
            "team": "Platform",
            "employmentType": "FullTime",
            "isListed": True,
            "compensation": {
                "summaryComponents": [
                    {
                        "compensationType": "Salary",
                        "minValue": 1800000,
                        "maxValue": 2500000,
                        "currencyCode": "INR",
                        "interval": "1 YEAR",
                    }
                ]
            },
        },
        {
            "id": "ashby-2",
            "title": "Staff Infrastructure Engineer",
            "location": "Remote - US",
            "isRemote": True,
            "publishedAt": NOW_ISO,
            "jobUrl": "https://jobs.ashbyhq.com/acme/ashby-2",
            "applyUrl": "https://jobs.ashbyhq.com/acme/ashby-2/application",
            "descriptionPlain": "US only role for Kubernetes expert.",
            "isListed": True,
        },
        {
            "id": "ashby-3",
            "title": "Full Stack Developer (Remote)",
            "location": "Remote - Global",
            "isRemote": True,
            "publishedAt": NOW_ISO,
            "jobUrl": "https://jobs.ashbyhq.com/acme/ashby-3",
            "descriptionPlain": "Build user interfaces with React, TypeScript, and Python.",
            "department": "Product",
            "isListed": True,
        },
        {
            "id": "ashby-4",
            "title": "Unlisted Draft Job",
            "location": "India",
            "isRemote": False,
            "jobUrl": "https://jobs.ashbyhq.com/acme/ashby-4",
            "descriptionPlain": "Draft",
            "isListed": False,
        },
    ]
}

MOCK_GREENHOUSE_RESPONSE = {
    "jobs": [
        {
            "id": 9001,
            "title": "Frontend Engineer, Design Systems",
            "location": {"name": "Bangalore, India"},
            "updated_at": NOW_ISO,
            "absolute_url": "https://job-boards.greenhouse.io/gitlab/jobs/9001",
            "content": "&lt;p&gt;Looking for React and CSS skills.&lt;/p&gt;",
            "departments": [{"name": "Frontend"}],
        },
        {
            "id": 9002,
            "title": "Regional Sales Director",
            "location": {"name": "Berlin, Germany"},
            "updated_at": NOW_ISO,
            "absolute_url": "https://job-boards.greenhouse.io/gitlab/jobs/9002",
            "content": "&lt;p&gt;Sales leadership.&lt;/p&gt;",
        },
    ]
}

MOCK_LEVER_RESPONSE = [
    {
        "id": "lever-1",
        "text": "Security Operations Analyst",
        "categories": {
            "commitment": "Full-time",
            "department": "Security",
            "location": "Hyderabad",
            "team": "SecOps",
        },
        "hostedUrl": "https://jobs.lever.co/zeta/lever-1",
        "descriptionPlain": "Analyze SIEM events with Python scripting.",
        "createdAt": NOW_MS,
        "salaryRange": {
            "min": 1200000,
            "max": 1600000,
            "currency": "INR",
            "interval": "per-year",
        },
    }
]


class TestAshbyExtractor:
    @pytest.mark.asyncio
    async def test_ashby_search_filtering_and_parsing(self):
        extractor = AshbyExtractor(companies=[{"company": "Acme", "slug": "acme"}])

        with patch.object(extractor, "_fetch_company_board", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_ASHBY_RESPONSE["jobs"]

            jobs = await extractor.search(keyword="python", location="India", limit=10)

            # ashby-1 (India, Python) and ashby-3 (Remote - Global, Python) should match.
            # ashby-2 (Remote - US) is excluded by non-India filter.
            # ashby-4 (isListed=False) is excluded.
            assert len(jobs) == 2
            titles = [j.title for j in jobs]
            assert "Software Engineer, Backend" in titles
            assert "Full Stack Developer (Remote)" in titles

            # Verify compensation conversion to LPA for INR
            job1 = next(j for j in jobs if j.title == "Software Engineer, Backend")
            assert job1.min_amount == 18.0
            assert job1.max_amount == 25.0
            assert job1.interval == "lpa"
            assert job1.site == "ashby"
            assert job1.scraper_engine == "ashby_harvester"

    @pytest.mark.asyncio
    async def test_ashby_location_matching(self):
        extractor = AshbyExtractor()

        # India / Indian cities
        assert extractor._matches_location("Bengaluru, India", False, "India") is True
        assert extractor._matches_location("Pune, Maharashtra", False, "India") is True

        # Remote Global
        assert extractor._matches_location("Remote - Global", True, "India") is True
        assert extractor._matches_location("Remote - Worldwide", True, "India") is True
        assert extractor._matches_location("Remote", True, "India") is True

        # Foreign / Restricted
        assert extractor._matches_location("Remote - US", True, "India") is False
        assert extractor._matches_location("Remote - European Union", True, "India") is False
        assert extractor._matches_location("London, United Kingdom", False, "India") is False


class TestGreenhouseExtractor:
    @pytest.mark.asyncio
    async def test_greenhouse_search_and_html_decode(self):
        extractor = GreenhouseExtractor(companies=[{"company": "GitLab", "slug": "gitlab"}])

        with patch.object(extractor, "_fetch_company_board", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_GREENHOUSE_RESPONSE["jobs"]

            jobs = await extractor.search(keyword="engineer", location="India", limit=10)
            assert len(jobs) == 1
            j = jobs[0]
            assert j.title == "Frontend Engineer, Design Systems"
            assert j.company == "GitLab"
            assert "Looking for React and CSS skills." in j.description
            assert j.site == "greenhouse"


class TestLeverExtractor:
    @pytest.mark.asyncio
    async def test_lever_search_and_salary(self):
        extractor = LeverExtractor(companies=[{"company": "Zeta", "slug": "zeta"}])

        with patch.object(extractor, "_fetch_company_board", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = MOCK_LEVER_RESPONSE

            jobs = await extractor.search(keyword="security", location="India", limit=10)
            assert len(jobs) == 1
            j = jobs[0]
            assert j.title == "Security Operations Analyst"
            assert j.company == "Zeta"
            assert j.min_amount == 12.0
            assert j.max_amount == 16.0
            assert j.interval == "lpa"
            assert j.site == "lever"


class TestAshbyATSProvider:
    def test_provider_registration_and_contract(self):
        provider = AshbyATSProvider()
        assert isinstance(provider, BaseJobScraperProvider)
        assert provider.name == "ashby_ats"
        assert "ashby" in provider.supported_sites
        assert "greenhouse" in provider.supported_sites
        assert "lever" in provider.supported_sites

        meta = provider.metadata()
        assert meta["name"] == "ashby_ats"
        assert "Ashby" in meta["display_name"]

    def test_composite_job_scraper_has_ashby_ats(self):
        composite = CompositeJobScraper()
        prov_names = [p["name"] for p in composite.get_available_providers()]
        assert "ashby_ats" in prov_names

    def test_taxonomy_normalization_on_ashby_job(self):
        tax = load_taxonomy()
        raw = {
            "title": "Junior Python Backend Developer",
            "company": "Linear",
            "location": "Bengaluru, India",
            "description": "Looking for freshers proficient in Python, SQL, Git, and Docker. 1 year experience required.",
            "job_url": "https://jobs.ashbyhq.com/linear/job-123",
            "site": "ashby",
            "scraper_engine": "ashby_ats",
            "min_amount": 14.0,
            "max_amount": 20.0,
            "interval": "lpa",
        }
        job = normalize_scraped_job(raw, tax)
        assert job.title == "Junior Python Backend Developer"
        assert job.company == "Linear"
        assert "python" in job.required_skills
        assert "sql" in job.required_skills or "git" in job.required_skills
        assert job.salary_lpa == [14, 20]
        assert job.salary_disclosed is True
        assert job.source == "ashby"
        assert job.scraper_engine == "ashby_ats"

    def test_ashby_usd_compensation_conversion(self):
        extractor = AshbyExtractor()
        raw_job = {
            "compensation": {
                "summaryComponents": [
                    {
                        "compensationType": "Salary",
                        "minValue": 207000,
                        "maxValue": 285000,
                        "currencyCode": "USD",
                    }
                ]
            }
        }
        lo, hi, iv = extractor._parse_compensation(raw_job)
        assert iv == "lpa"
        assert lo == pytest.approx(179.05, abs=0.5)
        assert hi == pytest.approx(246.52, abs=0.5)

        tax = load_taxonomy()
        raw = {
            "title": "US Remote Software Engineer",
            "company": "Figma",
            "location": "Remote",
            "description": "Python, SQL, AWS, Kubernetes.",
            "job_url": "https://jobs.ashbyhq.com/figma/job-999",
            "site": "ashby",
            "scraper_engine": "ashby_ats",
            "min_amount": lo,
            "max_amount": hi,
            "interval": iv,
        }
        job = normalize_scraped_job(raw, tax)
        assert job.salary_lpa == [179, 247]
        assert job.salary_disclosed is True
