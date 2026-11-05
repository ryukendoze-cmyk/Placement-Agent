import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from job_scraper.core.models import PlacementJob, RawScrapedJob
from job_scraper.core.normalizer import (
    clean_url,
    dedupe_jobs,
    extract_skills,
    generate_job_id,
    parse_date_posted,
    parse_experience_years,
    parse_salary_lpa,
    to_placement_agent_job,
)


class TestCleanUrl:
    def test_strips_utm_and_tracking_params(self):
        url = "https://www.naukri.com/job-listings-python-developer?utm_source=google&ref=abc&trk=xyz&gclid=123&id=keep"
        assert clean_url(url) == "https://www.naukri.com/job-listings-python-developer?id=keep"

    def test_strips_fragment_and_trailing_slash(self):
        assert clean_url("https://in.indeed.com/viewjob/?jk=abc123#apply") == "https://in.indeed.com/viewjob?jk=abc123"
        assert clean_url("https://foundit.in/s/software-jobs/") == "https://foundit.in/s/software-jobs"

    def test_strips_fbclid_and_ref_params(self):
        url = "https://www.linkedin.com/jobs/view/123?refId=xyz&trackingId=abc&fbclid=1"
        result = clean_url(url)
        assert "fbclid" not in result
        assert result == "https://www.linkedin.com/jobs/view/123"

    def test_preserves_essential_query_params(self):
        url = "https://in.indeed.com/viewjob?jk=4bb1c8f9&tok=abc"
        assert clean_url(url) == "https://in.indeed.com/viewjob?jk=4bb1c8f9&tok=abc"

    def test_deterministic_output(self):
        u = "https://www.naukri.com/job-listings-x?utm_source=a&utm_medium=b"
        assert clean_url(u) == clean_url(u + "&utm_campaign=c")


class TestGenerateJobId:
    def test_matches_spec_format(self):
        canonical = "https://www.naukri.com/job-listings-python-developer-sw"
        expected = "job-ext-" + hashlib.md5(canonical.encode()).hexdigest()[:10]
        assert generate_job_id(canonical) == expected

    def test_deterministic(self):
        u = "https://example-jobs.in/x"  # not a placeholder URL, just a domain name
        assert generate_job_id(u) == generate_job_id(u)

    def test_different_urls_different_ids(self):
        assert generate_job_id("https://a.com/1") != generate_job_id("https://a.com/2")


class TestParseSalaryLpa:
    @pytest.mark.parametrize(
        "min_amt,max_amt,interval,expected",
        [
            (10.0, 18.0, "lpa", [10, 18]),
            (100000, 180000, "monthly", [12, 22]),
            (1200000, 2400000, "annual", [12, 24]),
            (None, None, None, [0, 0]),
            (8, None, "lpa", [8, 8]),
        ],
    )
    def test_various_intervals(self, min_amt, max_amt, interval, expected):
        assert parse_salary_lpa(min_amt, max_amt, interval) == expected

    def test_monthly_inr_to_lpa(self):
        assert parse_salary_lpa(50000, 80000, "monthly") == [6, 10]

    def test_annual_inr_to_lpa(self):
        assert parse_salary_lpa(900000, 1500000, "annual") == [9, 15]

    def test_default_when_unstated(self):
        """Salary integrity (Milestone 8): absent salary -> [0, 0], NEVER synthetic [6, 12]."""
        assert parse_salary_lpa(None, None, None) == [0, 0]
        assert parse_salary_lpa(0, 0, "lpa") == [0, 0]
        # Explicitly reject the obsolete fresher heuristic default
        assert parse_salary_lpa(None, None, None) != [6, 12]
        assert parse_salary_lpa(0, 0, "lpa") != [6, 12]

    def test_empty_strings_unparseable_returns_zero(self):
        """Empty/unparseable inputs -> [0, 0] + never [6, 12]."""
        assert parse_salary_lpa("", "", "lpa") == [0, 0]
        assert parse_salary_lpa("abc", "xyz", "lpa") == [0, 0]

    def test_reversed_range_normalized(self):
        assert parse_salary_lpa(18, 10, "lpa") == [10, 18]


class TestExtractSkills:
    def test_extracts_known_tech_skills(self):
        text = "We use Python, FastAPI, PostgreSQL, Docker and AWS. React is a plus."
        skills = extract_skills(text)
        assert "python" in skills
        assert "fastapi" in skills
        assert "postgresql" in skills or "sql" in skills
        assert "docker" in skills
        assert "aws" in skills
        assert "react" in skills

    def test_case_insensitive(self):
        assert "java" in extract_skills("Experience with JAVA and Spring Boot")

    def test_does_not_match_substrings_word_boundary(self):
        # "java" should not match inside "javascript" incorrectly counted as java only when standalone
        skills = extract_skills("javascript developer")
        assert "javascript" in skills
        assert "java" not in skills

    def test_empty_text(self):
        assert extract_skills("") == []


class TestParseExperienceYears:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2-5 years", 2),
            ("0-2 years", 0),
            ("Fresher", 0),
            ("3+ years", 3),
            ("5 yrs experience", 5),
            ("Experienced 4 years", 4),
        ],
    )
    def test_experience_patterns(self, text, expected):
        assert parse_experience_years(text) == expected

    def test_none_when_unparseable(self):
        assert parse_experience_years("Senior role") is None


class TestParseDatePosted:
    def test_epoch_millis(self):
        result = parse_date_posted(1762073400000)
        assert result == "2025-11-02T08:50:00Z"

    def test_epoch_seconds(self):
        result = parse_date_posted(1762073400)
        assert result == "2025-11-02T08:50:00Z"

    def test_iso_string_with_z(self):
        assert parse_date_posted("2026-10-25T14:22:00.000Z") == "2026-10-25T14:22:00Z"

    def test_relative_days(self):
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        result = datetime.fromisoformat(parse_date_posted("3 days ago").replace("Z", "+00:00"))
        expected = now - timedelta(days=3)
        assert abs((result - expected).total_seconds()) < 3600

    def test_today_and_few_hours(self):
        from datetime import datetime, timezone

        result = parse_date_posted("Few Hours Ago")
        assert result is not None
        d = datetime.fromisoformat(result.replace("Z", "+00:00"))
        assert (datetime.now(timezone.utc) - d).total_seconds() < 3600 * 5
        assert parse_date_posted("Today") is not None

    def test_numeric_string_epoch(self):
        assert parse_date_posted("1762073400000") == "2025-11-02T08:50:00Z"

    def test_none_when_unparseable(self):
        assert parse_date_posted("unknown") is None
        assert parse_date_posted("") is None
        assert parse_date_posted(None) is None


class TestToPlacementAgentJob:
    def _raw(self):
        return {
            "title": "Backend Engineer (Python / FastAPI)",
            "company": "Swiggy",
            "location": "Bengaluru",
            "description": "Build microservices. Apply URL: https://www.naukri.com/job-listings-abc",
            "job_url": "https://www.naukri.com/job-listings-abc?utm_source=x",
            "site": "naukri",
            "scraper_engine": "stealth_harvester",
            "min_amount": 10.0,
            "max_amount": 18.0,
            "interval": "lpa",
            "is_remote": False,
            "experience_years": 1,
            "tags": ["python", "fastapi"],
        }

    def test_output_has_required_fields(self):
        job = to_placement_agent_job(self._raw())
        for key in (
            "id", "title", "company", "description", "required_skills",
            "preferred_skills", "experience_years", "location", "remote",
            "salary_lpa", "min_gpa", "allowed_branches", "job_url",
            "source", "scraper_engine",
        ):
            assert key in job, f"missing {key}"

    def test_id_is_md5_of_canonical_url(self):
        job = to_placement_agent_job(self._raw())
        canonical = clean_url("https://www.naukri.com/job-listings-abc?utm_source=x")
        assert job["id"] == generate_job_id(canonical)

    def test_description_appends_source_footer(self):
        job = to_placement_agent_job(self._raw())
        assert "Source: naukri (Scraped via stealth_harvester)" in job["description"]
        assert "Apply URL: https://www.naukri.com/job-listings-abc" in job["description"]

    def test_tags_are_mapped_through_taxonomy(self):
        raw = self._raw()
        raw["tags"] = ["Python", "FastAPI", "Business process improvement", "Analytical"]
        raw["description"] = "Short."
        job = to_placement_agent_job(raw)
        assert "python" in job["required_skills"]
        assert "fastapi" in job["required_skills"]
        # Non-taxonomy phrases must not leak into required skills.
        assert "business process improvement" not in job["required_skills"]

    def test_sparse_tags_supplemented_from_description(self):
        raw = self._raw()
        raw["tags"] = ["python"]
        raw["description"] = "We use FastAPI, PostgreSQL, Docker, AWS, Redis, Git daily."
        job = to_placement_agent_job(raw)
        all_skills = job["required_skills"] + job["preferred_skills"]
        assert "fastapi" in all_skills
        assert "docker" in all_skills

    def test_salary_is_integer_pair(self):
        job = to_placement_agent_job(self._raw())
        assert job["salary_lpa"] == [10, 18]
        assert all(isinstance(x, int) for x in job["salary_lpa"])

    def test_validates_against_placement_model(self):
        job = to_placement_agent_job(self._raw())
        PlacementJob(**job)  # should not raise

    def test_rejects_placeholder_url(self):
        raw = self._raw()
        raw["job_url"] = "https://example.com/job"
        raw["description"] = "x"
        with pytest.raises(Exception):
            to_placement_agent_job(raw)


class TestRawModelValidation:
    def test_rejects_example_com_url(self):
        with pytest.raises(Exception):
            RawScrapedJob(
                title="X", company="Y", job_url="https://example.com/j",
                site="naukri",
            )

    def test_rejects_relative_url(self):
        with pytest.raises(Exception):
            RawScrapedJob(title="X", company="Y", job_url="/job/1", site="naukri")


class TestDedupe:
    def test_dedupes_by_canonical_url(self):
        a = {"title": "Dev", "company": "A", "job_url": "https://x.com/j/1?utm_source=1", "description": "short"}
        b = {"title": "Dev", "company": "A", "job_url": "https://x.com/j/1?ref=y", "description": "much longer description here"}
        out = dedupe_jobs([a, b])
        assert len(out) == 1
        assert out[0]["description"] == "much longer description here"

    def test_soft_dedupe_title_company_prefers_longer_description(self):
        a = {"title": "Python Developer", "company": "Infosys", "location": "Bengaluru", "job_url": "https://x.com/j/1", "description": "a"}
        b = {"title": "Python Developer ", "company": "Infosys", "location": "Bengaluru", "job_url": "https://x.com/j/2", "description": "a much longer and more informative description"}
        out = dedupe_jobs([a, b])
        assert len(out) == 1
        assert out[0]["description"].startswith("a much longer")

    def test_keeps_distinct_jobs(self):
        a = {"title": "Dev", "company": "A", "job_url": "https://x.com/j/1"}
        b = {"title": "Dev", "company": "B", "job_url": "https://x.com/j/2"}
        assert len(dedupe_jobs([a, b])) == 2

    def test_soft_dedupe_does_not_collapse_same_role_different_locations(self):
        """Same title+company in different cities are distinct openings."""
        a = {"title": "Python Developer", "company": "Infosys", "location": "Bengaluru", "job_url": "https://x.com/j/1", "description": "short"}
        b = {"title": "Python Developer", "company": "Infosys", "location": "Pune", "job_url": "https://x.com/j/2", "description": "short"}
        assert len(dedupe_jobs([a, b])) == 2
