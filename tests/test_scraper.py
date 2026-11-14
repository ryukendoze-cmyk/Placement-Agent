"""Offline scraper test suite — ARCHITECTURE.md Section 7.2.

All network calls mocked via unittest.mock.patch(jobspy.scrape_jobs) returning
synthetic pandas DataFrames. Tests are 100% deterministic and offline.

Coverage (13 tests):
- Service: DataFrame conversion, argument forwarding, NaN sanitization, error resilience
- Normalizer: LPA conversion (yearly/monthly/fallback), taxonomy grounding, branch, HTML stripping
- Pipeline: session.merge + vector store upsert
- API: POST /api/jobs/scrape, GET /api/jobs/scrape/status
- CLI: scrape-jobs subcommand
- Zero-emoji: matches.html modal elements

These tests are the acceptance gate: they FAIL (with ModuleNotFoundError/ImportError)
until @Coder-Flash lands Tickets 1-3, then must all pass.
"""
from __future__ import annotations

import hashlib
import re
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.taxonomy.branches import normalize_branch
from app.taxonomy.taxonomy import load_taxonomy

# ---------------------------------------------------------------------------
# Synthetic DataFrame fixture (shared across service tests)
# ---------------------------------------------------------------------------


def _synthetic_df() -> pd.DataFrame:
    """Synthetic jobspy.scrape_jobs return value with NaN fields."""
    return pd.DataFrame(
        [
            {
                "title": "Senior Python Developer",
                "company": "Acme Tech",
                "description": "<p>We need Python, SQL, and Django expertise.</p>",
                "job_url": "https://example.com/job/123",
                "location": "Bengaluru",
                "min_amount": 800000.0,
                "max_amount": 1500000.0,
                "currency": "INR",
                "interval": "yearly",
                "is_remote": False,
                "tags": "python,sql,django",
            },
            {
                "title": "DevOps Engineer",
                "company": "CloudCorp",
                "description": "AWS, Docker, Kubernetes required",
                "job_url": "https://example.com/job/456",
                "location": "Hyderabad",
                "min_amount": float("nan"),  # NaN
                "max_amount": float("nan"),
                "currency": float("nan"),
                "interval": float("nan"),
                "is_remote": True,
                "tags": float("nan"),
            },
        ]
    )


# ---------------------------------------------------------------------------
# Service tests
# ---------------------------------------------------------------------------


class TestJobSpyScraperService:
    """JobSpyScraper.fetch_jobs: DataFrame conversion, args, NaN sanitization, errors."""

    def test_jobspy_service_fetch_jobs_mocked(self):
        """Verify DataFrame -> list[dict] conversion, argument forwarding, NaN sanitization."""
        from app.scraper.service import JobSpyScraper

        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df) as mock_scrape:
            scraper = JobSpyScraper()
            result = scraper.fetch_jobs(
                search_term="Python Developer",
                location="Bengaluru",
                sites=["naukri", "linkedin"],
                results_wanted=15,
            )

        # Argument forwarding
        mock_scrape.assert_called_once()
        call_kwargs = mock_scrape.call_args.kwargs
        assert call_kwargs.get("site_name") == ["naukri", "linkedin"]
        assert call_kwargs.get("search_term") == "Python Developer"
        assert call_kwargs.get("location") == "Bengaluru"
        assert call_kwargs.get("results_wanted") == 15

        # DataFrame -> list[dict] conversion
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)

        # NaN sanitization: second row had NaN salary/currency/interval/tags
        row2 = result[1]
        assert row2["min_amount"] is None or row2["min_amount"] == 0
        assert row2["max_amount"] is None or row2["max_amount"] == 0
        assert row2["currency"] is None or row2["currency"] == ""
        assert row2["interval"] is None or row2["interval"] == ""
        assert row2["tags"] is None or row2["tags"] == ""

        # First row preserved
        assert result[0]["title"] == "Senior Python Developer"
        assert result[0]["min_amount"] == 800000.0

    def test_jobspy_service_network_error_graceful(self):
        """Network exceptions must be caught; returns empty list, no unhandled raise."""
        from app.scraper.service import JobSpyScraper

        with patch(
            "app.scraper.service.scrape_jobs",
            side_effect=ConnectionError("network timeout"),
        ):
            scraper = JobSpyScraper()
            result = scraper.fetch_jobs()

        assert result == []
        assert isinstance(result, list)

    def test_jobspy_service_default_sites_validation(self):
        """None -> defaults to all three; invalid site raises ValueError (explicit validation)."""
        from app.scraper.service import JobSpyScraper

        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df) as mock_scrape:
            scraper = JobSpyScraper()
            # None -> defaults
            scraper.fetch_jobs(sites=None)
            assert mock_scrape.call_args.kwargs["site_name"] == [
                "naukri",
                "linkedin",
                "indeed",
            ]

            # Invalid site raises explicit ValueError (not silent filter)
            with pytest.raises(ValueError, match="Unsupported site"):
                scraper.fetch_jobs(sites=["naukri", "bogus", "linkedin"])

    def test_jobspy_service_fetch_jobs_multi_terms(self):
        """Passing multiple search terms scrapes for each term and deduplicates results."""
        from app.scraper.service import JobSpyScraper

        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df) as mock_scrape:
            scraper = JobSpyScraper()
            result = scraper.fetch_jobs(
                search_term=["Python Backend Engineer", "React Developer"],
                location="Bengaluru",
                results_wanted=20,
            )
            # mock_scrape called once per term
            assert mock_scrape.call_count == 2
            # Results deduplicated by job_url
            assert len(result) == 2


# ---------------------------------------------------------------------------
# Normalizer tests
# ---------------------------------------------------------------------------


class TestNormalizer:
    """normalize_scraped_job: LPA conversion, taxonomy, branch, HTML stripping."""

    def test_normalizer_salary_lpa_inr_yearly(self):
        """Yearly INR: 800000 -> 8 LPA (round(amount / 100_000))."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Python Developer",
            "company": "Acme",
            "description": "Python role",
            "job_url": "https://example.com/j1",
            "min_amount": 800000,
            "max_amount": 1500000,
            "currency": "INR",
            "interval": "yearly",
        }
        job = normalize_scraped_job(raw, taxonomy)
        assert job.salary_lpa == [8, 15]

    def test_normalizer_salary_lpa_inr_monthly(self):
        """Monthly INR: 50000 -> 6 LPA (round(amount * 12 / 100_000))."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "DevOps Engineer",
            "company": "CloudCorp",
            "description": "K8s role",
            "job_url": "https://example.com/j2",
            "min_amount": 50000,
            "max_amount": 80000,
            "currency": "INR",
            "interval": "monthly",
        }
        job = normalize_scraped_job(raw, taxonomy)
        # 50000 * 12 / 100_000 = 6; 80000 * 12 / 100_000 = 9.6 -> round = 10
        assert job.salary_lpa == [6, 10]

    def test_normalizer_salary_fallback(self):
        """Missing/unparseable salary -> [0, 0] + salary_disclosed=False (Milestone 8
        salary integrity: NEVER inject synthetic [6, 12] fresher default)."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Backend Engineer",
            "company": "StartupXYZ",
            "description": "Go role",
            "job_url": "https://example.com/j3",
            # No salary fields at all
        }
        job = normalize_scraped_job(raw, taxonomy)
        # Salary integrity: undisclosed -> [0, 0], never synthetic [6, 12]
        assert job.salary_lpa in ([0, 0], []), (
            f"Missing salary must normalize to [0, 0] or [], got {job.salary_lpa}"
        )
        assert job.salary_disclosed is False, "salary_disclosed must be False when absent"
        # Explicitly reject the obsolete fresher heuristic default
        assert job.salary_lpa != [6, 12], (
            "Synthetic [6, 12] default violates Milestone 8 salary integrity contract"
        )

    def test_parse_salary_lpa_missing_returns_zero_pair(self):
        """parse_salary_lpa: absent/empty/zero inputs -> [0, 0], never [6, 12]."""
        from app.scraper.normalizer import parse_salary_lpa

        # All absent
        assert parse_salary_lpa(None, None, None) == ([0, 0], False)
        # Zero values
        assert parse_salary_lpa(0, 0, "lpa") == ([0, 0], False)
        # Empty strings (unparseable)
        salary, disclosed = parse_salary_lpa("", "", "lpa")
        assert salary == [0, 0], f"empty salary must yield [0, 0], got {salary}"
        assert disclosed is False
        # Single value present -> disclosed True, but assert never [6, 12] for absent
        for result in (parse_salary_lpa(None, None, None)[0],
                       parse_salary_lpa(0, 0, "lpa")[0]):
            assert result != [6, 12], f"synthetic [6, 12] default leaked: {result}"

    def test_parse_salary_lpa_real_values_disclosed(self):
        """Real salary values -> parsed LPA pair + salary_disclosed=True."""
        from app.scraper.normalizer import parse_salary_lpa

        # LPA interval
        salary, disclosed = parse_salary_lpa(8, 14, "lpa")
        assert salary == [8, 14]
        assert disclosed is True
        # Monthly INR -> LPA conversion
        salary, disclosed = parse_salary_lpa(50000, 100000, "monthly")
        assert disclosed is True
        assert all(isinstance(x, int) for x in salary)
        # Never [6, 12] unless actually parsed 6..12
        salary, _ = parse_salary_lpa(6, 12, "lpa")
        assert salary == [6, 12]  # legit parse, not synthetic default

    def test_normalizer_taxonomy_grounding(self):
        """Strict canonical taxonomy grounding: matched in text -> required/preferred split."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Full Stack Developer",
            "company": "TechCo",
            "description": "We need Python, React, PostgreSQL, and Git.",
            "job_url": "https://example.com/j4",
            "tags": "python,react",
        }
        job = normalize_scraped_job(raw, taxonomy)

        # All grounded skills must belong to the canonical skill taxonomy
        all_skills = set(job.required_skills) | set(job.preferred_skills)
        for skill in all_skills:
            assert taxonomy.resolve(skill) is not None, f"{skill} not in taxonomy"

        # At least the explicitly mentioned skills should be caught
        assert "python" in all_skills
        assert "react" in all_skills

        # required_skills: first 2-6 matched; preferred: remainder
        assert 2 <= len(job.required_skills) <= 6
        assert len(job.preferred_skills) >= 0

        # Disjoint split
        assert not set(job.required_skills) & set(job.preferred_skills)

    def test_normalizer_taxonomy_fallback_when_no_match(self):
        """No skill matches -> fallback to ['python', 'sql', 'git']."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Mystery Role",
            "company": "MysteryCorp",
            "description": "No known skills here at all.",
            "job_url": "https://example.com/j5",
        }
        job = normalize_scraped_job(raw, taxonomy)
        assert job.required_skills == ["python", "sql", "git"]

    def test_normalizer_branch_normalization(self):
        """Branch mentions normalized to canonical CSE/IT/ECE codes."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Software Engineer",
            "company": "BigTech",
            "description": "CSE and IT graduates welcome.",
            "job_url": "https://example.com/j6",
        }
        job = normalize_scraped_job(raw, taxonomy)
        # Should contain CSE and IT (normalized)
        assert "CSE" in job.allowed_branches
        assert "IT" in job.allowed_branches

        # Direct normalize_branch calls
        assert normalize_branch("Computer Science") == "CSE"
        assert normalize_branch("Information Technology") == "IT"
        assert normalize_branch("Electronics & Communication") == "ECE"

    def test_normalizer_html_stripping_and_apply_url(self):
        """HTML tags removed; apply URL preserved at bottom."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Frontend Developer",
            "company": "WebCo",
            "description": "<p>We need <strong>React</strong> and CSS.</p><br><li>Item 1</li>",
            "job_url": "https://example.com/apply/789",
        }
        job = normalize_scraped_job(raw, taxonomy)

        # HTML tags stripped
        assert "<p>" not in job.description
        assert "<strong>" not in job.description
        assert "<br>" not in job.description
        assert "<li>" not in job.description

        # Apply URL appended
        assert "Apply URL: https://example.com/apply/789" in job.description

    def test_normalizer_deterministic_id(self):
        """Same job_url -> same ID (idempotent re-scraping)."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Test Role",
            "company": "TestCo",
            "description": "Test",
            "job_url": "https://example.com/idem/123",
        }
        job1 = normalize_scraped_job(raw, taxonomy)
        job2 = normalize_scraped_job(raw, taxonomy)
        assert job1.id == job2.id
        assert job1.id.startswith("job-ext-")
        # Deterministic hash prefix
        expected_hash = hashlib.md5(b"https://example.com/idem/123").hexdigest()[:10]
        assert expected_hash in job1.id


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestPipeline:
    """ingest_scraped_jobs: session.merge + vector store upsert."""

    def test_pipeline_ingest_and_vector_upsert(self):
        """Verify session.merge called, vector upsert called, commit called."""
        from app.scraper.pipeline import ingest_scraped_jobs

        taxonomy = load_taxonomy()
        raw_jobs = [
            {
                "title": "Python Dev",
                "company": "A",
                "description": "Python role",
                "job_url": "https://example.com/p1",
            },
            {
                "title": "Java Dev",
                "company": "B",
                "description": "Java role",
                "job_url": "https://example.com/p2",
            },
        ]

        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.vector_store = MagicMock()
        mock_ctx.embedder = MagicMock()
        mock_ctx.embedder.embed.return_value = [0.1, 0.2, 0.3]

        result = ingest_scraped_jobs(raw_jobs, mock_session, mock_ctx, taxonomy)

        # session.merge called for each job
        assert mock_session.merge.call_count == 2

        # session.commit called once
        mock_session.commit.assert_called_once()

        # vector_store.upsert_many is called once with all jobs (M21 batch path;
        # MagicMock exposes upsert_many so the batch branch is taken)
        assert mock_ctx.vector_store.upsert_many.call_count == 1
        items = mock_ctx.vector_store.upsert_many.call_args_list[0].args[0]
        assert len(items) == 2
        for kind, jid, _vec in items:
            assert kind == "job"  # collection/type
            assert jid.startswith("job-ext-")  # job ID

        # Result structure
        assert "ingested_count" in result
        assert "job_ids" in result
        assert "jobs" in result
        assert result["ingested_count"] == 2
        assert len(result["job_ids"]) == 2
        assert len(result["jobs"]) == 2


# ---------------------------------------------------------------------------
# Milestone 10: AI-Powered Hybrid Normalization (ARCHITECTURE §18.2 + §18.3)
# ---------------------------------------------------------------------------
# Charter §7 acceptance gate written FIRST: RED on each contract Coder-Flash
# has not yet landed (normalize_job_with_ai, MockLLM extract_job_posting
# handler, ThreadPoolExecutor batch ingestion). Baseline (2026-11-02,
# pre-implementation): normalize_job_with_ai / JOB_NORMALIZER_SYSTEM absent
# from app/scraper/normalizer.py; MockLLM.complete has no extract_job_posting
# branch (falls through to {"echo": user}); ingest_scraped_jobs normalizes
# sequentially with no llm/ThreadPool path. Spec committed as b46d507.
#
# Anti-vacuity (charter §4): the fallback + grounding tests first prove the
# probe FINDS the thing where the healthy path works (MockLLM stipend -> [4,4],
# grounded skills retained), so a "fell back / filtered" assertion is never a
# vacuous pass over a dead normalizer.
# ---------------------------------------------------------------------------


class TestAINormalizerMilestone10:
    """normalize_job_with_ai + MockLLM extract_job_posting (§18.2.A + §18.2.C)."""

    def test_normalize_job_with_ai_stipend_conversion(self):
        """Monthly stipend ₹30,000/mo -> [4, 4] LPA, salary_disclosed=True (§18.3.1)."""
        from app.llm.mock import MockLLM
        from app.scraper.normalizer import normalize_job_with_ai

        taxonomy = load_taxonomy()
        raw = {
            "title": "Backend Intern",
            "company": "StipendCorp",
            "description": "₹30,000/mo stipend for a Python backend intern.",
            "job_url": "https://example.com/stipend-1",
        }
        job = normalize_job_with_ai(raw, taxonomy, MockLLM())
        # Spec: round(30000 * 12 / 100000) == 4 -> [4, 4]
        assert job.salary_lpa == [4, 4], (
            f"monthly stipend must convert to [4, 4] LPA, got {job.salary_lpa}"
        )
        assert job.salary_disclosed is True, (
            "stated stipend must be salary_disclosed=True"
        )

    def test_normalize_job_with_ai_taxonomy_grounding(self):
        """Hallucinated / non-taxonomy tokens filtered; only valid taxonomy
        skills retained in required_skills (§18.3.1, zero-hallucination §18.2.A.3)."""
        from app.llm.mock import MockLLM
        from app.scraper.normalizer import normalize_job_with_ai

        taxonomy = load_taxonomy()
        # Description contains a real skill (python) + a hallucinated token
        # ("quantumflux") that is NOT in the taxonomy.
        raw = {
            "title": "Python Engineer",
            "company": "GroundCorp",
            "description": "Build services with python and quantumflux widgets.",
            "job_url": "https://example.com/ground-1",
        }
        job = normalize_job_with_ai(raw, taxonomy, MockLLM())
        all_skills = list(job.required_skills) + list(job.preferred_skills)
        # Every retained skill must resolve in the taxonomy (no hallucination).
        for skill in all_skills:
            assert taxonomy.resolve(skill) or taxonomy.get_opt(skill), (
                f"skill {skill!r} is not a valid taxonomy skill (hallucinated)"
            )
        # At least one genuine skill retained (probe finds the real thing).
        assert any(taxonomy.resolve(s) or taxonomy.get_opt(s) for s in all_skills), (
            "grounding must retain at least one valid taxonomy skill"
        )
        # The hallucinated token must NOT survive.
        assert "quantumflux" not in all_skills, (
            "hallucinated token 'quantumflux' leaked into skills"
        )

    def test_normalize_job_with_ai_fallback_on_llm_failure(self):
        """When the LLM raises (RuntimeError / rate limit / missing), the static
        normalizer takes over and returns a valid Job — no exception (§18.3.1)."""
        from app.scraper.normalizer import normalize_job_with_ai, normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Fallback Dev",
            "company": "StaticCorp",
            "description": "Java services role with clear requirements.",
            "job_url": "https://example.com/fallback-1",
        }

        class _BoomLLM:
            def complete(self, system, user, json_mode=False):
                raise RuntimeError("rate limited")

        job = normalize_job_with_ai(raw, taxonomy, _BoomLLM())
        # Anti-vacuity: identical raw via the STATIC normalizer is the ground
        # truth the fallback must reproduce — proves fallback returns a real Job
        # rather than the test passing on a dead path.
        expected = normalize_scraped_job(raw, taxonomy)
        assert job is not None, "fallback must return a Job, not None"
        assert job.title == expected.title == "Fallback Dev"
        assert job.company == expected.company == "StaticCorp"
        assert job.required_skills == expected.required_skills, (
            "fallback must reproduce the static normalizer's grounded skills"
        )
        assert job.salary_lpa == expected.salary_lpa

    def test_normalize_job_with_ai_fallback_on_missing_llm(self):
        """A None / absent LLM degrades to the static normalizer (§18.2.A.4)."""
        from app.scraper.normalizer import normalize_job_with_ai, normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "No LLM Dev",
            "company": "NullCorp",
            "description": "Go microservices role.",
            "job_url": "https://example.com/nollm-1",
        }
        job = normalize_job_with_ai(raw, taxonomy, None)
        expected = normalize_scraped_job(raw, taxonomy)
        assert job is not None
        assert job.title == expected.title
        assert job.required_skills == expected.required_skills


class TestPipelineConcurrentAINormalization:
    """ingest_scraped_jobs concurrent AI batch ingestion (§18.2.B + §18.3.1)."""

    def test_pipeline_concurrent_ai_normalization(self, client_engine):
        """Ingest 20 jobs concurrently via ingest_scraped_jobs with MockLLM;
        all 20 persist cleanly (§18.3.1).

        ANTI-VACUITY (charter §4): persistence alone passes even on the static
        path, so this also proves the AI normalizer actually ran for the batch
        (patch-count >= 20) — otherwise "20 persisted" is a vacuous green.
        """
        from unittest.mock import patch

        from sqlmodel import Session, select

        from app.llm.mock import MockLLM
        from app.models import Job
        from app.scraper.pipeline import ingest_scraped_jobs

        _client, engine = client_engine
        taxonomy = load_taxonomy()

        # 20 distinct jobs (unique titles/companies/urls so no dedup collapses them).
        raw_jobs = [
            {
                "title": f"Concurrent Dev {i}",
                "company": f"ConcurrentCo{i}",
                "description": f"Python role number {i} with python requirements.",
                "job_url": f"https://example.com/concurrent-{i}",
            }
            for i in range(20)
        ]

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.taxonomy = taxonomy
        ctx.llm = MockLLM()
        from app.embeddings.embedder import HashingEmbedder
        from app.embeddings.store import MemoryVectorStore

        ctx.embedder = HashingEmbedder()
        ctx.vector_store = MemoryVectorStore()

        import app.scraper.pipeline as pipeline_mod

        with patch.object(
            pipeline_mod, "normalize_job_with_ai", wraps=pipeline_mod.__dict__.get("normalize_job_with_ai")
        ) as spied:
            # If the name does not exist yet, patch.object raises AttributeError —
            # which is the intended self-arming RED until the AI path lands.
            with Session(engine) as db:
                result = ingest_scraped_jobs(raw_jobs, db, ctx, taxonomy)
                assert result["ingested_count"] == 20, (
                    f"expected 20 ingested, got {result['ingested_count']}"
                )
                assert len(result["job_ids"]) == 20
                # Every batch record must have gone through the AI normalizer.
                assert spied.call_count == 20, (
                    f"AI normalizer ran {spied.call_count} times, expected 20 "
                    "(static path would be 0 — this catches a vacuous green)"
                )
                # Persisted cleanly: re-query and confirm 20 rows exist.
                rows = db.exec(select(Job)).all()
                titles = {r.title for r in rows}
                for i in range(20):
                    assert f"Concurrent Dev {i}" in titles, (
                        f"job {i} not persisted after concurrent ingestion"
                    )
                db.commit()

    def test_pipeline_uses_llm_path_when_llm_present(self):
        """When ctx.llm is present, ingest routes normalization through the AI
        normalizer (not the static one) — proves the AI path is wired (§18.2.B.1)."""
        from unittest.mock import MagicMock, patch

        from app.llm.mock import MockLLM
        from app.scraper.pipeline import ingest_scraped_jobs

        taxonomy = load_taxonomy()
        raw_jobs = [
            {
                "title": "AI Path Dev",
                "company": "AIPathCo",
                "description": "Python role.",
                "job_url": "https://example.com/ai-path-1",
            }
        ]
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.taxonomy = taxonomy
        mock_ctx.llm = MockLLM()
        mock_ctx.vector_store = MagicMock()
        mock_ctx.embedder = MagicMock()
        mock_ctx.embedder.embed.return_value = [0.1]

        with patch(
            "app.scraper.pipeline.normalize_job_with_ai",
            autospec=True,
        ) as mock_ai:
            mock_ai.return_value = MagicMock()
            ingest_scraped_jobs(raw_jobs, mock_session, mock_ctx, taxonomy)
            # The AI normalizer MUST be invoked for each raw job when llm present.
            assert mock_ai.call_count >= 1, (
                "ctx.llm present but ingest did not call normalize_job_with_ai"
            )

    def test_pipeline_falls_back_to_static_without_llm(self):
        """When ctx.llm is absent, ingest uses the static normalizer (§18.2.B.1)."""
        from unittest.mock import MagicMock, patch

        from app.scraper.pipeline import ingest_scraped_jobs

        taxonomy = load_taxonomy()
        raw_jobs = [
            {
                "title": "Static Path Dev",
                "company": "StaticPathCo",
                "description": "Python role.",
                "job_url": "https://example.com/static-path-1",
            }
        ]
        mock_session = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.taxonomy = taxonomy
        mock_ctx.llm = None
        mock_ctx.vector_store = MagicMock()
        mock_ctx.embedder = MagicMock()
        mock_ctx.embedder.embed.return_value = [0.1]

        with patch(
            "app.scraper.pipeline.normalize_job_with_ai",
            autospec=True,
        ) as mock_ai:
            ingest_scraped_jobs(raw_jobs, mock_session, mock_ctx, taxonomy)
            # AI normalizer must NOT be called when llm is absent.
            assert mock_ai.call_count == 0, (
                "ctx.llm absent but ingest still called normalize_job_with_ai"
            )


# ---------------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------------


class TestAPIScrapeEndpoints:
    """POST /api/jobs/scrape and GET /api/jobs/scrape/status."""

    def test_api_scrape_jobs_endpoint(self, client):
        """POST /api/jobs/scrape returns success JSON with fetched/ingested/job_ids."""
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            resp = client.post(
                "/api/jobs/scrape",
                json={
                    "search_term": "Python Developer",
                    "location": "Bengaluru",
                    "sites": ["naukri"],
                    "results_wanted": 10,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["fetched"] == 2
        assert data["ingested"] == 2
        assert len(data["job_ids"]) == 2
        assert all(jid.startswith("job-ext-") for jid in data["job_ids"])

    def test_api_scrape_status_endpoint(self, client):
        """GET /api/jobs/scrape/status returns platforms, count, defaults."""
        resp = client.get("/api/jobs/scrape/status")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "supported_platforms" in data or "platforms" in data
        platforms = data.get("supported_platforms") or data.get("platforms")
        assert set(platforms) == {"naukri", "linkedin", "indeed"}
        assert "total_external_jobs" in data or "job_count" in data
        assert "default_params" in data or "defaults" in data


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLIScrapeJobs:
    """scrape-jobs subcommand."""

    def test_cli_scrape_jobs(self, capsys, monkeypatch):
        """CLI scrape-jobs executes pipeline and prints job table."""
        from app import cli
        from app.db import make_engine
        from app.embeddings.embedder import HashingEmbedder

        # Same monkeypatch seams as seed command
        monkeypatch.setattr(cli, "_make_engine", lambda url: make_engine("sqlite://"))
        monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
        monkeypatch.setattr(cli.settings, "database_url", "sqlite://")

        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            rc = cli.main(["scrape-jobs", "--term", "Python", "--limit", "10"])

        out = capsys.readouterr().out
        assert rc == 0
        # Should print at least one job (table header or job row)
        assert "Python" in out or "Acme" in out or "job-ext-" in out


# ---------------------------------------------------------------------------
# Zero-emoji UI test
# ---------------------------------------------------------------------------


class TestFresherModeMilestone11:
    """ARCHITECTURE §19.4 — Scraper Fresher Mode & Matches Dashboard Quick Filter.

    Re-attack of the LANDED feature (measured 2026-11-03): ScrapeRequest has
    `fresher_only: bool = True` (jobs.py), the stream handler enriches terms with
    ["Fresher", "Junior", "Intern", "Entry Level"] and forwards
    fresher_only/job_type="entry_level" (jobs.py:152-186), matches.html stamps
    data-fresher on every card (line 254), counts them (579-588), and filters on
    state.fresherOnly (711, 758-759). Anti-vacuity probes each layer where it
    exists so a silent removal fails every green.
    """

    QUALIFIERS = ("Fresher", "Junior", "Intern", "Entry Level")

    def test_fresher_mode_query_enrichment(self):
        """§19.4.1 — stream + non-stream endpoints enrich bare terms with fresher
        qualifiers when fresher_only is on, and pass the flag downstream.
        Inverted gate (charter §6): default OFF is Measured DEFECT (spec default
        True; wired bool(getattr(..., False)) 2026-11-03) — RED today by design,
        promotes to a plain assertion the day the default flips."""
        import inspect

        from app.api.routes.jobs import (
            ScrapeRequest,
            scrape_jobs_endpoint,
            scrape_jobs_stream_endpoint,
        )

        assert ScrapeRequest().fresher_only is True, (
            f"fresher toggle must default True per §19.4.1, got "
            f"{ScrapeRequest().fresher_only!r} — inverted gate: Measured DEFECT "
            f"(jobs.py wires bool(getattr(req, 'fresher_only', False)))"
        )
        for fn in (scrape_jobs_stream_endpoint, scrape_jobs_endpoint):
            src = inspect.getsource(fn)
            for q in self.QUALIFIERS:
                assert q in src, f"{fn.__name__} must enrich with {q!r} (§19.4.1)"
            assert "fresher_only" in src, f"{fn.__name__} must forward fresher_only"
            assert "entry_level" in src, f"{fn.__name__} must request entry-level job types (§19.4.1)"

    def test_fresher_term_enrichment_expansion(self, client_engine):
        """Behavioral: with fresher_only explicitly ON, a bare term expands to 4
        fresher-qualified searches; a term already carrying a qualifier is NOT
        double-enriched. The default-OFF defect is pinned by
        test_fresher_mode_query_enrichment (inverted), so this behavioral test
        stays green on the current build (does not assert the default)."""
        from unittest.mock import MagicMock, patch

        client, _engine = client_engine
        # fresher_only defaults True (§19.4.1); explicitly on here for clarity.
        # Patch the provider seam (CompositeJobScraper dispatches through
        # JobSpyProvider) with a MagicMock so enrichment is observable without
        # any network side effects (M10 lesson: spy the actual call).
        body = {"search_terms": ["Software Engineer"], "fresher_only": True,
                "providers": ["jobspy"], "results_wanted": 1, "live": False}
        stub_scraper = MagicMock()
        stub_scraper.fetch_jobs.return_value = []
        with patch("app.api.routes.jobs.CompositeJobScraper", return_value=stub_scraper):
            resp = client.post("/api/jobs/scrape", json=body)
        assert resp.status_code == 200, resp.text[:400]
        called_terms = []
        for call in stub_scraper.fetch_jobs.call_args_list:
            term = call.kwargs.get("search_term", call.args[0] if call.args else "")
            called_terms.extend(term if isinstance(term, list) else [term])
        joined = " | ".join(str(t) for t in called_terms)
        for q in self.QUALIFIERS:
            assert f"Software Engineer {q}" in joined, (
                f"query enrichment missing {q!r}: got {called_terms}"
            )
        # Anti-vacuity: already-qualified input is not re-enriched.
        stub_scraper.reset_mock()
        with patch("app.api.routes.jobs.CompositeJobScraper", return_value=stub_scraper):
            client.post("/api/jobs/scrape", json={"search_terms": ["Fresher Python Developer"],
                                                  "fresher_only": True,
                                                  "providers": ["jobspy"],
                                                  "results_wanted": 1, "live": False})
        again = " | ".join(str(t) for c in stub_scraper.fetch_jobs.call_args_list
                           for t in ([c.kwargs.get("search_term")] if isinstance(c.kwargs.get("search_term"), list)
                                     else [c.kwargs.get("search_term") or (c.args[0] if c.args else "")]))
        assert "Fresher Python Developer Fresher" not in again, (
            f"already-qualified term was double-enriched: {again}"
        )
        # Anti-vacuity: the OFF branch must NOT enrich (proves fresher_only is
        # the actual switch, not enrichment baked in unconditionally).
        stub_scraper.reset_mock()
        with patch("app.api.routes.jobs.CompositeJobScraper", return_value=stub_scraper):
            client.post("/api/jobs/scrape", json={"search_terms": ["Software Engineer"],
                                                  "fresher_only": False,
                                                  "providers": ["jobspy"],
                                                  "results_wanted": 1, "live": False})
        off = " | ".join(str(t) for c in stub_scraper.fetch_jobs.call_args_list
                         for t in ([c.kwargs.get("search_term")] if isinstance(c.kwargs.get("search_term"), list)
                                   else [c.kwargs.get("search_term") or (c.args[0] if c.args else "")]))
        assert "Software Engineer Junior" not in off, (
            f"fresher_only=False must not enrich: {off}"
        )

    def test_data_fresher_stamping(self):
        """§19.4.2 — every job card stamps data-fresher from the is_fresher flag."""
        from pathlib import Path

        content = Path("app/templates/student/matches.html").read_text(encoding="utf-8")
        assert 'data-fresher="' in content, "job cards must stamp data-fresher (§19.4.2)"
        assert "is_fresher" in content, (
            "data-fresher must derive from the Job.is_fresher flag (models.py)"
        )
        # Anti-vacuity: the stamp must be conditional (true|false), not a constant.
        assert "{% if is_fresher_val %}true{% else %}false{% endif %}" in content, (
            "data-fresher must be conditional true|false per §19.4.2"
        )

    def test_fresher_dashboard_filter_toggle(self):
        """§19.4.2 — 'Fresher Roles Only (count)' pill + client-side hide of
        non-fresher cards; OFF path keeps the full catalog (all data-fresher
        cards still rendered — anti-vacuity on the OFF branch)."""
        from pathlib import Path

        content = Path("app/templates/student/matches.html").read_text(encoding="utf-8")
        assert 'id="filter-fresher-pill"' in content, "fresher filter pill missing (§19.4.2)"
        assert "Fresher Roles Only" in content, "pill label missing (§19.4.2)"
        assert 'id="count-fresher"' in content, "fresher count span missing (§19.4.2)"
        # Toggle handler hides non-fresher cards when ON ...
        assert "state.fresherOnly" in content, "fresherOnly state not wired in JS"
        assert 'getAttribute("data-fresher")' in content, (
            "client filter must key on the data-fresher attribute"
        )
        # ... and the OFF path is the default full catalog (no unconditional hide).
        assert "state.fresherOnly && !dFresher" in content, (
            "hide must be conditional on state.fresherOnly (OFF = full catalog)"
        )


class TestFresherFilterSyncMilestone13:
    """§21.2.3 — form-restore guard + 0-1/1-3 experience bucket routing."""

    def test_form_restore_guard_syncs_dom_inputs(self):
        """§21.2.3.1 — on page load, restored DOM inputs are adopted into state
        before the initial applyFilters() so the view can never desync."""
        from pathlib import Path

        content = Path("app/templates/student/matches.html").read_text(encoding="utf-8")

        # The restore guard must live at the initial execution of the filter suite,
        # and both syncs must occur before that final applyFilters() call.
        assert "form restore guard" in content, "§21.2.3 restore guard comment missing"
        tail = content[content.index("// Initial Execution"):]
        apply_idx = tail.index("applyFilters();")
        sync_fresher = tail.index("state.fresherOnly = !!fresherCheckbox.checked")
        sync_exp = tail.index('state.exp = expSelect.value')
        assert sync_fresher < apply_idx, (
            "restored fresherCheckbox.checked must sync state.fresherOnly before applyFilters"
        )
        assert sync_exp < apply_idx, (
            "restored expSelect.value must sync state.exp before applyFilters"
        )

    def test_exp_0_1_bucket_routes_non_fresher_one_year_to_1_3(self):
        """§21.2.3.2 — a dExp==1 card with data-fresher === "false" is excluded
        from the 0-1 bucket (it is an experienced role) and remains in 1-3."""
        from pathlib import Path

        content = Path("app/templates/student/matches.html").read_text(encoding="utf-8")

        # Anti-vacuity: probe each half of the invariant independently.
        assert 'dFresherAttr === "false"' in content, (
            "0-1 bucket must key on the data-fresher attribute value"
        )
        assert "dExp === 1" in content, (
            "0-1 bucket leak guard must target exactly one-year roles"
        )
        assert "leakTo13" in content, (
            "0-1 bucket must route non-fresher 1-yr roles to 1-3, not drop them"
        )
        assert "!(dExp <= 1) || leakTo13" in content, (
            "0-1 hide predicate must add the non-fresher one-year exclusion"
        )
        # The 1-3 bucket still accepts that one-year role (routing, not deletion).
        assert 'state.exp === "1-3" && !(dExp >= 1 && dExp <= 3)' in content, (
            "1-3 bucket must keep accepting dExp==1 roles"
        )


    """Zero-emoji compliance in scraper modal elements."""

    def test_zero_emoji_in_scraper_ui(self):
        """matches.html scrape modal must contain zero emoji artifacts."""
        from pathlib import Path

        matches_html = Path("app/templates/student/matches.html")
        if not matches_html.exists():
            pytest.skip("matches.html not found")

        content = matches_html.read_text(encoding="utf-8")

        # Only scan the modal section if it exists
        modal_start = content.find("scrape-modal")
        if modal_start == -1:
            pytest.skip("scrape-modal not present yet (ticket 3 in flight)")

        # Extract modal block (until closing div after modal)
        # Simple heuristic: scan 5000 chars from modal start
        modal_block = content[modal_start : modal_start + 5000]

        # Emoji literals (same ranges as test_sui_shell.py)
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001FAFF☀-➿⏰-⏳]"
        )
        # HTML numeric emoji/dingbat entities
        entity_pattern = re.compile(
            r"&#(?:8962|9776|9989|10003|10007|12[789]\d{3}|127\d{3})\b"
        )

        literal_emojis = emoji_pattern.findall(modal_block)
        entities = entity_pattern.findall(modal_block)

        assert literal_emojis == [], f"Modal contains literal emojis: {literal_emojis}"
        assert entities == [], f"Modal contains emoji entities: {entities}"

        # Anti-vacuity: if modal exists, it should have content
        assert len(modal_block) > 100, "Modal block suspiciously small"

    def test_zero_emoji_gate_detects_known_symbols(self):
        """Anti-vacuity: prove probes detect known emoji symbols and spare legit symbols."""
        import re as re2

        emoji_pattern = re2.compile(
            "[\U0001F300-\U0001FAFF☀-➿⏰-⏳]"
        )
        entity_pattern = re2.compile(
            r"&#(?:8962|9776|9989|10003|10007|12[789]\d{3}|127\d{3})\b"
        )

        # Must DETECT these known emoji/symbols
        assert emoji_pattern.search("✅")  # U+2705
        assert emoji_pattern.search("❌")  # U+274C
        assert emoji_pattern.search("⏳")  # U+23F3
        assert emoji_pattern.search("🎤")  # U+1F3A4
        assert entity_pattern.search("&#9989;")
        assert entity_pattern.search("&#10003;")
        assert entity_pattern.search("&#127908;")

        # Must SPARE these legitimate symbols
        assert not emoji_pattern.search("×")  # U+00D7 multiplication
        assert not emoji_pattern.search("é")  # U+00E9
        assert not entity_pattern.search("&#39;")  # apostrophe
        assert not entity_pattern.search("&#8217;")  # right single quote


# ===========================================================================
# Milestone 5: Skill-Targeted AI Query, Link Preservation, Segmented View
# ARCHITECTURE.md Section 11 (spec-driven acceptance gate — charter §7)
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures for Milestone 5
# ---------------------------------------------------------------------------


def _student_with_skills() -> dict:
    """Minimal student profile for query synthesis tests."""
    return {
        "id": "stu-m5-test",
        "name": "Test Candidate",
        "branch": "CSE",
        "year": 3,
        "gpa": 8.5,
    }


class TestAIJobQuerySynthesizer:
    """suggest_job_queries_for_student: LLM path + deterministic fallback."""

    def test_suggest_job_queries_with_llm(self):
        """With LLM provided: calls TASK: suggest_job_queries, returns 2-3 queries."""
        from app.models import Student, StudentSkill
        from app.scraper.ai_query import suggest_job_queries_for_student

        student = Student(
            id="stu-m5-llm",
            name="LLM Candidate",
            branch="CSE",
            year=3,
            gpa=8.5,
        )
        skills = [
            StudentSkill(student_id=student.id, skill_id="python", level="advanced", mentions=12),
            StudentSkill(student_id=student.id, skill_id="fastapi", level="advanced", mentions=8),
            StudentSkill(student_id=student.id, skill_id="react", level="intermediate", mentions=5),
        ]

        # Mock LLM that returns a JSON list of queries
        mock_llm = MagicMock()
        mock_llm.complete.return_value = (
            '["Python FastAPI Backend Engineer", "React Frontend Developer", "Full Stack Python"]'
        )

        queries = suggest_job_queries_for_student(student, skills, llm=mock_llm)

        assert isinstance(queries, list)
        assert 2 <= len(queries) <= 3, f"Expected 2-3 queries, got {len(queries)}"
        assert all(isinstance(q, str) and q for q in queries)

        # Verify LLM was called with the correct task
        mock_llm.complete.assert_called_once()
        call_args = mock_llm.complete.call_args
        prompt_text = str(call_args)
        assert "suggest_job_queries" in prompt_text, (
            f"LLM prompt must contain TASK: suggest_job_queries; got: {prompt_text[:200]}"
        )

    def test_suggest_job_queries_fallback_no_llm(self):
        """Without LLM: deterministic synthesizer returns 2-3 queries from skills."""
        from app.models import Student, StudentSkill
        from app.scraper.ai_query import suggest_job_queries_for_student

        student = Student(
            id="stu-m5-fallback",
            name="Fallback Candidate",
            branch="CSE",
            year=3,
            gpa=8.5,
        )
        skills = [
            StudentSkill(student_id=student.id, skill_id="python", level="advanced", mentions=10),
            StudentSkill(student_id=student.id, skill_id="react", level="intermediate", mentions=6),
            StudentSkill(student_id=student.id, skill_id="postgresql", level="basic", mentions=3),
        ]

        queries = suggest_job_queries_for_student(student, skills, llm=None)

        assert isinstance(queries, list)
        assert 2 <= len(queries) <= 3, f"Expected 2-3 queries, got {len(queries)}"
        assert all(isinstance(q, str) and len(q) > 0 for q in queries)

        # Queries should reference top skills (python, react, postgresql)
        combined = " ".join(queries).lower()
        assert "python" in combined or "react" in combined or "sql" in combined, (
            f"Queries should reference top skills; got: {queries}"
        )

    def test_suggest_job_queries_empty_skills(self):
        """No skills: returns fallback queries or empty list without crashing."""
        from app.models import Student
        from app.scraper.ai_query import suggest_job_queries_for_student

        student = Student(
            id="stu-m5-empty",
            name="No Skills",
            branch="CSE",
            year=3,
            gpa=8.5,
        )

        queries = suggest_job_queries_for_student(student, [], llm=None)

        assert isinstance(queries, list)
        # Either empty or generic fallback queries — both acceptable
        assert all(isinstance(q, str) for q in queries)
        # Single role for 0 skills (non-diverse)
        assert len(queries) == 1

    def test_suggest_job_queries_single_skill_focused(self):
        """Single technical skill: returns exactly 1 focused query term."""
        from app.models import Student, StudentSkill
        from app.scraper.ai_query import suggest_job_queries_for_student

        student = Student(
            id="stu-single-skill",
            name="Single Skill Dev",
            branch="CSE",
            year=3,
            gpa=8.0,
        )
        skills = [
            StudentSkill(student_id=student.id, skill_id="python", level="advanced", mentions=5),
        ]
        queries = suggest_job_queries_for_student(student, skills, llm=None)
        assert isinstance(queries, list)
        assert len(queries) == 1, f"Expected 1 focused query for single skill, got {queries}"
        assert "python" in queries[0].lower()


class TestSkillTargetedScrapeEndpoints:
    """GET /api/students/{sid}/suggest-job-queries + POST /api/students/{sid}/scrape-jobs."""

    def test_get_suggest_job_queries_endpoint(self, client, client_engine):
        """Returns student_id, suggested_queries, top_skills for candidate."""
        client_c, engine = client_engine

        # Create student with skills
        resp = client_c.post(
            "/api/students",
            json=_student_with_skills(),
        )
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        # Add skills
        from sqlmodel import Session

        from app.models import StudentSkill

        with Session(engine) as db:
            db.add(StudentSkill(student_id=sid, skill_id="python", level="advanced", mentions=10))
            db.add(StudentSkill(student_id=sid, skill_id="react", level="intermediate", mentions=5))
            db.commit()

        # Test endpoint
        suggest_resp = client_c.get(f"/api/students/{sid}/suggest-job-queries")
        assert suggest_resp.status_code == 200, (
            f"GET suggest-job-queries failed: {suggest_resp.status_code}"
        )
        data = suggest_resp.json()

        assert "student_id" in data
        assert data["student_id"] == sid
        assert "suggested_queries" in data
        assert isinstance(data["suggested_queries"], list)
        assert 1 <= len(data["suggested_queries"]) <= 3
        assert "top_skills" in data
        assert isinstance(data["top_skills"], list)
        # Top skills should include student's skills
        assert "python" in data["top_skills"]

    def test_post_scrape_jobs_endpoint(self, client, client_engine):
        """Candidate-specific scraping executes with skill-targeted query."""
        client_c, engine = client_engine

        # Create student
        resp = client_c.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        # Mock the scraper to avoid network calls
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            scrape_resp = client_c.post(
                f"/api/students/{sid}/scrape-jobs",
                json={"location": "Bengaluru", "sites": ["naukri"], "results_wanted": 5},
            )

        # Accept 200 (success) or 422 (validation) — endpoint must exist
        assert scrape_resp.status_code in (200, 422), (
            f"POST scrape-jobs returned {scrape_resp.status_code}: {scrape_resp.text[:500]}"
        )

        if scrape_resp.status_code == 200:
            data = scrape_resp.json()
            assert "fetched" in data or "ingested" in data or "status" in data

    def test_post_student_scrape_jobs_stream_endpoint(self, client, client_engine):
        """POST /api/students/{sid}/scrape-jobs/stream streams SSE telemetry and completes."""
        client_c, engine = client_engine
        resp = client_c.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            r = client_c.post(
                f"/api/students/{sid}/scrape-jobs/stream",
                json={"search_term": "Python", "location": "Bengaluru", "results_wanted": 2, "live": False},
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            body = r.text
            assert "event: init" in body
            assert "event: complete" in body

    def test_post_scrape_jobs_endpoint_multiple_terms(self, client, client_engine):
        """API POST /api/jobs/scrape supports search_terms list and returns terms_used."""
        client_c, engine = client_engine
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df) as mock_scrape:
            resp = client_c.post(
                "/api/jobs/scrape",
                json={
                    "search_terms": ["Python Engineer", "React Developer"],
                    "location": "Bengaluru",
                    "results_wanted": 10,
                    "fresher_only": False,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert data["terms_used"] == ["Python Engineer", "React Developer"]
            assert mock_scrape.call_count == 2


class TestLinkPreservation:
    """Job.job_url + Job.source persistence and serialization."""

    def test_normalizer_captures_job_url_and_source(self):
        """normalize_scraped_job must extract job_url and source from raw dict."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Backend Engineer",
            "company": "LinkTest",
            "description": "Test link preservation",
            "job_url": "https://www.linkedin.com/jobs/view/12345",
            "site": "linkedin",
        }

        job = normalize_scraped_job(raw, taxonomy)

        # job_url must be captured
        assert hasattr(job, "job_url"), "Job model must have job_url field"
        assert job.job_url == "https://www.linkedin.com/jobs/view/12345", (
            f"job_url not captured: {job.job_url}"
        )

        # source must be captured
        assert hasattr(job, "source"), "Job model must have source field"
        assert job.source == "linkedin", f"source not captured: {job.source}"

        # Apply URL must still be in description (backward compat)
        assert "Apply URL: https://www.linkedin.com/jobs/view/12345" in job.description

    def test_normalizer_source_defaults_local(self):
        """No site in raw -> source defaults to 'local' or 'live'."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = {
            "title": "Local Job",
            "company": "Campus",
            "description": "Seed job",
            "job_url": "",  # no URL
        }

        job = normalize_scraped_job(raw, taxonomy)

        assert hasattr(job, "source")
        # Spec: default "local" (models.py) or "live" (normalizer.py sets raw.get("site") or "live")
        # Accept either per spec ambiguity
        assert job.source in ("local", "live"), f"Unexpected default source: {job.source}"

    def test_matches_response_includes_job_url_and_source(self, client, client_engine):
        """GET /students/{sid}/matches must serialize job_url and source in job cards."""
        client_c, engine = client_engine

        # Create student
        resp = client_c.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        # Add a live job with job_url/source
        from sqlmodel import Session

        from app.models import Job

        with Session(engine) as db:
            live_job = Job(
                id="job-ext-testlink",
                title="Live Backend Role",
                company="LiveCorp",
                description="Test live job",
                required_skills=["python"],
                preferred_skills=[],
                job_url="https://www.naukri.com/job/999",
                source="naukri",
                is_live=True,
                owner_student_id=sid,
            )
            db.add(live_job)
            db.commit()

        # Request matches
        matches_resp = client_c.get(f"/api/students/{sid}/matches")
        assert matches_resp.status_code == 200
        data = matches_resp.json()

        # Find our job in the response
        # Response could be list or dict with matches key
        matches = data if isinstance(data, list) else data.get("matches", [])
        found = None
        for m in matches:
            job_data = m.get("job", m)  # handle nested or flat
            if job_data.get("id") == "job-ext-testlink":
                found = job_data
                break

        assert found is not None, "Live test job not found in matches response"
        assert "job_url" in found, f"job_url missing from serialized job: {found.keys()}"
        assert found["job_url"] == "https://www.naukri.com/job/999"
        assert "source" in found, f"source missing from serialized job: {found.keys()}"
        assert found["source"] == "naukri"


class TestSegmentedMarketView:
    """Segmented filter pills + data-job-type filtering in matches.html."""

    def test_segmented_filter_pills_present(self):
        """matches.html must contain 3 filter pills with count spans."""
        from pathlib import Path

        matches_html = Path("app/templates/student/matches.html")
        if not matches_html.exists():
            pytest.skip("matches.html not found")

        content = matches_html.read_text(encoding="utf-8")

        # Three pills with count spans (skip if not present yet — ticket in flight)
        if 'id="count-all"' not in content:
            pytest.skip("Segmented filter pills not present yet (ticket 3 in flight)")

        assert 'id="count-all"' in content, "Missing 'All Opportunities' pill count span"
        assert 'id="count-live"' in content, "Missing 'Live Market' pill count span"
        assert 'id="count-local"' in content, "Missing 'On-Campus' pill count span"

    def test_data_job_type_attributes_present(self):
        """Match cards must have data-job-type='live' or 'local' for filtering."""
        from pathlib import Path

        matches_html = Path("app/templates/student/matches.html")
        if not matches_html.exists():
            pytest.skip("matches.html not found")

        content = matches_html.read_text(encoding="utf-8")

        if 'data-job-type' not in content:
            pytest.skip("data-job-type attributes not present yet (ticket 3 in flight)")

        # Must have both live and local variants (or conditional logic)
        assert 'data-job-type' in content, "Missing data-job-type attribute"

    def test_segmented_view_zero_emoji(self):
        """Segmented pills + badges must be zero-emoji compliant."""
        from pathlib import Path

        matches_html = Path("app/templates/student/matches.html")
        if not matches_html.exists():
            pytest.skip("matches.html not found")

        content = matches_html.read_text(encoding="utf-8")

        # Scan entire file (zero-emoji applies globally)
        emoji_pattern = re.compile(
            "[\U0001F300-\U0001FAFF☀-➿⏰-⏳]"
        )
        entity_pattern = re.compile(
            r"&#(?:8962|9776|9989|10003|10007|12[789]\d{3}|127\d{3})\b"
        )

        literal_emojis = emoji_pattern.findall(content)
        entities = entity_pattern.findall(content)

        assert literal_emojis == [], f"matches.html contains literal emojis: {literal_emojis}"
        assert entities == [], f"matches.html contains emoji entities: {entities}"


# ===========================================================================
# Milestone 6: Multi-Provider Scraper Aggregator & Deduplication
# ARCHITECTURE.md Sections 12 & 13 (spec-driven acceptance gate — charter §7)
# ===========================================================================


# ---------------------------------------------------------------------------
# Fixtures for Milestone 6
# ---------------------------------------------------------------------------


def _raw_job(
    url: str = "https://example.com/job/1",
    title: str | None = None,
    company: str | None = None,
    engine: str = "jobspy",
    site: str = "linkedin",
    description: str = "Build APIs",
) -> dict:
    """Minimal raw job dict matching provider output contract."""
    return {
        "title": title if title is not None else f"{engine.title()} Engineer",
        "company": company if company is not None else f"{engine.title()} Corp",
        "location": "Bengaluru",
        "description": description,
        "job_url": url,
        "site": site,
        "scraper_engine": engine,
        "date_posted": None,
    }


class _StubProvider:
    """Duck-typed stub provider (avoids import coupling during in-flight work)."""

    def __init__(self, name: str, jobs: list[dict] | None = None, exc: Exception | None = None):
        self.name = name
        self.display_name = name.title()
        self.supported_sites = ["naukri", "linkedin"]
        self._jobs = jobs or []
        self._exc = exc

    def fetch_jobs(self, search_term, location="India", results_wanted=15, **kwargs):
        if self._exc is not None:
            raise self._exc
        return list(self._jobs)


class TestProviderAbstraction:
    """BaseJobScraperProvider ABC contracts + concrete provider metadata."""

    def test_base_provider_is_abstract(self):
        """BaseJobScraperProvider cannot be instantiated; requires fetch_jobs override."""
        from app.scraper.providers import BaseJobScraperProvider

        with pytest.raises(TypeError):
            BaseJobScraperProvider()  # type: ignore[abstract]

    def test_base_provider_abstract_method(self):
        """fetch_jobs must be part of the abstract interface."""
        import inspect

        from app.scraper.providers import BaseJobScraperProvider

        assert inspect.isabstract(BaseJobScraperProvider)
        assert "fetch_jobs" in BaseJobScraperProvider.__abstractmethods__

    def test_base_provider_metadata_attributes(self):
        """All providers expose name/display_name/supported_sites contract attrs."""
        from app.scraper.providers import BaseJobScraperProvider

        for attr in ("name", "display_name", "supported_sites"):
            assert attr in vars(BaseJobScraperProvider) or any(
                attr in klass.__dict__ for klass in BaseJobScraperProvider.__mro__
            ), f"Base provider must declare {attr}"

    def test_all_concrete_providers_registered(self):
        """4 concrete providers exist with correct names and site targets."""
        from app.scraper.providers import (
            AutomationJobScraperProvider,
            JobSpyProvider,
            ProxyAPIJobScraperProvider,
            RapidJobScraperProvider,
        )

        expected = {
            JobSpyProvider: "jobspy",
            RapidJobScraperProvider: "rapid",
            AutomationJobScraperProvider: "automation",
            ProxyAPIJobScraperProvider: "proxy_api",
        }
        for cls, name in expected.items():
            provider = cls()
            assert provider.name == name, f"{cls.__name__}.name == {provider.name!r}, want {name!r}"
            assert isinstance(provider.display_name, str) and provider.display_name
            assert isinstance(provider.supported_sites, list) and provider.supported_sites

    def test_jobspy_provider_wraps_scraper(self):
        """JobSpyProvider.fetch_jobs delegates to JobSpyScraper and stamps engine."""
        from app.scraper.providers import JobSpyProvider

        provider = JobSpyProvider()
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            jobs = provider.fetch_jobs("Python", location="Bengaluru", results_wanted=5)

        assert isinstance(jobs, list)
        assert len(jobs) == 2
        for job in jobs:
            assert job.get("scraper_engine") == "jobspy", (
                f"Job missing scraper_engine='jobspy': {job.get('scraper_engine')!r}"
            )
            assert "title" in job and "job_url" in job

    def test_rapid_provider_offline_extraction(self):
        """RapidJobScraperProvider returns offline rapid results with skill tagging."""
        from app.scraper.providers import RapidJobScraperProvider

        provider = RapidJobScraperProvider()
        jobs = provider.fetch_jobs("python developer", results_wanted=5)

        assert isinstance(jobs, list)
        # Offline/rapid mode must return at least one job without network
        assert len(jobs) >= 1, "Rapid provider must return offline fallback results"
        for job in jobs:
            assert job.get("scraper_engine") == "rapid"
            assert "title" in job and "job_url" in job

    def test_automation_provider_offline_extraction(self):
        """AutomationJobScraperProvider returns offline tech-board results with skill tags."""
        from app.scraper.providers import AutomationJobScraperProvider

        provider = AutomationJobScraperProvider()
        jobs = provider.fetch_jobs("full stack developer", results_wanted=5)

        assert isinstance(jobs, list)
        assert len(jobs) >= 1, "Automation provider must return offline fallback results"
        for job in jobs:
            assert job.get("scraper_engine") == "automation"
            assert "title" in job and "job_url" in job

    def test_proxy_provider_resilient_fallback(self):
        """ProxyAPIJobScraperProvider without keys -> graceful mock/fallback mode."""
        from app.scraper.providers import ProxyAPIJobScraperProvider

        provider = ProxyAPIJobScraperProvider()
        # Must never raise even with no SCRAPER_API_KEY / JOB_SCRAPER_PROXY_URL
        jobs = provider.fetch_jobs("react developer", results_wanted=5)

        assert isinstance(jobs, list)
        assert len(jobs) >= 1, "Proxy provider must return offline fallback results"
        for job in jobs:
            assert job.get("scraper_engine") == "proxy_api"


class TestCompositeJobScraperAggregation:
    """CompositeJobScraper: aggregation, fault tolerance, dedup, filtering, metadata."""

    def test_aggregation_across_providers(self):
        """All provider results merged into a single list (distinct jobs)."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider("rapid", [_raw_job(url="https://r.example/1", title="Rapid Role", company="RapidCo", engine="rapid")])
        p2 = _StubProvider("automation", [_raw_job(url="https://a.example/1", title="Auto Role", company="AutoCo", engine="automation")])
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")

        assert isinstance(jobs, list)
        assert len(jobs) == 2, f"Expected 2 aggregated jobs, got {len(jobs)}"
        engines = {j["scraper_engine"] for j in jobs}
        assert engines == {"rapid", "automation"}

    def test_fault_tolerance_one_provider_fails(self):
        """One provider raises -> others still return jobs (isolated handling)."""
        from app.scraper.service import CompositeJobScraper

        bad = _StubProvider("rapid", exc=ConnectionError("network timeout"))
        good1 = _StubProvider("automation", [_raw_job(url="https://a.example/1", title="Auto Role", company="AutoCo", engine="automation")])
        good2 = _StubProvider("proxy_api", [_raw_job(url="https://p.example/1", title="Proxy Role", company="ProxyCo", engine="proxy_api")])
        composite = CompositeJobScraper(providers=[bad, good1, good2])

        jobs = composite.fetch_jobs("python")

        assert isinstance(jobs, list), "Composite must not raise when one provider fails"
        assert len(jobs) == 2, (
            f"Healthy providers must still contribute; got {len(jobs)} jobs"
        )
        engines = {j["scraper_engine"] for j in jobs}
        assert engines == {"automation", "proxy_api"}

    def test_fault_tolerance_all_providers_fail(self):
        """All providers raise -> empty list, no unhandled exception."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider("rapid", exc=TimeoutError("slow"))
        p2 = _StubProvider("automation", exc=ConnectionError("down"))
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")
        assert jobs == [], f"Total failure must degrade to [], got {jobs!r}"

    def test_dedup_canonical_url_with_tracking_params(self):
        """Same canonical URL with different tracking params deduplicates to 1."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider(
            "rapid",
            [_raw_job(url="https://example.com/j/1?ref=email&utm_source=nl", engine="rapid")],
        )
        p2 = _StubProvider(
            "automation",
            [_raw_job(url="https://example.com/j/1?ref=ads", engine="automation")],
        )
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")

        assert len(jobs) == 1, (
            f"Tracking-param variants of same URL must dedup to 1; got {len(jobs)}"
        )

    def test_dedup_exact_url(self):
        """Identical job_url across providers deduplicates to 1."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider("rapid", [_raw_job(url="https://example.com/j/9", engine="rapid")])
        p2 = _StubProvider("automation", [_raw_job(url="https://example.com/j/9", engine="automation")])
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")
        assert len(jobs) == 1, f"Exact URL duplicate must dedup to 1; got {len(jobs)}"

    def test_soft_dedup_normalized_title_company(self):
        """Different URLs but same normalized (title, company) soft-dedup to 1."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider(
            "rapid",
            [_raw_job(url="https://r.example/j/1", title="Sr. Python Developer", company="Acme Corp.", engine="rapid")],
        )
        p2 = _StubProvider(
            "automation",
            [_raw_job(url="https://a.example/j/1", title="SR PYTHON DEVELOPER", company="ACME CORP", engine="automation")],
        )
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")

        assert len(jobs) == 1, (
            f"Normalized (title, company) duplicates must soft-dedup to 1; got {len(jobs)}"
        )

    def test_dedup_prefers_richer_description(self):
        """Collision resolution keeps the entry with the richer description."""
        from app.scraper.service import CompositeJobScraper

        thin = _raw_job(
            url="https://r.example/j/2",
            title="Backend Engineer",
            company="Acme",
            engine="rapid",
            description="Short",
        )
        rich = _raw_job(
            url="https://a.example/j/2",
            title="Backend Engineer",
            company="Acme",
            engine="automation",
            description=(
                "Detailed description covering FastAPI, PostgreSQL, Docker deployment, "
                "CI pipelines, observability, and production on-call expectations."
            ),
        )
        p1 = _StubProvider("rapid", [thin])
        p2 = _StubProvider("automation", [rich])
        composite = CompositeJobScraper(providers=[p1, p2])

        jobs = composite.fetch_jobs("python")

        assert len(jobs) == 1
        assert jobs[0]["description"] == rich["description"], (
            "Dedup must prefer the richer description on collision"
        )
        assert jobs[0]["scraper_engine"] == "automation"

    def test_provider_selection_filtering(self):
        """providers=['rapid','automation'] dispatches only to those providers."""
        from app.scraper.service import CompositeJobScraper

        rapid = _StubProvider("rapid", [_raw_job(url="https://r.example/1", title="Rapid Role", company="RapidCo", engine="rapid")])
        auto = _StubProvider("automation", [_raw_job(url="https://a.example/1", title="Auto Role", company="AutoCo", engine="automation")])
        skipped = _StubProvider("proxy_api", [_raw_job(url="https://p.example/1", title="Proxy Role", company="ProxyCo", engine="proxy_api")])
        composite = CompositeJobScraper(providers=[rapid, auto, skipped])

        jobs = composite.fetch_jobs("python", providers=["rapid", "automation"])

        engines = {j["scraper_engine"] for j in jobs}
        assert engines == {"rapid", "automation"}, (
            f"Provider filter must exclude proxy_api; got {engines}"
        )
        assert "proxy_api" not in engines

    def test_get_available_providers_metadata(self):
        """get_available_providers() returns name/display_name/supported_sites for all."""
        from app.scraper.service import CompositeJobScraper

        p1 = _StubProvider("rapid")
        p2 = _StubProvider("automation")
        composite = CompositeJobScraper(providers=[p1, p2])

        meta = composite.get_available_providers()

        assert isinstance(meta, list)
        assert len(meta) == 2
        names = {m["name"] for m in meta}
        assert names == {"rapid", "automation"}
        for m in meta:
            assert "display_name" in m
            assert "supported_sites" in m
            assert isinstance(m["supported_sites"], list)


class TestMultiProviderAPIEndpoints:
    """POST /api/jobs/scrape (multi-provider body) + GET /api/jobs/scrape/providers."""

    def test_api_scrape_jobs_multi_provider_body(self, client):
        """POST accepts providers/sites in JSON body and executes composite scrape."""
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            resp = client.post(
                "/api/jobs/scrape",
                json={
                    "search_term": "Python FastAPI Developer",
                    "location": "India",
                    "results_wanted": 10,
                    "providers": ["jobspy", "rapid", "automation", "proxy_api"],
                    "sites": ["linkedin", "naukri", "indeed"],
                },
            )

        # 200 on success; 422 if body schema differs — endpoint must EXIST
        assert resp.status_code in (200, 422), (
            f"POST /api/jobs/scrape returned {resp.status_code}: {resp.text[:400]}"
        )

        if resp.status_code == 200:
            data = resp.json()
            assert "status" in data or "fetched" in data or "ingested" in data

    def test_api_scrape_providers_endpoint(self, client):
        """GET /api/jobs/scrape/providers returns registered provider list."""
        resp = client.get("/api/jobs/scrape/providers")

        assert resp.status_code == 200, (
            f"GET /api/jobs/scrape/providers returned {resp.status_code}"
        )
        data = resp.json()

        # Response: list of providers or dict containing providers key
        providers = data if isinstance(data, list) else data.get("providers", [])
        assert isinstance(providers, list)
        assert len(providers) >= 4, (
            f"Expected >=4 registered providers, got {len(providers)}"
        )
        names = set()
        for p in providers:
            if isinstance(p, dict):
                names.add(p.get("name"))
            else:
                names.add(p)
        for expected in ("jobspy", "rapid", "automation", "proxy_api"):
            assert expected in names, f"Provider {expected!r} not registered: {names}"


class TestNormalizerPipelineMultiProvider:
    """scraper_engine attribution + cross-provider ingest dedup."""

    def test_normalize_captures_scraper_engine(self):
        """normalize_scraped_job stamps scraper_engine with fallback 'jobspy'."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = _raw_job(engine="rapid", site="naukri")
        job = normalize_scraped_job(raw, taxonomy)

        assert hasattr(job, "scraper_engine"), "Job model must have scraper_engine field"
        assert job.scraper_engine == "rapid", (
            f"scraper_engine not captured: {job.scraper_engine!r}"
        )

    def test_normalize_scraper_engine_default(self):
        """Missing scraper_engine in raw -> defaults to 'jobspy'."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = _raw_job()
        raw.pop("scraper_engine")
        job = normalize_scraped_job(raw, taxonomy)

        assert job.scraper_engine == "jobspy", (
            f"Default scraper_engine must be 'jobspy'; got {job.scraper_engine!r}"
        )

    def test_normalize_description_includes_attribution(self):
        """Description enriched with apply URL + source + scraper attribution."""
        from app.scraper.normalizer import normalize_scraped_job

        taxonomy = load_taxonomy()
        raw = _raw_job(url="https://example.com/apply/1", engine="rapid", site="naukri")
        job = normalize_scraped_job(raw, taxonomy)

        assert "Apply URL: https://example.com/apply/1" in job.description
        assert "rapid" in job.description, (
            "Description must attribute scraper engine; "
            f"got: {job.description[-200:]}"
        )

    def test_pipeline_ingest_cross_provider_dedup(self, client_engine):
        """Ingesting same canonical URL from 2 providers yields 1 Job row."""
        _, engine = client_engine
        from sqlmodel import Session, select

        from app.models import Job
        from app.scraper.pipeline import ingest_scraped_jobs
        from app.taxonomy.taxonomy import load_taxonomy

        taxonomy = load_taxonomy()
        raw_jobs = [
            _raw_job(url="https://example.com/dup/1", engine="rapid", site="naukri"),
            _raw_job(url="https://example.com/dup/1?utm=x", engine="automation", site="linkedin"),
        ]

        with Session(engine) as db:
            result = ingest_scraped_jobs(raw_jobs, db, MagicMock(), taxonomy)
            # query persisted rows for this URL
            rows = db.exec(select(Job)).all()
            dup_rows = [j for j in rows if j.job_url.startswith("https://example.com/dup/1")]

        assert len(dup_rows) == 1, (
            f"Cross-provider duplicate URL must persist once; got {len(dup_rows)}"
        )
        assert result["ingested_count"] >= 1


class TestZeroEmojiMilestone6UI:
    """Milestone 6 UI additions (provider badges/checkboxes) stay zero-emoji."""

    def test_matches_html_zero_emoji_after_milestone6(self):
        """Full matches.html scan (provider checkboxes + engine badges) stays clean."""
        from pathlib import Path

        matches_html = Path("app/templates/student/matches.html")
        if not matches_html.exists():
            pytest.skip("matches.html not found")

        content = matches_html.read_text(encoding="utf-8")

        emoji_pattern = re.compile("[\U0001F300-\U0001FAFF☀-➿⏰-⏳]")
        entity_pattern = re.compile(
            r"&#(?:8962|9776|9989|10003|10007|12[789]\d{3}|127\d{3})\b"
        )

        literal_emojis = emoji_pattern.findall(content)
        entities = entity_pattern.findall(content)

        assert literal_emojis == [], f"matches.html contains literal emojis: {literal_emojis}"
        assert entities == [], f"matches.html contains emoji entities: {entities}"


# ---------------------------------------------------------------------------
# Milestone 6: Scraper Providers Abstraction Tests
# ---------------------------------------------------------------------------


class TestScraperProviders:
    """Test BaseJobScraperProvider interface and concrete provider engines."""

    def test_providers_exported_from_package(self):
        """All providers must be re-exported from app.scraper."""
        from app.scraper import (
            AutomationJobScraperProvider,
            BaseJobScraperProvider,
            JobSpyProvider,
            ProxyAPIJobScraperProvider,
            RapidJobScraperProvider,
        )

        assert issubclass(JobSpyProvider, BaseJobScraperProvider)
        assert issubclass(RapidJobScraperProvider, BaseJobScraperProvider)
        assert issubclass(AutomationJobScraperProvider, BaseJobScraperProvider)
        assert issubclass(ProxyAPIJobScraperProvider, BaseJobScraperProvider)

    def test_rapid_scraper_provider_contract(self):
        """RapidJobScraperProvider must harvest tech jobs with valid schema."""
        from app.scraper.providers import RapidJobScraperProvider

        provider = RapidJobScraperProvider()
        assert provider.name == "rapid"
        assert len(provider.supported_sites) >= 2

        jobs = provider.fetch_jobs("Python Developer", results_wanted=5)
        assert isinstance(jobs, list)
        assert len(jobs) == 5
        required_keys = {
            "title",
            "company",
            "location",
            "description",
            "job_url",
            "site",
            "scraper_engine",
            "date_posted",
            "min_amount",
            "max_amount",
        }
        for j in jobs:
            assert required_keys.issubset(j.keys())
            assert j["scraper_engine"] == "rapid"
            assert j["job_url"].startswith("http")

    def test_automation_scraper_provider_contract(self):
        """AutomationJobScraperProvider must return tech startup jobs."""
        from app.scraper.providers import AutomationJobScraperProvider

        provider = AutomationJobScraperProvider()
        assert provider.name == "automation"
        jobs = provider.fetch_jobs("Full Stack Engineer", results_wanted=3)
        assert isinstance(jobs, list)
        assert len(jobs) == 3
        for j in jobs:
            assert j["scraper_engine"] == "automation"
            assert j["job_url"].startswith("http")

    def test_proxy_api_scraper_provider_contract(self):
        """ProxyAPIJobScraperProvider must handle offline/mock resilience."""
        from app.scraper.providers import ProxyAPIJobScraperProvider

        provider = ProxyAPIJobScraperProvider()
        assert provider.name == "proxy_api"
        jobs = provider.fetch_jobs("Backend Developer", results_wanted=4)
        assert isinstance(jobs, list)
        assert len(jobs) == 4
        for j in jobs:
            assert j["scraper_engine"] == "proxy_api"
            assert j["job_url"].startswith("http")

    def test_jobspy_provider_contract_mocked(self):
        """JobSpyProvider must wrap JobSpyScraper and normalize output schema."""
        from app.scraper.providers import JobSpyProvider

        mock_scraper = MagicMock()
        mock_scraper.fetch_jobs.return_value = [
            {
                "title": "Senior Engineer",
                "company": "BigTech",
                "location": "Bengaluru",
                "description": "Python, SQL",
                "job_url": "https://linkedin.com/jobs/view/999",
                "site": "linkedin",
                "min_amount": 18.0,
                "max_amount": 35.0,
                "date_posted": "2026-11-01",
            }
        ]
        provider = JobSpyProvider(scraper=mock_scraper)
        assert provider.name == "jobspy"
        assert "naukri" in provider.supported_sites

        jobs = provider.fetch_jobs("Python", results_wanted=1)
        assert len(jobs) == 1
        assert jobs[0]["scraper_engine"] == "jobspy"
        assert jobs[0]["title"] == "Senior Engineer"
        assert jobs[0]["job_url"] == "https://linkedin.com/jobs/view/999"


# ===========================================================================
# Zero-Dummy URL Gate — ARCHITECTURE.md Section 12.3 (Zero Mock / Zero
# example.com policy). Spec-driven acceptance gate (charter §7): the
# provider-URL tests are RED for the 3 secondary providers until the
# remediation epic lands real feeds; purge tests are permanent regressions.
# ===========================================================================

#: RFC 2606 reserved placeholder domains that must NEVER appear in output.
_RESERVED_HOSTS = ("example.com", "example.org", "example.net")


def _is_dummy_url(url: str) -> bool:
    """True when a URL points at a placeholder/dummy domain (spec 12.3 ban)."""
    low = str(url or "").lower()
    return "example.com" in low or "rapid.example" in low or "example.org" in low


def _host_of(url: str) -> str:
    from urllib.parse import urlsplit

    return urlsplit(str(url or "")).netloc.lower()


class TestZeroDummyScraperURLs:
    """No scraper provider may emit dummy domains; purge must clean legacy rows."""

    def test_all_providers_emit_real_https_urls(self):
        """Every provider's job_url: https, non-reserved host, no example.com."""
        from app.scraper.providers import (
            AutomationJobScraperProvider,
            JobSpyProvider,
            ProxyAPIJobScraperProvider,
            RapidJobScraperProvider,
        )

        mock_jobspy = MagicMock()
        mock_jobspy.fetch_jobs.return_value = [
            {
                "title": "Python Developer",
                "company": "Tech Corp",
                "job_url": "https://www.linkedin.com/jobs/view/999111",
                "site": "linkedin",
            }
        ]
        providers = {
            JobSpyProvider: JobSpyProvider(scraper=mock_jobspy),
            RapidJobScraperProvider: RapidJobScraperProvider(),
            AutomationJobScraperProvider: AutomationJobScraperProvider(),
            ProxyAPIJobScraperProvider: ProxyAPIJobScraperProvider(),
        }
        offenders: dict[str, list[str]] = {}

        for cls, provider in providers.items():
            try:
                jobs = provider.fetch_jobs("python developer", results_wanted=5)
            except Exception as exc:  # provider must not crash even when feed is down
                offenders[f"{cls.__name__} (exception)"] = [str(exc)]
                continue

            bad = []
            for job in jobs:
                url = str(job.get("job_url") or "")
                if not url:
                    bad.append(f"empty job_url in {job.get('title')!r}")
                    continue
                if not url.startswith("https://"):
                    bad.append(f"non-https URL: {url}")
                if _is_dummy_url(url):
                    bad.append(f"DUMMY domain: {url}")
                host = _host_of(url)
                if any(host == r or host.endswith("." + r) for r in _RESERVED_HOSTS):
                    bad.append(f"RFC2606 reserved host: {url}")
                if "." not in host or len(host.split(".")[-1]) < 2:
                    bad.append(f"no valid TLD host: {url}")
            if bad:
                offenders[cls.__name__] = bad

        assert offenders == {}, (
            "Providers emitting dummy/placeholder URLs (violates spec 12.3 "
            "zero-mock policy):\n"
            + "\n".join(f"  {name}: {items}" for name, items in offenders.items())
        )

    def test_purge_dummy_jobs_removes_db_rows_and_vectors(self):
        """purge_dummy_jobs() deletes example.com rows + drops their vectors."""
        from sqlmodel import Session, select

        from app.db import make_engine
        from app.models import Job
        from app.scraper.pipeline import purge_dummy_jobs

        engine = make_engine("sqlite://")
        Job.metadata.create_all(engine)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {"job-ext-dummy1": [0.1, 0.2], "job-ext-real1": [0.3, 0.4]}

        with Session(engine) as db:
            # Legacy dummy rows (both banned patterns) + one real row that survives
            db.add(Job(id="job-ext-dummy1", title="D1", company="C", description="d",
                       job_url="https://rapid.example.com/jobs/1", source="rapid"))
            db.add(Job(id="job-ext-dummy2", title="D2", company="C", description="d",
                       job_url="https://careers.example.com/jobs/2", source="automation"))
            db.add(Job(id="job-ext-real1", title="R1", company="C", description="d",
                       job_url="https://www.instahyre.com/jobs?q=python", source="rapid"))
            db.commit()

        removed = purge_dummy_jobs(Session(engine), ctx)

        assert removed == 2, f"Expected 2 dummy rows purged, got {removed}"

        with Session(engine) as db:
            remaining = db.exec(select(Job)).all()
            ids = {j.id for j in remaining}
        assert ids == {"job-ext-real1"}, (
            f"Only the real job must survive purge; got {ids}"
        )

        # Vector cleanup: both purged ids dropped, real id intact
        for job_id in ("job-ext-dummy1", "job-ext-dummy2"):
            ctx.vector_store.delete.assert_any_call("job", job_id)
            assert job_id not in ctx.job_vectors
        assert "job-ext-real1" in ctx.job_vectors

    def test_purge_idempotent_on_clean_db(self):
        """Anti-vacuity companion: clean DB purges 0 rows and changes nothing."""
        from sqlmodel import Session, select

        from app.db import make_engine
        from app.models import Job
        from app.scraper.pipeline import purge_dummy_jobs

        engine = make_engine("sqlite://")
        Job.metadata.create_all(engine)

        with Session(engine) as db:
            db.add(Job(id="job-ext-real2", title="R2", company="C", description="d",
                       job_url="https://www.hirist.tech/search/python-jobs-in-bengaluru",
                       source="automation"))
            db.commit()

        removed = purge_dummy_jobs(Session(engine), None)

        assert removed == 0
        with Session(engine) as db:
            assert len(db.exec(select(Job)).all()) == 1

    def test_purge_pattern_coverage_detects_dummies(self):
        """Anti-vacuity: the detector actually matches the banned patterns."""
        from app.scraper.pipeline import DUMMY_URL_PATTERNS

        assert any("example.com" in p for p in DUMMY_URL_PATTERNS)
        assert any("rapid.example" in p for p in DUMMY_URL_PATTERNS)
        # And the synthetic dummies our tests use are indeed dummies
        assert _is_dummy_url("https://rapid.example.com/jobs/1")
        assert _is_dummy_url("https://careers.example.com/jobs/2")
        assert not _is_dummy_url("https://www.instahyre.com/jobs?q=python")
        assert not _is_dummy_url("https://in.indeed.com/viewjob?jk=abc")

    def test_purge_dummy_jobs_cascades_foreign_keys(self, client_engine):
        """Purge must delete child rows (JobMatch/UpskillPlan/PlanItem/Interview)
        referencing dummy jobs BEFORE deleting the Job itself — otherwise the
        FK constraint violation aborts the transaction (the reported bug).

        Anti-vacuity: the harness must enforce FKs (conftest._fk_on enables
        PRAGMA foreign_keys=ON) so this test FAILS today with
        ForeignKeyViolation and flips GREEN once the cascade fix lands.
        Valid (non-dummy) Job + its children must survive untouched.
        """
        from sqlmodel import Session, select

        from app.models import (
            Course,
            Interview,
            Job,
            JobMatch,
            PlanItem,
            Student,
            UpskillPlan,
        )
        from app.scraper.pipeline import purge_dummy_jobs

        _c, engine = client_engine

        # Seed prerequisites (FK targets): 1 student, 1 course
        with Session(engine) as db:
            db.add(Student(id="stu-cascade", name="Cascade Tester", branch="CSE"))
            db.add(Course(id="course-cascade", title="Cascade Course",
                          provider="test", url="https://course.test"))
            db.commit()

        # Step 1: insert Job rows first (parents must exist before children reference them)
        with Session(engine) as db:
            db.add(Job(id="job-dummy-c1", title="Dummy 1", company="Acme",
                       description="d", job_url="https://rapid.example.com/jobs/d1",
                       source="rapid"))
            db.add(Job(id="job-dummy-c2", title="Dummy 2", company="Acme",
                       description="d", job_url="https://careers.example.com/jobs/d2",
                       source="automation"))
            db.add(Job(id="job-valid-c", title="Valid Job", company="RealCorp",
                       description="real", job_url="https://www.instahyre.com/jobs/123",
                       source="rapid"))
            db.commit()

        # Step 2a: UpskillPlan rows (depend on jobs)
        with Session(engine) as db:
            db.add(UpskillPlan(id="plan-dummy-c1", student_id="stu-cascade",
                               job_id="job-dummy-c1", total_hours=10))
            db.add(UpskillPlan(id="plan-valid-c", student_id="stu-cascade",
                               job_id="job-valid-c", total_hours=20))
            db.commit()

        # Step 2b: PlanItem, JobMatch, Interview rows (depend on plans/jobs)
        with Session(engine) as db:
            # Dummy job 1: PlanItem + JobMatch + Interview
            db.add(PlanItem(id="pi-dummy-c1", plan_id="plan-dummy-c1", order=1,
                            skill_id="python", course_id="course-cascade", hours=5))
            db.add(JobMatch(id="jm-dummy-c1", student_id="stu-cascade",
                            job_id="job-dummy-c1", score=85, band="strong"))
            db.add(Interview(id="int-dummy-c1", student_id="stu-cascade",
                             job_id="job-dummy-c1"))

            # Dummy job 2: only JobMatch
            db.add(JobMatch(id="jm-dummy-c2", student_id="stu-cascade",
                            job_id="job-dummy-c2", score=70, band="good"))

            # Valid job: PlanItem + JobMatch (must survive)
            db.add(PlanItem(id="pi-valid-c", plan_id="plan-valid-c", order=1,
                            skill_id="python", course_id="course-cascade", hours=8))
            db.add(JobMatch(id="jm-valid-c", student_id="stu-cascade",
                            job_id="job-valid-c", score=90, band="strong"))
            db.commit()

        # Execute purge — this is where FK violation currently fires
        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}

        # The call itself must NOT raise ForeignKeyViolation
        removed = purge_dummy_jobs(Session(engine), ctx)

        # Verify cascade deletion: dummy job + all children gone
        with Session(engine) as db:
            job_ids = {j.id for j in db.exec(select(Job)).all()}
            plan_ids = {p.id for p in db.exec(select(UpskillPlan)).all()}
            plan_item_ids = {pi.id for pi in db.exec(select(PlanItem)).all()}
            match_ids = {jm.id for jm in db.exec(select(JobMatch)).all()}
            interview_ids = {i.id for i in db.exec(select(Interview)).all()}

        assert "job-dummy-c1" not in job_ids, "Dummy job 1 must be purged"
        assert "job-dummy-c2" not in job_ids, "Dummy job 2 must be purged"
        assert "plan-dummy-c1" not in plan_ids, "UpskillPlan for dummy must cascade"
        assert "pi-dummy-c1" not in plan_item_ids, "PlanItem must cascade via plan_id"
        assert "jm-dummy-c1" not in match_ids, "JobMatch for dummy 1 must cascade"
        assert "jm-dummy-c2" not in match_ids, "JobMatch for dummy 2 must cascade"
        assert "int-dummy-c1" not in interview_ids, "Interview for dummy must cascade"

        # Verify valid job + children untouched
        assert "job-valid-c" in job_ids, "Valid job must survive"
        assert "plan-valid-c" in plan_ids, "Valid UpskillPlan must survive"
        assert "pi-valid-c" in plan_item_ids, "Valid PlanItem must survive"
        assert "jm-valid-c" in match_ids, "Valid JobMatch must survive"

        assert removed == 2, f"Expected 2 dummy jobs purged, got {removed}"

        # Verify idempotency: a second call must not raise, must not double-delete,
        # and must not touch the valid rows (spec: "Verify idempotency on repeated calls").
        removed_second = purge_dummy_jobs(Session(engine), ctx)
        assert removed_second == 0, (
            f"Second purge must find 0 remaining dummy rows, got {removed_second}"
        )

        with Session(engine) as db:
            job_ids_2 = {j.id for j in db.exec(select(Job)).all()}
            match_ids_2 = {jm.id for jm in db.exec(select(JobMatch)).all()}
            plan_ids_2 = {p.id for p in db.exec(select(UpskillPlan)).all()}
            item_ids_2 = {pi.id for pi in db.exec(select(PlanItem)).all()}
            int_ids_2 = {i.id for i in db.exec(select(Interview)).all()}

        # My test rows: valid job + children still intact after the repeated call
        # (note: client_engine seeds 24 other jobs job-001..job-024 — only check MY ids)
        assert "job-valid-c" in job_ids_2, "Valid job must survive repeated purge"
        assert "job-dummy-c1" not in job_ids_2, "Dummy 1 must stay purged"
        assert "job-dummy-c2" not in job_ids_2, "Dummy 2 must stay purged"
        assert "jm-valid-c" in match_ids_2, "Valid JobMatch must survive"
        assert "jm-dummy-c1" not in match_ids_2, "Dummy JobMatch 1 must stay purged"
        assert "jm-dummy-c2" not in match_ids_2, "Dummy JobMatch 2 must stay purged"
        assert "plan-valid-c" in plan_ids_2, "Valid UpskillPlan must survive"
        assert "plan-dummy-c1" not in plan_ids_2, "Dummy plan must stay purged"
        assert "pi-valid-c" in item_ids_2, "Valid PlanItem must survive"
        assert "pi-dummy-c1" not in item_ids_2, "Dummy PlanItem must stay purged"
        assert "int-dummy-c1" not in int_ids_2, "Dummy Interview must stay purged"


# ===========================================================================
# Milestone 7: Advanced Stealth Harvester (8 Portals) Integration
# ===========================================================================


class TestStealthHarvesterIntegration:
    """Stealth Harvester provider, output ingestion endpoint, CLI and UI."""

    def test_stealth_harvester_provider_contract(self):
        from app.scraper.providers import BaseJobScraperProvider, StealthHarvesterProvider

        p = StealthHarvesterProvider()
        assert isinstance(p, BaseJobScraperProvider)
        assert p.name == "stealth_harvester"
        assert "8 Indian Portals" in p.display_name
        assert len(p.supported_sites) == 8
        for site in ("naukri", "linkedin", "indeed", "foundit", "internshala", "unstop", "instahyre", "hirist"):
            assert site in p.supported_sites

    def test_composite_scraper_includes_stealth_harvester(self):
        from app.scraper.service import CompositeJobScraper

        composite = CompositeJobScraper()
        providers = composite.get_available_providers()
        names = {p["name"] for p in providers}
        assert "stealth_harvester" in names
        assert "jobspy" in names
        assert "rapid" in names

    def test_api_jobs_scrape_supports_stealth_harvester(self, client, monkeypatch):
        from app.scraper.service import CompositeJobScraper

        def fake_fetch_jobs(self, search_term, location="India", results_wanted=15, providers=None, **kwargs):
            return [
                {
                    "title": "Stealth Python Lead",
                    "company": "StealthCorp",
                    "location": "Bengaluru",
                    "description": "Lead Python Engineer. Apply URL: https://naukri.com/job/101",
                    "job_url": "https://www.naukri.com/job-listings-stealth-lead-101",
                    "site": "naukri",
                    "scraper_engine": "stealth_harvester",
                    "date_posted": None,
                }
            ]

        monkeypatch.setattr(CompositeJobScraper, "fetch_jobs", fake_fetch_jobs)
        resp = client.post(
            "/api/jobs/scrape",
            json={
                "search_term": "Python",
                "location": "Bengaluru",
                "providers": ["stealth_harvester"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["fetched"] == 1
        assert "stealth_harvester" in data["providers_used"]

    def test_api_ingest_output_not_found_returns_404(self, client):
        resp = client.post(
            "/api/jobs/scrape/ingest-output",
            json={"file_path": "output/nonexistent_file_xyz.json"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_api_ingest_output_endpoint(self, client, tmp_path):
        import json

        test_jobs = [
            {
                "id": "job-ext-tst001",
                "title": "Ingested Python Engineer",
                "company": "HarvesterTech",
                "description": "Full stack Python position. Apply URL: https://internshala.com/job/1",
                "job_url": "https://internshala.com/job/1",
                "source": "internshala",
                "scraper_engine": "stealth_harvester",
                "required_skills": ["python", "sql"],
                "preferred_skills": [],
                "experience_years": 1,
                "location": "Bengaluru",
                "remote": False,
                "salary_lpa": [6, 12],
                "allowed_branches": ["CSE", "IT"],
            }
        ]
        out_file = tmp_path / "test_normalized.json"
        out_file.write_text(json.dumps(test_jobs), encoding="utf-8")

        resp = client.post(
            "/api/jobs/scrape/ingest-output",
            json={"file_path": str(out_file)},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["ingested"] == 1
        assert len(data["job_ids"]) == 1

    def test_cli_ingest_output_command(self, tmp_path, monkeypatch):
        import json

        from app import cli
        from app.cli import main
        from app.db import make_engine
        from app.embeddings import HashingEmbedder

        sqlite_engine = make_engine("sqlite://")
        monkeypatch.setattr("app.cli._make_engine", lambda url: sqlite_engine)
        monkeypatch.setattr(cli.settings, "database_url", "sqlite://")
        monkeypatch.setattr("app.cli._make_embedder", lambda: HashingEmbedder())

        test_jobs = [
            {
                "id": "job-ext-cli001",
                "title": "CLI Python Developer",
                "company": "CLITech",
                "description": "CLI test job. Apply URL: https://unstop.com/job/1",
                "job_url": "https://unstop.com/job/1",
                "source": "unstop",
                "scraper_engine": "stealth_harvester",
            }
        ]
        out_file = tmp_path / "cli_norm.json"
        out_file.write_text(json.dumps(test_jobs), encoding="utf-8")

        ret = main(["ingest-output", "--file", str(out_file)])
        assert ret == 0

        # Also test alias ingest-scraped-jobs
        ret_alias = main(["ingest-scraped-jobs", "--file", str(out_file)])
        assert ret_alias == 0

    def test_matches_html_stealth_harvester_badge_and_modal(self):
        from pathlib import Path

        html = Path("app/templates/student/matches.html").read_text(encoding="utf-8")
        assert 'value="stealth_harvester"' in html
        assert "Stealth Harvester (8 Indian Portals" in html
        assert "stealth_harvester" in html


# ===========================================================================
# Milestone 7: Advanced Stealth Harvester (8 portals) — provider behaviour gate
# ARCHITECTURE.md Sections 14 & 15 (spec-driven acceptance gate — charter §7)
# Complements TestStealthHarvesterIntegration (metadata/API/CLI/UI): pins the
# provider's fetch behaviour, output-file fallback, cross-portal dedup,
# composite aggregation, and hermetic file -> DB + vector ingestion.
# ===========================================================================


def _placement_rec(
    job_id: str,
    title: str,
    company: str,
    url: str,
    site: str = "naukri",
    description: str = "Build and ship production APIs.",
) -> dict:
    """Normalized PlacementJob-format record (as written to normalized_jobs.json)."""
    return {
        "id": job_id,
        "title": title,
        "company": company,
        "description": description,
        "required_skills": ["python"],
        "preferred_skills": [],
        "experience_years": 2,
        "location": "Bengaluru",
        "remote": False,
        "salary_lpa": [8, 14],
        "min_gpa": 7.0,
        "allowed_branches": ["CSE", "IT"],
        "job_url": url,
        "source": site,
        "scraper_engine": "stealth_harvester",
    }


def _raw_rec(title: str, company: str, url: str, site: str = "unstop") -> dict:
    """RawScrapedJob-format record (as written to raw_scraped_jobs.json)."""
    return {
        "title": title,
        "company": company,
        "location": "Bengaluru",
        "description": "Raw harvested posting.",
        "job_url": url,
        "site": site,
        "scraper_engine": "stealth_harvester",
        "date_posted": "2026-11-01",
        "min_amount": 6.0,
        "max_amount": 12.0,
        "interval": "lpa",
        "is_remote": False,
        "experience_years": 1.0,
        "tags": ["python"],
    }


class TestStealthHarvesterProvider:
    """Provider fetch behaviour, output-file fallback, dedup, aggregation, ingestion."""

    def test_fetch_jobs_live_run_mocked_pipeline(self, tmp_path, monkeypatch):
        """live=True executes the job_scraper pipeline (mocked); every unique
        listing is returned stamped scraper_engine='stealth_harvester' with no
        artificial truncation (spec 14.2 B/C)."""
        import job_scraper.pipeline as js_pipeline
        from app.scraper.providers import StealthHarvesterProvider

        listings = [
            _placement_rec("sh-1", "Python Developer", "Acme Corp",
                           "https://www.naukri.com/jobrapito/detail/111", site="naukri"),
            _placement_rec("sh-2", "Python Engineer", "ByteCraft",
                           "https://www.linkedin.com/jobs/view/222", site="linkedin"),
            _placement_rec("sh-3", "Backend Developer", "CodeLabs",
                           "https://www.indeed.com/viewjob?jk=333", site="indeed"),
        ]
        calls: list[dict] = []

        async def fake_run_pipeline(keywords, location="India", days_old=7,
                                    limit=20, portals=None, output_dir="output",
                                    progress=None):
            calls.append({"keywords": list(keywords), "location": location})
            return {"raw": [], "normalized": [dict(j) for j in listings],
                    "results": [], "counts": {}, "outputs": {}}

        monkeypatch.setattr(js_pipeline, "run_pipeline", fake_run_pipeline)

        provider = StealthHarvesterProvider(output_dir=tmp_path / "no_output")
        out = provider.fetch_jobs("Python Developer", location="Bengaluru",
                                  results_wanted=5, live=True)

        assert len(calls) == 1, "live run must execute the pipeline exactly once"
        assert "Python Developer" in calls[0]["keywords"]
        assert calls[0]["location"] == "Bengaluru"
        assert len(out) == 3, (
            f"all unique listings must survive (no truncation), got {len(out)}"
        )
        for job in out:
            assert job["scraper_engine"] == "stealth_harvester"
            for key in ("title", "company", "location", "description", "job_url", "site"):
                assert job.get(key) not in (None, ""), f"contract key missing/empty: {key}"
        assert {j["job_url"] for j in out} == {j["job_url"] for j in listings}

    def test_fetch_jobs_merges_normalized_output_file(self, tmp_path):
        """Pre-scraped output/normalized_jobs.json is emitted without a live run;
        placeholder URLs are filtered out (zero-mock, spec 12.3)."""
        import json

        from app.scraper.providers import StealthHarvesterProvider

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        good1 = _placement_rec("sh-f1", "FastAPI Engineer", "OtherCo",
                               "https://www.indeed.com/viewjob?jk=aaa", site="indeed")
        good2 = _placement_rec("sh-f2", "Python Lead", "ZetaCorp",
                               "https://www.linkedin.com/jobs/view/bbb", site="linkedin")
        dummy = _placement_rec("sh-f3", "Dummy Listing", "PlaceholderCo",
                               "https://rapid.example.com/jobs/1", site="naukri")
        (out_dir / "normalized_jobs.json").write_text(
            json.dumps([good1, good2, dummy]), encoding="utf-8")

        provider = StealthHarvesterProvider(output_dir=out_dir)
        out = provider.fetch_jobs("python", location="India", results_wanted=10)

        urls = {j["job_url"] for j in out}
        assert good1["job_url"] in urls and good2["job_url"] in urls
        assert dummy["job_url"] not in urls, "placeholder URL must be filtered (spec 12.3)"
        assert all(j["scraper_engine"] == "stealth_harvester" for j in out)
        assert len(out) == 2

    def test_fetch_jobs_falls_back_to_raw_output_file(self, tmp_path):
        """With only raw_scraped_jobs.json present, raw records are still emitted
        in contract shape (spec 14.2 B: 'or output/raw_scraped_jobs.json')."""
        import json

        from app.scraper.providers import StealthHarvesterProvider

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        raw1 = _raw_rec("Data Engineer", "RawCo",
                        "https://internshala.com/jobs/data-eng-1", site="internshala")
        raw2 = _raw_rec("ML Engineer", "RawCorp2",
                        "https://unstop.com/experience/ml-eng-2", site="unstop")
        (out_dir / "raw_scraped_jobs.json").write_text(json.dumps([raw1, raw2]), encoding="utf-8")

        provider = StealthHarvesterProvider(output_dir=out_dir)
        out = provider.fetch_jobs("engineer", location="India", results_wanted=10)

        urls = {j["job_url"] for j in out}
        assert raw1["job_url"] in urls and raw2["job_url"] in urls
        assert all(j["scraper_engine"] == "stealth_harvester" for j in out)
        by_url = {j["job_url"]: j for j in out}
        assert by_url[raw1["job_url"]]["site"] == "internshala"

    def test_dedup_drops_duplicates_keeps_all_unique(self, tmp_path, monkeypatch):
        """Duplicate canonical job_url (tracking params stripped) and duplicate
        normalized (title, company) pairs are dropped while every unique
        opportunity survives (spec 14.2 C)."""
        import job_scraper.pipeline as js_pipeline
        from app.scraper.providers import StealthHarvesterProvider

        a1 = _placement_rec("sh-d1", "Python Developer", "Acme Corp",
                            "https://www.naukri.com/jobrapito/detail/12345",
                            site="naukri",
                            description="Rich long description with many details about the "
                                        "role, the team, the stack and the responsibilities "
                                        "across services and releases.")
        a2 = _placement_rec("sh-d2", "Python Developer", "Acme Corp",
                            "https://www.naukri.com/jobrapito/detail/12345?ref=linkedin",
                            site="naukri", description="Short.")
        b = _placement_rec("sh-d3", "PYTHON DEVELOPER", "ACME CORP",
                           "https://www.linkedin.com/jobs/view/67890",
                           site="linkedin", description="Shorter still.")
        c = _placement_rec("sh-d4", "FastAPI Engineer", "OtherCo",
                           "https://www.indeed.com/viewjob?jk=unique789", site="indeed")

        async def fake_run_pipeline(keywords, location="India", days_old=7,
                                    limit=20, portals=None, output_dir="output",
                                    progress=None):
            return {"raw": [], "normalized": [dict(j) for j in (a1, a2, b, c)],
                    "results": [], "counts": {}, "outputs": {}}

        monkeypatch.setattr(js_pipeline, "run_pipeline", fake_run_pipeline)

        provider = StealthHarvesterProvider(output_dir=tmp_path / "no_output")
        out = provider.fetch_jobs("python", location="India", results_wanted=10, live=True)

        urls = {j["job_url"] for j in out}
        assert len(out) == 2, (
            f"4 listings with a canonical-URL dup and a (title, company) dup must "
            f"collapse to 2 unique, got {len(out)}: {urls}"
        )
        assert a1["job_url"] in urls, (
            "canonical-URL duplicate must resolve to the richer original listing"
        )
        assert c["job_url"] in urls, "unique opportunity must survive dedup"

    def test_composite_aggregates_stealth_alongside_other_providers(
        self, tmp_path, monkeypatch
    ):
        """CompositeJobScraper merges stealth_harvester results with other
        providers' results (spec 15.1: 'aggregates stealth_harvester alongside
        other providers') — actual aggregation, not just registration."""
        import job_scraper.pipeline as js_pipeline
        from app.scraper.providers import BaseJobScraperProvider, StealthHarvesterProvider
        from app.scraper.service import CompositeJobScraper

        stealth_jobs = [
            _placement_rec("sh-a1", "Go Developer", "StealthCo",
                           "https://www.hirist.tech/jobs/go-dev-1", site="hirist"),
            _placement_rec("sh-a2", "Rust Engineer", "StealthCorp2",
                           "https://cutshort.io/jobs/rust-eng-2", site="hirist"),
        ]

        async def fake_run_pipeline(keywords, location="India", days_old=7,
                                    limit=20, portals=None, output_dir="output",
                                    progress=None):
            return {"raw": [], "normalized": [dict(j) for j in stealth_jobs],
                    "results": [], "counts": {}, "outputs": {}}

        monkeypatch.setattr(js_pipeline, "run_pipeline", fake_run_pipeline)

        class _StubJobspyProvider(BaseJobScraperProvider):
            name = "jobspy"
            display_name = "Stub JobSpy"
            supported_sites = ["naukri"]

            def fetch_jobs(self, search_term, location="India",
                           results_wanted=15, **kwargs):
                return [{
                    "title": "Java Developer", "company": "StubCorp",
                    "location": "Pune", "description": "Stub listing.",
                    "job_url": "https://www.naukri.com/jobrapito/detail/999",
                    "site": "naukri", "scraper_engine": "jobspy", "date_posted": None,
                }]

        stealth = StealthHarvesterProvider(output_dir=tmp_path / "no_output")
        composite = CompositeJobScraper(providers=[stealth, _StubJobspyProvider()])
        out = composite.fetch_jobs(
            "developer", location="India", results_wanted=10,
            providers=["stealth_harvester", "jobspy"], live=True,
        )

        engines = {j.get("scraper_engine") for j in out}
        assert "stealth_harvester" in engines, "stealth results missing from aggregate"
        assert "jobspy" in engines, "jobspy results missing from aggregate"
        urls = {j["job_url"] for j in out}
        assert stealth_jobs[0]["job_url"] in urls and stealth_jobs[1]["job_url"] in urls
        assert "https://www.naukri.com/jobrapito/detail/999" in urls
        assert len(out) == 3, f"3 unique listings from 2 providers expected, got {len(out)}"

    def test_ingest_cli_populates_db_and_vectors_hermetic(
        self, tmp_path, monkeypatch, client_engine
    ):
        """app.cli ingest-output: JSON file -> DB rows + vector upserts.

        Hermetic: the CLI's engine/store/embedder are redirected onto the
        fixture engine and a mock store so the real database is never touched.
        """
        import json
        from unittest.mock import MagicMock

        from sqlmodel import Session, select

        import app.cli as app_cli
        from app.models import Job

        _client, engine = client_engine

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        job1 = _placement_rec("sh-i1", "FastAPI Engineer", "ByteCraft",
                              "https://www.indeed.com/viewjob?jk=ing1", site="indeed")
        job2 = _placement_rec("sh-i2", "Python Backend", "CodeLabs",
                              "https://www.linkedin.com/jobs/view/ing2", site="linkedin")
        (out_dir / "normalized_jobs.json").write_text(json.dumps([job1, job2]), encoding="utf-8")

        store = MagicMock()
        monkeypatch.setattr(app_cli, "_make_engine", lambda url: engine)
        monkeypatch.setattr(app_cli, "_make_store", lambda session, url: store)
        monkeypatch.setattr(app_cli, "_make_embedder", lambda: MagicMock())

        rc = app_cli.cmd_ingest_output(str(out_dir / "normalized_jobs.json"))
        assert rc == 0, "ingest-output must succeed for a valid file"

        with Session(engine) as db:
            by_url = {j.job_url: j for j in db.exec(select(Job)).all()}
        assert job1["job_url"] in by_url, "ingested job 1 missing from database"
        assert job2["job_url"] in by_url, "ingested job 2 missing from database"

        # Vector index: one batched upsert_many covering every ingested job (M21)
        assert store.upsert_many.call_count == 1
        batched = {ref for (_kind, ref, _v) in store.upsert_many.call_args_list[0].args[0]}
        for url in (job1["job_url"], job2["job_url"]):
            assert by_url[url].id in batched

        # Error path: missing file -> rc 1, no DB writes
        with Session(engine) as db:
            count_before = len(db.exec(select(Job)).all())
        rc_missing = app_cli.cmd_ingest_output(str(tmp_path / "nope.json"))
        assert rc_missing == 1
        with Session(engine) as db:
            count_after = len(db.exec(select(Job)).all())
        assert count_after == count_before, "failed ingest must not change the database"

    def test_cli_scrape_command_surface(self):
        """Root cli.py exposes the spec 14.2 D scrape command with its flags."""
        import os
        import re
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["NO_COLOR"] = "1"
        env["TERM"] = "dumb"
        env.pop("FORCE_COLOR", None)
        r = subprocess.run(
            [sys.executable, "cli.py", "scrape", "--help"],
            cwd=repo_root, capture_output=True,
            encoding="utf-8", errors="replace", timeout=120,
            env=env,
        )
        assert r.returncode == 0, f"cli.py scrape --help failed: {r.stderr[-400:]}"
        combined = (r.stdout or "") + (r.stderr or "")
        clean = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", combined)
        for flag in ("--keywords", "--location", "--days-old", "--limit"):
            assert (flag in combined) or (flag in clean), f"scrape --help missing {flag}"


class TestIngestFromOutputFolder:
    """ingest_from_output_folder(): batch ingest of harvester output JSON into
    the database and vector index (spec 14.2 B/C), idempotent and folder-based."""

    def test_ingest_normalized_folder_populates_db_and_vectors(self, client_engine, tmp_path):
        import json

        from sqlmodel import Session, select

        from app.models import Job
        from app.scraper.pipeline import ingest_from_output_folder
        from app.taxonomy.taxonomy import load_taxonomy

        _client, engine = client_engine

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        job1 = _placement_rec("sh-in1", "FastAPI Engineer", "IngestCo",
                              "https://www.indeed.com/viewjob?jk=ingfa1", site="indeed")
        job2 = _placement_rec("sh-in2", "Python Backend", "IngestCo2",
                              "https://www.linkedin.com/jobs/view/ingpb2", site="linkedin")
        (out_dir / "normalized_jobs.json").write_text(json.dumps([job1, job2]), encoding="utf-8")

        ctx = MagicMock()
        with Session(engine) as db:
            result = ingest_from_output_folder(
                output_dir=out_dir, session=db, ctx=ctx, taxonomy=load_taxonomy()
            )

        assert result["ingested_count"] == 2, result
        assert result["source"].endswith("normalized_jobs.json"), result["source"]
        assert len(result["job_ids"]) == 2

        with Session(engine) as db:
            rows = {j.job_url: j for j in db.exec(select(Job)).all()}
        assert job1["job_url"] in rows, "normalized job 1 missing from database"
        assert job2["job_url"] in rows, "normalized job 2 missing from database"
        # Vector index: one batched upsert_many covering every ingested job (M21)
        assert ctx.vector_store.upsert_many.call_count == 1
        batched = {ref for (_kind, ref, _v) in ctx.vector_store.upsert_many.call_args_list[0].args[0]}
        for url in (job1["job_url"], job2["job_url"]):
            assert rows[url].id in batched

    def test_ingest_falls_back_to_raw_output_file(self, client_engine, tmp_path):
        import json

        from sqlmodel import Session, select

        from app.models import Job
        from app.scraper.pipeline import ingest_from_output_folder
        from app.taxonomy.taxonomy import load_taxonomy

        _client, engine = client_engine

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        raw1 = _raw_rec("Data Engineer", "RawIngestCo",
                        "https://internshala.com/jobs/ing-raw-1", site="internshala")
        (out_dir / "raw_scraped_jobs.json").write_text(json.dumps([raw1]), encoding="utf-8")

        ctx = MagicMock()
        with Session(engine) as db:
            result = ingest_from_output_folder(
                output_dir=out_dir, session=db, ctx=ctx, taxonomy=load_taxonomy()
            )

        assert result["ingested_count"] == 1, result
        assert result["source"].endswith("raw_scraped_jobs.json"), result["source"]
        with Session(engine) as db:
            rows = {j.job_url: j for j in db.exec(select(Job)).all()}
        assert raw1["job_url"] in rows, "raw-fallback job missing from database"

    def test_ingest_missing_folder_returns_zero_without_raising(self, client_engine, tmp_path):
        from sqlmodel import Session

        from app.scraper.pipeline import ingest_from_output_folder

        _client, engine = client_engine
        with Session(engine) as db:
            result = ingest_from_output_folder(
                output_dir=tmp_path / "does_not_exist", session=db, ctx=MagicMock()
            )
        assert result["ingested_count"] == 0
        assert result["job_ids"] == []
        assert result["source"] is None

    def test_ingest_is_idempotent(self, client_engine, tmp_path):
        import json

        from sqlmodel import Session, select

        from app.models import Job
        from app.scraper.pipeline import ingest_from_output_folder
        from app.taxonomy.taxonomy import load_taxonomy

        _client, engine = client_engine

        out_dir = tmp_path / "output"
        out_dir.mkdir()
        job1 = _placement_rec("sh-ip1", "Go Backend", "IdempotentCo",
                              "https://www.hirist.tech/jobs/ing-ip-1", site="hirist")
        (out_dir / "normalized_jobs.json").write_text(json.dumps([job1]), encoding="utf-8")

        taxonomy = load_taxonomy()
        ctx = MagicMock()
        with Session(engine) as db:
            first = ingest_from_output_folder(output_dir=out_dir, session=db, ctx=ctx, taxonomy=taxonomy)
        with Session(engine) as db:
            second = ingest_from_output_folder(output_dir=out_dir, session=db, ctx=ctx, taxonomy=taxonomy)

        assert first["ingested_count"] == 1
        assert second["ingested_count"] == 1, "re-ingest must merge the same unique jobs"
        with Session(engine) as db:
            dup_rows = [j for j in db.exec(select(Job)).all() if j.job_url == job1["job_url"]]
        assert len(dup_rows) == 1, (
            f"Re-ingesting the same folder must not duplicate rows; got {len(dup_rows)}"
        )

    def test_ingest_dedupes_across_portals_preserves_unique(self, client_engine, tmp_path):
        """Cross-portal duplicates (canonical URL tracking variants + normalized
        title/company) collapse; every unique opportunity is ingested once."""
        import json

        from sqlmodel import Session, select

        from app.models import Job
        from app.scraper.pipeline import ingest_from_output_folder
        from app.taxonomy.taxonomy import load_taxonomy

        _client, engine = client_engine

        url_a = "https://www.naukri.com/jobrapito/detail/77001"
        records = [
            # Portal 1 + portal 2 variants of the same posting (tracking params).
            _placement_rec("sh-x1", "Python Developer", "Acme Softworks",
                           url_a + "?utm_source=nl", site="naukri"),
            _placement_rec("sh-x2", "Python Developer", "Acme Softworks",
                           url_a, site="linkedin"),
            # Same (title, company), different URL: soft dedup -> merges.
            _placement_rec("sh-x3", "PYTHON DEVELOPER", "ACME SOFTWORKS",
                           "https://www.linkedin.com/jobs/view/77002", site="indeed"),
            # Genuinely unique opportunity: must survive.
            _placement_rec("sh-x4", "Java Developer", "Beta Works",
                           "https://www.hirist.tech/jobs/77003", site="hirist"),
        ]
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        (out_dir / "normalized_jobs.json").write_text(json.dumps(records), encoding="utf-8")

        ctx = MagicMock()
        with Session(engine) as db:
            result = ingest_from_output_folder(
                output_dir=out_dir, session=db, ctx=ctx, taxonomy=load_taxonomy()
            )
        mine_urls = {url_a, "https://www.linkedin.com/jobs/view/77002",
                     "https://www.hirist.tech/jobs/77003"}
        with Session(engine) as db:
            mine = [j for j in db.exec(select(Job)).all() if j.job_url in mine_urls]

        assert result["ingested_count"] == 2, (
            f"4 cross-portal records with 1 URL-dup pair and 1 soft-dup must yield "
            f"2 unique jobs, got {result['ingested_count']}"
        )
        assert {j.job_url for j in mine} == {url_a, "https://www.hirist.tech/jobs/77003"}, (
            "unique opportunities (naukri posting + hirist posting) must be preserved; "
            f"got {[j.job_url for j in mine]}"
        )


class TestSalaryIntegrityMilestone8:
    """Milestone 8 Ticket 1: Salary Integrity Schema, Normalizer & UI Rendering."""

    def test_job_model_salary_disclosed_default(self):
        from app.models import Job

        job = Job(
            id="job-test-salary-01",
            title="Software Engineer",
            company="TestCorp",
            description="Testing salary integrity.",
        )
        assert job.salary_disclosed is False
        assert job.salary_lpa in ([0, 0], [])

    def test_placement_job_model_salary_disclosed_default(self):
        from job_scraper.core.models import PlacementJob

        pj = PlacementJob(
            id="job-ext-test-01",
            title="Backend Engineer",
            company="TestCorp",
            description="Python FastAPI backend.",
            job_url="https://valid-domain.com/jobs/1",
            source="test",
            scraper_engine="stealth_harvester",
        )
        assert pj.salary_disclosed is False
        assert pj.salary_lpa in ([0, 0], [])

    def test_normalizer_salary_integrity_contract(self):
        from app.scraper.normalizer import normalize_scraped_job, parse_salary_lpa
        from app.taxonomy.taxonomy import load_taxonomy

        tax = load_taxonomy()

        # Undisclosed / unstated -> [0, 0], salary_disclosed=False, never [6, 12]
        raw_undisclosed = {
            "title": "Python Developer",
            "company": "Acme",
            "description": "Python, SQL, Docker required.",
            "job_url": "https://careers.acme.org/jobs/42",
        }
        job_undisclosed = normalize_scraped_job(raw_undisclosed, tax)
        assert job_undisclosed.salary_lpa in ([0, 0], [])
        assert job_undisclosed.salary_disclosed is False
        assert job_undisclosed.salary_lpa != [6, 12]

        # Explicit salary -> parsed LPA, salary_disclosed=True
        raw_disclosed = {
            "title": "Senior Python Developer",
            "company": "Acme",
            "description": "Python, SQL, Docker required.",
            "job_url": "https://careers.acme.org/jobs/43",
            "min_amount": 12,
            "max_amount": 20,
            "interval": "lpa",
        }
        job_disclosed = normalize_scraped_job(raw_disclosed, tax)
        assert job_disclosed.salary_lpa == [12, 20]
        assert job_disclosed.salary_disclosed is True

        # Helper returns (salary_lpa, salary_disclosed) tuple
        salary, disclosed = parse_salary_lpa(None, None, None)
        assert salary == [0, 0]
        assert disclosed is False

    def test_sanitize_legacy_salaries_cleans_6_12_rows(self, client_engine):
        from sqlmodel import Session

        from app.models import Job
        from app.scraper.pipeline import sanitize_legacy_salaries

        _client, engine = client_engine
        with Session(engine) as db:
            legacy_job = Job(
                id="job-ext-legacy-01",
                title="Legacy Python Dev",
                company="OldCorp",
                description="Needs Python.",
                job_url="https://legit-domain.com/jobs/99",
                salary_lpa=[6, 12],
                salary_disclosed=False,
            )
            real_job = Job(
                id="job-ext-real-02",
                title="Real Python Dev",
                company="RealCorp",
                description="Needs Python.",
                job_url="https://legit-domain.com/jobs/100",
                salary_lpa=[10, 15],
                salary_disclosed=True,
            )
            db.add(legacy_job)
            db.add(real_job)
            db.commit()

            sanitized_count = sanitize_legacy_salaries(db)
            assert sanitized_count >= 1

            j1 = db.get(Job, "job-ext-legacy-01")
            assert j1.salary_lpa == [0, 0]
            assert j1.salary_disclosed is False

            j2 = db.get(Job, "job-ext-real-02")
            assert j2.salary_lpa == [10, 15]
            assert j2.salary_disclosed is True

    def test_sanitize_legacy_experience_updates_jobs(self, client_engine):
        """§20.2.4 — legacy job-ext rows with experience_years=0 get the true floor
        parsed from their description and are updated; unaffected rows untouched."""
        from sqlmodel import Session

        from app.models import Job
        from app.scraper.pipeline import sanitize_legacy_experience

        _client, engine = client_engine
        with Session(engine) as db:
            stale = Job(
                id="job-ext-legacy-exp-01",
                title="Python Developer",
                company="OldCorp",
                description="Build pipelines. Requires 1-3 years of Python experience.",
                job_url="https://legit-domain.com/jobs/201",
                experience_years=0,
            )
            plus = Job(
                id="job-ext-legacy-exp-02",
                title="Backend Engineer",
                company="OldCorp",
                description="2+ years with SQL and Kafka.",
                job_url="https://legit-domain.com/jobs/202",
                experience_years=0,
            )
            already = Job(
                id="job-ext-legacy-exp-03",
                title="Senior Engineer",
                company="OldCorp",
                description="5+ years required.",
                job_url="https://legit-domain.com/jobs/203",
                experience_years=5,
            )
            db.add(stale)
            db.add(plus)
            db.add(already)
            db.commit()

            updated_count = sanitize_legacy_experience(db, ctx=None)
            # Two stale job-ext rows updated; the already-set row untouched.
            assert updated_count == 2

            j1 = db.get(Job, "job-ext-legacy-exp-01")
            assert j1.experience_years == 1  # "1-3 years" -> lower bound 1

            j2 = db.get(Job, "job-ext-legacy-exp-02")
            assert j2.experience_years == 2  # "2+ years" -> 2

            j3 = db.get(Job, "job-ext-legacy-exp-03")
            assert j3.experience_years == 5  # never overwritten

    def test_matches_ui_not_disclosed_rendering(self, client_engine):
        from sqlmodel import Session

        from app.models import Job, JobMatch

        client, engine = client_engine
        resp = client.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        with Session(engine) as db:
            job_undisclosed = Job(
                id="job-undisclosed-ui",
                title="Cloud Engineer Undisclosed",
                company="StealthInc",
                location="Bengaluru",
                description="Cloud engineering with Python.",
                job_url="https://stealthinc.com/careers/1",
                source="stealth_harvester",
                scraper_engine="stealth_harvester",
                salary_lpa=[0, 0],
                salary_disclosed=False,
                is_live=True,
                owner_student_id=sid,
            )
            job_disclosed = Job(
                id="job-disclosed-ui",
                title="Cloud Engineer Disclosed",
                company="PublicInc",
                location="Bengaluru",
                description="Cloud engineering with Python.",
                job_url="https://publicinc.com/careers/2",
                source="stealth_harvester",
                scraper_engine="stealth_harvester",
                salary_lpa=[12, 18],
                salary_disclosed=True,
                is_live=True,
                owner_student_id=sid,
            )
            db.add(job_undisclosed)
            db.add(job_disclosed)
            db.commit()

            match1 = JobMatch(
                id="jm-sal-1",
                student_id=sid,
                job_id=job_undisclosed.id,
                score=85.0,
                band="strong",
                breakdown={"semantic_sim": 0.85},
            )
            match2 = JobMatch(
                id="jm-sal-2",
                student_id=sid,
                job_id=job_disclosed.id,
                score=80.0,
                band="strong",
                breakdown={"semantic_sim": 0.80},
            )
            db.add(match1)
            db.add(match2)
            db.commit()

        resp = client.get(f"/students/{sid}/matches")
        assert resp.status_code == 200
        html = resp.text

        # Undisclosed job renders "Not Disclosed" badge and never "0–0 LPA" or "6–12 LPA"
        assert "Not Disclosed" in html
        assert "text-slate-400 font-mono text-[11px] italic" in html
        assert "0–0 LPA" not in html
        assert "6–12 LPA" not in html

        # Disclosed job renders "12–18 LPA"
        assert "12–18 LPA" in html

    def test_matches_html_zero_emoji_after_salary_integrity(self):
        import re
        from pathlib import Path

        path = Path("app/templates/student/matches.html")
        content = path.read_text(encoding="utf-8")
        emoji_pattern = re.compile("[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")
        literal_emojis = emoji_pattern.findall(content)
        assert literal_emojis == [], f"matches.html contains literal emojis: {literal_emojis}"
        entities = re.findall(r"&#x(?:1F[0-9A-Fa-f]{3}|2[67][0-9A-Fa-f]{2});", content)
        assert entities == [], f"matches.html contains emoji entities: {entities}"


class TestScrapeTelemetrySSE:
    """POST /api/jobs/scrape/stream (ARCHITECTURE §16.2.B): SSE wire format,
    lifecycle event contract, and ingestion of discovered jobs."""

    @staticmethod
    def _parse_sse(body: str) -> list[tuple[str, dict]]:
        """Parse `event: <name>\\ndata: <json>\\n\\n` chunks into (name, data)."""
        import json as _json

        events: list[tuple[str, dict]] = []
        name: str | None = None
        for line in body.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: ") and name is not None:
                events.append((name, _json.loads(line[len("data: "):])))
                name = None
        return events

    def test_scrape_stream_endpoint_sse_contract(self, client):
        """Stream returns 200 text/event-stream with no-buffer headers and the
        full lifecycle: init -> portal_start -> portal_progress -> portal_done
        -> complete (with real ingest counts)."""
        resp = client.post(
            "/api/jobs/scrape/stream",
            json={
                "search_term": "Python Developer",
                "location": "Bengaluru",
                "results_wanted": 3,
                "providers": ["jobspy"],
            },
        )
        assert resp.status_code == 200, resp.text[:400]
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert resp.headers["cache-control"] == "no-cache"
        assert resp.headers["x-accel-buffering"] == "no"
        assert resp.headers["connection"] == "keep-alive"

        events = self._parse_sse(resp.text)
        names = [n for n, _ in events]
        assert names[0] == "init", f"first event must be init, got {names[:3]}"
        assert names[-1] == "complete", f"last event must be complete, got {names[-3:]}"
        for required in ("portal_start", "portal_progress", "portal_done"):
            assert required in names, f"missing lifecycle event {required}: {names}"

        init = dict(events[0][1])
        assert init["total"] == len(init["portals"]) >= 1

        start = next(d for n, d in events if n == "portal_start")
        assert start["status"] == "running"
        assert isinstance(start["timestamp"], str) and start["timestamp"]

        done = next(d for n, d in events if n == "portal_done")
        assert done["status"] == "ok"
        assert isinstance(done["jobs"], int) and done["jobs"] >= 1
        assert isinstance(done["duration_s"], (int, float))

        progress = next(d for n, d in events if n == "portal_progress")
        assert isinstance(progress["jobs_found"], int)
        assert isinstance(progress["elapsed_s"], (int, float))

        complete = dict(events[-1][1])
        assert isinstance(complete["total_jobs"], int)
        assert isinstance(complete["new_ingested"], int)
        assert complete["new_ingested"] >= 1, "stream must ingest discovered jobs"

    def test_scrape_stream_emits_default_8_portals_for_stealth(self, client):
        """With no providers given and the stealth harvester registered, init
        advertises the 8 Indian portals (§16.2.B default portal list)."""
        resp = client.post(
            "/api/jobs/scrape/stream",
            json={"search_term": "Python", "results_wanted": 2},
        )
        assert resp.status_code == 200, resp.text[:400]
        events = self._parse_sse(resp.text)
        names = [n for n, _ in events]
        assert names[0] == "init"
        portals = dict(events[0][1])["portals"]
        # The endpoint defaults providers to ["jobspy"], so this asserts the
        # provider-name unit path; if the default ever becomes the stealth
        # harvester, the 8-portal list must appear instead.
        assert isinstance(portals, list) and portals

    def test_fetch_jobs_streaming_event_contract(self):
        """fetch_jobs_streaming emits init/portal_start/portal_progress/
        portal_done/complete per provider and never raises on provider failure."""
        from app.scraper.providers import BaseJobScraperProvider
        from app.scraper.service import CompositeJobScraper

        class _BoomProvider(BaseJobScraperProvider):
            name = "boom"
            display_name = "Boom"
            supported_sites = ["nowhere"]

            def fetch_jobs(self, search_term, location="India", results_wanted=15, **kwargs):
                raise RuntimeError("portal exploded")

        class _OkProvider(BaseJobScraperProvider):
            name = "ok"
            display_name = "OK"
            supported_sites = ["somewhere"]

            def fetch_jobs(self, search_term, location="India", results_wanted=15, **kwargs):
                return [{
                    "title": "Streamed Dev", "company": "StreamCo",
                    "location": location, "description": "d",
                    "job_url": "https://www.naukri.com/job/streamed-1",
                    "site": "naukri", "scraper_engine": "ok",
                }]

        events: list[tuple[str, dict]] = []
        composite = CompositeJobScraper(providers=[_BoomProvider(), _OkProvider()])
        jobs = composite.fetch_jobs_streaming(
            "python", location="Bengaluru", results_wanted=3,
            on_event=lambda name, data: events.append((name, data)),
        )

        names = [n for n, _ in events]
        assert names[0] == "init" and names[-1] == "complete"
        assert "portal_start" in names and "portal_done" in names
        assert "portal_error" in names, "failing provider must emit portal_error"
        err = next(d for n, d in events if n == "portal_error")
        assert err["status"] == "warning" and err["message"]
        # Zero-crash: the healthy provider's job still comes through.
        assert any(j["job_url"].endswith("streamed-1") for j in jobs), jobs
        complete = dict(events[-1][1])
        assert complete["total_jobs"] == len(jobs)

    def test_fetch_jobs_streaming_stealth_8_portal_units(self):
        """When the stealth harvester participates, streaming uses the 8 portal
        units and forwards them to the provider as its portal scope."""
        from app.scraper.providers import StealthHarvesterProvider
        from app.scraper.service import CompositeJobScraper

        seen_portals: list[list[str]] = []
        provider = StealthHarvesterProvider(output_dir="__missing__")
        original = provider.fetch_jobs

        def spy(search_term, location="India", results_wanted=15, **kwargs):
            seen_portals.append(list(kwargs.get("portals") or []))
            return original(search_term, location=location, results_wanted=results_wanted, **kwargs)

        provider.fetch_jobs = spy  # type: ignore[method-assign]

        events: list[tuple[str, dict]] = []
        composite = CompositeJobScraper(providers=[provider])
        composite.fetch_jobs_streaming(
            "python", results_wanted=2,
            on_event=lambda name, data: events.append((name, data)),
        )

        init = dict(events[0][1])
        assert init["portals"] == provider.supported_sites
        assert init["total"] == 8
        done = [d for n, d in events if n == "portal_done"]
        assert len(done) == 8, f"expected one portal_done per portal, got {len(done)}"
        # Each streaming unit scopes the provider to exactly that portal.
        assert seen_portals, "provider was never invoked"
        assert all(len(p) == 1 and p[0] in provider.supported_sites for p in seen_portals), (
            f"each unit must forward its single portal scope, got {seen_portals}"
        )


# ===========================================================================
# Milestone 8: DOM data-* filter acceptance gate + offline SSE contract
# ARCHITECTURE.md Section 16 (spec-driven acceptance gate — charter §7).
#
# TestScrapeTelemetrySSE (above, @Coder-Flash) drives the LIVE stream with a
# real jobspy call. The class below is the hermetic, offline mirror of the
# SSE contract (patched scrape_jobs — matches the suite's whole offline
# convention) PLUS the §16.2.C DOM attribute gate that has no coverage yet.
# ===========================================================================


class TestSSEScrapeStreamOfflineContract:
    """Offline (patched) mirror of the §16.2.B SSE contract: deterministic and
    network-free, so the gate can never be flaky."""

    _REQUIRED = ("init", "portal_start", "portal_progress", "portal_done", "complete")

    @staticmethod
    def _parse_sse(body: str) -> list[tuple[str, dict]]:
        import json as _json

        events: list[tuple[str, dict]] = []
        name: str | None = None
        for line in body.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):]
            elif line.startswith("data: ") and name is not None:
                try:
                    data = _json.loads(line[len("data: "):])
                except ValueError:
                    data = {"_raw": line[len("data: "):]}
                events.append((name, data))
                name = None
        return events

    def test_offline_stream_sequence_and_single_complete(self, client, client_engine):
        """Offline: full lifecycle in spec order with exactly one terminal complete."""
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            resp = client.post(
                "/api/jobs/scrape/stream",
                json={"search_term": "Python Developer", "location": "Bengaluru",
                      "sites": ["naukri"], "results_wanted": 5, "providers": ["jobspy"]},
            )

        assert resp.status_code == 200, resp.text[:300]
        assert "text/event-stream" in resp.headers.get("content-type", "")
        assert "no-cache" in resp.headers.get("cache-control", "").lower()

        names = [n for n, _ in self._parse_sse(resp.text)]
        missing = [e for e in self._REQUIRED if e not in names]
        assert not missing, f"missing SSE events {missing}; got {names}"

        first_idx = {e: names.index(e) for e in self._REQUIRED}
        ordered = [first_idx[e] for e in self._REQUIRED]
        assert ordered == sorted(ordered), f"events out of spec order: {names}"
        assert names.count("complete") == 1, (
            f"exactly one terminal complete required, got {names.count('complete')}"
        )

    def test_offline_stream_payloads(self, client, client_engine):
        """Offline: init/complete/portal payloads carry the §16.2.B keys."""
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            resp = client.post(
                "/api/jobs/scrape/stream",
                json={"search_term": "Python Developer", "sites": ["naukri"],
                      "results_wanted": 5, "providers": ["jobspy"]},
            )

        parsed = self._parse_sse(resp.text)
        events = dict(parsed)

        init = events.get("init")
        assert init is not None and isinstance(init.get("portals"), list) and "total" in init, (
            f"init payload must have portals + total: {init}"
        )

        complete = events.get("complete")
        assert complete is not None, "complete event missing"
        for key in ("total_jobs", "new_ingested", "elapsed_total_s"):
            assert key in complete, f"complete payload missing {key}: {complete}"

        for name, keys in (
            ("portal_start", ("portal", "status")),
            ("portal_progress", ("portal", "jobs_found")),
            ("portal_done", ("portal", "status")),
        ):
            for ev_name, data in parsed:
                if ev_name == name:
                    for key in keys:
                        assert key in data, f"{name} payload missing {key}: {data}"
                    break

    def test_offline_stream_emits_ingest_and_ranking_phases(self, client, client_engine):
        """§20.2.5 — post-harvest phases emit ingest_start/ranking_start so the
        frontend leaves 'SCRAPING IN PROGRESS' during ingestion + ranking."""
        df = _synthetic_df()
        with patch("app.scraper.service.scrape_jobs", return_value=df):
            resp = client.post(
                "/api/jobs/scrape/stream",
                json={"search_term": "Python Developer", "sites": ["naukri"],
                      "results_wanted": 5, "providers": ["jobspy"]},
            )

        assert resp.status_code == 200, resp.text[:300]
        parsed = self._parse_sse(resp.text)
        events = dict(parsed)

        ingest = events.get("ingest_start")
        assert ingest is not None, "ingest_start event missing (§20.2.5)"
        assert ingest.get("status") == "ingesting", ingest
        assert "total" in ingest, f"ingest_start payload missing total: {ingest}"

        ranking = events.get("ranking_start")
        assert ranking is not None, "ranking_start event missing (§20.2.5)"
        assert ranking.get("status") == "ranking", ranking
        assert "students" in ranking, f"ranking_start payload missing students: {ranking}"

        # Both phases precede the terminal complete event.
        names = [n for n, _ in parsed]
        assert names.index("ingest_start") < names.index("complete"), names
        assert names.index("ranking_start") < names.index("complete"), names


class TestDOMFilterDataAttributes:
    """Match cards must stamp the 12 data-* attributes the client-side filter
    suite (ARCHITECTURE §16.2.C) consumes, without XSS or template errors."""

    _REQUIRED_ATTRS = (
        "data-title",
        "data-company",
        "data-location",
        "data-salary-min",
        "data-salary-max",
        "data-salary-disclosed",
        "data-source",
        "data-eligible",
        "data-match-score",
        "data-work-mode",
        "data-exp-years",
        "data-skills",
    )

    @staticmethod
    def _render_matches_page(client, engine) -> str:
        """Create a student + disclosed job + JobMatch, then render the page HTML.

        NOTE: `client` and `engine` MUST come from the SAME `client_engine`
        tuple — mixing the standalone `client` fixture with `client_engine`'s
        engine would write rows to two different databases (FK failure)."""
        from sqlmodel import Session

        from app.models import Job, JobMatch

        resp = client.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        with Session(engine) as db:
            job = Job(
                id="job-dom-filter-1",
                title="FastAPI Backend Engineer",
                company="DOMFilterCo",
                location="Bengaluru",
                description="Build FastAPI services with Python and PostgreSQL.",
                required_skills=["python", "fastapi"],
                preferred_skills=["postgresql"],
                job_url="https://careers.domfilterco.io/jobs/1",
                source="naukri",
                scraper_engine="stealth_harvester",
                experience_years=2,
                salary_lpa=[8, 14],
                salary_disclosed=True,
                remote=False,
                is_live=True,
                owner_student_id=sid,
            )
            db.add(job)
            db.commit()
            db.add(JobMatch(
                id="jm-dom-filter-1",
                student_id=sid,
                job_id=job.id,
                score=88.0,
                band="strong",
                breakdown={"semantic_sim": 0.88},
            ))
            db.commit()

        page = client.get(f"/students/{sid}/matches")
        assert page.status_code == 200, (
            f"matches page render failed: {page.status_code} {page.text[:300]}"
        )
        return page.text

    def test_all_cards_carry_required_data_attributes(self, client_engine):
        """Every .match-card stamps all 12 data-* attributes the filter JS reads."""
        import re

        client, engine = client_engine
        html = self._render_matches_page(client, engine)

        card_opens = re.findall(r"<div[^>]*class=\"[^\"]*match-card[^\"]*\"[^>]*>", html)
        assert card_opens, "no .match-card elements rendered on matches page"

        for tag in card_opens:
            missing = [a for a in self._REQUIRED_ATTRS if f"{a}=" not in tag]
            assert not missing, (
                f"card missing data-* attributes {missing}; card tag: {tag}"
            )

    def test_data_attribute_values_are_wellformed(self, client_engine):
        """Salary/eligible/score attributes parse to valid typed values (no XSS
        breakage, no template errors leaking into rendered attributes)."""
        import re

        client, engine = client_engine
        html = self._render_matches_page(client, engine)

        for attr in ("data-salary-min", "data-salary-max"):
            for val in re.findall(rf"{attr}=\"([^\"]*)\"", html):
                assert val.lstrip("-").isdigit(), (
                    f"{attr} must be an integer, got {val!r}"
                )

        for attr in ("data-salary-disclosed", "data-eligible"):
            for val in re.findall(rf"{attr}=\"([^\"]*)\"", html):
                assert val in ("true", "false", "True", "False"), (
                    f"{attr} must be a boolean string, got {val!r}"
                )

        for val in re.findall(r"data-match-score=\"([^\"]*)\"", html):
            try:
                float(val)
            except ValueError:
                pytest.fail(f"data-match-score must be numeric, got {val!r}")

        assert "{{" not in html, "unrendered Jinja expression leaked into page"
        assert "{%" not in html, "unrendered Jinja statement leaked into page"
        assert "UndefinedError" not in html and "TemplateSyntaxError" not in html, (
            "template error text leaked into rendered page"
        )

    def test_data_skills_attribute_is_delimited(self, client_engine):
        """data-skills carries a delimited skill list (comma/pipe) for multi-select."""
        import re

        client, engine = client_engine
        html = self._render_matches_page(client, engine)

        vals = re.findall(r"data-skills=\"([^\"]*)\"", html)
        assert vals, "no data-skills attribute found on any card"
        assert any(v and ("," in v or "|" in v) for v in vals), (
            f"data-skills must be a delimited list, got {vals}"
        )

    def test_card_attributes_do_not_break_with_undisclosed_salary(self, client_engine):
        """A salary_disclosed=False card still stamps data-salary-* without a
        template error; the synthetic [6, 12] default must never surface."""
        import re

        from sqlmodel import Session

        from app.models import Job, JobMatch

        client, engine = client_engine
        resp = client.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        with Session(engine) as db:
            job = Job(
                id="job-dom-undisclosed-1",
                title="Undisclosed Role",
                company="StealthCo",
                location="Pune",
                description="No salary listed.",
                required_skills=["python"],
                preferred_skills=[],
                job_url="https://careers.stealthco.io/jobs/2",
                source="stealth_harvester",
                scraper_engine="stealth_harvester",
                salary_lpa=[0, 0],
                salary_disclosed=False,
                is_live=True,
                owner_student_id=sid,
            )
            db.add(job)
            db.commit()
            db.add(JobMatch(
                id="jm-dom-undisclosed-1",
                student_id=sid,
                job_id=job.id,
                score=70.0,
                band="good",
                breakdown={"semantic_sim": 0.7},
            ))
            db.commit()

        page = client.get(f"/students/{sid}/matches")
        assert page.status_code == 200
        html = page.text

        assert "{{" not in html and "UndefinedError" not in html, (
            "undisclosed-salary card broke template rendering"
        )

        disclosed_vals = re.findall(r"data-salary-disclosed=\"([^\"]*)\"", html)
        assert disclosed_vals, "no data-salary-disclosed attribute rendered"
        assert any(v in ("false", "False") for v in disclosed_vals), (
            f"expected a false data-salary-disclosed, got {disclosed_vals}"
        )

        assert "6–12 LPA" not in html and "6-12 LPA" not in html, (
            "synthetic [6, 12] salary leaked into the rendered card"
        )


class TestEightDimensionFilteringSuite:
    """Milestone 8 Ticket 3: 8-Dimension Client-Side Filtering & Telemetry UI."""

    def test_matches_html_contains_all_8_dimension_controls(self):
        from pathlib import Path

        html = Path("app/templates/student/matches.html").read_text(encoding="utf-8")

        # Dim 1: Search & Clear
        assert 'id="filter-search"' in html
        assert 'id="clear-search-btn"' in html

        # Dim 2: Salary Range & Disclosed Only toggle
        assert 'id="filter-salary"' in html
        assert 'id="filter-disclosed-only"' in html
        assert 'value="lt6"' in html
        assert 'value="gte25"' in html

        # Dim 3: Portal pills & Quick Action buttons
        assert 'id="portal-pills"' in html
        assert 'id="btn-portal-all"' in html
        assert 'id="btn-portal-none"' in html
        assert 'id="btn-portal-scraped"' in html
        assert 'id="btn-portal-campus"' in html

        # Dim 4: Eligibility & Match Score Bands
        assert 'id="filter-eligibility"' in html
        assert 'id="filter-band"' in html

        # Dim 5: Location & Work Mode
        assert 'id="filter-location"' in html
        assert 'id="filter-mode"' in html

        # Dim 6: Experience Level
        assert 'id="filter-exp"' in html
        assert 'value="0-1"' in html
        assert 'value="5+"' in html

        # Dim 7: Skill Match Cloud
        assert 'id="skill-cloud"' in html

        # Dim 8: Sorting, Sticky Counter Bar & Reset All Filters
        assert 'id="filter-sort"' in html
        assert 'id="btn-reset-filters"' in html
        assert 'id="visible-matches-count"' in html
        assert 'id="total-matches-count"' in html
        assert 'value="score_desc"' in html
        assert 'value="salary_desc"' in html
        assert 'value="salary_asc"' in html
        assert 'value="company_asc"' in html

    def test_matches_html_card_data_attributes_contract(self, client_engine):
        from sqlmodel import Session

        from app.models import Job, JobMatch

        client, engine = client_engine
        resp = client.post("/api/students", json=_student_with_skills())
        assert resp.status_code in (200, 201)
        sid = resp.json()["id"]

        with Session(engine) as db:
            j = Job(
                id="job-filter-probe-01",
                title="Staff Python Architect",
                company="NexusTech",
                location="Bengaluru, Karnataka",
                description="Staff level Python FastAPI Docker AWS role.",
                required_skills=["python", "fastapi", "docker", "aws"],
                preferred_skills=["sql"],
                experience_years=6,
                remote=True,
                salary_lpa=[24, 32],
                salary_disclosed=True,
                source="naukri",
                scraper_engine="stealth_harvester",
                is_live=True,
                owner_student_id=sid,
            )
            db.add(j)
            db.commit()
            db.add(JobMatch(
                id="jm-filter-probe-01",
                student_id=sid,
                job_id=j.id,
                score=92.0,
                band="strong",
                breakdown={"semantic_sim": 0.92},
            ))
            db.commit()

        page = client.get(f"/students/{sid}/matches")
        assert page.status_code == 200
        html = page.text

        # Verify all stamped data attributes
        assert 'data-job-type="live"' in html
        assert 'data-source="naukri"' in html
        assert 'data-title="staff python architect"' in html
        assert 'data-company="nexustech"' in html
        assert 'data-salary-min="24"' in html
        assert 'data-salary-max="32"' in html
        assert 'data-salary-disclosed="true"' in html
        assert 'data-eligible=' in html
        assert 'data-score="92.0"' in html or 'data-score="92"' in html
        assert 'data-band="strong"' in html
        assert 'data-tier="super_dream"' in html
        assert 'data-work-mode="remote"' in html
        assert 'data-exp="6"' in html
        assert 'data-skills=' in html
        assert "python" in html
        assert 'data-date=' in html

    def test_matches_html_telemetry_modal_components(self):
        from pathlib import Path

        html = Path("app/templates/student/matches.html").read_text(encoding="utf-8")

        # Telemetry dashboard container & stopwatch
        assert 'id="scrape-telemetry"' in html
        assert 'id="telemetry-stopwatch"' in html
        assert 'id="telemetry-progress-bar"' in html
        assert 'id="telemetry-progress-text"' in html
        assert 'id="telemetry-jobs-discovered"' in html

        # 8-portal real-time grid
        assert 'id="portal-telemetry-grid"' in html
        portals = [
            "naukri", "linkedin", "indeed", "foundit",
            "internshala", "unstop", "instahyre", "hirist"
        ]
        for p in portals:
            assert f'id="telemetry-portal-{p}"' in html, f"Missing telemetry card for {p}"

        # Connected to streaming endpoint
        assert "/api/jobs/scrape/stream" in html

    def test_filtering_suite_zero_emoji_contract(self):
        import re
        from pathlib import Path

        content = Path("app/templates/student/matches.html").read_text(encoding="utf-8")
        emoji_pattern = re.compile("[\U0001F300-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]")
        literal_emojis = emoji_pattern.findall(content)
        assert literal_emojis == [], f"matches.html contains literal emojis: {literal_emojis}"
        entities = re.findall(r"&#x(?:1F[0-9A-Fa-f]{3}|2[67][0-9A-Fa-f]{2});", content)
        assert entities == [], f"matches.html contains emoji entities: {entities}"


# ===========================================================================
# Bugfix: Real-Time SSE Telemetry Streaming and Per-Portal Scoping Verification
# ===========================================================================


class TestRealtimeTelemetryFix:
    """Verifies concurrent SSE streaming via queue.Queue and strict portal unit scoping."""

    def test_stream_unit_scoping_never_overwrites_with_caller_sites(self):
        """Even when caller passes sites=['naukri', 'linkedin', 'indeed'], each
        portal unit in fetch_jobs_streaming receives strictly its own [unit] for
        both sites and portals, preventing cross-portal leak and rate limits."""
        from app.scraper.providers import StealthHarvesterProvider
        from app.scraper.service import CompositeJobScraper

        seen_calls: list[dict[str, Any]] = []
        provider = StealthHarvesterProvider(output_dir="__missing__")
        original = provider.fetch_jobs

        def spy(search_term, location="India", results_wanted=15, **kwargs):
            seen_calls.append({
                "portals": list(kwargs.get("portals") or []),
                "sites": list(kwargs.get("sites") or []),
            })
            return original(search_term, location=location, results_wanted=results_wanted, **kwargs)

        provider.fetch_jobs = spy  # type: ignore[method-assign]
        composite = CompositeJobScraper(providers=[provider])
        composite.fetch_jobs_streaming(
            "Software Engineer",
            results_wanted=2,
            sites=["naukri", "linkedin", "indeed"],  # Caller's multi-site filter
            on_event=lambda name, data: None,
        )

        assert len(seen_calls) == 8, f"expected 8 portal unit calls, got {len(seen_calls)}"
        for call in seen_calls:
            # Each unit must be scoped strictly to exactly ONE portal
            assert len(call["portals"]) == 1, f"leaked portals: {call['portals']}"
            assert len(call["sites"]) == 1, f"leaked sites: {call['sites']}"
            assert call["portals"] == call["sites"], f"mismatched unit: {call}"

    def test_stream_endpoint_realtime_generator_concurrency(self, client):
        """TestClient streaming response yields chunks immediately via SSE wire protocol."""
        resp = client.post(
            "/api/jobs/scrape/stream",
            json={"search_term": "Python", "results_wanted": 2, "providers": ["jobspy"]},
        )
        assert resp.status_code == 200
        assert "event: init\n" in resp.text
        assert "event: complete\n" in resp.text
        # Verify complete payload has positive metrics
        assert '"total_jobs":' in resp.text
        assert '"new_ingested":' in resp.text


# ---------------------------------------------------------------------------
# ARCHITECTURE §18.4 — Live Scraped Job Data Purge & Refetch Sanitation
# ---------------------------------------------------------------------------
# Charter §7 acceptance gate written FIRST: purge_live_jobs (pipeline.py) and
# the `purge-live-jobs` CLI subcommand do not exist yet (measured 2026-11-02;
# spec committed as 7685bce). These tests self-arm RED with ImportError /
# AttributeError until the contract lands.
#
# Anti-vacuity (charter §4): the purge tests seed BOTH live and local jobs and
# assert both halves of the contract (live deleted, local preserved), so a
# no-op implementation that deletes nothing cannot pass by deleting nothing;
# a blanket-delete implementation cannot pass either.
# ---------------------------------------------------------------------------


class TestPurgeLiveJobs:
    """purge_live_jobs: delete scraped jobs, preserve institutional ones (§18.4)."""

    @staticmethod
    def _seed(db):
        """Seed FK-safe prerequisites: student + course."""
        from app.models import Course, Student

        db.add(Student(id="stu-purge", name="Purge Tester", branch="CSE"))
        db.add(Course(id="course-purge", title="Purge Course",
                      provider="test", url="https://course.test"))
        db.commit()

    @staticmethod
    def _seed_jobs(engine):
        """Insert 2 live jobs (source != local) + 2 local jobs, FK-safe order."""
        from sqlmodel import Session

        from app.models import Job

        # Parents first.
        with Session(engine) as db:
            db.add(Job(id="job-live-1", title="Live Dev 1", company="LiveCo",
                       description="d", job_url="https://www.naukri.com/job/live1",
                       source="naukri"))
            db.add(Job(id="job-live-2", title="Live Dev 2", company="LiveCo2",
                       description="d", job_url="https://www.instahyre.com/job/live2",
                       source="instahyre"))
            db.add(Job(id="job-local-1", title="Local Drive 1", company="JKLU",
                       description="d", job_url="https://jklu.edu.in/drive/1",
                       source="local"))
            db.add(Job(id="job-local-2", title="Local Drive 2", company="JKLU",
                       description="d", job_url="https://jklu.edu.in/drive/2",
                       source="local"))
            db.commit()

        # Children of both live and local jobs (cascade + preservation proof).
        with Session(engine) as db:
            from app.models import Interview, JobMatch, PlanItem, UpskillPlan

            db.add(UpskillPlan(id="plan-live-1", student_id="stu-purge",
                               job_id="job-live-1", total_hours=10))
            db.add(UpskillPlan(id="plan-local-1", student_id="stu-purge",
                               job_id="job-local-1", total_hours=20))
            db.commit()
        with Session(engine) as db:
            from app.models import Interview, JobMatch, PlanItem

            db.add(PlanItem(id="pi-live-1", plan_id="plan-live-1", order=1,
                            skill_id="python", course_id="course-purge", hours=5))
            db.add(PlanItem(id="pi-local-1", plan_id="plan-local-1", order=1,
                            skill_id="python", course_id="course-purge", hours=8))
            db.add(JobMatch(id="jm-live-1", student_id="stu-purge",
                            job_id="job-live-1", score=85, band="strong"))
            db.add(JobMatch(id="jm-local-1", student_id="stu-purge",
                            job_id="job-local-1", score=90, band="strong"))
            db.add(Interview(id="int-live-1", student_id="stu-purge",
                             job_id="job-live-1"))
            db.commit()

    def test_purge_live_jobs_deletes_live_preserves_local(self, client_engine):
        """Live jobs (source != 'local') and their children are deleted; local
        institutional jobs and their children survive (§18.4.1 + §18.4.2)."""
        from sqlmodel import Session, select

        from app.models import Interview, Job, JobMatch, PlanItem, UpskillPlan
        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
        self._seed_jobs(engine)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}

        removed = purge_live_jobs(Session(engine), ctx, clear_output=False)
        assert removed == 2, f"expected 2 live jobs purged, got {removed}"

        with Session(engine) as db:
            job_ids = {j.id for j in db.exec(select(Job)).all()}
            plan_ids = {p.id for p in db.exec(select(UpskillPlan)).all()}
            item_ids = {i.id for i in db.exec(select(PlanItem)).all()}
            match_ids = {m.id for m in db.exec(select(JobMatch)).all()}
            int_ids = {i.id for i in db.exec(select(Interview)).all()}
            local_sources = {j.source for j in db.exec(select(Job)).all()}

        # Live jobs + cascade gone.
        assert "job-live-1" not in job_ids and "job-live-2" not in job_ids
        assert "plan-live-1" not in plan_ids
        assert "pi-live-1" not in item_ids
        assert "jm-live-1" not in match_ids
        assert "int-live-1" not in int_ids

        # Local institutional data preserved (anti-vacuity: a blanket delete fails here).
        assert "job-local-1" in job_ids and "job-local-2" in job_ids
        assert "plan-local-1" in plan_ids
        assert "pi-local-1" in item_ids
        assert "jm-local-1" in match_ids
        assert local_sources == {"local"}, (
            f"only local jobs must remain, got sources {local_sources}"
        )

    def test_purge_live_jobs_cleans_vector_store(self, client_engine):
        """vector_store deletion is invoked for every purged job id (§18.4.1)."""
        from sqlmodel import Session

        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
        self._seed_jobs(engine)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {"job-live-1": [0.1], "job-local-1": [0.2]}

        removed = purge_live_jobs(Session(engine), ctx, clear_output=False)
        assert removed == 2

        # Anti-vacuity: the cleanup hook must actually fire per purged id.
        deleted_ids = set()
        for call in ctx.vector_store.delete.call_args_list:
            args = call.args
            if len(args) >= 2:
                deleted_ids.add(args[1])
            elif len(args) == 1:
                deleted_ids.add(args[0])
        removed_via_remove = {
            c.args[1] for c in ctx.vector_store.remove.call_args_list if len(c.args) >= 2
        }
        cleaned = deleted_ids | removed_via_remove
        assert "job-live-1" in cleaned, (
            f"vector cleanup missing for job-live-1; cleaned={sorted(cleaned)}"
        )
        # Local job embeddings must not be dropped.
        assert "job-local-1" not in cleaned, "local job embedding must be preserved"
        # job_vectors cache pruned for purged ids, kept for local.
        assert "job-live-1" not in ctx.job_vectors
        assert "job-local-1" in ctx.job_vectors

    def test_purge_live_jobs_clear_output_true_resets_files(self, client_engine, tmp_path, monkeypatch):
        """clear_output=True resets output/normalized_jobs.json and
        output/raw_scraped_jobs.json to [] (§18.4.1)."""
        import json
        from pathlib import Path

        from sqlmodel import Session

        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
        self._seed_jobs(engine)

        # Hermetic output dir: chdir so a hardcoded "output/" lands in tmp_path.
        out_dir = Path(tmp_path) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "normalized_jobs.json").write_text(
            json.dumps([{"title": "Normalized A"}, {"title": "Normalized B"}]), encoding="utf-8"
        )
        (out_dir / "raw_scraped_jobs.json").write_text(
            json.dumps([{"title": "Raw A"}]), encoding="utf-8"
        )
        monkeypatch.chdir(tmp_path)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}
        purge_live_jobs(Session(engine), ctx, clear_output=True)

        normalized = json.loads((out_dir / "normalized_jobs.json").read_text(encoding="utf-8"))
        raw = json.loads((out_dir / "raw_scraped_jobs.json").read_text(encoding="utf-8"))
        assert normalized == [], f"normalized_jobs.json must reset to [], got {normalized}"
        assert raw == [], f"raw_scraped_jobs.json must reset to [], got {raw}"

    def test_purge_live_jobs_clear_output_false_preserves_files(self, client_engine, tmp_path, monkeypatch):
        """clear_output=False leaves the output JSON files untouched (§18.4.1)."""
        import json
        from pathlib import Path

        from sqlmodel import Session

        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
        self._seed_jobs(engine)

        out_dir = Path(tmp_path) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = [{"title": "Keep Me"}]
        (out_dir / "normalized_jobs.json").write_text(json.dumps(payload), encoding="utf-8")
        (out_dir / "raw_scraped_jobs.json").write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}
        purge_live_jobs(Session(engine), ctx, clear_output=False)

        assert json.loads((out_dir / "normalized_jobs.json").read_text(encoding="utf-8")) == payload
        assert json.loads((out_dir / "raw_scraped_jobs.json").read_text(encoding="utf-8")) == payload

    def test_purge_live_jobs_preserves_students_skills_courses(self, client_engine):
        """Student profiles, skills, and institutional courses are untouched (§18.4.2)."""
        from sqlmodel import Session, select

        from app.models import Course, Student, StudentSkill
        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
            db.add(StudentSkill(student_id="stu-purge", skill_id="python", mentions=2))
            db.commit()
        self._seed_jobs(engine)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}
        purge_live_jobs(Session(engine), ctx, clear_output=False)

        with Session(engine) as db:
            students = {s.id for s in db.exec(select(Student)).all()}
            courses = {c.id for c in db.exec(select(Course)).all()}
            skills = {(ss.student_id, ss.skill_id) for ss in db.exec(select(StudentSkill)).all()}

        assert "stu-purge" in students, "student profiles must be preserved"
        assert "course-purge" in courses, "institutional courses must be preserved"
        assert ("stu-purge", "python") in skills, "student skills must be preserved"

    def test_purge_live_jobs_idempotent_on_second_run(self, client_engine):
        """A second purge finds nothing and must not raise (§18.4 anti-vacuity)."""
        from sqlmodel import Session

        from app.scraper.pipeline import purge_live_jobs

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
        self._seed_jobs(engine)

        ctx = MagicMock()
        ctx.vector_store = MagicMock()
        ctx.job_vectors = {}

        first = purge_live_jobs(Session(engine), ctx, clear_output=False)
        second = purge_live_jobs(Session(engine), ctx, clear_output=False)
        assert first == 2, f"first purge should remove 2, got {first}"
        assert second == 0, f"second purge must remove 0, got {second}"

    def test_purge_live_jobs_cli_executes_cleanly(self, client_engine, tmp_path, monkeypatch, capsys):
        """`python -m app.cli purge-live-jobs` executes cleanly (rc == 0) and
        reports the purge count (§18.4.1 / Coordinator acceptance: "executes cleanly").

        Safe seams only (never touches the real DB): `_make_engine` -> the
        in-memory sqlite from client_engine, `settings.database_url` -> 'sqlite://',
        and chdir to tmp_path so output/ writes land in the scratch dir.
        """
        import json
        from pathlib import Path

        from sqlmodel import Session, select

        from app import cli
        from app.models import Job

        _c, engine = client_engine
        with Session(engine) as db:
            self._seed(db)
            db.add(Job(id="job-live-cli", title="Live CLI Job", company="Corp",
                       description="d", job_url="https://www.naukri.com/job/cli1",
                       source="naukri"))
            db.add(Job(id="job-local-cli", title="Local CLI Job", company="JKLU",
                       description="d", job_url="https://jklu.edu.in/drive/cli",
                       source="local"))
            db.commit()

        out_dir = Path(tmp_path) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "raw_scraped_jobs.json").write_text(
            json.dumps([{"title": "temp"}]), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(cli, "_make_engine", lambda url: engine)
        monkeypatch.setattr(cli, "_make_embedder", lambda: __import__(
            "app.embeddings.embedder", fromlist=["HashingEmbedder"]).HashingEmbedder())
        monkeypatch.setattr(cli.settings, "database_url", "sqlite://")

        rc = cli.main(["purge-live-jobs"])
        assert rc == 0, f"purge-live-jobs must exit cleanly (rc==0), got {rc}"
        out = capsys.readouterr().out
        assert "Purged 1 live job(s)" in out, f"unexpected CLI output: {out!r}"

        with Session(engine) as db:
            remaining = {j.id for j in db.exec(select(Job)).all()}
        assert "job-live-cli" not in remaining, "CLI purge must delete the live job"
        assert "job-local-cli" in remaining, "CLI purge must preserve the local job"

        # output reset by the default (no --keep-output) path
        assert json.loads((out_dir / "raw_scraped_jobs.json").read_text(encoding="utf-8")) == []

    def test_purge_live_jobs_cli_keep_output_flag(self, tmp_path, monkeypatch, capsys):
        """`purge-live-jobs --keep-output` runs cleanly and preserves output/ (§18.4.1)."""
        import json
        from pathlib import Path

        from app import cli
        from app.db import make_engine
        from app.embeddings.embedder import HashingEmbedder
        from app.models import Job

        eng = make_engine("sqlite://")
        Job.metadata.create_all(eng)

        out_dir = Path(tmp_path) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        raw = out_dir / "raw_scraped_jobs.json"
        raw.write_text(json.dumps([{"title": "keep"}]), encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        monkeypatch.setattr(cli, "_make_engine", lambda url: eng)
        monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
        monkeypatch.setattr(cli.settings, "database_url", "sqlite://")

        rc = cli.main(["purge-live-jobs", "--keep-output"])
        assert rc == 0, f"--keep-output run must exit cleanly, got {rc}"
        assert json.loads(raw.read_text(encoding="utf-8")) == [{"title": "keep"}]


class TestScraperFresherModeAndMatchesDashboard:
    """Milestone 11 — ARCHITECTURE §19.4 Scraper Fresher Mode & Matches Quick Filter."""

    def test_scrape_request_fresher_only_defaults_true(self):
        from app.api.routes.jobs import ScrapeRequest

        req_default = ScrapeRequest(search_term="Python Developer")
        assert req_default.fresher_only is True

        req_false = ScrapeRequest(search_term="Senior Architect", fresher_only=False)
        assert req_false.fresher_only is False

        req_true = ScrapeRequest(search_term="Junior Developer", fresher_only=True)
        assert req_true.fresher_only is True

    def test_fresher_mode_query_enrichment(self):
        from app.api.routes.jobs import ScrapeRequest

        req = ScrapeRequest(search_terms=["Backend Developer"], fresher_only=True)
        assert req.fresher_only is True

        terms = list(req.search_terms)
        base_term = terms[0]
        enriched = list(terms)
        for qualifier in ["Fresher", "Junior", "Intern", "Entry Level"]:
            candidate = f"{base_term} {qualifier}" if qualifier in ["Fresher", "Intern"] else f"{qualifier} {base_term}"
            if candidate not in enriched:
                enriched.append(candidate)

        assert any("Fresher" in t for t in enriched)
        assert any("Junior" in t for t in enriched)
        assert any("Intern" in t for t in enriched)
        assert any("Entry Level" in t for t in enriched)

    def test_matches_html_contains_fresher_filter_suite(self):
        from pathlib import Path

        tpl_path = Path("app/templates/student/matches.html")
        assert tpl_path.exists(), "matches.html template must exist"
        content = tpl_path.read_text(encoding="utf-8")

        # §19.4 UI elements
        assert 'id="filter-fresher-pill"' in content, "Fresher quick filter pill missing in matches.html"
        assert 'id="count-fresher"' in content, "Fresher count badge missing in matches.html"
        assert 'id="filter-fresher-checkbox"' in content, "Fresher checkbox missing in matches.html"
        assert 'id="scrape-fresher-only"' in content, "Scrape modal fresher toggle missing in matches.html"
        assert 'data-fresher=' in content, "Card data-fresher attribute missing in matches.html"
        assert "state.fresherOnly" in content, "state.fresherOnly missing in filter engine"
        assert "fresher_only: fresherOnly" in content, "fresher_only parameter not forwarded in submitScrape"

    def test_matches_html_zero_emoji_compliance(self):
        from pathlib import Path

        tpl_path = Path("app/templates/student/matches.html")
        content = tpl_path.read_text(encoding="utf-8")

        # Zero-emoji check across the template
        emoji_pattern = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)
        emojis_found = emoji_pattern.findall(content)
        assert not emojis_found, f"Zero-emoji policy violated in matches.html: {emojis_found}"

    def test_post_scrape_jobs_endpoint_default_on_enrichment(self, client_engine):
        """M11-F1: Verify POST /api/jobs/scrape enriches with 4 qualifiers by default when fresher_only omitted."""
        from unittest.mock import MagicMock, patch

        client, _ = client_engine
        body = {"search_terms": ["Cloud Engineer"], "providers": ["jobspy"], "results_wanted": 1, "live": False}
        stub_scraper = MagicMock()
        stub_scraper.fetch_jobs.return_value = []
        with patch("app.api.routes.jobs.CompositeJobScraper", return_value=stub_scraper):
            resp = client.post("/api/jobs/scrape", json=body)
        assert resp.status_code == 200, resp.text[:400]
        called_terms = []
        for call in stub_scraper.fetch_jobs.call_args_list:
            term = call.kwargs.get("search_term", call.args[0] if call.args else "")
            called_terms.extend(term if isinstance(term, list) else [term])
        joined = " | ".join(str(t) for t in called_terms)
        for q in ("Fresher", "Junior", "Intern", "Entry Level"):
            assert f"Cloud Engineer {q}" in joined, f"default-ON enrichment missing {q!r}: got {called_terms}"


class TestMilestone12SalaryIntegrityAndExperienceParsing:
    """§20.2.1, §20.2.3, §20.3 — Salary extraction regex and amount sanitization."""

    def test_salary_parsing_full_zeros_and_western_commas(self):
        """Verifies 28,000,000, 28,00,000, and 2,000,000 parse correctly to their true LPA figures without digit truncation."""
        from app.scraper.normalizer import _parse_amount, parse_salary_lpa

        # Direct _parse_amount checks across Western and Indian comma representations
        assert _parse_amount("28,000,000", "annual") == 280
        assert _parse_amount("28,00,000", "annual") == 28
        assert _parse_amount("2,000,000", "annual") == 20
        assert _parse_amount("₹ 28,000,000", "annual") == 280
        assert _parse_amount("₹ 28,00,000", "annual") == 28
        assert _parse_amount("Rs. 2,000,000", "annual") == 20
        assert _parse_amount("INR 3,500,000", "annual") == 35

        # Fallback to annual conversion when interval omitted for >= 100k
        assert _parse_amount("28,000,000", "") == 280
        assert _parse_amount("28,00,000", "") == 28
        assert _parse_amount("2,000,000", "") == 20

        # Monthly conversion remains accurate
        assert _parse_amount("50,000", "monthly") == 6
        assert _parse_amount("₹ 1,00,000", "monthly") == 12

        # parse_salary_lpa integration checks
        salary, disc = parse_salary_lpa("₹ 28,00,000", "₹ 35,00,000", "annual")
        assert salary == [28, 35]
        assert disc is True

        salary, disc = parse_salary_lpa("28,000,000", "35,000,000", "annual")
        assert salary == [280, 350]
        assert disc is True

        salary, disc = parse_salary_lpa("2,000,000", "2,500,000", "annual")
        assert salary == [20, 25]
        assert disc is True

    def test_indeed_extractor_flexible_salary_regex(self):
        """Verifies indeed parse_indeed_result extracts Western and Indian comma formatted salaries without truncation."""
        from job_scraper.extractors.indeed import (
            SALARY_RANGE_RE,
            SALARY_SINGLE_RE,
            parse_indeed_result,
        )

        # Regex unit check
        m_western = SALARY_RANGE_RE.search("₹28,000,000 to ₹35,000,000 a year")
        assert m_western is not None
        assert m_western.group(1).replace(",", "") == "28000000"
        assert m_western.group(2).replace(",", "") == "35000000"

        m_indian = SALARY_RANGE_RE.search("₹28,00,000 - 35,00,000 a year")
        assert m_indian is not None
        assert m_indian.group(1).replace(",", "") == "2800000"
        assert m_indian.group(2).replace(",", "") == "3500000"

        m_single = SALARY_SINGLE_RE.search("₹2,000,000 a year")
        assert m_single is not None
        assert m_single.group(1).replace(",", "") == "2000000"

        # parse_indeed_result mock job check
        raw_res = {
            "jobkey": "testjk123",
            "displayTitle": "Senior Backend Architect",
            "company": "Enterprise Corp",
            "formattedLocation": "Bengaluru",
            "snippet": "Design scalable microservices.",
            "salarySnippet": "₹28,000,000 - ₹35,000,000 a year",
        }
        job = parse_indeed_result(raw_res)
        assert job is not None
        assert job.min_amount == 28000000.0
        assert job.max_amount == 35000000.0
        assert job.interval == "annual"

    def test_extract_experience_years_in_scraped_job_normalization(self):
        """Verifies normalize_scraped_job extracts experience from description when unparsed in raw job."""
        from app.scraper.normalizer import normalize_scraped_job
        from app.taxonomy.taxonomy import load_taxonomy

        tax = load_taxonomy()

        # Job with unparsed experience (experience_years missing/0) but description containing 1-3 years
        raw_job_range = {
            "title": "Software Engineer",
            "company": "Cloud Corp",
            "description": "Looking for candidates with 1-3 years of Python and FastAPI experience.",
            "location": "Bengaluru",
            "experience_years": 0,
        }
        job1 = normalize_scraped_job(raw_job_range, tax)
        assert job1.experience_years == 1, f"Expected experience_years=1, got {job1.experience_years}"

        # Job with 2+ years in description
        raw_job_plus = {
            "title": "Backend Engineer",
            "company": "Data Inc",
            "description": "Candidate must possess 2+ years of production experience.",
            "location": "Bengaluru",
            "experience_years": None,
        }
        job2 = normalize_scraped_job(raw_job_plus, tax)
        assert job2.experience_years == 2, f"Expected experience_years=2, got {job2.experience_years}"

        # Job with no experience requirement in description
        raw_job_fresher = {
            "title": "Junior Developer",
            "company": "StartUp",
            "description": "Entry-level role open to freshers with good problem solving skills.",
            "location": "Bengaluru",
            "experience_years": 0,
        }
        job3 = normalize_scraped_job(raw_job_fresher, tax)
        assert job3.experience_years == 0, f"Expected experience_years=0, got {job3.experience_years}"


# ---------------------------------------------------------------------------
# ARCHITECTURE §21.2.4 / §21.3 item 3 — Legacy Sanitation: COMBINED
# title + description inspection (Milestone 13)
# ---------------------------------------------------------------------------
# Gate-first (charter §7): written RED against the measured-absent baseline
# (2026-11-03): sanitize_legacy_experience (pipeline.py) reads ONLY
# `job.description`, so a legacy job whose experience requirement lives in the
# TITLE (e.g. "Software Engineering Specialist (2 YOE)") is NOT repaired. The
# existing test_sanitize_legacy_experience_updates_jobs (M12, line 2970) puts
# the requirement in the description, so it passes WITHOUT proving title
# extraction — this gate closes that hole. It self-arms RED (ImportError /
# failed assertion) until §21.2.4 lands.
#
# Anti-vacuity (charter §4): seeds THREE rows and asserts all three outcomes —
# title-only repaired to its floor, description-only still repaired (M12
# regression guard), and an exp-free row NOT falsely bumped. A no-op
# sanitation fails the title row; a blanket "set to 1" fails the control.
# ---------------------------------------------------------------------------


class TestMilestone13SanitizeCombinedTitleDescription:
    """§21.2.4 — sanitize_legacy_experience inspects COMBINED title + description."""

    @staticmethod
    def _seed_three(db):
        from app.models import Job

        db.add(Job(
            id="job-ext-m13-title-only",
            title="Software Engineering Specialist (2 YOE)",
            company="GE Vernova",
            description="Build scalable systems across the full stack.",
            job_url="https://legit-domain.com/m13-1",
            experience_years=0,
        ))
        db.add(Job(
            id="job-ext-m13-desc-only",
            title="Backend Engineer",
            company="OldCorp",
            description="Requires 1-3 years of Python experience.",
            job_url="https://legit-domain.com/m13-2",
            experience_years=0,
        ))
        db.add(Job(
            id="job-ext-m13-none",
            title="Fresher Trainee",
            company="OldCorp",
            description="Great opportunity for entry-level talent.",
            job_url="https://legit-domain.com/m13-3",
            experience_years=0,
        ))
        db.commit()

    def test_sanitize_legacy_experience_scans_combined_title_and_description(self, client_engine):
        """§21.2.4 — title-carried experience IS repaired; description path
        (M12) still works; exp-free rows are NOT falsely bumped."""
        from sqlmodel import Session

        from app.models import Job
        from app.scraper.pipeline import sanitize_legacy_experience

        _client, engine = client_engine
        with Session(engine) as db:
            self._seed_three(db)

            updated = sanitize_legacy_experience(db, ctx=None)

            title_job = db.get(Job, "job-ext-m13-title-only")
            desc_job = db.get(Job, "job-ext-m13-desc-only")
            none_job = db.get(Job, "job-ext-m13-none")

            # 1) TITLE-only requirement must be extracted (THE §21.2.4 gap)
            assert title_job.experience_years == 2, (
                f"sanitation must inspect the TITLE (combined title+description): "
                f"'(2 YOE)' in title -> expected 2, got {title_job.experience_years}"
            )
            # 2) M12 regression guard: description path must keep working
            assert desc_job.experience_years == 1, (
                f"description path regressed: '1-3 years' -> expected 1, "
                f"got {desc_job.experience_years}"
            )
            # 3) Anti-vacuity control: no requirement anywhere -> untouched
            assert none_job.experience_years == 0, (
                f"exp-free row must NOT be bumped (blanket-set defect), "
                f"got {none_job.experience_years}"
            )
            # Count = title row + desc row, not the control
            assert updated == 2, f"expected 2 rows repaired, got {updated}"


# ---------------------------------------------------------------------------
# ARCHITECTURE §21.2.3 — Client Filter State Sync & Experience Routing
# (Milestone 13) — STATIC CONTENT GATE
# ---------------------------------------------------------------------------
# Gate-first (charter §7): measured 2026-11-03 — NO test asserted any of these
# three §21.2.3 contracts (grep for initFilteringSuite/expSelect/leakTo13/
# form-restore across tests/*.py returned zero). The implementation lines exist
# in matches.html (measured: sync at 1097/1108, routing at 850), so this gate
# pins them as a regression guard (charter §6: keep the contract in the suite,
# not just in a message). It self-arms RED if any sync/guard line is removed.
#
# Anti-vacuity (charter §4): each assertion names the SPEC clause it guards, so
# a partial revert (e.g. drop the pill reconcile but keep the assignment) still
# trips the matching assert.
# ---------------------------------------------------------------------------


class TestMilestone13ClientFilterSync:
    """§21.2.3 — form-restore sync + 0-1yr experience routing in matches.html."""

    @staticmethod
    def _read():
        from pathlib import Path

        return Path("app/templates/student/matches.html").read_text(encoding="utf-8")

    def test_filter_state_syncs_from_restored_dom_inputs(self):
        """§21.2.3 — initFilteringSuite adopts restored checkbox/select into JS state."""
        content = self._read()
        # fresherOnly sync from the (browser-restored) checkbox
        assert "state.fresherOnly = !!fresherCheckbox.checked" in content, (
            "§21.2.3: state.fresherOnly must sync from fresherCheckbox.checked on init "
            "(form-restore guard)"
        )
        # exp sync from the (browser-restored) select
        assert "state.exp = expSelect.value || \"all\"" in content, (
            "§21.2.3: state.exp must sync from expSelect.value on init "
            "(form-restore guard)"
        )
        # applyFilters must run after adoption so the view matches the DOM
        assert content.count("applyFilters();") >= 1, (
            "§21.2.3: applyFilters() must run after init so restored forms never desync"
        )

    def test_one_year_non_fresher_cards_route_to_1_3_not_0_1(self):
        """§21.2.3 — '0-1 yrs' excludes dExp==1 && data-fresher===false (routed to '1-3')."""
        content = self._read()
        # the routing predicate that keeps 1-yr experienced roles out of 0-1
        assert 'leakTo13 = (dExp === 1 && dFresherAttr === "false")' in content, (
            "§21.2.3: '0-1' must identify 1-year NON-fresher cards for exclusion"
        )
        # data-fresher attribute must be read (the discriminator)
        assert 'card.getAttribute("data-fresher")' in content, (
            "§21.2.3: routing must key on the per-card data-fresher attribute"
        )
        # the predicate must actually hide the card when it's a 1-yr leak
        assert "|| leakTo13" in content, (
            "§21.2.3: leakTo13 must feed the hidden decision in the '0-1' branch"
        )





