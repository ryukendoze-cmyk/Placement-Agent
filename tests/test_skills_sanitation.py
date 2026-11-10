"""Tests for skill taxonomy dumping prevention and database skill sanitation."""

from unittest.mock import MagicMock

from sqlmodel import Session

from app.db import init_db, make_engine
from app.models import Job
from app.scraper.normalizer import normalize_job_with_ai, reset_ai_normalizer_circuit
from app.scraper.pipeline import sanitize_legacy_skills
from app.taxonomy.taxonomy import load_taxonomy


def test_ai_normalizer_does_not_dump_unmentioned_taxonomy():
    """Verify normalize_job_with_ai drops unmentioned skills and caps lists to <= 6."""
    reset_ai_normalizer_circuit()
    taxonomy = load_taxonomy()
    all_taxonomy_ids = [s.id for s in taxonomy.all()]

    mock_llm = MagicMock()
    # Mock LLM returning 2 valid required, plus docker and 50 dumped preferred skills
    mock_llm.complete.return_value = (
        '{\n'
        '  "required_skills": ["python", "machine_learning"],\n'
        '  "preferred_skills": ' + str(["docker"] + all_taxonomy_ids[:50]).replace("'", '"') + ',\n'
        '  "salary_lpa": [6, 12],\n'
        '  "salary_disclosed": false,\n'
        '  "experience_years": 0,\n'
        '  "allowed_branches": ["CSE"],\n'
        '  "min_gpa": 7.0\n'
        '}'
    )

    raw_job = {
        "title": "AI Intern",
        "company": "Test AI Corp",
        "description": "Looking for an intern skilled in Python and Machine Learning. Nice to have Docker.",
        "location": "Bengaluru",
        "job_url": "https://example.org/jobs/test-1",
        "source": "naukri",
    }

    job = normalize_job_with_ai(raw_job, taxonomy=taxonomy, llm=mock_llm)

    assert "python" in job.required_skills
    assert "machine_learning" in job.required_skills
    assert len(job.required_skills) <= 6

    # Docker is mentioned in text ("Nice to have Docker"), so it is allowed in preferred
    assert "docker" in job.preferred_skills
    # Unmentioned skills (like typescript, c, rust, etc.) must NOT be in preferred
    assert "rust" not in job.preferred_skills
    assert "swift" not in job.preferred_skills
    assert len(job.preferred_skills) <= 6


def test_sanitize_legacy_skills():
    """Verify sanitize_legacy_skills cleans bloated jobs and leaves local curated jobs intact."""
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)

    with Session(engine) as session:
        # 1. Bloated external job with 100 preferred skills
        bloated_job = Job(
            id="job-ext-bloated-1",
            title="Intern Bachelors AI",
            company="Sparta Systems",
            description="Work with machine learning models and data visualization. Apply URL: https://naukri.com/1",
            job_url="https://naukri.com/1",
            source="naukri",
            required_skills=["python", "machine_learning"],
            preferred_skills=["javascript", "typescript", "java", "c", "cplusplus", "go", "rust", "kotlin", "ruby"],
        )
        session.add(bloated_job)

        # 2. Local curated job with valid skills (should be preserved)
        local_job = Job(
            id="job-local-1",
            title="Backend Engineer",
            company="TechNova",
            description="Build backend APIs in Python.",
            job_url="https://internal.jklu.edu/1",
            source="local",
            required_skills=["python", "fastapi", "postgresql", "git"],
            preferred_skills=["docker", "redis"],
        )
        session.add(local_job)
        session.commit()

        sanitized_count = sanitize_legacy_skills(session)
        assert sanitized_count == 1

        updated_bloated = session.get(Job, "job-ext-bloated-1")
        assert updated_bloated is not None
        # Preferred skills unmentioned in description must be purged
        assert len(updated_bloated.preferred_skills) <= 6
        assert "go" not in updated_bloated.preferred_skills
        assert "rust" not in updated_bloated.preferred_skills

        # Local job must not be altered
        preserved_local = session.get(Job, "job-local-1")
        assert preserved_local is not None
        assert preserved_local.required_skills == ["python", "fastapi", "postgresql", "git"]
        assert preserved_local.preferred_skills == ["docker", "redis"]
