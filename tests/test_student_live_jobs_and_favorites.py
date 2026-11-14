"""Tests for V4 student-owned live jobs, favorites schema/migration, legacy audit email scrub, and courses hygiene."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text
from sqlmodel import Session, select

from app.db import init_db, make_engine
from app.models import AuditLog, Job, JobApplication, JobMatch, Student
from app.scraper.pipeline import ingest_scraped_jobs, purge_live_jobs
from app.taxonomy.taxonomy import load_taxonomy


def test_ingest_scraped_jobs_isolates_per_student_owner():
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    ctx = SimpleNamespace(taxonomy=tax, llm=None, vector_store=None, embedder=None)

    raw_same_url = [
        {
            "title": "Backend Python Intern",
            "company": "Acme Cloud",
            "location": "Jaipur",
            "description": "Build FastAPI services with Python, SQL, and Docker for analytics workloads.",
            "job_url": "https://careers.acmecloud.in/jobs/backend-intern",
            "source": "linkedin",
            "salary_min": 6,
            "salary_max": 8,
            "salary_unit": "lpa",
        }
    ]
    raw_alt = [
        {
            "title": "Data Analyst Intern",
            "company": "Beta Labs",
            "location": "Remote",
            "description": "Analyze metrics using Python, Pandas, and PostgreSQL.",
            "job_url": "https://betalabs.in/careers/data-analyst",
            "source": "internshala",
        }
    ]

    with Session(eng) as session:
        campus_job = Job(
            id="job-campus-1",
            title="On-Campus SDE",
            company="JKLU Partner Tech",
            description="Institutional drive for CSE/IT.",
            source="local",
            is_live=False,
            owner_student_id=None,
        )
        session.add(campus_job)
        session.commit()

        res_alice = ingest_scraped_jobs(raw_same_url, session, ctx, taxonomy=tax, owner_student_id="stu-alice")
        res_bob = ingest_scraped_jobs(raw_same_url, session, ctx, taxonomy=tax, owner_student_id="stu-bob")

        assert res_alice["ingested_count"] == 1
        assert res_bob["ingested_count"] == 1

        alice_job = res_alice["jobs"][0]
        bob_job = res_bob["jobs"][0]
        assert alice_job.id != bob_job.id
        assert alice_job.owner_student_id == "stu-alice"
        assert bob_job.owner_student_id == "stu-bob"
        assert alice_job.is_live is True
        assert bob_job.is_live is True
        assert alice_job.id.endswith(hashlib.sha1(b"stu-alice").hexdigest()[:6])
        assert bob_job.id.endswith(hashlib.sha1(b"stu-bob").hexdigest()[:6])

        # Re-ingesting with replace_existing_live=True for Alice replaces only Alice's live jobs
        res_alice_v2 = ingest_scraped_jobs(
            raw_alt,
            session,
            ctx,
            taxonomy=tax,
            owner_student_id="stu-alice",
            replace_existing_live=True,
        )
        assert res_alice_v2["ingested_count"] == 1

        all_jobs = {j.id: j for j in session.exec(select(Job)).all()}
        assert "job-campus-1" in all_jobs
        assert alice_job.id not in all_jobs
        assert bob_job.id in all_jobs
        assert res_alice_v2["job_ids"][0] in all_jobs

        # Scoped purge removes only Bob's live job
        purged = purge_live_jobs(session, ctx, clear_output=False, owner_student_id="stu-bob")
        assert purged == 1
        remaining_ids = {j.id for j in session.exec(select(Job)).all()}
        assert "job-campus-1" in remaining_ids
        assert res_alice_v2["job_ids"][0] in remaining_ids
        assert bob_job.id not in remaining_ids


def test_favorites_schema_and_sqlite_legacy_migration(tmp_path):
    db_file = tmp_path / "legacy_v4.db"
    eng = make_engine(f"sqlite:///{db_file}")

    with eng.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE student (id TEXT PRIMARY KEY, name TEXT NOT NULL, branch TEXT NOT NULL DEFAULT '', "
                "year INTEGER NOT NULL DEFAULT 3, gpa REAL NOT NULL DEFAULT 0.0, backlogs INTEGER NOT NULL DEFAULT 0, "
                "pii_redacted_count INTEGER NOT NULL DEFAULT 0, resume_reupload_unlocked BOOLEAN NOT NULL DEFAULT 0, "
                "resume_text TEXT NOT NULL DEFAULT '', created_at TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE job (id TEXT PRIMARY KEY, title TEXT NOT NULL, company TEXT NOT NULL, "
                "description TEXT NOT NULL DEFAULT '', required_skills TEXT NOT NULL DEFAULT '[]', "
                "preferred_skills TEXT NOT NULL DEFAULT '[]', experience_years INTEGER NOT NULL DEFAULT 0, "
                "location TEXT NOT NULL DEFAULT '', remote BOOLEAN NOT NULL DEFAULT 0, "
                "salary_lpa TEXT NOT NULL DEFAULT '[0, 0]', salary_disclosed BOOLEAN NOT NULL DEFAULT 0, "
                "min_gpa REAL NOT NULL DEFAULT 0.0, allowed_branches TEXT NOT NULL DEFAULT '[]', "
                "max_backlogs INTEGER DEFAULT NULL, job_url TEXT NOT NULL DEFAULT '', "
                "source TEXT NOT NULL DEFAULT 'local', scraper_engine TEXT NOT NULL DEFAULT '')"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE jobmatch (id TEXT PRIMARY KEY, student_id TEXT NOT NULL, job_id TEXT NOT NULL, "
                "score INTEGER NOT NULL DEFAULT 0, band TEXT NOT NULL DEFAULT 'low', breakdown TEXT NOT NULL DEFAULT '{}', "
                "created_at TIMESTAMP)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE jobapplication (id TEXT PRIMARY KEY, student_id TEXT NOT NULL, job_id TEXT NOT NULL, "
                "status TEXT NOT NULL DEFAULT 'submitted', tier TEXT NOT NULL DEFAULT 'standard', "
                "match_score INTEGER NOT NULL DEFAULT 0, eligible_at_apply BOOLEAN NOT NULL DEFAULT 1, "
                "eligibility_checks TEXT NOT NULL DEFAULT '{}', approval_id TEXT DEFAULT NULL, "
                "endorsed_by TEXT NOT NULL DEFAULT '', approved_by TEXT NOT NULL DEFAULT '', "
                "notes TEXT NOT NULL DEFAULT '', created_at TIMESTAMP, updated_at TIMESTAMP)"
            )
        )

    init_db(eng)

    insp = sa_inspect(eng)
    job_cols = {c["name"] for c in insp.get_columns("job")}
    jm_cols = {c["name"] for c in insp.get_columns("jobmatch")}
    ja_cols = {c["name"] for c in insp.get_columns("jobapplication")}
    assert {"is_live", "owner_student_id"}.issubset(job_cols)
    assert "is_favorite" in jm_cols
    assert "is_favorite" in ja_cols

    with Session(eng) as session:
        session.add(Student(id="stu-fav", name="Kavya", branch="CSE", year=4, gpa=8.7))
        session.add(Job(id="job-fav-1", title="Cloud Intern", company="Nimbus", description="AWS & Python", is_live=True, owner_student_id="stu-fav"))
        session.add(JobMatch(id="jm-fav-1", student_id="stu-fav", job_id="job-fav-1", score=88, band="strong", is_favorite=True))
        session.add(JobApplication(id="app-fav-1", student_id="stu-fav", job_id="job-fav-1", is_favorite=True))
        session.commit()

        jm = session.get(JobMatch, "jm-fav-1")
        ja = session.get(JobApplication, "app-fav-1")
        assert jm is not None and jm.is_favorite is True
        assert ja is not None and ja.is_favorite is True


def test_scrub_legacy_audit_emails_runs_before_audit_guard(tmp_path):
    db_file = tmp_path / "legacy_audit.db"
    eng = make_engine(f"sqlite:///{db_file}")

    # Create tables cleanly without calling init_db first, then seed legacy email-bearing AuditLog rows
    from app.models import SQLModel

    SQLModel.metadata.create_all(eng)
    with eng.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO auditlog (id, action, entity_type, entity_id, actor_role, detail, created_at) "
                "VALUES (:id1, 'auth.login', 'user', 'Student.User@jklu.edu.in', 'student', :det1, CURRENT_TIMESTAMP), "
                "(:id2, 'auth.denied', 'auth', 'other@gmail.com', 'anonymous', :det2, CURRENT_TIMESTAMP)"
            ),
            {
                "id1": "aud-old-1",
                "det1": json.dumps({"email": "Student.User@jklu.edu.in", "actor_ref": "t***@jklu.edu.in", "role": "student"}),
                "id2": "aud-old-2",
                "det2": json.dumps({"reason": "blocked for ops@company.in", "nested": ["admin@jklu.edu.in", "r***@jklu.edu.in"]}),
            },
        )

    init_db(eng)

    expected_stu_tok = f"id-{hashlib.sha256(b'student.user@jklu.edu.in').hexdigest()[:8]}"
    expected_masked_tok = f"id-{hashlib.sha256(b't***@jklu.edu.in').hexdigest()[:8]}"
    expected_other_tok = f"id-{hashlib.sha256(b'other@gmail.com').hexdigest()[:8]}"
    expected_ops_tok = f"id-{hashlib.sha256(b'ops@company.in').hexdigest()[:8]}"
    expected_admin_tok = f"id-{hashlib.sha256(b'admin@jklu.edu.in').hexdigest()[:8]}"
    expected_masked2_tok = f"id-{hashlib.sha256(b'r***@jklu.edu.in').hexdigest()[:8]}"

    with Session(eng) as session:
        a1 = session.get(AuditLog, "aud-old-1")
        a2 = session.get(AuditLog, "aud-old-2")
        assert a1 is not None and a2 is not None
        assert "@" not in a1.entity_id
        assert a1.entity_id == expected_stu_tok
        assert a1.detail["email"] == expected_stu_tok
        assert a1.detail["actor_ref"] == expected_masked_tok
        assert a2.entity_id == expected_other_tok
        assert expected_ops_tok in a2.detail["reason"]
        assert a2.detail["nested"] == [expected_admin_tok, expected_masked2_tok]
        assert "@" not in json.dumps(a1.detail)
        assert "@" not in json.dumps(a2.detail)

        # Audit guard remains active after init_db scrubbing
        a1.action = "tampered"
        session.add(a1)
        with pytest.raises(RuntimeError, match="immutable"):
            session.commit()
        session.rollback()


def test_courses_catalog_has_no_youtube_provider():
    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    for c in courses:
        assert "youtube" not in (c.get("provider") or "").lower(), c["id"]
        assert "youtube" not in (c.get("url") or "").lower(), c["id"]
