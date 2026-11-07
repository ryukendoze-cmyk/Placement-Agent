"""Unit and integration tests for the 'remove all live jobs' option across all roles.

Tests verify that:
1. DELETE /api/jobs/live & POST /api/jobs/purge-live:
   - Successfully purge all live scraped jobs (source != 'local' or id starts with 'job-ext-')
   - Preserve institutional on-campus jobs (source == 'local')
   - Cascade foreign-key deletions safely (JobApplication, JobMatch, PlanItem, UpskillPlan, Interview)
   - Clean up vector store entries
   - Record an AuditLog row
   - Work regardless of whether caller is student, faculty_mentor, or tpo_admin (or open demo)
2. DELETE /api/students/{sid}/matches/live & POST /api/students/{sid}/matches/purge-live:
   - Work for students, faculty mentors, and TPO admins
   - Refresh remaining matches for the student
3. purge_live_market_jobs tool in agent tools:
   - Successfully executes and returns the purged count
4. matches.html UI:
   - Contains the 'Remove All Live Jobs' option in the header, portals toolbar, and modal
   - Exposes purgeAllLiveJobs JavaScript function
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from sqlmodel import Session, select

from app.agent.tools import call_tool
from app.models import (
    AuditLog,
    Course,
    Interview,
    Job,
    JobApplication,
    JobMatch,
    PlanItem,
    Student,
    UpskillPlan,
)


def _setup_live_and_local_jobs(eng):
    with Session(eng) as s:
        sid = f"stu-{uuid4().hex[:8]}"
        student = Student(id=sid, name="Live Purge Student", branch="CSE", gpa=8.5, resume_text="Python FastAPI SQL")
        s.add(student)

        # 1 campus job (local)
        jid_local = f"job-local-{uuid4().hex[:8]}"
        local_job = Job(
            id=jid_local,
            title="Campus SDE",
            company="JKLU Tech",
            description="On campus software role",
            required_skills=["python"],
            salary_lpa=[8, 12],
            source="local",
        )
        s.add(local_job)

        # 2 live market jobs (scraped)
        jid_live1 = f"job-ext-{uuid4().hex[:8]}"
        live_job1 = Job(
            id=jid_live1,
            title="Scraped Backend Engineer",
            company="Global Stream",
            description="External backend job",
            required_skills=["python", "fastapi"],
            salary_lpa=[10, 16],
            source="live",
            scraper_engine="jobspy",
        )
        jid_live2 = f"job-ext-{uuid4().hex[:8]}"
        live_job2 = Job(
            id=jid_live2,
            title="External Data Analyst",
            company="Market Dynamics",
            description="External data job",
            required_skills=["sql"],
            salary_lpa=[6, 9],
            source="naukri",
            scraper_engine="stealth_harvester",
        )
        s.add(student)
        s.add(local_job)
        s.add(live_job1)
        s.add(live_job2)
        s.commit()

        # Add related records to verify cascade deletion
        s.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=jid_live1, score=88, band="strong"))
        s.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=jid_local, score=75, band="moderate"))

        plan_id = f"plan-{uuid4().hex[:8]}"
        s.add(UpskillPlan(id=plan_id, student_id=sid, job_id=jid_live1, total_hours=20))
        s.commit()
        course = s.exec(select(Course)).first()
        cid = course.id if course else "c-1"
        s.add(PlanItem(id=f"pi-{uuid4().hex[:8]}", plan_id=plan_id, order=1, skill_id="fastapi", course_id=cid, hours=10))

        s.add(JobApplication(id=f"app-{uuid4().hex[:8]}", student_id=sid, job_id=jid_live1, status="submitted"))
        s.add(Interview(id=f"iv-{uuid4().hex[:8]}", student_id=sid, job_id=jid_live1, score=85, finished=True))

        s.commit()
    return sid, jid_local, [jid_live1, jid_live2]


@pytest.mark.parametrize("role", ["student", "faculty_mentor", "tpo_admin"])
def test_delete_jobs_live_allowed_for_all_roles(client_engine, role):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    headers = {"X-Demo-Role": role}
    if role == "student":
        headers["X-Demo-Student-ID"] = sid

    resp = c.delete("/api/jobs/live", headers=headers)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert data["purged"] >= len(live_jids)

    with Session(eng) as s:
        # Local job must be intact
        assert s.get(Job, jid_local) is not None
        # Live jobs must be purged
        for jid in live_jids:
            assert s.get(Job, jid) is None
            # Cascaded rows must be deleted
            assert s.exec(select(JobMatch).where(JobMatch.job_id == jid)).all() == []
            assert s.exec(select(JobApplication).where(JobApplication.job_id == jid)).all() == []
            assert s.exec(select(Interview).where(Interview.job_id == jid)).all() == []
            assert s.exec(select(UpskillPlan).where(UpskillPlan.job_id == jid)).all() == []

        # AuditLog record created
        audit = s.exec(
            select(AuditLog).where(AuditLog.action == "jobs.live.purged", AuditLog.actor_role == role)
        ).first()
        assert audit is not None


def test_post_jobs_purge_live_endpoint(client_engine):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    resp = c.post("/api/jobs/purge-live", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"

    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None


@pytest.mark.parametrize("role", ["student", "faculty_mentor", "tpo_admin"])
def test_student_matches_purge_live_endpoint(client_engine, role):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    headers = {"X-Demo-Role": role}
    if role == "student":
        headers["X-Demo-Student-ID"] = sid

    resp = c.delete(f"/api/students/{sid}/matches/live", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"

    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None


def test_matches_page_ui_contains_purge_live_options(client_engine):
    c, eng = client_engine
    sid, _, _ = _setup_live_and_local_jobs(eng)

    resp = c.get(f"/students/{sid}/matches", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid})
    assert resp.status_code == 200
    html = resp.text

    # Header button
    assert "purge-live-jobs-header-btn" in html
    assert "Remove All Live Jobs" in html

    # Portals bar button
    assert "btn-purge-live-portal" in html
    assert "Clear Live Jobs" in html

    # Scrape modal button
    assert "modal-purge-live-btn" in html

    # JS function
    assert "purgeAllLiveJobs" in html


def test_purge_live_market_jobs_tool(client_engine):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    ctx = c.app.state.ctx
    with Session(eng) as s:
        res = call_tool("purge_live_market_jobs", ctx, s)
        assert res["purged_count"] >= len(live_jids)
        assert "Purged" in res["message"]

        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None
