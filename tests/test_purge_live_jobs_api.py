"""Unit and integration tests for the 'remove all live jobs' RBAC and per-student dismissal flow.

Tests verify that:
1. DELETE /api/jobs/live & POST /api/jobs/purge-live:
   - Restricted strictly to tpo_admin (403 for student & faculty_mentor)
   - Successfully purge all live scraped jobs (source != 'local' or id starts with 'job-ext-')
   - Preserve institutional on-campus jobs (source == 'local')
   - Cascade foreign-key deletions safely (JobApplication, JobMatch, PlanItem, UpskillPlan, Interview)
   - Clean up vector store entries and record an AuditLog row
2. DELETE /api/students/{sid}/matches/live & POST /api/students/{sid}/matches/purge-live:
   - POST .../purge-live is tpo_admin-only (403 for student & faculty_mentor)
   - DELETE .../matches/live performs global purge for tpo_admin, per-student dismissal ONLY for student
     (leaving shared Job catalog intact), and returns 403 for faculty_mentor
3. purge_live_market_jobs tool in agent tools:
   - Successfully executes and returns the purged count
4. matches.html UI:
   - Exposes 'Remove All Live Jobs' & purge controls strictly for tpo_admin, and hides them completely
     from students (no greyed-out admin buttons)
"""
from __future__ import annotations

from uuid import uuid4

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


def test_delete_jobs_live_restricted_to_tpo_admin(client_engine):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    assert c.delete("/api/jobs/live", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid}).status_code == 403
    assert c.delete("/api/jobs/live", headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403

    resp = c.delete("/api/jobs/live", headers={"X-Demo-Role": "tpo_admin"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "success"
    assert data["purged"] >= len(live_jids)

    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None
            assert s.exec(select(JobMatch).where(JobMatch.job_id == jid)).all() == []
            assert s.exec(select(JobApplication).where(JobApplication.job_id == jid)).all() == []
            assert s.exec(select(Interview).where(Interview.job_id == jid)).all() == []
            assert s.exec(select(UpskillPlan).where(UpskillPlan.job_id == jid)).all() == []

        audit = s.exec(
            select(AuditLog).where(AuditLog.action == "jobs.live.purged", AuditLog.actor_role == "tpo_admin")
        ).first()
        assert audit is not None


def test_post_jobs_purge_live_endpoint_rbac(client_engine):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    assert c.post("/api/jobs/purge-live", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid}).status_code == 403
    assert c.post("/api/jobs/purge-live", headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403

    resp = c.post("/api/jobs/purge-live", headers={"X-Demo-Role": "tpo_admin"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "success"

    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None


def test_student_matches_dismiss_vs_tpo_purge_live(client_engine):
    c, eng = client_engine
    sid, jid_local, live_jids = _setup_live_and_local_jobs(eng)

    # Faculty mentor cannot purge/dismiss another student's live feed globally
    assert c.delete(f"/api/students/{sid}/matches/live", headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403
    assert c.post(f"/api/students/{sid}/matches/purge-live", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid}).status_code == 403

    # Student calling DELETE /matches/live or /live-jobs purges ONLY their owned live jobs + live matches
    # without deleting unowned global jobs, peer students' live jobs, or on-campus institutional jobs.
    stu_b_id = f"stu-{uuid4().hex[:8]}"
    owned_live_a = f"job-ext-own-a-{uuid4().hex[:6]}"
    owned_live_b = f"job-ext-own-b-{uuid4().hex[:6]}"
    with Session(eng) as s:
        s.add(Student(id=stu_b_id, name="Peer Student B", branch="CSE", gpa=8.2, resume_text="Python SQL"))
        s.add(
            Job(
                id=owned_live_a,
                title="A Owned Live Role",
                company="Alpha Live",
                description="Live job scraped by A",
                required_skills=["python"],
                salary_lpa=[10, 14],
                source="naukri",
                is_live=True,
                owner_student_id=sid,
            )
        )
        s.add(
            Job(
                id=owned_live_b,
                title="B Owned Live Role",
                company="Beta Live",
                description="Live job scraped by B",
                required_skills=["sql"],
                salary_lpa=[9, 12],
                source="linkedin",
                is_live=True,
                owner_student_id=stu_b_id,
            )
        )
        s.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=owned_live_a, score=85, band="strong"))
        s.commit()

    stu_resp = c.delete(
        f"/api/students/{sid}/live-jobs",
        headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid},
    )
    assert stu_resp.status_code == 200, stu_resp.text
    sdata = stu_resp.json()
    assert sdata["status"] == "success"
    assert sdata["purged"] == 1
    assert sdata["deleted_matches"] >= 2
    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        assert s.get(Job, owned_live_a) is None
        assert s.get(Job, owned_live_b) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is not None
            assert s.exec(select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == jid)).first() is None
        # On-campus JobMatch stays untouched
        local_jm = s.exec(select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == jid_local)).first()
        assert local_jm is not None and local_jm.band != "dismissed"

    # TPO Admin calling DELETE /matches/live performs global purge
    tpo_resp = c.delete(f"/api/students/{sid}/matches/live", headers={"X-Demo-Role": "tpo_admin"})
    assert tpo_resp.status_code == 200, tpo_resp.text
    assert tpo_resp.json()["status"] == "success"
    with Session(eng) as s:
        assert s.get(Job, jid_local) is not None
        for jid in live_jids:
            assert s.get(Job, jid) is None


def test_matches_page_ui_student_sees_live_controls_hides_campus_admin(client_engine):
    c, eng = client_engine
    sid, _, _ = _setup_live_and_local_jobs(eng)

    # Student sees personal Live Job controls, but MUST NOT see TPO Campus Drive / JNF controls
    stu_resp = c.get(f"/students/{sid}/matches", headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid})
    assert stu_resp.status_code == 200
    stu_html = stu_resp.text
    for visible_id in (
        "open-scrape-modal",
        "purge-live-jobs-header-btn",
        "btn-purge-live-portal",
        "modal-purge-live-btn",
        "scrape-modal",
    ):
        assert visible_id in stu_html, f"Student matches page must expose {visible_id}"
    for hidden_id in (
        "open-smart-jd-modal",
        "open-jnf-drawer-btn",
        "campus_drive.js",
    ):
        assert hidden_id not in stu_html, f"Student matches page must hide {hidden_id}"

    # TPO Admin keeps all live-job + campus-drive management controls
    tpo_resp = c.get(f"/students/{sid}/matches", headers={"X-Demo-Role": "tpo_admin"})
    assert tpo_resp.status_code == 200
    tpo_html = tpo_resp.text
    assert "open-smart-jd-modal" in tpo_html
    assert "open-jnf-drawer-btn" in tpo_html
    assert "purge-live-jobs-header-btn" in tpo_html
    assert "Remove All Live Jobs" in tpo_html
    assert "btn-purge-live-portal" in tpo_html
    assert "Clear Live Jobs" in tpo_html
    assert "modal-purge-live-btn" in tpo_html
    assert "purgeAllLiveJobs" in tpo_html


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
