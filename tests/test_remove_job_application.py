"""Unit tests for removing / withdrawing job applications and dismissing / deleting jobs
for all roles (Student, Faculty Mentor, TPO Admin).
"""
from __future__ import annotations

from uuid import uuid4

from sqlmodel import Session, select

from app.models import (
    ApprovalRequest,
    AuditLog,
    Job,
    JobApplication,
    JobMatch,
    Student,
)


def _seed_job_and_student(client, eng):
    with Session(eng) as s:
        sid = f"stu-{uuid4().hex[:8]}"
        jid = f"job-ext-{uuid4().hex[:8]}"
        s.add(Student(id=sid, name="Test Removal Student", branch="CSE", gpa=8.5, resume_text="Python FastAPI"))
        s.add(Job(id=jid, title="Closed or Unwanted Role", company="Acme Corp", description="Python developer",
                  required_skills=["python"], salary_lpa=[10, 15], source="live"))
        s.commit()
    return sid, jid


def test_student_can_dismiss_job_match(client, client_engine):
    c, eng = client_engine
    sid, jid = _seed_job_and_student(c, eng)

    with Session(eng) as s:
        mid = f"m-{uuid4().hex[:8]}"
        s.add(JobMatch(id=mid, student_id=sid, job_id=jid, score=85, band="strong", breakdown={}))
        s.commit()

    # Student dismisses job match
    resp = c.delete(f"/api/students/{sid}/matches/{jid}")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "dismissed"
    assert data["job_id"] == jid

    with Session(eng) as s:
        m = s.exec(select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == jid)).first()
        assert m is not None
        assert m.band == "dismissed"
        assert m.score == -1
        assert m.breakdown.get("dismissed") is True

        audit = s.exec(select(AuditLog).where(AuditLog.entity_id == jid, AuditLog.action == "job_match.dismissed")).first()
        assert audit is not None


def test_matches_page_filters_out_dismissed_matches(client, client_engine):
    c, eng = client_engine
    sid, jid = _seed_job_and_student(c, eng)

    with Session(eng) as s:
        mid = f"m-{uuid4().hex[:8]}"
        s.add(JobMatch(id=mid, student_id=sid, job_id=jid, score=90, band="dismissed", breakdown={"dismissed": True}))
        s.commit()

    resp = c.get(f"/students/{sid}/matches")
    assert resp.status_code == 200
    assert "Closed or Unwanted Role" not in resp.text


def test_student_and_admin_can_withdraw_application(client, client_engine):
    c, eng = client_engine
    sid, jid = _seed_job_and_student(c, eng)

    app_id = f"app-{uuid4().hex[:8]}"
    ap_id = f"appr-{uuid4().hex[:8]}"
    with Session(eng) as s:
        ap = ApprovalRequest(id=ap_id, kind="job_application_nomination", target_id=sid,
                             title="Nomination", summary="Test", payload={}, status="pending_review")
        s.add(ap)
        s.add(JobApplication(id=app_id, student_id=sid, job_id=jid, status="awaiting_tpo_approval",
                             tier="dream", match_score=85, approval_id=ap_id))
        s.commit()

    # Student withdraws / removes application
    resp = c.delete(f"/api/students/{sid}/applications/{app_id}")
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "withdrawn"

    with Session(eng) as s:
        app_row = s.get(JobApplication, app_id)
        assert app_row.status == "withdrawn"
        ap_row = s.get(ApprovalRequest, ap_id)
        assert ap_row.status == "rejected"


def test_tpo_can_permanently_delete_job_and_clean_matches(client, client_engine):
    c, eng = client_engine
    sid, jid = _seed_job_and_student(c, eng)

    with Session(eng) as s:
        s.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=jid, score=80, band="strong", breakdown={}))
        s.add(JobApplication(id=f"app-{uuid4().hex[:8]}", student_id=sid, job_id=jid, status="submitted", tier="standard"))
        s.commit()

    # Permanent removal with tpo_admin role
    resp = c.delete(f"/api/students/{sid}/matches/{jid}?permanent=true", headers={"X-Role": "tpo_admin"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "deleted_permanently"

    with Session(eng) as s:
        assert s.get(Job, jid) is None
        matches = s.exec(select(JobMatch).where(JobMatch.job_id == jid)).all()
        assert len(matches) == 0
        apps = s.exec(select(JobApplication).where(JobApplication.job_id == jid)).all()
        for a in apps:
            assert a.status == "withdrawn"


def test_delete_admin_application_endpoint(client, client_engine):
    c, eng = client_engine
    sid, jid = _seed_job_and_student(c, eng)

    app_id = f"app-{uuid4().hex[:8]}"
    with Session(eng) as s:
        s.add(JobApplication(id=app_id, student_id=sid, job_id=jid, status="submitted", tier="standard"))
        s.commit()

    resp = c.delete(f"/api/admin/applications/{app_id}", headers={"X-Role": "tpo_admin"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "withdrawn"

    with Session(eng) as s:
        app_row = s.get(JobApplication, app_id)
        assert app_row.status == "withdrawn"
