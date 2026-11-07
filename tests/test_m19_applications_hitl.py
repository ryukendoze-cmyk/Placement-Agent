"""Milestone 19: campus drive application pipeline (JobApplication) and
two-tier HITL governance (endorse -> approve).

Covers 19.1-19.5 of the master plan:
- standard tier eligible -> auto-nominate (no HITL)
- dream/super_dream eligible -> pending ApprovalRequest, nominated only after approve
- ineligible + no waiver -> HTTP 422 structured block with SOP citations
- ineligible + waiver -> awaiting_mentor_endorsement -> endorse -> awaiting_tpo_approval -> approve -> nominated
- rejection -> linked JobApplication rejected
- delete_student_core purges JobApplication rows (FK safety)
- admin applications endpoint: faculty_mentor department-scoped, tpo_admin all
- student isolation via enforce_student_scope
"""
from __future__ import annotations

from uuid import uuid4

from sqlmodel import Session, select

from app.api.routes.applications import list_admin_applications
from app.api.routes.students import delete_student_core
from app.models import (
    ApprovalRequest,
    AuditLog,
    Job,
    JobApplication,
    ProgressEvent,
    Student,
)


def _make_job(session: Session, *, job_id: str, salary: list[int],
              min_gpa: float = 7.0, branches: list[str] | None = None,
              required: list[str] | None = None) -> None:
    session.add(Job(
        id=job_id, title=f"Role {job_id}", company="Acme Corp",
        description="Build things. " + ", ".join(required or ["python"]),
        required_skills=required or ["python"],
        preferred_skills=[], salary_lpa=salary, min_gpa=min_gpa,
        allowed_branches=branches or [], job_url="https://example.com/j",
        source="local",
    ))
    session.commit()


def _seed_client_student(client, *, name: str = "M19 Probe",
                         branch: str = "CSE", gpa: float = 8.0) -> str:
    body = ("Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n"
            "Backend internship building REST APIs\n")
    r = client.post("/api/students",
                    files={"file": ("r.txt", body.encode(), "text/plain")},
                    data={"name": name, "branch": branch, "gpa": str(gpa)})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _add_mock_interview(eng, sid: str, job_id: str, score: int) -> None:
    """Seed a completed mock interview so the SOP-04 dream gate passes."""
    from app.models import Interview
    with Session(eng) as db:
        db.add(Interview(id=f"iv-{uuid4().hex[:8]}", student_id=sid,
                         job_id=job_id, transcript=[], score=score,
                         finished=True))
        db.commit()


def _make_standard_job(session: Session, *, job_id: str) -> None:
    """A standard-tier job the seeded resume already satisfies (SOP-05 pass).

    required_skills=[python] at intermediate depth -> coverage 100% and
    non-early band, so evaluate_student_eligibility returns eligible=True
    without needing a mock interview (standard tier skips SOP-04).
    """
    session.add(Job(
        id=job_id, title="Junior Python Developer", company="Acme Corp",
        description=("Junior Python developer role building REST APIs with "
                     "Python, FastAPI and PostgreSQL. Fresher friendly."),
        required_skills=["python"], preferred_skills=["fastapi"],
        salary_lpa=[5, 7], min_gpa=0.0, allowed_branches=[],
        job_url="https://example.com/j", source="local",
    ))
    session.commit()


def _client_engine(client_engine):
    client, eng = client_engine
    return client, eng


def _get_app_row(eng, app_id: str) -> JobApplication | None:
    with Session(eng) as db:
        return db.get(JobApplication, app_id)


def _apps_for(eng, sid: str) -> list[JobApplication]:
    with Session(eng) as db:
        return db.exec(select(JobApplication)
                       .where(JobApplication.student_id == sid)).all()


def _audits(eng) -> list[AuditLog]:
    with Session(eng) as db:
        return db.exec(select(AuditLog)).all()


# ------------------------------------------------------------- 19.1 model

def test_model_and_endorse_columns_exist(client_engine):
    """19.1: JobApplication table + ApprovalRequest endorsement columns exist."""
    client, eng = client_engine
    sid = _seed_client_student(client)
    with Session(eng) as db:
        _make_job(db, job_id="job-model-1", salary=[6, 7])
        ap = ApprovalRequest(id=f"appr-{uuid4().hex[:8]}",
                             kind="job_application_nomination",
                             target_id=sid, title="t", summary="s",
                             payload={}, status="pending_review",
                             requested_by="agent")
        assert ap.endorsed_by == ""
        assert ap.endorsed_at is None
        db.add(ap)
        db.flush()
        app = JobApplication(id=f"app-{uuid4().hex[:8]}", student_id=sid,
                             job_id="job-model-1", status="submitted",
                             tier="standard",
                             match_score=70, eligible_at_apply=True,
                             eligibility_checks={"ok": True},
                             approval_id=None, endorsed_by="", approved_by="",
                             notes="")
        db.add(app)
        db.commit()
        assert db.get(ApprovalRequest, ap.id).endorsed_by == ""
        row = db.get(JobApplication, app.id)
        assert row is not None and row.status == "submitted"


def test_delete_student_core_purges_applications(client_engine):
    """19.1: delete_student_core must delete JobApplication rows first (FK safety)."""
    client, eng = client_engine
    sid = _seed_client_student(client)
    with Session(eng) as db:
        _make_job(db, job_id="job-del-1", salary=[10, 12])
        db.add(JobApplication(id=f"app-{uuid4().hex[:8]}", student_id=sid,
                              job_id="job-del-1", status="nominated",
                              tier="dream", match_score=80,
                              eligible_at_apply=True, eligibility_checks={},
                              approval_id=None, endorsed_by="", approved_by="",
                              notes=""))
        db.commit()
    with Session(eng) as db:
        assert delete_student_core(db, client.app.state.ctx, sid) is True
    assert _apps_for(eng, sid) == []
    with Session(eng) as db:
        assert db.get(Student, sid) is None


# ------------------------------------------------------- 19.2 endpoints

def test_standard_eligible_application_auto_nominates(client_engine):
    """Eligible + standard tier -> JobApplication(status=nominated) + audit,
    no ApprovalRequest, no HITL."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=8.5)
    with Session(eng) as db:
        _make_standard_job(db, job_id="job-std-1")
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-std-1"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["application"]["status"] == "nominated"
    assert body["application"]["tier"] == "standard"
    assert body["eligibility"]["eligible"] is True
    with Session(eng) as db:
        apps = db.exec(select(JobApplication)
                       .where(JobApplication.student_id == sid)).all()
        assert len(apps) == 1
        assert apps[0].status == "nominated"
        assert apps[0].approval_id is None
        aps = db.exec(select(ApprovalRequest)).all()
        assert aps == []
        evs = db.exec(select(ProgressEvent)
                      .where(ProgressEvent.student_id == sid)).all()
        assert any(e.kind == "nomination" for e in evs)
    assert any(a.action == "application.nominated" for a in _audits(eng))


def test_dream_application_creates_pending_approval(client_engine):
    """Eligible + dream tier -> ApprovalRequest(pending_review) +
    JobApplication(awaiting_tpo_approval); nominated only after approve."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=8.5)
    # job-001 (seeded, dream tier) passes SOP-03/05 for this resume; the
    # SOP-04 mock gate needs a completed interview >= 65.
    _add_mock_interview(eng, sid, "job-001", 75)
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-001"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    assert body["application"]["status"] == "awaiting_tpo_approval"
    ap_id = body["application"]["approval_id"]
    assert ap_id
    with Session(eng) as db:
        ap = db.get(ApprovalRequest, ap_id)
        assert ap is not None
        assert ap.status == "pending_review"
        assert ap.kind == "job_application_nomination"
        assert ap.payload["student_id"] == sid
        assert ap.payload["job_id"] == "job-001"
        assert ap.payload["tier"] == "dream"
        assert int(ap.payload["readiness"]) >= 0
        evs = db.exec(select(ProgressEvent)
                      .where(ProgressEvent.student_id == sid)).all()
        assert not any(e.kind == "nomination" for e in evs)
    # approve -> nominated
    r2 = client.post(f"/api/approvals/{ap_id}/approve", json={"note": "ok"})
    assert r2.status_code == 200, r2.text
    row = _get_app_row(eng, body["application"]["id"])
    assert row is not None and row.status == "nominated"
    assert row.approved_by in ("", "faculty_mentor", "tpo_admin")
    with Session(eng) as db:
        evs = db.exec(select(ProgressEvent)
                      .where(ProgressEvent.student_id == sid)).all()
        assert any(e.kind == "nomination" for e in evs)


def test_ineligible_without_waiver_is_blocked(client_engine):
    """Ineligible + request_waiver=False -> HTTP 422 structured block citing SOPs."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=6.0, branch="CSE")
    with Session(eng) as db:
        # super_dream tier + restrictive branch + low GPA -> blocked on SOP-03/09
        _make_job(db, job_id="job-block-1", salary=[20, 25], min_gpa=9.0,
                  branches=["ECE"])
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-block-1", "request_waiver": False})
    assert r.status_code == 422, r.text
    body = r.json()["detail"]  # FastAPI wraps HTTPException detail
    assert body["status"] == "blocked"
    elig = body["eligibility"]
    assert elig["eligible"] is False
    sops = set(elig.get("cited_sops") or [])
    assert "SOP-03" in sops or "SOP-09" in sops
    assert _apps_for(eng, sid) == []
    assert not any(a.action == "application.nominated" for a in _audits(eng))


def test_ineligible_waiver_full_two_tier_flow(client_engine):
    """Blocked + waiver -> awaiting_mentor_endorsement -> endorse ->
    awaiting_tpo_approval -> approve -> nominated, with AuditLog trail."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=6.0, branch="CSE")
    with Session(eng) as db:
        _make_job(db, job_id="job-waiver-1", salary=[20, 25], min_gpa=9.0,
                  branches=["ECE"])
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-waiver-1", "request_waiver": True,
                          "note": "strong referral"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    app_id = body["application"]["id"]
    ap_id = body["application"]["approval_id"]
    assert body["application"]["status"] == "awaiting_mentor_endorsement"
    assert ap_id
    assert body["eligibility"]["eligible"] is False

    # mentor endorsement advances the application
    r2 = client.post(f"/api/approvals/{ap_id}/endorse", json={"note": "ok"})
    assert r2.status_code == 200, r2.text
    row = _get_app_row(eng, app_id)
    assert row is not None and row.status == "awaiting_tpo_approval"
    assert row.endorsed_by in ("faculty_mentor", "tpo_admin")
    with Session(eng) as db:
        ap = db.get(ApprovalRequest, ap_id)
        assert ap is not None and ap.endorsed_at is not None
    assert any(a.action == "approval.endorsed" and a.entity_id == ap_id
               for a in _audits(eng))

    # TPO approval nominates
    r3 = client.post(f"/api/approvals/{ap_id}/approve", json={"note": "go"})
    assert r3.status_code == 200, r3.text
    row = _get_app_row(eng, app_id)
    assert row is not None and row.status == "nominated"
    assert row.approved_by in ("faculty_mentor", "tpo_admin")
    with Session(eng) as db:
        evs = db.exec(select(ProgressEvent)
                      .where(ProgressEvent.student_id == sid)).all()
        assert any(e.kind == "nomination" for e in evs)
    assert any(a.action == "approval.approved" and a.entity_id == ap_id
               for a in _audits(eng))


def test_rejection_marks_application_rejected(client_engine):
    """reject() on a linked approval marks the JobApplication rejected."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=8.5)
    _add_mock_interview(eng, sid, "job-001", 75)
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-001"})
    assert r.status_code in (200, 201), r.text
    body = r.json()
    app_id = body["application"]["id"]
    ap_id = body["application"]["approval_id"]
    r2 = client.post(f"/api/approvals/{ap_id}/reject", json={"note": "no"})
    assert r2.status_code == 200, r2.text
    row = _get_app_row(eng, app_id)
    assert row is not None and row.status == "rejected"


def test_duplicate_active_application_conflicts(client_engine):
    """A second application for the same (student, job) while one is active
    returns HTTP 409."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=8.5)
    with Session(eng) as db:
        _make_standard_job(db, job_id="job-dup-1")
    r1 = client.post(f"/api/students/{sid}/applications",
                     json={"job_id": "job-dup-1"})
    assert r1.status_code in (200, 201), r1.text
    r2 = client.post(f"/api/students/{sid}/applications",
                     json={"job_id": "job-dup-1"})
    assert r2.status_code == 409, r2.text


def test_unknown_job_returns_404(client_engine):
    """Missing job id must surface 404, never an unhandled ValueError/500."""
    client, eng = client_engine
    sid = _seed_client_student(client)
    r = client.post(f"/api/students/{sid}/applications",
                    json={"job_id": "job-does-not-exist"})
    assert r.status_code == 404, r.text


def test_student_isolation_on_apply(client_engine):
    """enforce_student_scope: student A cannot apply as student B."""
    client, eng = client_engine
    sid_a = _seed_client_student(client, name="A")
    sid_b = _seed_client_student(client, name="B")
    with Session(eng) as db:
        _make_standard_job(db, job_id="job-iso-1")
    hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_a}
    r = client.post(f"/api/students/{sid_b}/applications",
                    json={"job_id": "job-iso-1"}, headers=hdr)
    assert r.status_code == 403, r.text
    assert _apps_for(eng, sid_b) == []


def test_list_student_applications(client_engine):
    """GET /api/students/{sid}/applications returns rows with job + HITL status."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=8.5)
    with Session(eng) as db:
        _make_standard_job(db, job_id="job-list-1")
    client.post(f"/api/students/{sid}/applications",
                json={"job_id": "job-list-1"})
    r = client.get(f"/api/students/{sid}/applications")
    assert r.status_code == 200, r.text
    items = r.json()
    assert len(items) == 1
    it = items[0]
    assert it["job_id"] == "job-list-1"
    assert it["status"] == "nominated"
    assert it["job_title"] == "Junior Python Developer"
    assert "eligibility" in it


# ------------------------------------------------- 19.3 admin scoping

class _Req:
    """Minimal request stand-in carrying a CurrentUser for scoping tests.

    Exposes `headers`/`cookies` so `get_current_user_optional` resolves the
    declared role + department (X-Demo-Role / X-Demo-Department).
    """

    def __init__(self, user):
        self.headers = {"x-demo-role": user.role}
        if user.department:
            self.headers["x-demo-department"] = user.department
        if user.role == "student" and user.student_id:
            self.headers["x-demo-student-id"] = user.student_id
        self.cookies = {}
        self.state = type("S", (), {})()
        self.state.user = user


def test_admin_applications_department_scoped(client_engine):
    """faculty_mentor sees only their department's applications; tpo_admin sees all."""
    from app.security.rbac import CurrentUser

    client, eng = client_engine
    sid_cse = _seed_client_student(client, name="CSE Kid", branch="CSE")
    sid_ece = _seed_client_student(client, name="ECE Kid", branch="ECE")
    with Session(eng) as db:
        _make_standard_job(db, job_id="job-admin-1")
        _make_standard_job(db, job_id="job-admin-2")
    client.post(f"/api/students/{sid_cse}/applications",
                json={"job_id": "job-admin-1"})
    client.post(f"/api/students/{sid_ece}/applications",
                json={"job_id": "job-admin-2"})

    mentor = CurrentUser(user_id="u1", email="m@jkl.ac.in", name="Mentor",
                         role="faculty_mentor", department="cse")
    req = _Req(mentor)
    rows = list_admin_applications(request=req, session=Session(eng),
                                   role="faculty_mentor")
    sids = {r["student_id"] for r in rows}
    assert sids == {sid_cse}

    tpo = CurrentUser(user_id="u2", email="t@jkl.ac.in", name="TPO",
                      role="tpo_admin")
    rows_all = list_admin_applications(request=_Req(tpo),
                                       session=Session(eng), role="tpo_admin")
    assert {r["student_id"] for r in rows_all} == {sid_cse, sid_ece}


def test_endorse_requires_mentor_or_tpo(client_engine):
    """POST endorse with an explicit student identity is forbidden (403)."""
    client, eng = client_engine
    sid = _seed_client_student(client)
    with Session(eng) as db:
        ap = ApprovalRequest(id=f"appr-{uuid4().hex[:8]}",
                             kind="job_application_nomination",
                             target_id=sid, title="t", summary="s",
                             payload={"student_id": sid, "job_id": "j"},
                             status="pending_review", requested_by="agent")
        db.add(ap)
        db.commit()
        ap_id = ap.id
    hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid}
    r = client.post(f"/api/approvals/{ap_id}/endorse", json={}, headers=hdr)
    assert r.status_code == 403, r.text


def test_separation_of_duties_blocks_faculty_mentor_from_final_approval(client_engine):
    """Faculty mentor can endorse (Tier 1) but cannot final-approve (Tier 2) a drive nomination."""
    client, eng = client_engine
    sid = _seed_client_student(client, gpa=6.0, branch="CSE")
    with Session(eng) as db:
        _make_job(db, job_id="job-sod-1", salary=[20, 25], min_gpa=9.0, branches=["ECE"])
    r = client.post(
        f"/api/students/{sid}/applications",
        json={"job_id": "job-sod-1", "request_waiver": True, "note": "strong projects"},
    )
    assert r.status_code in (200, 201), r.text
    ap_id = r.json()["application"]["approval_id"]

    mentor_hdr = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"}
    # Tier 1 endorsement succeeds for faculty_mentor
    r_end = client.post(f"/api/approvals/{ap_id}/endorse", json={"note": "endorsed"}, headers=mentor_hdr)
    assert r_end.status_code == 200, r_end.text

    # Tier 2 final approval is blocked (403) for faculty_mentor (separation of duties)
    r_blk = client.post(f"/api/approvals/{ap_id}/approve", json={"note": "self-approve"}, headers=mentor_hdr)
    assert r_blk.status_code == 403, r_blk.text

    # Tier 2 final approval succeeds for tpo_admin
    tpo_hdr = {"X-Demo-Role": "tpo_admin"}
    r_ok = client.post(f"/api/approvals/{ap_id}/approve", json={"note": "tpo sign-off"}, headers=tpo_hdr)
    assert r_ok.status_code == 200, r_ok.text


def test_copilot_and_student_delete_emit_audit_logs(client_engine):
    """Verify copilot.asked and student.deleted events are recorded in AuditLog."""
    client, eng = client_engine
    sid = _seed_client_student(client, name="Audit Check", gpa=8.2)
    r_cop = client.post(
        "/api/copilot/ask",
        json={"question": "What is the CGPA cutoff for Dream tier drives?", "student_id": sid},
    )
    assert r_cop.status_code == 200, r_cop.text
    audits = _audits(eng)
    assert any(a.action == "copilot.asked" for a in audits)

    r_del = client.delete(f"/api/students/{sid}", headers={"X-Demo-Role": "tpo_admin"})
    assert r_del.status_code == 200, r_del.text
    audits_after = _audits(eng)
    assert any(a.action == "student.deleted" and a.entity_id == sid for a in audits_after)

