"""Phase 4: PII scrubbing, RBAC, HITL approvals, immutable audit tests."""

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import (
    ApprovalRequest,
    AuditLog,
    JobMatch,
    PlanItem,
    Student,
    StudentSkill,
)
from app.security.pii import detect_pii, redact_pii, scrub_for_llm
from app.security.rbac import ROLES, current_role, normalize_role, require_role, role_allows
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _make_client(cohort: bool = False) -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
        if cohort:
            seed_demo_cohort(db, taxonomy=taxonomy, embedder=ctx.embedder,
                             vector_store=ctx.vector_store, llm=MockLLM())
    return TestClient(create_app(ctx))


def _seed_student(client, name="Sec Probe", branch="CSE", gpa=8.0, resume_extra="") -> str:
    body = ("Skills: Python, FastAPI, PostgreSQL, Docker, Git, Kubernetes\n"
            "Backend internship building REST APIs\n" + resume_extra)
    r = client.post("/api/students",
                    files={"file": ("r.txt", body.encode(), "text/plain")},
                    data={"name": name, "branch": branch, "gpa": str(gpa)})
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------- PII ----------------

def test_pii_detects_all_kinds():
    text = ("Email aarav@example.com phone 9876543210 "
            "Aadhaar 1234 5678 9012 PAN ABCDE1234F "
            "I live at 12 MG Road, Jaipur")
    kinds = {h["kind"] for h in detect_pii(text)}
    assert {"EMAIL", "PHONE", "AADHAAR", "PAN", "ADDRESS"} <= kinds


def test_pii_redaction_replaces_and_counts():
    text = "Reach me at x@y.com or 9876543210"
    out, count, per = redact_pii(text)
    assert count == 2
    assert per["EMAIL"] == 1 and per["PHONE"] == 1
    assert "x@y.com" not in out and "9876543210" not in out
    assert "[REDACTED_EMAIL]" in out and "[REDACTED_PHONE]" in out


def test_pii_leaves_non_pii_text_intact():
    text = "Skills: Python, FastAPI. GPA 8.5 in year 2024, score 65 of 100."
    out, count, _ = redact_pii(text)
    assert count == 0
    assert out == text


def test_pii_aadhaar_beats_phone_pattern():
    out, count, per = redact_pii("Aadhaar 123456789012")
    assert per.get("AADHAAR") == 1 and per.get("PHONE", 0) == 0
    assert "123456789012" not in out


def test_pii_scrub_for_llm_signature():
    out, count = scrub_for_llm("mail me at a@b.co")
    assert count == 1 and "a@b.co" not in out


def test_resume_upload_scrubs_pii_and_counts():
    client = _make_client()
    with client:
        extra = "Contact: 9876543210, aarav@example.com\n"
        sid = _seed_student(client, resume_extra=extra)
        with Session(client.app.state.ctx.engine) as db:
            s = db.get(Student, sid)
            assert s.pii_redacted_count >= 2
            assert "aarav@example.com" not in s.resume_text
            assert "9876543210" not in s.resume_text
            assert "[REDACTED_EMAIL]" in s.resume_text or "[REDACTED_PHONE]" in s.resume_text


def test_resume_reupload_scrubs_pii():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        body = "Skills: Python\nCall 9123456789\n"
        r = client.post(f"/students/{sid}/resume",
                        files={"file": ("r.txt", body.encode(), "text/plain")})
        assert r.status_code in (200, 201), r.text
        with Session(client.app.state.ctx.engine) as db:
            s = db.get(Student, sid)
            assert s.pii_redacted_count >= 1
            assert "9123456789" not in s.resume_text


# ---------------- RBAC ----------------

def test_rbac_normalize_and_implies():
    assert normalize_role("TPO-Admin") == "tpo_admin"
    assert normalize_role("Faculty Mentor") == "faculty_mentor"
    assert normalize_role("hacker") is None
    assert normalize_role("") is None
    assert role_allows("tpo_admin", "student") is True
    assert role_allows("student", "tpo_admin") is False
    assert role_allows("faculty_mentor", "tpo_admin") is False
    assert set(ROLES) == {"student", "faculty_mentor", "tpo_admin"}


def test_rbac_admin_open_by_default_then_enforced():
    client = _make_client()
    with client:
        # no explicit role => open (demo mode) => allowed
        assert client.get("/api/admin/analytics").status_code == 200
        # explicit student role => 403 on admin
        r = client.get("/api/admin/analytics", headers={"X-Demo-Role": "student"})
        assert r.status_code == 403
        # explicit faculty => 200 on analytics (department scoped, M9 §17.2.B.3), 403 on audit
        assert client.get("/api/admin/analytics",
                          headers={"X-Demo-Role": "faculty_mentor"}).status_code == 200
        assert client.get("/api/admin/audit",
                          headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403
        # explicit tpo_admin => 200
        assert client.get("/api/admin/analytics",
                          headers={"X-Demo-Role": "tpo_admin"}).status_code == 200


def test_rbac_admin_page_html_blocked_for_student():
    client = _make_client()
    with client:
        assert client.get("/admin", headers={"X-Demo-Role": "student"}).status_code == 403
        assert client.get("/admin", headers={"X-Demo-Role": "tpo_admin"}).status_code == 200


def test_auth_switch_role_sets_cookie_and_validates():
    client = _make_client()
    with client:
        r = client.post("/api/auth/switch-role", json={"role": "student"})
        assert r.status_code == 200
        assert r.json()["role"] == "student"
        # cookie is now set -> role endpoint reflects it
        assert client.get("/api/auth/role").json()["role"] == "student"
        # invalid role -> 400
        assert client.post("/api/auth/switch-role", json={"role": "admin"}).status_code == 400
        # after switching to tpo_admin, admin analytics allowed again
        client.post("/api/auth/switch-role", json={"role": "tpo_admin"})
        assert client.get("/api/admin/analytics").status_code == 200


def test_approvals_require_faculty_or_tpo_when_role_explicit():
    client = _make_client()
    with client:
        # explicit student role cannot list approvals
        assert client.get("/api/approvals", headers={"X-Demo-Role": "student"}).status_code == 403
        # open (no role) allowed
        assert client.get("/api/approvals").status_code == 200
        # tpo allowed
        assert client.get("/api/approvals", headers={"X-Demo-Role": "tpo_admin"}).status_code == 200


def test_current_role_prefers_header_over_cookie():
    class _Req:
        headers = {"x-demo-role": "faculty_mentor"}
        cookies = {"ps06_role": "student"}
    assert current_role(_Req()) == "faculty_mentor"


def test_require_role_open_when_unset():
    class _Req:
        headers = {}
        cookies = {}
    dep = require_role("tpo_admin")
    # no explicit role => allowed (returns the resolved default)
    assert dep(_Req()) in ROLES


# ---------------- CurrentUser identity model (M9 §17.2.A) ----------------

def test_current_user_model_requires_student_id_for_student_role():
    import pytest

    from app.security.rbac import CurrentUser
    with pytest.raises(ValueError):
        CurrentUser(user_id="u", email="e@x.y", name="N", role="student", student_id=None)
    with pytest.raises(ValueError):
        CurrentUser(user_id="u", email="e@x.y", name="N", role="student", student_id="  ")
    u = CurrentUser(user_id="u", email="e@x.y", name="N", role="student", student_id="STU-1")
    assert u.student_id == "STU-1"
    # non-student roles drop any stray student_id
    f = CurrentUser(user_id="u", email="e@x.y", name="N", role="faculty_mentor",
                    student_id="STU-1", department="CSE")
    assert f.student_id is None and f.department == "CSE"


def test_session_cookie_roundtrip_and_malformed_is_none():
    from app.security.rbac import decode_session, encode_session
    claims = {"role": "student", "user_id": "STU-1", "student_id": "STU-1",
              "department": "CSE"}
    token = encode_session(claims)
    # M15: encode_session now emits a SIGNED token (<payload>.<hmac>) carrying
    # injected iat/exp, so round-trip preserves all original claims (plus the
    # freshness timestamps) instead of byte-identical equality.
    decoded = decode_session(token)
    assert decoded is not None
    assert all(decoded[k] == v for k, v in claims.items())
    assert "iat" in decoded and "exp" in decoded
    assert decode_session(None) is None
    assert decode_session("") is None
    assert decode_session("!!!not-base64!!!") is None
    # base64 of a non-dict JSON -> None
    import base64
    import json
    assert decode_session(base64.urlsafe_b64encode(json.dumps([1, 2]).encode()).decode()) is None


def test_get_current_user_prefers_header_then_session_then_legacy_then_default():
    from app.security.rbac import encode_session, get_current_user, get_current_user_optional

    class _Req:
        def __init__(self, headers=None, cookies=None):
            self.headers = headers or {}
            self.cookies = cookies or {}

    # open demo mode: nothing declared
    assert get_current_user_optional(_Req()) is None
    u = get_current_user(_Req())
    assert u.role == "tpo_admin" and u.student_id is None

    # legacy single-value cookie
    u = get_current_user(_Req(cookies={"ps06_role": "faculty_mentor"}))
    assert u.role == "faculty_mentor"

    # multi-claim session cookie beats legacy
    sess = encode_session({"role": "student", "user_id": "STU-7",
                           "student_id": "STU-7", "department": "CSE",
                           "email": "st@jklu.edu.in", "name": "Aarav"})
    u = get_current_user(_Req(cookies={"ps06_session": sess, "ps06_role": "tpo_admin"}))
    assert u.role == "student" and u.student_id == "STU-7"
    assert u.email == "st@jklu.edu.in" and u.name == "Aarav" and u.department == "CSE"

    # header override beats session for role AND student_id (locked to header identity)
    u = get_current_user(_Req(headers={"x-demo-role": "student",
                                       "x-demo-student-id": "HDR-9"},
                              cookies={"ps06_session": sess}))
    assert u.role == "student" and u.student_id == "HDR-9"
    assert get_current_user_optional(_Req(headers={"x-demo-role": "student"})) is not None

    # student role with no sid claim gets a non-empty locked default
    u = get_current_user(_Req(headers={"x-demo-role": "student"}))
    assert u.student_id and u.student_id.strip()


def test_get_current_user_optional_none_only_in_open_mode():
    from app.security.rbac import get_current_user_optional

    class _Req:
        def __init__(self, headers=None, cookies=None):
            self.headers = headers or {}
            self.cookies = cookies or {}

    assert get_current_user_optional(_Req()) is None
    assert get_current_user_optional(_Req(headers={"x-demo-role": "student"})) is not None
    assert get_current_user_optional(_Req(cookies={"ps06_role": "tpo_admin"})) is not None


def test_switch_role_endpoint_sets_multi_claim_session_and_reports_user(client):
    r = client.post("/api/auth/switch-role",
                    json={"role": "student", "student_id": "STU-2022-001",
                          "department": "CSE"})
    assert r.status_code == 200
    assert r.json()["role"] == "student"
    # both cookies present
    from app.security.rbac import decode_session
    sess = client.cookies.get("ps06_session")
    assert sess, "ps06_session cookie not set"
    claims = decode_session(sess)
    assert claims["role"] == "student"
    assert claims["student_id"] == "STU-2022-001"
    assert claims["department"] == "CSE"
    assert client.cookies.get("ps06_role") == "student"   # legacy cookie kept
    # role endpoint exposes the resolved CurrentUser
    body = client.get("/api/auth/role").json()
    assert body["role"] == "student"
    assert body["user"]["student_id"] == "STU-2022-001"
    assert body["user"]["department"] == "CSE"
    # student role without explicit sid still yields a locked non-empty sid
    client.post("/api/auth/switch-role", json={"role": "student"})
    body = client.get("/api/auth/role").json()
    assert body["user"]["student_id"]
    # invalid role still 400
    assert client.post("/api/auth/switch-role", json={"role": "admin"}).status_code == 400


# ---------------- HITL approvals ----------------

def _create_approval(client, kind, target_id, payload=None, title="t", summary="s"):
    with Session(client.app.state.ctx.engine) as db:
        from uuid import uuid4
        appr_id = f"appr-{uuid4().hex[:8]}"
        db.add(ApprovalRequest(id=appr_id, kind=kind, target_id=target_id, title=title,
                               summary=summary, payload=payload or {},
                               status="pending_review", requested_by="agent"))
        db.commit()
        return appr_id


def test_hitl_tool_persists_real_approval_request():
    client = _make_client()
    with client:
        from app.agent.tools import call_tool
        with Session(client.app.state.ctx.engine) as session:
            out = call_tool("propose_hitl_approval", client.app.state.ctx, session,
                            kind="job_application_nomination", target_id="job-001",
                            title="Nominate", summary="for drive",
                            payload={"student_id": "x"})
            session.commit()
        with Session(client.app.state.ctx.engine) as session:
            ap = session.get(ApprovalRequest, out["approval_id"])
            assert ap is not None
            assert ap.status == "pending_review"
            assert ap.kind == "job_application_nomination"
            assert ap.target_id == "job-001"


def test_approval_nomination_approval_writes_progress_event_and_audit():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        appr_id = _create_approval(client, "job_application_nomination", "job-001",
                                   payload={"student_id": sid, "job_id": "job-001",
                                            "readiness": 42})
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "go"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "approved"
        assert body["reviewed_by"] in ROLES
        with Session(client.app.state.ctx.engine) as session:
            evs = session.exec(select(ProgressEvent if False else __import__("app.models", fromlist=["ProgressEvent"]).ProgressEvent)
                               .where(__import__("app.models", fromlist=["ProgressEvent"]).ProgressEvent.student_id == sid)).all()
            assert any(e.kind == "nomination" for e in evs)
            auds = session.exec(select(AuditLog)).all()
            assert any(a.action == "approval.approved" and a.entity_id == appr_id for a in auds)


def test_approval_proficiency_waiver_bumps_skill_and_recomputes_match():
    client = _make_client()
    with client:
        sid = _seed_student(client, resume_extra="Skills: Docker\n")
        with Session(client.app.state.ctx.engine) as session:
            # ensure a StudentSkill row for python exists (from upload)
            ss = session.exec(select(StudentSkill).where(
                StudentSkill.student_id == sid)).first()
            assert ss is not None
            skill_id = ss.skill_id
        appr_id = _create_approval(client, "proficiency_waiver", skill_id,
                                   payload={"student_id": sid, "skill_id": skill_id,
                                            "level": "advanced"})
        # ensure matches exist first
        client.get(f"/api/students/{sid}/matches")
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "waiver granted"})
        assert r.status_code == 200, r.text
        with Session(client.app.state.ctx.engine) as session:
            ss2 = session.exec(select(StudentSkill).where(
                StudentSkill.student_id == sid, StudentSkill.skill_id == skill_id)).first()
            assert ss2 is not None and ss2.level == "advanced"
            after = session.exec(select(JobMatch).where(
                JobMatch.student_id == sid)).all()
            assert len(after) > 0
        assert r.json()["executed"]["new_level"] == "advanced"


def test_approval_cohort_bootcamp_publish_updates_plan_items():
    client = _make_client(cohort=True)
    with client:
        # pick a pending PlanItem from a seeded cohort student
        with Session(client.app.state.ctx.engine) as session:
            item = session.exec(select(PlanItem).where(PlanItem.status == "pending")).first()
            assert item is not None, "cohort seeding should create pending plan items"
            skill_id = item.skill_id
        appr_id = _create_approval(client, "cohort_bootcamp_publish", skill_id,
                                   payload={"skill_id": skill_id, "course_id": "course-011"})
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "publish"})
        assert r.status_code == 200, r.text
        with Session(client.app.state.ctx.engine) as session:
            updated = session.exec(select(PlanItem).where(PlanItem.skill_id == skill_id,
                                                          PlanItem.status == "published")).all()
            assert updated, "approved bootcamp should publish matching plan items"
            assert all(i.course_id == "course-011" for i in updated)


def test_approval_reject_is_recorded_and_audited():
    client = _make_client()
    with client:
        appr_id = _create_approval(client, "nomination", "job-002",
                                   payload={"student_id": "x"})
        r = client.post(f"/api/approvals/{appr_id}/reject", json={"note": "not now"})
        assert r.status_code == 200
        assert r.json()["status"] == "rejected"
        with Session(client.app.state.ctx.engine) as session:
            auds = session.exec(select(AuditLog)).all()
            assert any(a.action == "approval.rejected" and a.entity_id == appr_id for a in auds)


def test_approval_execute_failure_still_audited_and_no_500():
    client = _make_client()
    with client:
        # proficiency waiver for a non-existent student -> handler 404 but approve returns 200
        appr_id = _create_approval(client, "proficiency_waiver", "python",
                                   payload={"student_id": "stu-none", "skill_id": "python",
                                            "level": "advanced"})
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "try"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "approved"
        assert body.get("error")           # failure surfaced, not swallowed
        with Session(client.app.state.ctx.engine) as session:
            auds = session.exec(select(AuditLog)).all()
            assert any(a.action == "approval.approved" and a.entity_id == appr_id
                       and "error" in a.detail for a in auds)


def test_approve_idempotent_and_conflicts():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        appr_id = _create_approval(client, "job_application_nomination", "job-001",
                                   payload={"student_id": sid, "job_id": "job-001",
                                            "readiness": 1})
        assert client.post(f"/api/approvals/{appr_id}/approve", json={}).status_code == 200
        # re-approve -> 200 idempotent, no second audit row
        assert client.post(f"/api/approvals/{appr_id}/approve", json={}).status_code == 200
        with Session(client.app.state.ctx.engine) as session:
            n = len([a for a in session.exec(select(AuditLog)).all()
                     if a.entity_id == appr_id and a.action == "approval.approved"])
            assert n == 1
        # approve after reject -> 409
        appr2 = _create_approval(client, "nomination", "job-003",
                                 payload={"student_id": sid})
        client.post(f"/api/approvals/{appr2}/reject", json={})
        assert client.post(f"/api/approvals/{appr2}/approve", json={}).status_code == 409


def test_unknown_approval_404():
    client = _make_client()
    with client:
        assert client.post("/api/approvals/appr-none/approve", json={}).status_code == 404
        assert client.post("/api/approvals/appr-none/reject", json={}).status_code == 404


def test_agent_nomination_approval_payload_binds_student_and_executes():
    """Gate C: agent-proposed nomination must carry student_id so the mutation runs."""
    client = _make_client()
    with client:
        from app.agent.orchestrator import run_agent
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            run = run_agent(client.app.state.ctx, session,
                            goal="Nominate this student for the placement drive",
                            student_id=sid, actor_role="tpo_admin")
            session.commit()
            from app.models import AgentStep
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            hitl = next(s for s in steps if s.tool_name == "propose_hitl_approval")
            appr_id = hitl.tool_output["approval_id"]
            ap = session.get(ApprovalRequest, appr_id)
            # payload must bind the student (not fall back to a job- id)
            assert ap.payload.get("student_id") == sid
        # approve must execute the real ProgressEvent mutation with no error
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "go"})
        assert r.status_code == 200
        assert "error" not in r.json()
        with Session(client.app.state.ctx.engine) as session:
            from app.models import ProgressEvent
            evs = session.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)).all()
            assert any(e.kind == "nomination" for e in evs)


def test_agent_cohort_bootcamp_publish_resolves_course_and_publishes():
    """Gate C: cohort_bootcamp_publish without course_id must resolve from catalog."""
    client = _make_client(cohort=True)
    with client:
        with Session(client.app.state.ctx.engine) as session:
            item = session.exec(select(PlanItem).where(PlanItem.status == "pending")).first()
            assert item is not None
            skill_id = item.skill_id
        # payload deliberately omits course_id -> handler must resolve it
        appr_id = _create_approval(client, "cohort_bootcamp_publish", skill_id,
                                   payload={"skill_id": skill_id})
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "publish"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "error" not in body
        # course resolved from catalog (a real course covering the skill)
        assert body["executed"]["course_id"], "course_id must be resolved"
        with Session(client.app.state.ctx.engine) as session:
            published = session.exec(select(PlanItem).where(
                PlanItem.skill_id == skill_id, PlanItem.status == "published")).all()
            assert published


def test_agent_approval_run_id_linked_for_provenance():
    """Gate C: approvals proposed by an agent run must carry run_id."""
    client = _make_client()
    with client:
        from app.agent.orchestrator import run_agent
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            run = run_agent(client.app.state.ctx, session,
                            goal="Nominate this student for the placement drive",
                            student_id=sid, actor_role="tpo_admin")
            session.commit()
            from app.models import AgentStep
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            hitl = next(s for s in steps if s.tool_name == "propose_hitl_approval")
            ap = session.get(ApprovalRequest, hitl.tool_output["approval_id"])
            assert ap is not None
            assert ap.run_id == run.id, "approval must link back to its agent run"


def test_nomination_falls_back_to_run_student_when_payload_empty():
    """Gate C: even with empty payload, student_id resolves via approval.run_id."""
    client = _make_client()
    with client:
        from app.agent.orchestrator import run_agent
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            run = run_agent(client.app.state.ctx, session,
                            goal="Nominate this student for the placement drive",
                            student_id=sid, actor_role="tpo_admin")
            session.commit()
            from app.models import AgentStep
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            hitl = next(s for s in steps if s.tool_name == "propose_hitl_approval")
            appr_id = hitl.tool_output["approval_id"]
            # wipe payload to simulate a legacy/agent approval without bindings
            ap = session.get(ApprovalRequest, appr_id)
            ap.payload = {}
            ap.target_id = "job-001"   # job id, not a student id
            session.add(ap)
            session.commit()
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "go"})
        assert r.status_code == 200
        assert "error" not in r.json()
        with Session(client.app.state.ctx.engine) as session:
            from app.models import ProgressEvent
            evs = session.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)).all()
            assert any(e.kind == "nomination" for e in evs)


def test_student_profile_exposes_pii_redacted_count():
    client = _make_client()
    with client:
        sid = _seed_student(client, resume_extra="Email leak@example.com\n")
        r = client.get(f"/api/students/{sid}")
        assert r.status_code == 200
        assert r.json()["pii_redacted_count"] >= 1


# ---------------- audit immutability + endpoint ----------------

def test_audit_log_rows_are_immutable():
    client = _make_client()
    with client:
        with Session(client.app.state.ctx.engine) as session:
            a = AuditLog(id="aud-imm001", action="test.append", entity_type="t",
                         entity_id="e", actor_role="tpo_admin", detail={"n": 1})
            session.add(a)
            session.commit()
            # UPDATE must raise
            try:
                a.action = "mutated"
                session.add(a)
                session.commit()
                raise AssertionError("AuditLog update was allowed")
            except RuntimeError:
                session.rollback()
            # DELETE must raise
            try:
                session.delete(a)
                session.commit()
                raise AssertionError("AuditLog delete was allowed")
            except RuntimeError:
                session.rollback()
            # append still works
            session.add(AuditLog(id="aud-imm002", action="test.append2", entity_type="t",
                                 entity_id="e", actor_role="tpo_admin", detail={}))
            session.commit()
            n = len(session.exec(select(AuditLog).where(AuditLog.id.in_(
                ["aud-imm001", "aud-imm002"]))).all())
            assert n == 2


def test_audit_endpoint_lists_rows_readonly():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        appr_id = _create_approval(client, "nomination", "job-001",
                                   payload={"student_id": sid, "job_id": "job-001",
                                            "readiness": 0})
        client.post(f"/api/approvals/{appr_id}/approve", json={"note": "x"})
        rows = client.get("/api/admin/audit").json()
        assert rows
        assert {"id", "action", "entity_type", "entity_id", "actor_role",
                "detail", "created_at"} <= set(rows[0].keys())
        # there is no DELETE endpoint for audit (append-only surface)
        assert client.delete(f"/api/admin/audit/{rows[0]['id']}").status_code in (404, 405)


def test_agent_run_nomination_tool_creates_pending_approval():
    """Agent propose_hitl_approval now persists a real pending ApprovalRequest."""
    client = _make_client()
    with client:
        from app.agent.orchestrator import run_agent
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            run = run_agent(client.app.state.ctx, session,
                            goal="Nominate this student for the placement drive",
                            student_id=sid, actor_role="tpo_admin")
            session.commit()
            steps = session.exec(select(__import__("app.models", fromlist=["AgentStep"]).AgentStep)
                                 .where(__import__("app.models", fromlist=["AgentStep"]).AgentStep.run_id == run.id)).all()
            hitl = next((s for s in steps if s.tool_name == "propose_hitl_approval"), None)
            if hitl is not None:
                ap_id = hitl.tool_output.get("approval_id")
                ap = session.get(ApprovalRequest, ap_id)
                assert ap is not None, "HITL tool must persist a real ApprovalRequest"
                assert ap.status == "pending_review"
            assert run.status == "awaiting_approval"
