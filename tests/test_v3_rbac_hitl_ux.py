"""Tests for JKLU CareerOS V3 RBAC, HITL resume lock, on-campus dismiss block, cohort filters, and audit PII privacy."""
from __future__ import annotations

import json
import re
from contextlib import contextmanager
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.api.main import AppContext, create_app
from app.config import settings
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import ApprovalRequest, AuditLog, Job, JobMatch, Student
from app.security.pii import audit_actor_ref, mask_email, sanitize_audit_entry, sanitize_audit_text
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


def _make_client() -> TestClient:
    engine = make_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _fk(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    init_db(engine)
    tax = load_taxonomy()
    ctx = AppContext(
        engine=engine,
        taxonomy=tax,
        embedder=HashingEmbedder(dim=64),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    with Session(engine) as session:
        seed(session, tax)
    return TestClient(create_app(ctx))


@contextmanager
def _auth_mode(mode: str, demo_switch: bool):
    old_mode = settings.auth_mode
    old_sw = settings.demo_role_switch
    settings.auth_mode = mode
    settings.demo_role_switch = demo_switch
    try:
        yield
    finally:
        settings.auth_mode = old_mode
        settings.demo_role_switch = old_sw


def _seed_student(client: TestClient, name: str = "Aarav Verma", branch: str = "CSE", gpa: float = 8.4) -> str:
    resp = client.post(
        "/api/students",
        headers={"X-Demo-Role": "tpo_admin"},
        files={"file": ("resume.txt", b"Skills: Python, FastAPI, Docker\n", "text/plain")},
        data={"name": name, "branch": branch, "year": "3", "gpa": str(gpa)},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_resume_reupload_hitl_gate_single_use_and_audit():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        stu_headers = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid}
        fac_headers = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"}
        tpo_headers = {"X-Demo-Role": "tpo_admin"}

        # 1. Student with existing resume is blocked on BOTH API and HTMX re-upload endpoints
        r_api_blk = client.post(
            f"/api/students/{sid}/resume",
            headers=stu_headers,
            files={"file": ("new.txt", b"Skills: Python, Rust\n", "text/plain")},
        )
        assert r_api_blk.status_code == 403, r_api_blk.text

        r_htmx_blk = client.post(
            f"/students/{sid}/resume",
            headers={**stu_headers, "HX-Request": "true"},
            files={"file": ("new.txt", b"Skills: Python, Rust\n", "text/plain")},
        )
        assert r_htmx_blk.status_code == 403
        assert "TPO Admin permission" in r_htmx_blk.text

        # 2. Student submits approval request; duplicate request returns 409
        r_req = client.post(
            f"/api/students/{sid}/resume-reupload-request",
            headers=stu_headers,
            json={"reason": "Added summer internship and AWS cert"},
        )
        assert r_req.status_code == 201, r_req.text
        apr_id = r_req.json()["id"]
        assert r_req.json()["kind"] == "resume_reupload_request"

        r_dup = client.post(
            f"/api/students/{sid}/resume-reupload-request",
            headers=stu_headers,
            json={"reason": "Duplicate click"},
        )
        assert r_dup.status_code == 409

        # 3. Profile page receives pending status cleanly (reupload_unlocked=False initially)
        r_prof = client.get(f"/students/{sid}", headers=stu_headers)
        assert r_prof.status_code == 200

        # 4. Faculty mentor cannot approve or reject resume_reupload_request (403)
        assert client.post(f"/api/approvals/{apr_id}/approve", headers=fac_headers, json={"note": "no"}).status_code == 403
        assert client.post(f"/api/approvals/{apr_id}/reject", headers=fac_headers, json={"note": "no"}).status_code == 403

        # 5. TPO Admin approves -> Student.resume_reupload_unlocked becomes True
        r_appr = client.post(f"/api/approvals/{apr_id}/approve", headers=tpo_headers, json={"note": "Verified update"})
        assert r_appr.status_code == 200, r_appr.text
        assert r_appr.json()["executed"]["mutated"] == "Student.resume_reupload_unlocked=True"

        with Session(client.app.state.ctx.engine) as db:
            st = db.get(Student, sid)
            assert st is not None and st.resume_reupload_unlocked is True

        # 6. Student succeeds once, unlock is consumed immediately, and second attempt is blocked (403)
        r_ok = client.post(
            f"/api/students/{sid}/resume",
            headers=stu_headers,
            files={"file": ("v2.txt", b"Skills: Python, AWS, Kubernetes\n", "text/plain")},
        )
        assert r_ok.status_code == 200, r_ok.text

        with Session(client.app.state.ctx.engine) as db:
            st = db.get(Student, sid)
            assert st is not None and st.resume_reupload_unlocked is False
            actions = {a.action for a in db.exec(select(AuditLog).where(AuditLog.entity_id == sid)).all()}
            assert "student.resume_reupload_requested" in actions
            assert "student.resume_reupload_consumed" in actions

        r_second = client.post(
            f"/api/students/{sid}/resume",
            headers=stu_headers,
            files={"file": ("v3.txt", b"Skills: Python, Go\n", "text/plain")},
        )
        assert r_second.status_code == 403

        # 7. TPO Admin can always re-upload without unlock
        r_tpo = client.post(
            f"/api/students/{sid}/resume",
            headers=tpo_headers,
            files={"file": ("admin.txt", b"Skills: Python, Docker, Go\n", "text/plain")},
        )
        assert r_tpo.status_code == 200


def test_student_cannot_dismiss_on_campus_jobs_or_see_dismiss_button():
    client = _make_client()
    with client:
        sid = _seed_student(client, name="Riya Campus", branch="CSE", gpa=8.6)
        stu_headers = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid}
        tpo_headers = {"X-Demo-Role": "tpo_admin"}

        on_campus_jid = f"job-{uuid4().hex[:8]}"
        live_jid = f"job-ext-{uuid4().hex[:8]}"
        with Session(client.app.state.ctx.engine) as db:
            db.add(
                Job(
                    id=on_campus_jid,
                    title="Campus SDE Drive",
                    company="JKLU Partner Tech",
                    description="Python FastAPI",
                    required_skills=["python"],
                    salary_lpa=[10, 12],
                    source="local",
                )
            )
            db.add(
                Job(
                    id=live_jid,
                    title="External Backend Role",
                    company="External Corp",
                    description="Python Docker",
                    required_skills=["python"],
                    salary_lpa=[8, 10],
                    source="live",
                    is_live=True,
                    owner_student_id=sid,
                )
            )
            db.commit()
            db.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=on_campus_jid, score=88, band="strong", breakdown={}))
            db.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=live_jid, score=76, band="good", breakdown={}))
            db.commit()

        # Student cannot dismiss on-campus job via any API endpoint
        assert client.delete(f"/api/students/{sid}/matches/{on_campus_jid}", headers=stu_headers).status_code == 403
        assert client.post(f"/api/students/{sid}/matches/{on_campus_jid}/dismiss", headers=stu_headers).status_code == 403
        assert client.delete(f"/api/jobs/{on_campus_jid}", headers=stu_headers).status_code == 403

        # Student CAN still dismiss live external job
        assert client.delete(f"/api/students/{sid}/matches/{live_jid}", headers=stu_headers).status_code == 200

        # Restore live match for UI button check
        with Session(client.app.state.ctx.engine) as db:
            m_live = db.exec(select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == live_jid)).first()
            assert m_live is not None
            m_live.band = "good"
            m_live.score = 76
            m_live.breakdown = {}
            db.add(m_live)
            db.commit()

        # HTML /matches for student hides Dismiss button on On-Campus card, shows it ONLY on Live card
        html_stu = client.get(f"/students/{sid}/matches", headers=stu_headers).text
        assert f'data-job-remove="{on_campus_jid}"' not in html_stu
        assert f'data-job-remove="{live_jid}"' in html_stu

        # TPO Admin retains Remove button on On-Campus card and can delete
        html_tpo = client.get(f"/students/{sid}/matches", headers=tpo_headers).text
        assert f'data-job-remove="{on_campus_jid}"' in html_tpo
        assert client.delete(f"/api/jobs/{on_campus_jid}", headers=tpo_headers).status_code == 200


def test_cohort_alphabetical_order_branch_normalization_and_filters():
    client = _make_client()
    with client:
        s_zoya = _seed_student(client, name="Zoya Khan", branch="ECE", gpa=6.5)
        s_aarav = _seed_student(client, name="Aarav Patel", branch="Computer Science & Engineering", gpa=8.3)
        s_meera = _seed_student(client, name="Meera Nair", branch="AI & Data Science", gpa=9.1)

        tpo_headers = {"X-Demo-Role": "tpo_admin"}
        r_home = client.get("/", headers=tpo_headers)
        assert r_home.status_code == 200
        html = r_home.text

        # Dynamic taxonomy count + filter toolbar + canonical onboarding branch select
        expected_tax_count = len(load_taxonomy().all())
        assert f"Taxonomy: {expected_tax_count} skills" in html
        assert 'id="cohort-search"' in html
        assert 'id="cohort-branch-filter"' in html
        assert 'id="cohort-gpa-filter"' in html
        assert 'id="cohort-reset-btn"' in html
        assert 'id="upload-branch-select"' in html

        # Alphabetical ordering on /
        pos_aarav = html.find("Aarav Patel")
        pos_meera = html.find("Meera Nair")
        pos_zoya = html.find("Zoya Khan")
        assert 0 < pos_aarav < pos_meera < pos_zoya

        # API GET /api/students sorted alphabetically + filters (q, branch alias, min_gpa)
        all_rows = client.get("/api/students", headers=tpo_headers).json()
        names_lower = [r["name"].lower() for r in all_rows]
        assert names_lower == sorted(names_lower)

        cse_rows = client.get("/api/students?branch=Computer+Science", headers=tpo_headers).json()
        cse_ids = {r["id"]: r for r in cse_rows}
        assert s_aarav in cse_ids
        assert cse_ids[s_aarav]["branch"] == "CSE"
        assert s_zoya not in cse_ids

        gpa_rows = client.get("/api/students?min_gpa=8.0", headers=tpo_headers).json()
        gpa_ids = {r["id"] for r in gpa_rows}
        assert {s_aarav, s_meera} <= gpa_ids
        assert s_zoya not in gpa_ids

        q_rows = client.get("/api/students?q=meera", headers=tpo_headers).json()
        assert any(r["id"] == s_meera for r in q_rows)
        assert all("meera" in f"{r['name']} {r['id']} {r['branch']}".lower() for r in q_rows)

        # Faculty mentor scoped to normalized department; student sees only self
        fac_rows = client.get(
            "/api/students",
            headers={"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "Computer Science"},
        ).json()
        assert all(r["branch"] == "CSE" for r in fac_rows)

        stu_rows = client.get(
            "/api/students",
            headers={"X-Demo-Role": "student", "X-Demo-Student-ID": s_meera},
        ).json()
        assert len(stu_rows) == 1 and stu_rows[0]["id"] == s_meera

        # Unused ApprovalRequest import guard for ruff
        assert ApprovalRequest is not None


def test_audit_trail_pii_redaction_and_session_toggle():
    masked_jklu = mask_email("aarav.sharma@jklu.edu.in")
    masked_ext = mask_email("secret.user@gmail.com")
    assert "@" not in masked_jklu and "jklu.edu.in" not in masked_jklu
    assert "@" not in masked_ext and "gmail.com" not in masked_ext
    assert re.match(r"^id-[0-9a-f]{8}$", masked_jklu)
    assert re.match(r"^id-[0-9a-f]{8}$", masked_ext)
    ref1 = audit_actor_ref("secret.user@gmail.com")
    ref2 = audit_actor_ref("secret.user@gmail.com")
    assert re.match(r"^id-[0-9a-f]{8}$", ref1) and ref1 == ref2

    scrubbed = sanitize_audit_text("Attempt by hacker.secret@gmail.com, aarav@jklu.edu.in, t***@jklu.edu.in, and s***@[external-domain]")
    assert "@" not in scrubbed and "gmail.com" not in scrubbed and "jklu.edu.in" not in scrubbed
    ent, det = sanitize_audit_entry(
        "user:rohit@jklu.edu.in",
        {"actor_ref": "r***@jklu.edu.in", "nested": [{"mail": "other@gmail.com"}]},
    )
    assert "@" not in ent
    assert "@" not in json.dumps(det)

    client = _make_client()
    with client, _auth_mode("strict", False):
        # 1. Non-JKLU email failure must NEVER leak raw email in AuditLog
        r1 = client.post("/auth/login", json={"email": "hacker.secret@gmail.com", "pin": "2026"})
        assert r1.status_code == 403

        # 2. Invalid PIN on registered account must NEVER leak raw email in AuditLog
        r2 = client.post("/auth/login", json={"email": "rohit.shetty@jklu.edu.in", "pin": "0000"})
        assert r2.status_code == 401

        # 3. Valid login + logout must NEVER leak raw email in AuditLog
        r3 = client.post("/auth/login", json={"email": "rohit.shetty@jklu.edu.in", "pin": "2026"})
        assert r3.status_code == 200
        client.post("/auth/logout", follow_redirects=False)

        # 4. Login as TPO Admin to inspect /admin/audit HTML UI
        r4 = client.post("/auth/login", json={"email": "tpo.admin@jklu.edu.in", "pin": "2026"})
        assert r4.status_code == 200
        audit_html = client.get("/admin/audit").text
        assert 'id="audit-include-sessions"' in audit_html
        assert "Include routine auth sessions" in audit_html
        assert "auth.role_switched" in audit_html
        assert "demo_role_switch_disabled" in audit_html

        with Session(client.app.state.ctx.engine) as db:
            logs = db.exec(select(AuditLog).where(AuditLog.action.in_(["auth.denied", "auth.login", "auth.logout"]))).all()
            assert len(logs) >= 4
            for row in logs:
                blob = json.dumps(row.detail or {}) + " " + (row.entity_id or "")
                assert "@" not in blob
                assert "hacker.secret@gmail.com" not in blob
                assert "rohit.shetty@jklu.edu.in" not in blob
                assert "tpo.admin@jklu.edu.in" not in blob
                assert "email" not in (row.detail or {})


def test_hitl_approvals_render_human_readable_context_without_raw_ids():
    client = _make_client()
    with client:
        sid = _seed_student(client, name="Riya Context", branch="CSE", gpa=8.6)
        with Session(client.app.state.ctx.engine) as db:
            st = db.get(Student, sid)
            job = db.exec(select(Job)).first()
            assert st is not None and job is not None
            jid = job.id
            st_name, j_title, j_comp = st.name, job.title, job.company

        tpo_headers = {"X-Demo-Role": "tpo_admin"}
        r_cr = client.post(
            "/api/approvals",
            headers=tpo_headers,
            json={
                "kind": "job_application_nomination",
                "target_id": sid,
                "title": "Human Context Test",
                "payload": {"student_id": sid, "job_id": jid, "skill_id": "system_design"},
            },
        )
        assert r_cr.status_code == 201, r_cr.text
        created = r_cr.json()
        assert created["payload"]["student_name"] == st_name
        assert created["payload"]["job_title"] == j_title
        assert created["payload"]["job_company"] == j_comp

        page_html = client.get("/admin/approvals", headers=tpo_headers).text
        assert "Target Job:" not in page_html
        assert "Target Candidate:" not in page_html
        assert "a.payload.student_name" in page_html
        assert "a.payload.job_title" in page_html
