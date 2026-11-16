"""Regression tests for security, auth/RBAC, DB concurrency, policy RAG, parser, and API validation fixes."""
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.pool import QueuePool
from sqlmodel import Session, select

from app.config import Settings, settings
from app.models import ApprovalRequest, AuditLog, Job, JobApplication, Student, StudentSkill


def _sign_payload(payload: dict, secret: str) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    payload_b64 = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def test_strict_mode_defaults_and_startup_secret_validation(monkeypatch, tmp_path):
    s_strict = Settings(auth_mode="strict", demo_role_switch=True, _env_file=None)
    assert s_strict.demo_role_switch is False

    s_strict_ws = Settings(auth_mode="  STRICT ", _env_file=None)
    assert s_strict_ws.demo_role_switch is False

    s_demo = Settings(auth_mode="demo", _env_file=None)
    assert s_demo.demo_role_switch is True

    from app.api.main import AppContext, create_app
    from app.db import init_db, make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    from app.llm.mock import MockLLM
    from app.taxonomy.taxonomy import load_taxonomy

    eng = make_engine(f"sqlite:///{tmp_path / 'strict_check.db'}")
    init_db(eng)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(dim=128),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )

    monkeypatch.setattr(settings, "auth_mode", "strict")
    monkeypatch.setattr(settings, "session_secret_key", Settings().session_secret_key)
    with pytest.raises(RuntimeError, match=r"(?i)session_secret_key"):
        create_app(ctx)

    monkeypatch.setattr(settings, "session_secret_key", "too-short-secret")
    with pytest.raises(RuntimeError, match=r"(?i)session_secret_key"):
        create_app(ctx)

    monkeypatch.setattr(settings, "session_secret_key", "a" * 32)
    app = create_app(ctx)
    assert app is not None


def test_auth_login_persona_lock_fast_login_block_and_logout_jti_revocation(client_engine, monkeypatch):
    client, _eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")
    monkeypatch.setattr(settings, "session_secret_key", "test-strict-secret-key-at-least-32-chars")

    # fast_login must be blocked unconditionally in strict mode
    r_fast = client.post(
        "/auth/login",
        json={"email": "rohit.shetty@jklu.edu.in", "fast_login": True},
    )
    assert r_fast.status_code == 401

    # Known persona locks role/department/name against body spoofing
    r_login = client.post(
        "/auth/login",
        json={
            "email": "rohit.shetty@jklu.edu.in",
            "pin": "2026",
            "role": "tpo_admin",
            "department": "ECE",
            "name": "Spoofed Admin",
        },
    )
    assert r_login.status_code == 200, r_login.text
    claims_resp = r_login.json()["claims"]
    assert claims_resp["role"] == "student"
    assert claims_resp["department"] == "CSE"
    assert claims_resp["name"] == "Rohit Shetty"

    set_cookies = r_login.headers.get_list("set-cookie")
    assert any("Max-Age=" in c for c in set_cookies)
    session_tok = client.cookies.get("ps06_session")
    assert session_tok

    from app.security.rbac import decode_session

    claims = decode_session(session_tok)
    assert claims is not None
    assert claims.get("jti")
    assert isinstance(claims.get("exp"), int)

    # Logout revokes the token's jti immediately
    r_out = client.post("/auth/logout")
    assert r_out.status_code == 200
    assert decode_session(session_tok) is None

    # Replaying the revoked cookie in strict mode must fail with 401
    r_replay = client.get(
        "/api/students",
        headers={"Cookie": f"ps06_session={session_tok}"},
    )
    assert r_replay.status_code == 401


def test_decode_session_enforces_numeric_exp_and_case_insensitive_strict(monkeypatch):
    from app.security.rbac import decode_session, encode_session

    secret = "s" * 32
    monkeypatch.setattr(settings, "session_secret_key", secret)
    tok = encode_session({"role": "student"})
    claims = decode_session(tok)
    assert claims is not None and isinstance(claims["exp"], int) and claims.get("jti")

    # Token missing exp must be rejected even with valid HMAC
    assert decode_session(_sign_payload({"role": "tpo_admin"}, secret)) is None

    # Boolean or string exp must also be rejected
    assert decode_session(_sign_payload({"role": "tpo_admin", "exp": True}, secret)) is None
    assert decode_session(_sign_payload({"role": "tpo_admin", "exp": "9999999999"}, secret)) is None

    # Case-insensitive strict mode rejects unsigned payloads
    monkeypatch.setattr(settings, "auth_mode", "  STRICT ")
    raw = json.dumps({"role": "tpo_admin", "exp": 9999999999}).encode("utf-8")
    unsigned = base64.urlsafe_b64encode(raw).decode("ascii")
    assert decode_session(unsigned) is None


def test_faculty_mentor_fail_closed_exact_role_and_copilot_drive_scoping(client_engine):
    client, eng = client_engine
    with Session(eng) as s:
        cse_sid = f"stu-cse-{uuid4().hex[:8]}"
        ece_sid = f"stu-ece-{uuid4().hex[:8]}"
        blank_sid = f"stu-non-{uuid4().hex[:8]}"
        jid = f"job-{uuid4().hex[:8]}"
        s.add(Student(id=cse_sid, name="Asha CSE", branch="CSE", gpa=8.5, resume_text="Python SQL"))
        s.add(Student(id=ece_sid, name="Vikram ECE", branch="ECE", gpa=8.2, resume_text="VLSI Embedded"))
        s.add(Student(id=blank_sid, name="No Branch", branch="", gpa=8.0, resume_text="Java"))
        s.add(Job(id=jid, title="Backend SDE", company="Acme", description="Python backend", required_skills=["python"], salary_lpa=[10, 14], source="local"))
        s.commit()

    # Faculty mentor without department fails closed
    no_dept = {"X-Demo-Role": "faculty_mentor"}
    assert client.get(f"/api/students/{cse_sid}", headers=no_dept).status_code == 403
    assert client.get("/api/students", headers=no_dept).json() == []
    assert client.get("/api/approvals", headers=no_dept).json() == []

    # Branch alias normalization ("Computer Science" == "CSE") allows CSE, blocks ECE & empty branch
    cse_alias = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "Computer Science"}
    assert client.get(f"/api/students/{cse_sid}", headers=cse_alias).status_code == 200
    assert client.get(f"/api/students/{ece_sid}", headers=cse_alias).status_code == 403
    assert client.get(f"/api/students/{blank_sid}", headers=cse_alias).status_code == 403

    # Exact-role endpoints must reject faculty_mentor even on same-department student
    assert client.post(
        f"/api/students/{cse_sid}/matches/{jid}/favorite",
        headers=cse_alias,
        json={"is_favorite": True},
    ).status_code == 403
    assert client.post(
        f"/api/students/{cse_sid}/scrape-jobs",
        headers=cse_alias,
        json={"term": "python"},
    ).status_code == 403
    assert client.post(
        f"/api/students/{cse_sid}/scrape-jobs/stream",
        headers=cse_alias,
        json={"term": "python"},
    ).status_code == 403

    # Copilot ask & stream enforce student/faculty scope
    assert client.post(
        "/api/copilot/ask",
        headers=cse_alias,
        json={"question": "Summarize readiness", "student_id": ece_sid},
    ).status_code == 403
    assert client.post(
        "/api/copilot/ask/stream",
        headers=cse_alias,
        json={"question": "Summarize readiness", "student_id": ece_sid},
    ).status_code == 403

    # Streaming for allowed CSE student succeeds cleanly without cross-thread SQLite errors
    r_stream = client.post(
        "/api/copilot/ask/stream",
        headers=cse_alias,
        json={"question": "What are my top skills?", "student_id": cse_sid},
    )
    assert r_stream.status_code == 200
    assert "event:" in r_stream.text

    # Drive simulator scopes cohort to mentees for faculty_mentor
    r_sim_no_dept = client.post(
        "/api/admin/simulate-drive",
        headers=no_dept,
        json={"required_skills": ["python"], "min_gpa": 6.0},
    )
    assert r_sim_no_dept.status_code == 200
    assert r_sim_no_dept.json()["total_students"] == 0

    r_sim_cse = client.post(
        "/api/admin/simulate-drive",
        headers=cse_alias,
        json={"required_skills": ["python"], "allowed_branches": ["CSE"], "min_gpa": 6.0},
    )
    assert r_sim_cse.status_code == 200
    sim_json = r_sim_cse.json()
    sim_ids = {s_row["id"] for s_row in sim_json.get("eligible_students", [])}
    assert ece_sid not in sim_ids
    assert sim_json["total_students"] == 1
    assert sim_json["bottleneck_breakdown"]["branch_restriction"] == 0


def test_ratelimit_trusted_proxies_pruning_and_auth_login_limit(client_engine, monkeypatch):
    from app.security import ratelimit as rl_mod
    from app.security.ratelimit import RateLimiter, _identity

    # Bounded key pruning
    lim = RateLimiter(max_keys=5)
    for i in range(10):
        lim.allow(f"k-{i}", "test_bucket", max_requests=2, window_seconds=60)
    assert len(lim._hits) <= 5

    # X-Forwarded-For ignored unless client.host in trusted_proxies
    req = SimpleNamespace(
        headers={"x-forwarded-for": "1.2.3.4"},
        client=SimpleNamespace(host="10.0.0.99"),
    )
    monkeypatch.setattr(settings, "trusted_proxies", [])
    assert _identity(req) == "10.0.0.99"

    monkeypatch.setattr(settings, "trusted_proxies", ["10.0.0.99"])
    assert _identity(req) == "1.2.3.4"

    # auth_login rate limit triggers 429 on burst using isolated RateLimiter
    client, _eng = client_engine
    isolated_limiter = RateLimiter()
    monkeypatch.setattr(rl_mod, "limiter", isolated_limiter)
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    statuses = []
    for _ in range(16):
        resp = client.post("/auth/login", json={"email": "unknown@jklu.edu.in", "pin": "0000"})
        statuses.append(resp.status_code)
    assert statuses[:15] == [401] * 15
    assert statuses[15] == 429


def test_application_visibility_on_campus_duplicate_and_withdrawal_guards(client_engine):
    client, eng = client_engine
    with Session(eng) as s:
        s1 = f"stu-a-{uuid4().hex[:8]}"
        s2 = f"stu-b-{uuid4().hex[:8]}"
        campus_jid = f"job-cam-{uuid4().hex[:8]}"
        live_s1 = f"job-live-1-{uuid4().hex[:8]}"
        live_s2 = f"job-live-2-{uuid4().hex[:8]}"
        s.add(Student(id=s1, name="Student One", branch="CSE", gpa=8.5, resume_text="Python"))
        s.add(StudentSkill(student_id=s1, skill_id="python", level="intermediate", mentions=2, evidence=["Python"]))
        s.add(Student(id=s2, name="Student Two", branch="CSE", gpa=8.5, resume_text="Python"))
        s.add(Job(id=campus_jid, title="Campus SWE", company="TechCo", description="Python", required_skills=["python"], salary_lpa=[4, 6], source="local"))
        s.add(Job(id=live_s1, title="Live Own", company="Ext1", description="Python", required_skills=["python"], salary_lpa=[8, 12], source="live", owner_student_id=s1))
        s.add(Job(id=live_s2, title="Live Other", company="Ext2", description="Python", required_skills=["python"], salary_lpa=[8, 12], source="live", owner_student_id=s2))
        s.commit()

    h_s1 = {"X-Demo-Role": "student", "X-Demo-Student-Id": s1}

    # Private live job of s2 is forbidden across application, simulate, and dismiss
    assert client.post(f"/api/students/{s1}/applications", headers=h_s1, json={"job_id": live_s2}).status_code == 403
    assert client.post(f"/api/students/{s1}/matches/{live_s2}/simulate", headers=h_s1, json={"skills": ["python"]}).status_code == 403
    assert client.delete(f"/api/students/{s1}/matches/{live_s2}", headers=h_s1).status_code == 403

    # Missing job returns 404
    assert client.post(f"/api/students/{s1}/matches/missing-job/simulate", headers=h_s1, json={"skills": []}).status_code == 404
    assert client.delete(f"/api/students/{s1}/matches/missing-job", headers=h_s1).status_code == 404

    # External live job owned by s1 cannot create an on-campus JobApplication
    r_ext = client.post(f"/api/students/{s1}/applications", headers=h_s1, json={"job_id": live_s1})
    assert r_ext.status_code == 400

    # Valid campus application succeeds once (standard tier -> nominated), duplicate returns 409
    r_ok = client.post(f"/api/students/{s1}/applications", headers=h_s1, json={"job_id": campus_jid})
    assert r_ok.status_code == 201, r_ok.text
    nom_id = r_ok.json()["application"]["id"]
    assert r_ok.json()["application"]["status"] == "nominated"
    assert client.post(f"/api/students/{s1}/applications", headers=h_s1, json={"job_id": campus_jid}).status_code == 409

    # Non-TPO cannot withdraw a finalized nominated application (403)
    assert client.delete(f"/api/students/{s1}/applications/{nom_id}", headers=h_s1).status_code == 403

    # Pending (non-nominated) application can be withdrawn once, second withdrawal fails with 409
    with Session(eng) as s:
        pend_id = f"app-pend-{uuid4().hex[:8]}"
        s.add(JobApplication(id=pend_id, student_id=s2, job_id=campus_jid, status="awaiting_tpo_approval", tier="dream"))
        s.commit()
    h_s2 = {"X-Demo-Role": "student", "X-Demo-Student-Id": s2}
    assert client.delete(f"/api/students/{s2}/applications/{pend_id}", headers=h_s2).status_code == 200
    assert client.delete(f"/api/students/{s2}/applications/{pend_id}", headers=h_s2).status_code == 409


def test_concurrent_approve_and_resume_reupload_toctou(tmp_path, monkeypatch):
    from app.api.main import AppContext, create_app
    from app.db import init_db, make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    from app.llm.mock import MockLLM
    from app.seed import seed
    from app.taxonomy.taxonomy import load_taxonomy

    eng = make_engine(f"sqlite:///{tmp_path / 'concurrent.db'}")
    init_db(eng)
    taxonomy = load_taxonomy()
    sid = f"stu-conc-{uuid4().hex[:8]}"
    appr_id = f"apr-conc-{uuid4().hex[:8]}"
    with Session(eng) as s:
        seed(s, taxonomy)
        s.add(
            Student(
                id=sid,
                name="Concurrent Student",
                branch="CSE",
                gpa=8.8,
                resume_text="Python SQL",
                resume_uploaded=True,
                resume_reupload_unlocked=True,
            )
        )
        s.add(
            ApprovalRequest(
                id=appr_id,
                kind="cohort_bootcamp_publish",
                target_id="python",
                title="Bootcamp",
                payload={"skill_id": "python", "course_id": "course-011"},
                status="pending_review",
            )
        )
        s.commit()

    monkeypatch.setattr(settings, "auth_mode", "demo")
    monkeypatch.setattr(settings, "demo_role_switch", True)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(dim=128),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    app = create_app(ctx)

    with TestClient(app) as client:
        # Concurrent approval claims execute once
        def _do_approve(_idx):
            return client.post(
                f"/api/approvals/{appr_id}/approve",
                headers={"X-Demo-Role": "tpo_admin"},
                json={"note": "ok"},
            ).status_code

        with ThreadPoolExecutor(max_workers=4) as pool:
            appr_codes = list(pool.map(_do_approve, range(4)))
        assert all(c == 200 for c in appr_codes)
        with Session(eng) as s:
            auds = s.exec(
                select(AuditLog).where(AuditLog.entity_id == appr_id, AuditLog.action == "approval.approved")
            ).all()
            assert len(auds) == 1

        # Concurrent resume-reupload-request creates only one pending ApprovalRequest
        def _do_req(_idx):
            return client.post(
                f"/api/students/{sid}/resume-reupload-request",
                headers={"X-Demo-Role": "student", "X-Demo-Student-Id": sid},
                json={"reason": "updated projects"},
            ).status_code

        with ThreadPoolExecutor(max_workers=4) as pool:
            req_codes = list(pool.map(_do_req, range(4)))
        assert req_codes.count(201) == 1
        assert all(c in (201, 409) for c in req_codes)

        # Concurrent resume upload consumes resume_reupload_unlocked atomically once
        def _do_upload(_idx):
            return client.post(
                f"/api/students/{sid}/resume",
                headers={"X-Demo-Role": "student", "X-Demo-Student-Id": sid},
                files={"file": ("resume.txt", b"Python FastAPI SQL Docker", "text/plain")},
            ).status_code

        with ThreadPoolExecutor(max_workers=3) as pool:
            up_codes = list(pool.map(_do_upload, range(3)))
        assert up_codes.count(200) == 1
        assert up_codes.count(403) == 2


def test_sqlite_engine_pragmas_and_linkedin_host_validation(tmp_path, monkeypatch):
    from app.db import _sanitize_legacy_experience, init_db, make_engine
    from app.scraper import normalizer

    eng = make_engine(f"sqlite:///{tmp_path / 'pragmas.db'}")
    assert isinstance(eng.pool, QueuePool)
    init_db(eng)
    with eng.connect() as conn:
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert str(conn.execute(text("PRAGMA journal_mode")).scalar()).lower() == "wal"
        assert conn.execute(text("PRAGMA busy_timeout")).scalar() == 5000

    # Deceptive substring URL must NOT trigger LinkedIn external fetch
    fetched_urls: list[str] = []
    monkeypatch.setattr(normalizer, "fetch_external_job_description", lambda url, timeout=3.0: fetched_urls.append(url) or "")
    with Session(eng) as s:
        jid = f"job-badurl-{uuid4().hex[:8]}"
        s.add(
            Job(
                id=jid,
                title="Role",
                company="Co",
                description="Short",
                required_skills=["python"],
                salary_lpa=[8, 12],
                source="live",
                job_url="https://evil.com/?redirect=linkedin.com/jobs/view/123",
            )
        )
        s.commit()
    _sanitize_legacy_experience(eng)
    assert fetched_urls == []


def test_security_headers_and_cors(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "SAMEORIGIN"
    assert r.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"

    from fastapi.middleware.cors import CORSMiddleware

    cors_entries = [m for m in client.app.user_middleware if m.cls is CORSMiddleware]
    assert len(cors_entries) == 1
    assert "*" not in (cors_entries[0].kwargs.get("allow_methods") or [])
    assert "*" not in (cors_entries[0].kwargs.get("allow_headers") or [])


def test_policy_rag_cache_key_mixed_case_and_sim_isolation(client_engine):
    from app.embeddings.embedder import HashingEmbedder
    from app.rag.policy_rag import evaluate_student_eligibility, index_sops, verify_policy_citations

    e128 = HashingEmbedder(dim=128)
    e256 = HashingEmbedder(dim=256)
    idx1 = index_sops(e128)
    idx2 = index_sops(e256)
    assert len(idx1[0]["embedding"]) == 128
    assert len(idx2[0]["embedding"]) == 256

    verdict = verify_policy_citations("Per [sop-01] and SOP-999, proceed.", known={"SOP-01"})
    assert "SOP-01" in verdict["verified"]
    assert "SOP-999" in verdict["unverified"]
    assert "SOP-999 [UNVERIFIED: SOP-999]" in verdict["annotated_text"]
    assert "UNVERIFIED: SOP-01" not in verdict["annotated_text"]

    client, eng = client_engine
    ctx = client.app.state.ctx
    with Session(eng) as s:
        sid = f"stu-sim-{uuid4().hex[:8]}"
        s.add(Student(id=sid, name="Sim Student", branch="CSE", gpa=8.5, resume_text="Python"))
        s.commit()
        sim_job = Job(
            id="sim-ephemeral-1",
            title="Sim Role",
            company="SimCo",
            description="Python",
            required_skills=["python"],
            salary_lpa=[10, 12],
        )
        res = evaluate_student_eligibility(s, ctx, sid, sim_job)
        assert isinstance(res["eligible"], bool)
        assert ctx.vector_store.get("job", "sim-ephemeral-1") is None
        assert "sim-ephemeral-1" not in ctx.job_vectors


def test_resume_parser_zip_bomb_and_corrupt_pdf():
    from app.resume.parser import _get_tessdata_dir, parse_resume_bytes

    # Zip bomb exceeding uncompressed cap raises ValueError
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "A" * (26 * 1024 * 1024))
    with pytest.raises(ValueError, match="maximum decompressed size"):
        parse_resume_bytes(buf.getvalue(), ".docx")

    # Corrupt PDF raises ValueError instead of leaking raw pypdf exceptions
    with pytest.raises(ValueError, match=r"(?i)could not open pdf|failed to parse pdf"):
        parse_resume_bytes(b"%PDF-1.4\nnot-a-valid-pdf-stream", ".pdf")

    # _get_tessdata_dir performs no runtime download when env/system paths absent
    td = _get_tessdata_dir()
    assert td is None or Path(td).exists()


def test_student_input_validation_policy_bounds_and_delete_cascade(client_engine):
    client, eng = client_engine
    tpo = {"X-Demo-Role": "tpo_admin"}

    # Malformed JSON & out-of-range fields
    r_bad_json = client.post(
        "/api/students",
        headers={**tpo, "Content-Type": "application/json"},
        content=b"{bad json",
    )
    assert r_bad_json.status_code == 400

    assert client.post("/api/students", headers=tpo, json={"name": "Bad Year", "year": 99}).status_code == 422
    assert client.post("/api/students", headers=tpo, json={"name": "Bad GPA", "gpa": 12.5}).status_code == 422

    # Policy min_gpa bounds validation
    assert client.post(
        "/api/admin/policies",
        headers=tpo,
        json={"id": "SOP-BAD", "title": "Bad", "body": "Text", "params": {"min_gpa": 11.0}},
    ).status_code == 422

    # Student delete cascades owned jobs, approval requests, and vector store
    sid = f"stu-del-{uuid4().hex[:8]}"
    jid = f"job-del-{uuid4().hex[:8]}"
    aid = f"apr-del-{uuid4().hex[:8]}"
    ctx = client.app.state.ctx
    with Session(eng) as s:
        s.add(Student(id=sid, name="Delete Me", branch="CSE", gpa=8.0, resume_text="Python"))
        s.add(Job(id=jid, title="Owned Live", company="Co", description="Python", required_skills=["python"], salary_lpa=[6, 8], source="live", owner_student_id=sid))
        s.add(ApprovalRequest(id=aid, kind="resume_reupload_request", target_id=sid, title="Req", payload={"student_id": sid}, status="pending_review"))
        s.commit()
    ctx.vector_store.upsert("student", sid, [0.1] * ctx.embedder.dim)

    r_del = client.delete(f"/api/students/{sid}", headers=tpo)
    assert r_del.status_code == 200
    assert ctx.vector_store.get("student", sid) is None
    with Session(eng) as s:
        assert s.get(Student, sid) is None
        assert s.get(Job, jid) is None
        assert s.get(ApprovalRequest, aid) is None
