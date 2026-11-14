"""End-to-end verification tests for JKLU CareerOS V2 (Campus Drive, RBAC, Offline, Packaging)."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from sqlmodel import Session, select

from app.campus.agent import parse_campus_jd
from app.campus.policy import tier_for
from app.config import settings
from app.embeddings.embedder import LocalEmbedder
from app.llm.base import get_llm
from app.models import AuditLog, CampusDrive, Job, JobMatch, Student
from app.security.rbac import encode_session
from app.taxonomy.taxonomy import load_taxonomy

FIXTURE_DIR = Path("tests/fixtures/jd")
ROOT = Path(".")


def _login_as(client, email: str, pin: str = "2026") -> None:
    resp = client.post("/auth/login", json={"email": email, "pin": pin})
    assert resp.status_code == 200, f"Login failed for {email}: {resp.status_code} {resp.text}"


def test_e2e_strict_rbac_personas_and_spoof_blocking(client_engine, monkeypatch):
    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")

    # 1. Header spoof without signed session must fail
    r_spoof = client.get("/api/campus-drives", headers={"X-Demo-Role": "tpo_admin"})
    assert r_spoof.status_code in (401, 403)

    r_html = client.get("/", headers={"X-Demo-Role": "tpo_admin"}, follow_redirects=False)
    assert r_html.status_code == 303
    assert r_html.headers["location"] == "/login"

    # 2. Wrong PIN and non-JKLU email blocked
    r_bad_pin = client.post("/auth/login", json={"email": "tpo.admin@jklu.edu.in", "pin": "0000"})
    assert r_bad_pin.status_code == 401
    r_non_jklu = client.post("/auth/login", json={"email": "hacker@gmail.com", "pin": "2026"})
    assert r_non_jklu.status_code == 403

    # 3. Login per persona with PIN 2026
    for email in (
        "rohit.shetty@jklu.edu.in",
        "faculty.mentor@jklu.edu.in",
        "tpo.admin@jklu.edu.in",
    ):
        client.cookies.clear()
        _login_as(client, email, "2026")
        r_me = client.get("/api/campus-drives")
        if email == "rohit.shetty@jklu.edu.in":
            assert r_me.status_code == 403
        else:
            assert r_me.status_code == 200

    with Session(eng) as db:
        actions = {a.action for a in db.exec(select(AuditLog)).all()}
        assert "auth.login" in actions
        assert "auth.denied" in actions


def test_e2e_parse_all_jd_fixtures_and_sop01_grounding(client):
    taxonomy = load_taxonomy()
    valid_skill_ids = {s.id for s in taxonomy.all()}

    fixtures = sorted(FIXTURE_DIR.glob("*"))
    assert len(fixtures) >= 8

    for fx in fixtures:
        raw = fx.read_bytes()
        for mode in ("auto", "static"):
            r = client.post(
                "/api/campus-drives/parse-jd",
                headers={"X-Demo-Role": "tpo_admin"},
                files={"file": (fx.name, raw, "application/octet-stream")},
                data={"force_parser": mode, "simulate_failure": "none"},
            )
            assert r.status_code == 200, f"{fx.name} ({mode}) failed: {r.text}"
            payload = r.json()
            assert set(payload["required_skills"]).issubset(valid_skill_ids)
            assert set(payload["preferred_skills"]).issubset(valid_skill_ids)
            assert "Blockchain-Quantum-Rust++" not in payload["required_skills"]
            assert "Blockchain-Quantum-Rust++" not in payload["preferred_skills"]
            if "08_fake_unmapped_skills" in fx.name:
                unmapped_joined = " ".join(payload.get("unmapped_skill_mentions") or [])
                assert "Blockchain-Quantum-Rust++" in unmapped_joined
            if "04_internship_stipend" in fx.name:
                assert payload["experience_level"] == "Internship"
                assert payload["policy_tier"] is None

        for sim in ("timeout", "invalid_json"):
            r_sim = client.post(
                "/api/campus-drives/parse-jd",
                headers={"X-Demo-Role": "tpo_admin"},
                files={"file": (fx.name, raw, "application/octet-stream")},
                data={"force_parser": "llm", "simulate_failure": sim},
            )
            assert r_sim.status_code == 200
            sim_data = r_sim.json()
            assert sim_data["parser_metadata"]["engine_used"] == "static_fallback"
            expected_reason = "llm_timeout" if sim == "timeout" else "llm_invalid_json"
            assert sim_data["parser_metadata"]["fallback_reason"] == expected_reason

    # Exact SOP-01 boundary checks
    assert tier_for(7.99)[0] == "standard"
    assert tier_for(8.0)[0] == "dream"
    assert tier_for(14.99)[0] == "dream"
    assert tier_for(15.0)[0] == "super_dream"
    assert tier_for(12.0, experience_level="Internship")[0] is None


def test_e2e_bogus_openai_endpoint_recovers_fast(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "openai")
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-bogus")
    monkeypatch.setattr(settings, "openai_base_url", "http://127.0.0.1:9/v1")
    llm = get_llm(settings)
    taxonomy = load_taxonomy()

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.llm = llm
    ctx.taxonomy = taxonomy

    sample_jd = (
        "Company: Acme Systems\n"
        "Role: Backend Engineer\n"
        "CTC: 12 LPA\n"
        "Skills: Python, FastAPI, PostgreSQL\n"
    )
    t0 = time.monotonic()
    res = parse_campus_jd(sample_jd, mode="llm", ctx=ctx, actor_role="tpo_admin")
    elapsed = time.monotonic() - t0

    assert elapsed < 8.0, f"Recovery took too long: {elapsed:.2f}s"
    assert res["parser_metadata"]["engine_used"] == "static_fallback"
    assert res["parser_metadata"]["fallback_reason"] is not None
    assert res["policy_tier"] == "dream"


def test_e2e_hitl_governance_publish_rematch_and_audit_trace(client_engine, monkeypatch):
    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")

    # TPO parses & submits a campus drive (PENDING_TPO_APPROVAL)
    _login_as(client, "tpo.admin@jklu.edu.in", "2026")
    jd_text = (
        "Company: FinTech Edge\n"
        "Role: Platform Backend Engineer\n"
        "Location: Bengaluru (Hybrid)\n"
        "CTC: 16 LPA\n"
        "Eligibility: CGPA >= 7.0, B.Tech CSE/AI&DS, no active backlogs, 2027 batch\n"
        "Required Skills: Python, FastAPI, PostgreSQL, Docker\n"
        "Preferred Skills: Redis\n"
    )
    r_parse = client.post(
        "/api/campus-drives/parse-jd",
        data={"text": jd_text, "force_parser": "auto", "simulate_failure": "timeout"},
    )
    assert r_parse.status_code == 200
    draft = r_parse.json()
    trace_id = draft["trace_id"]

    with Session(eng) as db:
        jobs_before = len(db.exec(select(Job)).all())

    r_create = client.post(
        "/api/campus-drives",
        json={**draft, "publish_now": False, "confirm": False},
    )
    assert r_create.status_code == 201, r_create.text
    created = r_create.json()
    drive_id = created["id"]
    assert created["approval_status"] == "PENDING_TPO_APPROVAL"

    # No Job row created before TPO approval
    with Session(eng) as db:
        assert len(db.exec(select(Job)).all()) == jobs_before

    # Student and faculty_mentor MUST get 403 on approve
    for email in ("rohit.shetty@jklu.edu.in", "faculty.mentor@jklu.edu.in"):
        client.cookies.clear()
        _login_as(client, email, "2026")
        r_deny = client.post(f"/api/campus-drives/{drive_id}/approve", json={"note": "try"})
        assert r_deny.status_code == 403, f"Expected 403 for {email}, got {r_deny.status_code}"

    # TPO approves -> Job(source="local") created + cohort re-match runs
    client.cookies.clear()
    _login_as(client, "tpo.admin@jklu.edu.in", "2026")
    r_app = client.post(f"/api/campus-drives/{drive_id}/approve", json={"note": "LGTM"})
    assert r_app.status_code == 200, r_app.text
    app_data = r_app.json()
    job_id = app_data["executed"]["job_id"]
    task_id = app_data["executed"]["task_id"]
    assert job_id.startswith("job-campus-")

    # Poll task queue if needed
    for _ in range(30):
        t_resp = client.get(f"/api/tasks/{task_id}")
        if t_resp.status_code == 200 and t_resp.json().get("status") in ("done", "completed", "succeeded"):
            break
        time.sleep(0.05)

    with Session(eng) as db:
        job = db.get(Job, job_id)
        assert job is not None
        assert job.source == "local"
        assert job.max_backlogs == 0  # SOP-03 Super-Dream default/explicit 0

        matches = db.exec(select(JobMatch).where(JobMatch.job_id == job_id)).all()
        assert len(matches) >= 1

        rohit = db.exec(select(Student).where(Student.name == "Rohit Shetty")).first()
        assert rohit is not None
        rohit_match = db.exec(
            select(JobMatch).where(JobMatch.job_id == job_id, JobMatch.student_id == rohit.id)
        ).first()
        assert rohit_match is not None

        drive = db.get(CampusDrive, drive_id)
        assert drive is not None
        assert drive.approval_status == "PUBLISHED"
        assert (drive.cohort_summary or {}).get("students_total", 0) >= 1

        # AuditLog rows share the same trace_id
        audits = db.exec(select(AuditLog)).all()
        traced_actions = {
            a.action for a in audits if isinstance(a.detail, dict) and a.detail.get("trace_id") == trace_id
        }
        assert "campus_drive.jd_extracted" in traced_actions
        assert "campus_drive.submitted" in traced_actions
        assert "campus_drive.published" in traced_actions
        assert "campus_drive.cohort_rematched" in traced_actions


def test_e2e_public_jnf_token_security_and_intake(client_engine, monkeypatch):
    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")

    # Bad and expired tokens -> 401
    assert client.get("/jnf/invalid-token-123").status_code == 401
    expired_tok = encode_session({"kind": "jnf", "exp": int(time.time()) - 60})
    assert client.get(f"/jnf/{expired_tok}").status_code == 401
    r_bad_post = client.post(
        "/api/campus-drives/public-jnf",
        data={"token": "invalid-token-123", "company": "BadCo", "role_title": "Dev"},
    )
    assert r_bad_post.status_code == 401

    # TPO creates valid JNF link
    _login_as(client, "tpo.admin@jklu.edu.in", "2026")
    r_link = client.post("/api/campus-drives/jnf-links", json={"expires_hours": 24})
    assert r_link.status_code == 200
    token = r_link.json()["token"]
    jnf_url = r_link.json()["url"]

    # Unauthenticated recruiter opens JNF page & submits JD
    client.cookies.clear()
    r_page = client.get(jnf_url)
    assert r_page.status_code == 200
    assert "http://" not in r_page.text and "https://" not in r_page.text

    with Session(eng) as db:
        jobs_before = len(db.exec(select(Job)).all())

    r_sub = client.post(
        "/api/campus-drives/public-jnf",
        data={
            "token": token,
            "company": "Jaipur Cloud Works",
            "role_title": "Graduate Cloud Engineer",
            "location": "Jaipur",
            "work_mode": "On-site",
            "experience_level": "Fresher",
            "ctc_lpa_max": "9.5",
            "salary_disclosed": "true",
            "raw_jd_text": "Company: Jaipur Cloud Works\nCTC: 9.5 LPA\nSkills: Python, Docker, AWS",
        },
    )
    assert r_sub.status_code == 201, r_sub.text
    sub_data = r_sub.json()
    assert sub_data["approval_status"] == "PENDING_TPO_APPROVAL"
    assert sub_data["origin"] == "recruiter_jnf"

    # Must NOT create Job before TPO approval
    with Session(eng) as db:
        assert len(db.exec(select(Job)).all()) == jobs_before


def test_e2e_offline_pwa_electron_and_capacitor_assets(client_engine, monkeypatch):
    client, _ = client_engine
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"

    # Embedder boots offline without network
    emb = LocalEmbedder()
    vec = emb.embed("python fastapi docker")
    assert len(vec) == 384

    # PWA endpoints
    r_manifest = client.get("/manifest.webmanifest")
    assert r_manifest.status_code == 200
    assert "JKLU CareerOS" in r_manifest.text

    r_sw = client.get("/sw.js")
    assert r_sw.status_code == 200
    assert "PRECACHE_ASSETS" in r_sw.text

    # Rendered pages have zero external http/https asset links
    _login_as(client, "tpo.admin@jklu.edu.in", "2026")
    ext_re = re.compile(r"""(?:src|href)\s*=\s*["']https?://""", re.IGNORECASE)
    for path in ("/login", "/", "/admin/approvals", "/admin/audit"):
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} returned {resp.status_code}"
        assert not ext_re.search(resp.text), f"External asset URL found in {path}"

    # Electron & Capacitor packaging artifacts exist
    expected_files = [
        ROOT / "desktop" / "package.json",
        ROOT / "desktop" / "main.js",
        ROOT / "desktop" / "preload.js",
        ROOT / "desktop" / "splash.html",
        ROOT / "mobile" / "package.json",
        ROOT / "mobile" / "capacitor.config.json",
        ROOT / "mobile" / "www" / "index.html",
        ROOT / "mobile" / "android" / "app" / "src" / "main" / "AndroidManifest.xml",
        ROOT / "scripts" / "serve_lan.ps1",
        ROOT / "app" / "static" / "icons" / "icon-192.png",
        ROOT / "app" / "static" / "icons" / "icon-512.png",
    ]
    for p in expected_files:
        assert p.exists(), f"Missing packaging file: {p}"
