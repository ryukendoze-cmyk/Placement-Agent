"""Platform P0 RBAC strict-mode hardening, personas, PIN auth, and audit tests."""
from __future__ import annotations

import json
from pathlib import Path

from sqlmodel import Session, select

from app.models import AuditLog, Student


def test_personas_json_has_pbkdf2_hashes_for_all_three_roles():
    data = json.loads(Path("data/personas.json").read_text(encoding="utf-8"))
    personas = {p["email"]: p for p in data["personas"]}
    assert "rohit.shetty@jklu.edu.in" in personas
    assert "faculty.mentor@jklu.edu.in" in personas
    assert "tpo.admin@jklu.edu.in" in personas
    assert personas["rohit.shetty@jklu.edu.in"]["role"] == "student"
    assert personas["faculty.mentor@jklu.edu.in"]["role"] == "faculty_mentor"
    assert personas["tpo.admin@jklu.edu.in"]["role"] == "tpo_admin"
    for p in personas.values():
        assert str(p["pin_hash"]).startswith("pbkdf2_sha256$")
        assert "2026" not in str(p["pin_hash"])


def test_demo_cohort_has_rohit_shetty_and_twelve_students():
    from app.seed import _DEMO_COHORT

    assert len(_DEMO_COHORT) == 12
    assert any(row[0] == "Rohit Shetty" for row in _DEMO_COHORT)


def test_strict_mode_blocks_unauthenticated_and_header_spoofs(client_engine, monkeypatch):
    from app.config import settings

    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")

    # HTML routes redirect 303 -> /login
    r_home = client.get("/", follow_redirects=False)
    assert r_home.status_code == 303
    assert r_home.headers["location"] == "/login"

    # Header spoofing X-Demo-Role: tpo_admin must be completely ignored in strict mode
    r_spoof_html = client.get(
        "/admin/audit",
        headers={"X-Demo-Role": "tpo_admin"},
        follow_redirects=False,
    )
    assert r_spoof_html.status_code == 303
    assert r_spoof_html.headers["location"] == "/login"

    # API routes return 401 even with X-Demo-Role spoof or unsigned ps06_role cookie
    r_spoof_api = client.get(
        "/api/admin/audit",
        headers={"X-Demo-Role": "tpo_admin"},
    )
    assert r_spoof_api.status_code == 401

    client.cookies.set("ps06_role", "tpo_admin")
    r_cookie_spoof = client.get("/api/admin/audit")
    client.cookies.clear()
    assert r_cookie_spoof.status_code == 401

    with Session(eng) as db:
        denied = db.exec(select(AuditLog).where(AuditLog.action == "auth.denied")).all()
        assert len(denied) >= 1
        assert all((a.detail or {}).get("trace_id") for a in denied)


def test_strict_mode_public_allowlist_stays_accessible(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "auth_mode", "strict")
    for path in (
        "/login",
        "/healthz",
        "/readyz",
        "/manifest.webmanifest",
        "/sw.js",
        "/static/responsive.css",
    ):
        r = client.get(path, follow_redirects=False)
        assert r.status_code == 200, f"Public allowlisted path {path} returned {r.status_code}"


def test_strict_mode_login_pin_validation_and_audit(client_engine, monkeypatch):
    from app.config import settings

    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")
    monkeypatch.setattr(settings, "demo_role_switch", True)

    # Non-JKLU email -> 403 + auth.denied
    r_bad_domain = client.post(
        "/auth/login",
        json={"email": "outsider@gmail.com", "pin": "2026"},
    )
    assert r_bad_domain.status_code == 403

    # Wrong PIN -> 401 + auth.denied
    r_bad_pin = client.post(
        "/auth/login",
        json={"email": "tpo.admin@jklu.edu.in", "pin": "9999"},
    )
    assert r_bad_pin.status_code == 401

    # Missing PIN without fast_login in strict mode -> 401
    r_no_pin = client.post(
        "/auth/login",
        json={"email": "tpo.admin@jklu.edu.in"},
    )
    assert r_no_pin.status_code == 401

    # Valid Rohit Shetty login binds to DB Student "Rohit Shetty" and sets HttpOnly session
    r_rohit = client.post(
        "/auth/login",
        json={"email": "rohit.shetty@jklu.edu.in", "pin": "2026"},
    )
    assert r_rohit.status_code == 200
    assert "ps06_session" in r_rohit.cookies
    claims = r_rohit.json()["claims"]
    assert claims["role"] == "student"
    assert claims["name"] == "Rohit Shetty"
    assert claims["student_id"].startswith("stu-")

    with Session(eng) as db:
        rohit_row = db.get(Student, claims["student_id"])
        assert rohit_row is not None
        assert rohit_row.name == "Rohit Shetty"

    # Authenticated student CANNOT elevate via X-Demo-Role header in strict mode (403 + rate-limited audit)
    for _ in range(3):
        r_elev = client.get(
            "/api/admin/audit",
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r_elev.status_code == 403

    # Logout clears session and audits auth.logout
    r_logout = client.post("/auth/logout", follow_redirects=False)
    assert r_logout.status_code == 303

    # Faculty Mentor login -> 200 on approvals, 403 on audit
    r_fac = client.post(
        "/auth/login",
        json={"email": "faculty.mentor@jklu.edu.in", "pin": "2026"},
    )
    assert r_fac.status_code == 200
    assert client.get("/admin/approvals").status_code == 200
    assert client.get("/admin/audit").status_code == 403

    # TPO Admin login -> 200 on audit
    r_tpo = client.post(
        "/auth/login",
        json={"email": "tpo.admin@jklu.edu.in", "pin": "2026"},
    )
    assert r_tpo.status_code == 200
    assert client.get("/admin/audit").status_code == 200

    with Session(eng) as db:
        rows = db.exec(select(AuditLog)).all()
        actions = {a.action for a in rows}
        assert {"auth.login", "auth.logout", "auth.denied"} <= actions
        # Rate-limit check: 3 identical 403 hits on /api/admin/audit should not log 3 separate rows
        audit_403_rows = [
            a for a in rows
            if a.action == "auth.denied" and a.entity_id == "/api/admin/audit"
        ]
        assert len(audit_403_rows) == 1


def test_strict_mode_demo_role_switch_toggle_and_audit(client_engine, monkeypatch):
    from app.config import settings

    client, eng = client_engine
    monkeypatch.setattr(settings, "auth_mode", "strict")

    # Enabled -> 200 + audited
    monkeypatch.setattr(settings, "demo_role_switch", True)
    r_ok = client.post("/api/auth/switch-role", json={"role": "faculty_mentor"})
    assert r_ok.status_code == 200
    assert r_ok.json()["role"] == "faculty_mentor"

    # Disabled -> 403 + auth.denied
    monkeypatch.setattr(settings, "demo_role_switch", False)
    r_denied = client.post("/api/auth/switch-role", json={"role": "tpo_admin"})
    assert r_denied.status_code == 403

    # Fast-login also blocked when demo_role_switch=False in strict mode
    r_fast_denied = client.post(
        "/auth/login",
        json={"email": "tpo.admin@jklu.edu.in", "fast_login": True},
    )
    assert r_fast_denied.status_code == 401

    with Session(eng) as db:
        actions = [a.action for a in db.exec(select(AuditLog)).all()]
        assert "auth.role_switched" in actions
        assert "auth.denied" in actions


def test_cors_explicit_origin_allowlist(client):
    r_ok = client.options(
        "/api/auth/role",
        headers={
            "Origin": "http://localhost:8010",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_ok.headers.get("access-control-allow-origin") == "http://localhost:8010"
    assert r_ok.headers.get("access-control-allow-credentials") == "true"

    r_bad = client.options(
        "/api/auth/role",
        headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r_bad.headers.get("access-control-allow-origin") is None
