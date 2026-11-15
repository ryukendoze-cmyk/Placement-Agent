"""Tests for the TPO-Admin-only /know-more System Engineering Manual and strict RBAC."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
KNOW_MORE_TEMPLATE = ROOT / "app" / "templates" / "know_more.html"
KNOW_MORE_PARTIALS_DIR = ROOT / "app" / "templates" / "know_more"

EXPECTED_PARTIALS = (
    "_shell_auth_agent.html",
    "_cohort_profile.html",
    "_matches_scraper_drives.html",
    "_plan_progress_interview.html",
    "_copilot_admin_ops.html",
)


def _extract_sidebar(html: str) -> str:
    m = re.search(r'<nav id="sidebar".*?</nav>', html, re.DOTALL)
    assert m, "sidebar <nav id='sidebar'> missing from rendered HTML"
    return m.group(0)


def _seed_student(client: TestClient) -> str:
    res = client.post(
        "/api/students",
        headers={"X-Demo-Role": "tpo_admin"},
        files={"file": ("resume.txt", b"Skills: Python, FastAPI, SQL\n", "text/plain")},
        data={"name": "KnowMore Probe", "branch": "CSE", "year": "4", "gpa": "8.4"},
    )
    assert res.status_code == 201, res.text
    return res.json()["id"]


def test_know_more_renders_all_partials_controls_and_dark_theme(client: TestClient):
    for partial in EXPECTED_PARTIALS:
        path = KNOW_MORE_PARTIALS_DIR / partial
        assert path.is_file(), f"missing partial {path}"
        assert "alert(" not in path.read_text(encoding="utf-8")
    assert "alert(" not in KNOW_MORE_TEMPLATE.read_text(encoding="utf-8")

    res = client.get("/know-more", headers={"X-Demo-Role": "tpo_admin"})
    assert res.status_code == 200
    html = res.text

    # No unrendered Jinja tokens
    assert "{{" not in html
    assert "{%" not in html

    # Master shell controls, TOC, search, role filters, expand/collapse, theme toggle
    for required_id in (
        'id="km-search"',
        'id="km-expand-all"',
        'id="km-collapse-all"',
        'id="km-toc"',
        'id="km-visible-count"',
        'id="km-total-count"',
        'id="theme-toggle"',
        "/static/theme.css",
        "/static/theme.js",
    ):
        assert required_id in html, f"missing {required_id} in /know-more"

    assert html.count("km-role-btn") >= 5
    for role_filter in ("all", "student", "faculty_mentor", "tpo_admin", "public"):
        assert f'data-role-filter="{role_filter}"' in html

    # All 6 top-level sections (overview + 5 partials)
    for sec_id in (
        'id="km-architecture"',
        'id="km-shell-auth"',
        'id="km-cohort-profile"',
        'id="km-matches-drives"',
        'id="km-plan-interview"',
        'id="km-copilot-admin"',
    ):
        assert sec_id in html, f"missing section {sec_id}"

    # Representative content from each of the 5 partials
    for marker in (
        "Left Sidebar, Topbar Shell, Command Bar &amp; Obsidian Theme",
        "Home Cohort Directory",
        "Job Matching Engine &amp; Incremental Auto-Rank",
        "Upskill Plan Generator, BKT Mastery &amp; CPM/RCPSP Solver",
        "Placement Copilot &amp; Policy RAG",
    ):
        assert marker in html, f"partial content missing: {marker}"

    # Sidebar exposes /know-more to tpo_admin
    sidebar = _extract_sidebar(html)
    assert 'href="/know-more"' in sidebar
    assert "Know More" in sidebar

    # Dark-mode CSS overrides for Know More elements are served
    css_res = client.get("/static/theme.css")
    assert css_res.status_code == 200
    for selector in (
        "html.dark #km-toolbar",
        "html.dark details.km-card",
        "html.dark .km-table",
        "html.dark .km-role-btn",
        "html.dark .km-toc-link",
    ):
        assert selector in css_res.text, f"missing dark mode CSS selector: {selector}"


def test_know_more_demo_mode_rbac_and_sidebar_hiding(client: TestClient):
    sid = _seed_student(client)

    # tpo_admin -> 200 and sidebar link visible
    r_tpo = client.get("/know-more", headers={"X-Demo-Role": "tpo_admin"})
    assert r_tpo.status_code == 200
    assert 'href="/know-more"' in _extract_sidebar(client.get("/", headers={"X-Demo-Role": "tpo_admin"}).text)

    # faculty_mentor -> 403 and sidebar link hidden
    r_fac = client.get("/know-more", headers={"X-Demo-Role": "faculty_mentor"})
    assert r_fac.status_code == 403
    fac_sidebar = _extract_sidebar(client.get("/admin/approvals", headers={"X-Demo-Role": "faculty_mentor"}).text)
    assert "/know-more" not in fac_sidebar
    assert "Know More" not in fac_sidebar

    # student -> 403 and sidebar link hidden
    r_stu = client.get(
        "/know-more",
        headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid},
    )
    assert r_stu.status_code == 403
    stu_sidebar = _extract_sidebar(
        client.get(
            f"/students/{sid}",
            headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid},
        ).text
    )
    assert "/know-more" not in stu_sidebar
    assert "Know More" not in stu_sidebar


def test_know_more_strict_mode_rbac_and_redirects(client: TestClient, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "auth_mode", "strict")

    # 1. Unauthenticated -> 303 redirect to /login (even with spoofed header or cookie)
    r_anon = client.get("/know-more", follow_redirects=False)
    assert r_anon.status_code == 303
    assert r_anon.headers["location"] == "/login"

    r_header_spoof = client.get(
        "/know-more",
        headers={"X-Demo-Role": "tpo_admin"},
        follow_redirects=False,
    )
    assert r_header_spoof.status_code == 303
    assert r_header_spoof.headers["location"] == "/login"

    client.cookies.set("ps06_role", "tpo_admin")
    r_cookie_spoof = client.get("/know-more", follow_redirects=False)
    client.cookies.clear()
    assert r_cookie_spoof.status_code == 303
    assert r_cookie_spoof.headers["location"] == "/login"

    # 2. Signed-in student -> 403 Forbidden & hidden from sidebar
    r_stu_login = client.post(
        "/auth/login",
        json={"email": "rohit.shetty@jklu.edu.in", "pin": "2026"},
    )
    assert r_stu_login.status_code == 200
    stu_sid = r_stu_login.json()["claims"]["student_id"]
    assert client.get("/know-more", follow_redirects=False).status_code == 403
    assert "/know-more" not in _extract_sidebar(client.get(f"/students/{stu_sid}").text)
    client.post("/auth/logout", follow_redirects=False)

    # 3. Signed-in faculty_mentor -> 403 Forbidden & hidden from sidebar
    r_fac_login = client.post(
        "/auth/login",
        json={"email": "faculty.mentor@jklu.edu.in", "pin": "2026"},
    )
    assert r_fac_login.status_code == 200
    assert client.get("/know-more", follow_redirects=False).status_code == 403
    assert "/know-more" not in _extract_sidebar(client.get("/admin/approvals").text)
    client.post("/auth/logout", follow_redirects=False)

    # 4. Signed-in tpo_admin -> 200 OK & visible in sidebar
    r_tpo_login = client.post(
        "/auth/login",
        json={"email": "tpo.admin@jklu.edu.in", "pin": "2026"},
    )
    assert r_tpo_login.status_code == 200
    r_ok = client.get("/know-more", follow_redirects=False)
    assert r_ok.status_code == 200
    assert 'href="/know-more"' in _extract_sidebar(r_ok.text)


def test_know_more_documents_key_routes_and_guards_shell_navigation(client: TestClient):
    res = client.get("/know-more", headers={"X-Demo-Role": "tpo_admin"})
    assert res.status_code == 200
    html = res.text

    for expected_route in (
        "/login",
        "/auth/login",
        "/auth/logout",
        "/copilot",
        "/admin",
        "/admin/approvals",
        "/admin/policies",
        "/admin/audit",
        "/know-more",
        "/jnf/{token}",
        "/healthz",
        "/readyz",
        "/api/agent/run/stream",
        "/api/campus-drives/public-jnf",
        "/api/students/{sid}/matches",
        "/api/plans/{plan_id}/simulate",
    ):
        assert expected_route in html, f"route {expected_route} missing from /know-more documentation"

    # Base shell guards /know-more in switchDemoRole and Ctrl+K command bar
    base_html = (ROOT / "app" / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'path.indexOf("/know-more") === 0' in base_html
    assert 'location.href = "/know-more"' in base_html
    assert 'activeRole !== "tpo_admin"' in base_html
