"""Tests for the TPO-Admin-only /know-more User & Operations Guide and strict RBAC."""
from __future__ import annotations

import html as htmllib
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

    # Master shell controls, TOC, focus banner, search, role filters, expand/collapse, theme toggle
    for required_id in (
        'id="km-search"',
        'id="km-expand-all"',
        'id="km-collapse-all"',
        'id="km-toc"',
        'id="km-focus-banner"',
        'id="km-visible-count"',
        'id="km-total-count"',
        'id="theme-toggle"',
        "/static/theme.css",
        "/static/theme.js",
    ):
        assert required_id in html, f"missing {required_id} in /know-more"

    for chapter in (
        'data-chapter="all"',
        'data-chapter="km-architecture"',
        'data-chapter="km-shell-auth"',
        'data-chapter="km-cohort-profile"',
        'data-chapter="km-matches-drives"',
        'data-chapter="km-plan-interview"',
        'data-chapter="km-copilot-admin"',
    ):
        assert chapter in html, f"missing TOC chapter link {chapter}"

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

    # Representative content from each of the 5 redesigned partials
    for marker in (
        "Navigation, Sign-In, Agent Brain &amp; Multi-Device Access",
        "Home Cohort Directory, AI Resume Onboarding &amp; Student Profiles",
        "Job Matches, Live Market Scraper, Applications &amp; Campus Drives",
        "Upskill Planner, Progress Center &amp; Mock Interview Studio",
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
        "html.dark .km-ui-card",
        "html.dark .km-ui-btn-primary",
        "html.dark .km-ui-btn-secondary",
        "html.dark .km-ui-btn-danger",
        "html.dark .km-ui-toggle",
        "html.dark .km-ui-input",
        "html.dark .km-focus-banner",
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


def test_know_more_visual_plain_english_no_code_dumps_and_shell_guards(client: TestClient):
    res = client.get("/know-more", headers={"X-Demo-Role": "tpo_admin"})
    assert res.status_code == 200
    html = res.text

    main_match = re.search(r"<main\b[^>]*>(.*?)</main>", html, re.DOTALL)
    assert main_match, "<main> block missing from /know-more"
    main_no_script = re.sub(r"<script\b[^>]*>.*?</script>", "", main_match.group(1), flags=re.DOTALL)
    visible_text = htmllib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", main_no_script)))

    # Visual UI button & control previews rendered across the guide
    for ui_cls in (
        "km-ui-card",
        "km-ui-btn-primary",
        "km-ui-btn-secondary",
        "km-ui-btn-danger",
        "km-ui-toggle",
        "km-ui-input",
    ):
        assert ui_cls in main_no_script, f"missing visual preview class {ui_cls}"

    # Real user-facing button labels present in plain English
    for label in (
        "Dark Mode",
        "Agent Brain",
        "Upload & Run AI Extraction",
        "Fetch Live Jobs",
        "Remove All Live Jobs",
        "Re-Solve Plan",
        "Start interview (top match)",
        "Print scorecard",
        "Run Simulation",
        "Endorse (Tier 1)",
        "Approve & Execute (Tier 2)",
        "Reset Core Defaults",
    ):
        assert label in visible_text, f"missing plain-English button label {label!r}"

    # Zero raw code dumps (endpoints, .py files, HTTP verbs, CSS IDs) in visible <main> guide text
    for forbidden_pat, desc in (
        (r"/api/", "raw /api/ path"),
        (r"/auth/", "raw /auth/ path"),
        (r"/admin/audit", "raw /admin/audit path"),
        (r"/jnf/\{", "raw /jnf/{token} path"),
        (r"\b[a-zA-Z0-9_]+\.py\b", ".py filename"),
        (r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/", "HTTP method + path"),
        (r"#(?:menu-btn|cohort-search|theme-toggle)\b", "raw CSS selector ID"),
    ):
        assert not re.search(forbidden_pat, visible_text), f"found {desc} in /know-more visible text"

    # Base shell guards /know-more in switchDemoRole and Ctrl+K command bar
    base_html = (ROOT / "app" / "templates" / "base.html").read_text(encoding="utf-8")
    assert 'path.indexOf("/know-more") === 0' in base_html
    assert 'location.href = "/know-more"' in base_html
    assert 'activeRole !== "tpo_admin"' in base_html
