"""Enterprise RBAC & Row-Level Security acceptance gate — ARCHITECTURE §17.2 + §17.3.

Milestone 9 Ticket (charter §7 — spec-driven acceptance gate written FIRST):

These tests PIN the §17.2 contracts (CurrentUser identity model, object-level
row-level security, dynamic sidebar gating, institutional login portal) and the
§17.3 quality gates. They are SELF-ARMING: they go RED for each contract
@Coder-Flash has not yet landed, defining "done" before implementation begins.

Baseline (measured 2026-11-02, pre-implementation):
  - /login, app/api/routes/auth.py, app/templates/auth/login.html  -> MISSING
  - student A reads student B /students/{B} etc. -> 200 (no row-level security)
  - copilot: student A queries B -> response echoes sidB (no self-binding)
  - base.html Platform admin links + student switcher -> shown to all roles
  - POST /api/jobs/scrape as student -> 200 (no tpo_admin restriction)
  - Role-level gates (/admin/audit 403 faculty/student, /admin/approvals 200
    faculty) -> ALREADY WORK (regression-pinned here, measured-green).

Anti-vacuity (charter §4): every "missing/blocked" probe below first proves the
probe FINDS the thing where it DOES exist (e.g. own-profile 200, audit 403 for
the right reason), so a 403/404 assertion can never be vacuous.
"""
from __future__ import annotations

import re

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_students(client, n: int = 2) -> list[str]:
    """Create n students via POST /api/students; return their ids."""
    ids = []
    for i in range(n):
        r = client.post(
            "/api/students",
            json={"name": f"Candidate {i}", "branch": "CSE", "year": 3, "gpa": 8.0},
        )
        assert r.status_code in (200, 201), r.text[:200]
        ids.append(r.json()["id"])
    return ids


def _student_h(sid: str) -> dict:
    """Header override declaring an authenticated student identity (§17.2.A)."""
    return {"X-Demo-Role": "student", "X-Demo-Student-ID": sid}


# ===========================================================================
# §17.2.D — Institutional Login Portal (GET /login, POST /auth/login, logout)
# ===========================================================================


class TestJKLULoginPortal:
    """app/api/routes/auth.py + app/templates/auth/login.html (§17.2.D)."""

    def test_login_page_renders(self, client):
        """GET /login returns 200 with a JKLU-branded portal."""
        r = client.get("/login")
        assert r.status_code == 200, f"GET /login -> {r.status_code}: {r.text[:300]}"
        html = r.text
        assert "JKLU" in html, "login page must be branded JKLU"
        assert "Institutional" in html or "institutional" in html, (
            "login page must offer 'Sign in with JKLU Institutional Account'"
        )
        assert "@jklu.edu.in" in html, "login page must reference the @jklu.edu.in domain"

    def test_login_page_has_fast_login_role_selector(self, client):
        """One-click demo/eval role fast-login buttons (§17.2.D.1)."""
        r = client.get("/login")
        assert r.status_code == 200
        html = r.text
        # The demo fast-login must expose all three roles.
        for role in ("student", "faculty_mentor", "tpo_admin"):
            assert role in html, f"fast-login selector missing role {role!r}"
        # Must be a form/post that actually logs in (not a dead link).
        assert "POST" in html or "post" in html or "fetch(" in html, (
            "fast-login must POST to /auth/login"
        )

    def test_post_auth_login_sets_session_cookie(self, client):
        """POST /auth/login sets ps06_session and redirects to / (§17.2.D.2)."""
        r = client.post("/auth/login", json={"email": "2022btech01@jklu.edu.in",
                                             "role": "student"})
        # Either a redirect (303/302) or 200 with the cookie set.
        assert r.status_code in (200, 302, 303), (
            f"POST /auth/login -> {r.status_code}: {r.text[:300]}"
        )
        assert "ps06_session" in r.cookies, (
            f"ps06_session cookie not set; cookies={dict(r.cookies)}"
        )

    def test_post_auth_login_rejects_non_jklu_domain(self, client):
        """Institutional flow validates @jklu.edu.in (§17.2.D.2)."""
        r = client.post("/auth/login", json={"email": "someone@evil.com",
                                             "role": "student"})
        assert r.status_code != 404, "POST /auth/login endpoint does not exist yet"
        # A non-JKLU email must NOT establish a session.
        if r.status_code < 400:
            assert "ps06_session" not in r.cookies, (
                "non-JKLU email must not receive a session cookie"
            )

    def test_post_auth_logout_clears_session(self, client):
        """POST /auth/logout clears ps06_session and redirects to /login (§17.2.D.2)."""
        # Establish a session first.
        client.post("/auth/login", json={"email": "2022btech01@jklu.edu.in",
                                         "role": "student"})
        r = client.post("/auth/logout")
        assert r.status_code in (200, 302, 303), (
            f"POST /auth/logout -> {r.status_code}: {r.text[:300]}"
        )
        # Cookie cleared (expired) or absent.
        cleared = "ps06_session" not in r.cookies or (
            r.cookies.get("ps06_session") in ("", None)
        )
        # A Set-Cookie with Max-Age=0/expired also counts as cleared.
        set_cookie = r.headers.get("set-cookie", "")
        expired = "ps06_session" in set_cookie and (
            "max-age=0" in set_cookie.lower() or "expires=" in set_cookie.lower()
        )
        assert cleared or expired, (
            f"logout must clear ps06_session; cookies={dict(r.cookies)} "
            f"set-cookie={set_cookie!r}"
        )


# ===========================================================================
# §17.2.B.1 — Student endpoint row-level security (isolation)
# ===========================================================================


class TestStudentRowLevelIsolation:
    """Student A must 403 on student B's profile/matches/plan/interviews."""

    _CROSS_PATHS = (
        "/students/{b}",
        "/students/{b}/matches",
        "/students/{b}/plan",
        "/students/{b}/interviews",
    )

    @pytest.fixture()
    def two_students(self, client):
        a, b = _mk_students(client, 2)
        return client, a, b

    def test_student_can_read_own_records(self, two_students):
        """Anti-vacuity baseline: the student CAN read their own records (200),
        so a 403 on another student proves isolation, not a blanket block."""
        client, a, _b = two_students
        h = _student_h(a)
        r = client.get(f"/students/{a}", headers=h)
        assert r.status_code == 200, (
            f"own profile must be readable: {r.status_code} {r.text[:200]}"
        )

    @pytest.mark.parametrize("path_tpl", _CROSS_PATHS)
    def test_student_cannot_read_other_student(self, two_students, path_tpl):
        """Student A -> student B's records: 403 (§17.2.B.1)."""
        client, a, b = two_students
        h = _student_h(a)
        path = path_tpl.format(b=b)
        r = client.get(path, headers=h)
        assert r.status_code == 403, (
            f"{path} as student A must be 403 (row-level security), got "
            f"{r.status_code}: {r.text[:200]}"
        )

    def test_403_reason_is_row_level_not_role(self, two_students):
        """The 403 message must be the row-level isolation message, proving the
        denial is object-level (own-student scope) and not a blanket role block."""
        client, a, b = two_students
        r = client.get(f"/students/{b}", headers=_student_h(a))
        assert r.status_code == 403
        body = r.text.lower()
        assert "own" in body or "placement record" in body or "access denied" in body, (
            f"403 must state row-level isolation reason, got: {r.text[:200]}"
        )


# ===========================================================================
# §17.2.B.3 — Student blocked from audit + scrape; faculty blocked from audit
# ===========================================================================


class TestStudentBlockedFromAdminAndScrape:
    def test_student_blocked_from_audit(self, client):
        """Student gets 403 on /admin/audit and /api/admin/audit (§17.3.1)."""
        a = _mk_students(client, 1)[0]
        h = _student_h(a)
        for path in ("/admin/audit", "/api/admin/audit"):
            r = client.get(path, headers=h)
            assert r.status_code == 403, f"{path} as student must be 403, got {r.status_code}"

    def test_student_blocked_from_scrape(self, client):
        """POST /api/jobs/scrape strictly restricted to tpo_admin (§17.2.B.3).
        A student triggering a live external scrape must be 403."""
        a = _mk_students(client, 1)[0]
        h = _student_h(a)
        # Mock the underlying scrape so the test is offline/deterministic.
        from unittest.mock import patch

        with patch("app.scraper.service.scrape_jobs") as mock_scrape:
            r = client.post("/api/jobs/scrape",
                            json={"search_term": "Python", "results_wanted": 1},
                            headers=h)
        assert r.status_code == 403, (
            f"POST /api/jobs/scrape as student must be 403, got {r.status_code}"
        )
        # And no scrape actually executed (blocked before the provider ran).
        mock_scrape.assert_not_called()

    def test_student_blocked_from_scrape_stream(self, client):
        """POST /api/jobs/scrape/stream also tpo_admin-only (live external work)."""
        from unittest.mock import patch

        a = _mk_students(client, 1)[0]
        h = _student_h(a)
        with patch("app.scraper.service.scrape_jobs"):
            r = client.post("/api/jobs/scrape/stream",
                            json={"search_term": "Python", "results_wanted": 1},
                            headers=h)
        assert r.status_code == 403, (
            f"POST /api/jobs/scrape/stream as student must be 403, got {r.status_code}"
        )


class TestFacultyVsAdminSeparation:
    """Faculty mentor: 403 on audit, 200 on approvals (§17.2.B.3 / §17.3.1).
    These are ALREADY WORKING (measured-green baseline) — pinned as regressions."""

    def test_faculty_blocked_from_audit(self, client):
        h = {"X-Demo-Role": "faculty_mentor"}
        for path in ("/admin/audit", "/api/admin/audit"):
            r = client.get(path, headers=h)
            assert r.status_code == 403, f"{path} as faculty must be 403, got {r.status_code}"

    def test_faculty_allowed_on_approvals(self, client):
        h = {"X-Demo-Role": "faculty_mentor"}
        r = client.get("/admin/approvals", headers=h)
        assert r.status_code == 200, (
            f"faculty must reach /admin/approvals (first-level endorser), got {r.status_code}"
        )

    def test_faculty_readonly_market_scraper_metadata(self, client):
        """Faculty gets 200 on read-only scraper metadata (§17.2.C.2),
        but not on live scrape execution."""
        h = {"X-Demo-Role": "faculty_mentor"}
        r = client.get("/api/jobs/scrape/status", headers=h)
        assert r.status_code == 200, f"faculty scrape status (read-only) got {r.status_code}"

    def test_tpo_admin_reaches_audit(self, client):
        """Anti-vacuity: the audit endpoint IS reachable for the right role,
        so the faculty/student 403 proves the gate, not a dead route."""
        h = {"X-Demo-Role": "tpo_admin"}
        r = client.get("/admin/audit", headers=h)
        assert r.status_code == 200, f"tpo_admin must reach /admin/audit, got {r.status_code}"


# ===========================================================================
# §17.2.B.2 — Copilot strictly bound to self
# ===========================================================================


class TestCopilotSelfBinding:
    """A student's copilot query must be forcibly bound to their own student_id."""

    def test_student_cannot_query_another_students_context(self, client):
        a, b = _mk_students(client, 2)
        h = _student_h(a)
        r = client.post("/api/copilot/ask",
                        json={"question": "What is my GPA?", "student_id": b},
                        headers=h)
        assert r.status_code == 403, (
            f"copilot must reject cross-student request with 403; got {r.status_code}: {r.text[:300]}"
        )
        assert "Candidate 1" not in r.text, (
            "copilot 403 response must not leak student B's context"
        )

    def test_copilot_requires_identity_header_for_binding(self, client):
        """Anti-vacuity: with an explicit student identity the binding applies;
        proves the hook reads X-Demo-Student-ID (not a hard-coded always-A)."""
        a = _mk_students(client, 1)[0]
        h = _student_h(a)
        r = client.post("/api/copilot/ask",
                        json={"question": "What is my GPA?", "student_id": a},
                        headers=h)
        assert r.status_code == 200
        assert r.json().get("student_id") == a, (
            "caller's own query must stay bound to self"
        )


# ===========================================================================
# §17.2.C — Dynamic UI / sidebar gating
# ===========================================================================


class TestDynamicSidebarGating:
    """base.html must hide admin links + global switcher in student mode."""

    @staticmethod
    def _sidebar_block(html: str) -> str:
        """Extract the <nav id="sidebar">...</nav> block (the sidebar only,
        excluding the command-palette JS which legitimately routes to /admin)."""
        m = re.search(r'<nav id="sidebar".*?</nav>', html, re.DOTALL)
        assert m, "sidebar <nav id=\"sidebar\"> block not found in rendered page"
        return m.group(0)

    @staticmethod
    def _render_home(client, headers: dict) -> str:
        r = client.get("/", headers=headers)
        assert r.status_code == 200, f"home render failed: {r.status_code}"
        return r.text

    def test_student_mode_hides_admin_links(self, client):
        """Student sidebar shows ZERO Platform admin links (§17.2.C.1)."""
        a = _mk_students(client, 1)[0]
        html = self._render_home(client, _student_h(a))
        sidebar = self._sidebar_block(html)
        for href in ('href="/admin"', "/admin/approvals", "/admin/audit"):
            assert href not in sidebar, (
                f"student sidebar leaked admin link {href!r}"
            )
        # The whole Platform section label must be gone from the sidebar.
        assert "Platform" not in sidebar, (
            "student sidebar must not show the Platform section"
        )

    def test_student_mode_hides_global_switcher(self, client):
        """Student view replaces the Active Student dropdown with a badge."""
        a = _mk_students(client, 1)[0]
        html = self._render_home(client, _student_h(a))
        sidebar = self._sidebar_block(html)
        assert 'id="switch-student"' not in sidebar, (
            "student mode must hide the global student switcher"
        )
        assert "Logged in as" in sidebar or "role-badge" in html, (
            "student mode must show a 'Logged in as' badge"
        )

    def test_tpo_admin_mode_shows_admin_links(self, client):
        """Anti-vacuity: the admin links DO render for tpo_admin, so the student
        absence proves gating, not a dead template."""
        html = self._render_home(client, {"X-Demo-Role": "tpo_admin"})
        sidebar = self._sidebar_block(html)
        assert "/admin" in sidebar, "tpo_admin sidebar must show Platform admin links"
        assert 'id="switch-student"' in sidebar, (
            "tpo_admin keeps the global student switcher"
        )

    def test_student_mode_shows_student_workspace_links(self, client):
        """Student sidebar keeps only student workspace links (§17.2.C.1)."""
        a = _mk_students(client, 1)[0]
        html = self._render_home(client, _student_h(a))
        sidebar = self._sidebar_block(html)
        for link in ('href="/"', "/copilot"):
            assert link in sidebar, f"student workspace link {link!r} missing"


# ===========================================================================
# M9-fix gate — Student Persona Binding, Login Navigation, Copilot Lockdown
# (@item:l/fix-student-persona-binding-login-navigation-and, Coordinator spec)
# Self-arming: RED on every contract @Coder-Flash has not yet landed.
#
# Baseline (measured 2026-11-02, pre-fix):
#   - POST /api/auth/switch-role role=student, no student_id -> ps06_session
#     student_id == "demo-student" (NOT a real DB student)
#   - GET /api/auth/role -> student_id "demo-student", name "Student",
#     department None (no DB binding)
#   - /copilot student: VISIBLE #copilot-student select with 2 peer options,
#     no Locked badge
#   - /copilot faculty (x-demo-department CSE): shows ALL students (not
#     dept-filtered)
#   - /login: 0 student cards (only 3 static fast-login role buttons)
#   - base.html: no /login portal link, no /auth/logout signout
# ===========================================================================


def _real_students(client) -> list[dict]:
    """Real DB students created via POST /api/students (hermetic, no reliance
    on demo-cohort seeding which the base fixture doesn't run). Creates one
    CSE and one ECE student so dept-scoping is testable.
    Returns [{'id','name','branch'}]. Anti-vacuity: created ids come back."""
    specs = [
        {"name": "Alpha Student", "branch": "CSE", "year": 3, "gpa": 8.0},
        {"name": "Beta Student", "branch": "ECE", "year": 3, "gpa": 7.5},
    ]
    out = []
    for spec in specs:
        r = client.post("/api/students", json=spec)
        assert r.status_code in (200, 201), r.text[:200]
        sid = r.json()["id"]
        out.append({"id": sid, "name": spec["name"], "branch": spec["branch"]})
    return out


def _decode_session(token: str) -> dict:
    """Decode the base64 JSON ps06_session cookie value (signed or legacy).

    Milestone 15 tokens are ``<base64url_payload>.<hmac_sig>``; split off the
    signature before base64-decoding the payload so both signed and legacy
    unsigned cookies decode.
    """
    import base64
    import json
    raw = token.strip('"').split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(raw).decode())


class TestStudentPersonaBinding:
    """switch-role + /api/auth/role must bind students to REAL DB identities."""

    def test_student_persona_binding_uses_real_student_id(self, client):
        """switch-role role=student WITHOUT student_id must set a REAL DB
        student id, not the literal 'demo-student' placeholder (spec #1)."""
        real = _real_students(client)
        assert real, "seeded cohort must exist for anti-vacuity probe"
        r = client.post("/api/auth/switch-role", json={"role": "student"})
        assert r.status_code == 200, r.text[:200]
        assert "ps06_session" in r.cookies, "switch-role must set ps06_session"
        claims = _decode_session(r.cookies["ps06_session"])
        sid = str(claims.get("student_id") or "")
        assert sid and sid != "demo-student", (
            f"student_id must be a real DB id, got {sid!r} (still the demo placeholder)"
        )
        # And the id must actually exist in the seeded cohort.
        real_ids = {s["id"] for s in real}
        assert sid in real_ids, (
            f"bound student_id {sid!r} not among real students {sorted(real_ids)}"
        )

    def test_get_api_auth_role_reports_real_student_id(self, client):
        """GET /api/auth/role reports a real student id + name + department
        after switch-role (spec #1). Anti-vacuity: /api/auth/role itself works
        (200 + role shape) so a wrong-value assertion isn't a dead route."""
        real = _real_students(client)
        assert real
        r = client.post("/api/auth/switch-role", json={"role": "student"})
        sess = r.cookies["ps06_session"]
        client.cookies.set("ps06_session", sess)
        r = client.get("/api/auth/role")
        assert r.status_code == 200, r.text[:200]
        body = r.json()
        assert body.get("role") == "student"
        user = body.get("user") or {}
        sid = str(user.get("student_id") or "")
        assert sid and sid != "demo-student", (
            f"/api/auth/role student_id must be real, got {sid!r}"
        )
        # name must be a real person name from the cohort (not the generic
        # 'Student' placeholder), department/branch must be present.
        real_names = {s["name"] for s in real}
        assert user.get("name") in real_names, (
            f"role name {user.get('name')!r} not a real cohort student "
            f"(expected one of {sorted(real_names)})"
        )
        assert user.get("department") not in (None, ""), (
            "role department must be populated for a bound student"
        )

    def test_student_can_switch_between_real_student_personas(self, client):
        """switch-role WITH an explicit student_id binds to THAT student's id,
        name, and branch (spec #2)."""
        real = _real_students(client)
        assert real
        target = real[0]
        r = client.post(
            "/api/auth/switch-role",
            json={"role": "student", "student_id": target["id"]},
        )
        assert r.status_code == 200, r.text[:200]
        claims = _decode_session(r.cookies["ps06_session"])
        assert claims.get("student_id") == target["id"], (
            f"explicit student_id {target['id']!r} not bound, got "
            f"{claims.get('student_id')!r}"
        )
        assert claims.get("name") == target["name"], (
            f"switched persona name {claims.get('name')!r} != "
            f"target {target['name']!r}"
        )
        # branch is stored as department claim (Student.branch -> persona)
        dept = str(claims.get("department") or "")
        assert target["branch"] and target["branch"] in dept or dept == target["branch"], (
            f"switched persona branch/department {dept!r} != target "
            f"{target['branch']!r}"
        )


class TestStudentWorkspaceAccess:
    """Authenticated student can reach their OWN workspace (anti-complement to
    the row-level 403s — spec #3)."""

    def test_student_workspace_links_accessible_without_403(self, client):
        real = _real_students(client)
        assert real
        sid = real[0]["id"]
        h = _student_h(sid)
        for path in (
            f"/students/{sid}",
            f"/students/{sid}/matches",
            f"/students/{sid}/plan",
            f"/students/{sid}/progress",
            f"/students/{sid}/interviews",
        ):
            r = client.get(path, headers=h)
            assert r.status_code == 200, (
                f"own workspace {path} as student must be 200, got "
                f"{r.status_code}: {r.text[:200]}"
            )


class TestCopilotLockdown:
    """Copilot student context locked to self; faculty filtered to department
    (spec #4, #5)."""

    def test_copilot_student_dropdown_hidden_and_locked_for_student_role(self, client):
        real = _real_students(client)
        assert real
        me, _peer = real[0], real[1]
        h = _student_h(me["id"])
        r = client.get("/copilot", headers=h)
        assert r.status_code == 200, r.text[:200]
        html = r.text
        # The peer selector must not be VISIBLE (hidden input or absent select).
        sel = re.search(r'<select id="copilot-student"[^>]*>', html)
        if sel:
            assert "hidden" in sel.group(0), (
                "student copilot #copilot-student select must be hidden, "
                f"got visible tag: {sel.group(0)!r}"
            )
        # No peer-student options (other than self) rendered for a student.
        opts = re.findall(r'<option value="(stu-[^"]+)"', html)
        peer_ids = {s["id"] for s in real}
        leaked = {o for o in opts if o != me["id"] and o in peer_ids}
        assert not leaked, (
            f"student copilot leaked peer student selector options: {sorted(leaked)}"
        )
        # A locked context badge showing the student's name must be present.
        assert "ocked" in html or "Locked" in html, (
            "student copilot must render a Locked context badge"
        )
        assert me["name"] in html, (
            f"locked badge must show student name {me['name']!r}"
        )

    def test_copilot_faculty_dropdown_filtered_to_department(self, client):
        real = _real_students(client)
        assert real
        # CSE cohort students vs a non-CSE one (Meera Iyer is ECE in seed).
        cse = [s for s in real if s["branch"].upper() == "CSE"]
        non_cse = [s for s in real if s["branch"].upper() != "CSE"]
        assert cse and non_cse, "seed must have both CSE and non-CSE students"
        h = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"}
        r = client.get("/copilot", headers=h)
        assert r.status_code == 200, r.text[:200]
        opts = set(re.findall(r'<option value="(stu-[^"]+)"', r.text))
        cse_ids = {s["id"] for s in cse}
        non_cse_ids = {s["id"] for s in non_cse}
        leaked = opts & non_cse_ids
        assert not leaked, (
            f"faculty copilot dropdown leaked non-CSE students: {sorted(leaked)}"
        )
        # Anti-vacuity: at least one CSE student IS offered (filter works, not
        # an empty dropdown).
        assert opts & cse_ids, (
            "faculty copilot dropdown must offer at least one CSE mentee"
        )


class TestLoginPersonasAndNavigation:
    """Login renders real student personas; base shell has portal + signout
    (spec #6, #7)."""

    def test_login_page_renders_student_personas_from_db(self, client):
        real = _real_students(client)
        assert real
        r = client.get("/login")
        assert r.status_code == 200, r.text[:200]
        html = r.text
        # At least one real student id, name, and branch must appear as a card.
        sid = real[0]["id"]
        assert sid in html, (
            f"login page must render real student id {sid!r} from DB"
        )
        assert real[0]["name"] in html, (
            f"login page must render student name {real[0]['name']!r}"
        )
        if real[0]["branch"]:
            assert real[0]["branch"] in html, (
                f"login page must render student branch {real[0]['branch']!r}"
            )

    def test_base_html_has_portal_login_and_signout_navigation(self, client):
        """Rendered base shell contains visible links for JKLU Portal /login
        and sign-out /auth/logout (spec #7)."""
        r = client.get("/", headers={"X-Demo-Role": "tpo_admin"})
        assert r.status_code == 200, r.text[:200]
        html = r.text
        # Portal link to /login must be present as an href.
        assert re.search(r'href="/login"', html), (
            "base shell must link to JKLU Portal /login"
        )
        # Sign-out must point at /auth/logout.
        assert "/auth/logout" in html, (
            "base shell must expose a sign-out action to /auth/logout"
        )
        # And it must be a visible control (not just a JS string): look for
        # sign-out/logout text on an anchor/button.
        assert re.search(
            r">(Sign ?[Oo]ut|Log ?[Oo]ut|Logout)<", html
        ), "base shell must show a visible Sign out / Logout control"


# ===========================================================================
# §17.3.3 — Full regression / zero-emoji gates are asserted at run time; this
# file only pins the RBAC-specific contracts. (sui_shell + ruff run separately.)
# ===========================================================================
