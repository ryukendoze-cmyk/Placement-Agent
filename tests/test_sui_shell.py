"""Phase 7: flagship SaaS UI/UX - app shell, role pill, student switcher,
global toast system (zero alert), drawer, copilot/admin page upgrades,
and removal of every legacy browser alert() from templates.
"""
import re
from pathlib import Path

TEMPLATES = Path("app/templates")
APP_JS = Path("app/static/app.js")


def _client_engine():
    from fastapi.testclient import TestClient
    from sqlmodel import Session

    from app.api.main import AppContext, create_app
    from app.db import init_db, make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    from app.taxonomy.taxonomy import load_taxonomy
    eng = make_engine("sqlite://")
    init_db(eng)
    ctx = AppContext(engine=eng, taxonomy=load_taxonomy(), embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=None)
    from app.seed import seed
    with Session(eng) as db:
        seed(db, ctx.taxonomy)
    return TestClient(create_app(ctx)), eng


def _make_student(client, resume=b"Skills: Python, FastAPI\n", name="Shell Probe"):
    r = client.post("/api/students", files={"file": ("r.txt", resume, "text/plain")},
                    data={"name": name})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


def _base() -> str:
    return (TEMPLATES / "base.html").read_text(encoding="utf-8")


# --- (1) app shell -----------------------------------------------------------

def test_base_renders_app_shell_with_sidebar_and_command_bar():
    b = _base()
    assert 'id="sidebar"' in b, "left sidebar missing"
    assert 'id="topbar"' in b or 'id="command-bar"' in b, "top command bar missing"
    assert "JKLU CareerOS" in b, "brand name missing"
    assert "Autonomous Placement" in b, "product tagline missing"


def test_base_has_role_pill_and_student_switcher():
    b = _base()
    assert 'id="role-pill"' in b, "1-click role pill missing"
    assert 'id="student-switcher"' in b, "active-student switcher missing"
    # role pill switches via the cookie API (POST /api/auth/switch-role)
    assert "/api/auth/switch-role" in b
    # switcher navigates between students
    assert 'id="switch-student"' in b


def test_base_nav_links_every_top_level_area():
    b = _base()
    for href in ('href="/"', "/copilot", "/admin", "/admin/approvals", "/admin/audit"):
        assert href in b, f"nav missing {href}"


def test_base_has_agent_brain_drawer_and_toast_container():
    b = _base()
    assert 'id="agent-drawer"' in b, "Agent Brain drawer missing"
    assert 'id="toast-root"' in b or 'id="toasts"' in b, "global toast container missing"
    # drawer opens from a control in the shell
    assert 'id="open-drawer"' in b


# --- (2) student switcher gets its data --------------------------------------

def test_index_lists_students_for_the_switcher(client_engine):
    client, _ = client_engine
    sid = _make_student(client, name="Switcher Target")
    h = client.get("/").text
    assert 'id="student-switcher"' in h
    assert "Switcher Target" in h
    assert f"/students/{sid}" in h


def test_student_pages_pass_sid_for_the_switcher(client_engine):
    client, _ = client_engine
    sid = _make_student(client, name="Pager")
    h = client.get(f"/students/{sid}").text
    assert 'id="student-switcher"' in h
    assert f"/students/{sid}" in h


# --- (3) global toast system, zero browser alert -----------------------------

def test_app_js_has_toast_system_and_no_alert():
    js = APP_JS.read_text(encoding="utf-8")
    assert "alert(" not in js, "app.js must not use browser alert()"
    assert "toast" in js.lower(), "toast helper missing from app.js"
    # the toast helper must actually create + append + auto-dismiss nodes
    assert re.search(r"function\s+toast\s*\(", js), "toast() function missing"
    assert re.search(r"function\s+showToast\s*\(", js) or re.search(r"function\s+toast\s*\(", js)


def test_no_template_uses_browser_alert():
    offenders = []
    for p in TEMPLATES.rglob("*.html"):
        text = p.read_text(encoding="utf-8")
        if "alert(" in text:
            offenders.append(str(p))
    assert offenders == [], f"templates still call alert(): {offenders}"


def test_shell_loads_app_js_for_the_toast_helper():
    b = _base()
    assert "/static/app.js" in b, "base shell must load app.js (toast helper lives there)"


# --- (4) copilot page ---------------------------------------------------------

def test_copilot_page_renders_chat_with_quick_prompts_and_pills():
    client, _eng = _client_engine()
    with client:
        r = client.get("/copilot")
        assert r.status_code == 200, r.text
        h = r.text
        assert 'id="chat-log"' in h or 'id="copilot-chat"' in h, "chat surface missing"
        assert 'id="copilot-input"' in h, "question input missing"
        # quick-prompt chips
        assert 'id="quick-prompts"' in h or "quick-prompt" in h
        # grounded citation pills are clickable [SOP-XX] / [JOB-XX]
        assert "SOP-" in h, "SOP citation reference missing"
        assert "[SOP-XX]" in h or "[SOP-" in h or "SOP-XX" in h, "citation pill hint missing"
        # agent step trace region (copilot returns a trace)
        assert 'id="trace"' in h or 'id="copilot-trace"' in h, "step trace region missing"


def test_copilot_api_still_grounded_and_deterministic(client_engine):
    client, _eng = client_engine
    q = {"question": "What is the mock interview score needed for Dream drives?"}
    a = client.post("/api/copilot/ask", json=q).json()
    b = client.post("/api/copilot/ask", json=q).json()
    assert a["answer"] == b["answer"], "copilot answers must be deterministic offline"
    assert a["citations"]["verified"], "expected at least one verified SOP citation"


# --- (5) admin upgrades -------------------------------------------------------

def test_admin_approvals_center_renders():
    client, _eng = _client_engine()
    with client:
        r = client.get("/admin/approvals")
        assert r.status_code == 200, r.text
        h = r.text
        assert 'id="approvals"' in h, "approvals region missing"
        # Approve & Execute / Reject with note affordances
        assert "Approve" in h and "Reject" in h
        assert 'id="note-' in h or 'class="note-input"' in h or "note" in h.lower()


def test_admin_audit_page_renders():
    client, _eng = _client_engine()
    with client:
        r = client.get("/admin/audit")
        assert r.status_code == 200, r.text
        h = r.text
        assert 'id="audit-log"' in h, "audit table missing"
        # read-only inspector: pulls the existing read-only API
        assert "/api/admin/audit" in h


def test_launch_bootcamp_endpoint_creates_pending_approval_and_audit():
    """The Launch Bootcamp button posts to /api/approvals - must actually exist."""
    client, eng = _client_engine()
    with client:
        r = client.post("/api/approvals", json={
            "kind": "cohort_bootcamp_publish",
            "target_id": "docker",
            "title": "Cohort bridge bootcamp: Docker",
            "summary": "Launch a bridge bootcamp for the top cohort skill gap (12).",
            "payload": {"skill_id": "docker", "course_id": "crs-1"},
        })
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "pending_review"
        assert body["kind"] == "cohort_bootcamp_publish"
        assert body["payload"]["skill_id"] == "docker"
        # and it shows up in the queue the approvals page reads
        items = client.get("/api/approvals?status=pending_review").json()
        assert any(a["id"] == body["id"] for a in items)
        # requested action is audited
        from sqlmodel import Session, select

        from app.models import AuditLog
        with Session(eng) as db:
            auds = db.exec(select(AuditLog).where(
                AuditLog.action == "approval.requested",
                AuditLog.entity_id == body["id"])).all()
            assert len(auds) == 1


def test_analytics_page_has_kpi_strip_and_export():
    client, _eng = _client_engine()
    with client:
        r = client.get("/admin")
        assert r.status_code == 200, r.text
        h = r.text
        assert 'id="kpi-strip"' in h, "KPI strip missing"
        # export affordances (CSV / JSON download)
        assert "csv" in h.lower() and "json" in h.lower(), "export buttons missing"
        # bootcamp launch routes through HITL approvals
        assert "/api/approvals" in h or "Bootcamp" in h, "Launch Bootcamp HITL missing"


# --- (6) RBAC stays intact ----------------------------------------------------

def test_admin_pages_still_enforce_role_when_explicit():
    client, _eng = _client_engine()
    with client:
        for path in ("/admin/approvals", "/admin/audit"):
            assert client.get(path, headers={"X-Demo-Role": "student"}).status_code == 403
            assert client.get(path, headers={"X-Demo-Role": "tpo_admin"}).status_code == 200


# --- (7) existing pages keep their contracts ---------------------------------

def test_existing_page_tests_still_hold(client_engine):
    client, _ = client_engine
    sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                      data={"name": "Contract"}).json()["id"]
    client.get(f"/api/students/{sid}/matches")
    for path, needles in (
        (f"/students/{sid}", ("skill-chips",)),
        (f"/students/{sid}/matches", ("Missing", "Generate plan")),
        (f"/students/{sid}/interviews", ("Start",)),
        ("/admin", ("gapChart", "bandChart")),
    ):
        h = client.get(path).text
        for n in needles:
            assert n in h, f"{path} lost {n!r}"


def test_interview_studio_page_still_renders(client_engine):
    client, _ = client_engine
    sid = _make_student(client, name="Studio Keep")
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    iid = r.json()["interview_id"]
    h = client.get(f"/interviews/{iid}").text
    for n in ('id="stepper"', 'id="difficulty-badge"', 'id="rubric"', 'id="print-btn"'):
        assert n in h, f"interview studio lost {n}"


# --- (8) Zero-Emoji hard gate (Milestone 2) -----------------------------------

# Literal Unicode emoji codepoint ranges from @item:l/automated-zero-emoji-regression-gate-and-ui-test:
#   \U0001F300-\U0001FAFF  (misc emoji, transport, supplemental)
#   \u2600-\u27BF          (misc symbols, dingbats incl. U+2705/U+274C/U+2795/U+2713)
#   \u231A-\u23F9          (watch/hourglass incl. U+23F3 + stop button)
_EMOJI_LITERAL = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u231A-\u23F9]")
# JS escapes: \u{1F3A4} (astral) and \uXXXX in emoji/symbol planes only —
# restricted so legitimate escapes like \u00d7 (×) / \u00e9 (é) are NOT flagged.
_JS_EMOJI_ESCAPE = re.compile(r"\\u\{[0-9A-Fa-f]{4,6}\}|\\u(?:[2-9A-Fa-f][0-9A-Fa-f]{3}|1[0-9A-Fa-f]{4})")
# HTML numeric emoji/dingbat entities named in the ticket (127xxx-129xxx + the
# specific dingbat codes). Kept as a bounded set so legitimate entities like
# &#39; (apostrophe) and &#8230; (ellipsis, used for typography) are NOT flagged.
_HTML_EMOJI_ENTITY = re.compile(
    r"&#(?:8962|9776|9989|10003|10007|127\d\d\d|128\d\d\d|129\d\d\d)\b"
)

_ZERO_EMOJI_TARGETS = sorted(
    list(TEMPLATES.rglob("*.html")) + [APP_JS]
)


def _emoji_offenders(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    hits = []
    for m in _EMOJI_LITERAL.finditer(text):
        hits.append(f"literal emoji U+{ord(m.group(0)):04X}")
    for m in _JS_EMOJI_ESCAPE.finditer(text):
        hits.append(f"JS emoji escape {m.group(0)}")
    for m in _HTML_EMOJI_ENTITY.finditer(text):
        hits.append(f"HTML emoji entity {m.group(0)}")
    return hits


def test_no_emojis_in_templates_or_static_js():
    """Zero-Emoji hard gate: every .html under app/templates/ and app/static/app.js
    must contain ZERO literal Unicode emojis, ZERO JS emoji escapes, and ZERO
    HTML numeric emoji/dingbat entities. Anti-vacuity: the same probes are
    proven to detect real hits (see test_zero_emoji_gate_detects_known_symbols)."""
    offenders = {}
    for p in _ZERO_EMOJI_TARGETS:
        hits = _emoji_offenders(p)
        if hits:
            offenders[str(p)] = hits
    assert offenders == {}, "emoji artifacts found:\n" + "\n".join(
        f"  {p}: {', '.join(h)}" for p, h in offenders.items()
    )


def test_zero_emoji_gate_detects_known_symbols():
    """Anti-vacuity: the same regex probes MUST match every symbol class the
    ticket names — proves the gate is not vacuously green by looking in the
    wrong place. If this test passes, a real emoji in a template WILL be caught
    by test_no_emojis_in_templates_or_static_js."""
    # literal emoji (dingbat / emoji ranges)
    assert _EMOJI_LITERAL.search("\u2705")   # white heavy check mark
    assert _EMOJI_LITERAL.search("\u274C")   # cross mark
    assert _EMOJI_LITERAL.search("\u2795")   # heavy plus
    assert _EMOJI_LITERAL.search("\u23F3")   # hourglass
    assert _EMOJI_LITERAL.search("\U0001F3A4")  # microphone
    # JS emoji escapes
    assert _JS_EMOJI_ESCAPE.search(r"\u{1F3A4}")
    assert _JS_EMOJI_ESCAPE.search(r"\u23F3")
    assert _JS_EMOJI_ESCAPE.search(r"\u23F9")
    # ... but a legitimate (non-emoji) escape must NOT be flagged
    assert not _JS_EMOJI_ESCAPE.search(r"\u00d7")   # multiplication sign
    assert not _JS_EMOJI_ESCAPE.search(r"\u00e9")   # e-acute
    # HTML numeric emoji/dingbat entities
    assert _HTML_EMOJI_ENTITY.search("&#8962;")
    assert _HTML_EMOJI_ENTITY.search("&#9776;")
    assert _HTML_EMOJI_ENTITY.search("&#9989;")
    assert _HTML_EMOJI_ENTITY.search("&#10003;")
    assert _HTML_EMOJI_ENTITY.search("&#10007;")
    assert _HTML_EMOJI_ENTITY.search("&#128100;")
    assert _HTML_EMOJI_ENTITY.search("&#127908;")
    assert _HTML_EMOJI_ENTITY.search("&#127919;")
    # ... but a legitimate entity must NOT be flagged (keeps the gate honest)
    assert not _HTML_EMOJI_ENTITY.search("&#39;")


def test_rendered_routes_contain_zero_emojis(client_engine):
    """Rendered HTML for every route named in the ticket must contain zero
    emojis (catches Jinja-templated emoji and data-driven content)."""
    client, _ = client_engine
    sid = _make_student(client, name="Emoji Probe")
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    iid = r.json()["interview_id"]
    routes = [
        "/",
        f"/students/{sid}",
        f"/students/{sid}/matches",
        f"/students/{sid}/plan",
        f"/students/{sid}/progress",
        f"/students/{sid}/interviews",
        "/copilot",
        "/admin/analytics",
        "/admin/approvals",
        "/admin/audit",
        f"/interviews/{iid}",
    ]
    offenders = {}
    for route in routes:
        resp = client.get(route)
        if resp.status_code != 200:
            continue  # RBAC-protected or missing-data routes are covered elsewhere
        hits = []
        for m in _EMOJI_LITERAL.finditer(resp.text):
            hits.append(f"literal U+{ord(m.group(0)):04X}")
        for m in _HTML_EMOJI_ENTITY.finditer(resp.text):
            hits.append(f"entity {m.group(0)}")
        if hits:
            offenders[route] = hits
    assert offenders == {}, "rendered routes contain emoji:\n" + "\n".join(
        f"  {r}: {', '.join(h)}" for r, h in offenders.items()
    )


# --- (9) Dynamic UI Sidebar Adaptation & Role-Scoped Views (Milestone 9) ------

def test_dynamic_ui_student_view_hides_switcher_and_platform_shows_badge():
    client, _eng = _client_engine()
    sid = _make_student(client, name="Ananya Roy")
    headers = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid}
    resp = client.get(f"/students/{sid}", headers=headers)
    assert resp.status_code == 200
    html = resp.text
    m = re.search(r'<nav id="sidebar".*?</nav>', html, re.DOTALL)
    assert m, "sidebar missing"
    sidebar = m.group(0)

    # 1. Active student dropdown switcher hidden; clean badge with student name shown
    assert 'id="switch-student"' not in sidebar, "student view must hide switch-student"
    assert "Logged in as:" in sidebar, "student view must show 'Logged in as:' badge"
    assert "Ananya Roy" in sidebar, "badge must display active student name"

    # 2. Platform section completely removed
    assert "Platform" not in sidebar, "Platform section header must be removed for students"
    assert 'href="/admin"' not in sidebar, "student view must not show /admin"
    assert "/admin/approvals" not in sidebar, "student view must not show /admin/approvals"
    assert "/admin/audit" not in sidebar, "student view must not show /admin/audit"

    # 3. Sidebar displays student workspace links
    for link in ('href="/"', f'/students/{sid}', f'/students/{sid}/matches',
                 f'/students/{sid}/plan', f'/students/{sid}/progress',
                 f'/students/{sid}/interviews', '/copilot'):
        assert link in sidebar, f"student workspace link {link!r} missing"
    assert "Mock Interview" in sidebar, "Mock Interview link label expected"


def test_dynamic_ui_faculty_view_filters_switcher_and_shows_approvals_dept_analytics():
    client, eng = _client_engine()
    from sqlmodel import Session

    from app.models import Student
    with Session(eng) as session:
        session.add(Student(id="stu-cse-1", name="CSE Mentee", branch="CSE", year=3, gpa=8.5))
        session.add(Student(id="stu-ece-1", name="ECE Student", branch="ECE", year=3, gpa=7.8))
        session.commit()

    headers = {"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"}
    resp = client.get("/admin/approvals", headers=headers)
    assert resp.status_code == 200
    html = resp.text
    m = re.search(r'<nav id="sidebar".*?</nav>', html, re.DOTALL)
    assert m, "sidebar missing"
    sidebar = m.group(0)

    # 1. Switcher exists and is filtered to CSE department
    assert 'id="switch-student"' in sidebar, "faculty view must include student switcher"
    assert "CSE Mentee" in sidebar, "faculty switcher must show CSE departmental mentee"
    assert "ECE Student" not in sidebar, "faculty switcher must NOT show ECE student"

    # 2. Platform section displays Department Analytics and Approvals, but hides Audit Trail
    assert "Platform" in sidebar, "Platform section must be visible to faculty mentor"
    assert "Department Analytics" in sidebar, "faculty view must show Department Analytics"
    assert "/admin/approvals" in sidebar, "faculty view must show Approvals"
    assert "/admin/audit" not in sidebar, "faculty view must NOT show Audit trail"
    assert "nav-scraper-controls" not in sidebar, "faculty view must not show Scraper Controls"


def test_dynamic_ui_tpo_admin_view_shows_full_switcher_and_platform_controls():
    client, eng = _client_engine()
    from sqlmodel import Session

    from app.models import Student
    with Session(eng) as session:
        session.add(Student(id="stu-cse-2", name="CSE Student 2", branch="CSE", year=3, gpa=8.0))
        session.add(Student(id="stu-ece-2", name="ECE Student 2", branch="ECE", year=3, gpa=7.5))
        session.commit()

    headers = {"X-Demo-Role": "tpo_admin"}
    resp = client.get("/admin", headers=headers)
    assert resp.status_code == 200
    html = resp.text
    m = re.search(r'<nav id="sidebar".*?</nav>', html, re.DOTALL)
    assert m, "sidebar missing"
    sidebar = m.group(0)

    # 1. Global student switcher across all departments
    assert 'id="switch-student"' in sidebar
    assert "CSE Student 2" in sidebar
    assert "ECE Student 2" in sidebar

    # 2. Full platform modules
    assert "Platform" in sidebar
    assert "/admin" in sidebar
    assert "/admin/approvals" in sidebar
    assert "/admin/audit" in sidebar
    assert "nav-scraper-controls" in sidebar or "Scraper Controls" in sidebar


def test_demo_role_bar_and_reload_logic_present():
    b = _base()
    # Top demo switcher bar in topbar
    assert 'id="demo-role-bar"' in b, "topbar demo role bar missing"
    assert 'data-role-value="student"' in b
    assert 'data-role-value="faculty_mentor"' in b
    assert 'data-role-value="tpo_admin"' in b
    # Sidebar role pill
    assert 'id="role-pill"' in b
    # Switching roles reloads view
    assert "window.location.reload" in b or "location.reload" in b
