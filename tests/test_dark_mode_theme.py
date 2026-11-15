"""Tests for JKLU CareerOS dark mode toggle, CSS contrast overrides, and persistence across all pages."""
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session

from app.api.main import AppContext, create_app
from app.api.routes.campus_drives import _create_jnf_token
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _make_client() -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(
        engine=eng,
        taxonomy=taxonomy,
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    return TestClient(create_app(ctx))


def _seed_student(client: TestClient) -> str:
    r = client.post(
        "/api/students",
        files={"file": ("resume.txt", b"Skills: Python, FastAPI, SQL\n", "text/plain")},
        data={"name": "Theme Tester", "branch": "CSE", "year": "4", "gpa": "8.5"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_theme_static_assets_served_and_complete():
    client = _make_client()
    with client:
        css_resp = client.get("/static/theme.css")
        assert css_resp.status_code == 200
        css = css_resp.text
        for required in (
            "html.dark body",
            "html.dark #sidebar",
            "html.dark .bg-white",
            "html.dark .editorial-card",
            "html.dark .bg-\\[\\#E3F2EA\\]\\/60",
            "html.dark .bg-\\[\\#EEF5FB\\]\\/70",
            "html.dark .bg-\\[\\#E3F2EA\\]\\/50",
            "html.dark .text-\\[\\#1C1E1B\\]",
            "html.dark .text-\\[\\#5C5A52\\]",
            "html.dark .text-\\[\\#1D5C42\\]",
            "@media print",
        ):
            assert required in css, f"missing dark rule: {required}"

        js_resp = client.get("/static/theme.js")
        assert js_resp.status_code == 200
        js = js_resp.text
        assert "careeros-theme" in js
        assert "CareerOSTheme" in js
        assert "syncChartTheme" in js
        assert "Dark Mode" in js
        assert "Light Mode" in js


def test_all_ui_shells_expose_theme_toggle_and_assets():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        client.get(f"/api/students/{sid}/matches")
        interview = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
        assert interview.status_code in (200, 201), interview.text
        iid = interview.json()["interview_id"]

        routes = [
            "/",
            "/login",
            "/copilot",
            "/admin",
            "/admin/policies",
            "/admin/approvals",
            "/admin/audit",
            f"/students/{sid}",
            f"/students/{sid}/matches",
            f"/students/{sid}/plan",
            f"/students/{sid}/progress",
            f"/students/{sid}/interviews",
            f"/interviews/{iid}",
            f"/jnf/{_create_jnf_token()}",
        ]
        for path in routes:
            res = client.get(path)
            assert res.status_code == 200, f"{path} failed: {res.status_code}"
            html = res.text
            assert 'id="theme-toggle"' in html, f"{path} missing #theme-toggle"
            assert "/static/theme.css" in html, f"{path} missing theme.css"
            assert "/static/theme.js" in html, f"{path} missing theme.js"
