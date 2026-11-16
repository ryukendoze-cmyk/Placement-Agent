"""Regression checks for frontend security, accessibility, PWA, Electron/Mobile, and DevOps audit fixes."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import AppContext, create_app
from app.config import Settings
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


def _make_client(*, with_seed: bool = False) -> TestClient:
    eng = make_engine("sqlite://")
    init_db(eng)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    if with_seed:
        from sqlmodel import Session

        with Session(eng) as db:
            seed(db, ctx.taxonomy)
    return TestClient(create_app(ctx))


def test_fe01_xss_hardening_in_templates() -> None:
    matches_html = (ROOT / "app" / "templates" / "student" / "matches.html").read_text(encoding="utf-8")
    plan_html = (ROOT / "app" / "templates" / "student" / "plan.html").read_text(encoding="utf-8")
    copilot_html = (ROOT / "app" / "templates" / "copilot.html").read_text(encoding="utf-8")
    interviews_html = (ROOT / "app" / "templates" / "student" / "interviews.html").read_text(encoding="utf-8")
    profile_html = (ROOT / "app" / "templates" / "student" / "profile.html").read_text(encoding="utf-8")

    # Inline onclick Jinja interpolation replaced with data-* delegation
    assert 'onclick="genPlan(' not in matches_html
    assert 'onclick="applyJob(' not in matches_html
    assert 'onclick="removeJobOpportunity(' not in matches_html
    assert 'onclick="selectMasterCard(' not in matches_html
    assert 'onclick="toggleFavoriteJob(' not in matches_html
    assert 'data-action="gen-plan"' in matches_html
    assert 'data-action="apply-job"' in matches_html
    assert 'data-action="remove-job"' in matches_html
    assert "CSS.escape" in matches_html
    assert "encodeURIComponent(sid)" in matches_html

    assert 'onclick="openVerifyModal(' not in plan_html
    assert 'onclick="openWaiverModal(' not in plan_html
    assert 'onclick="toggleGuide(' not in plan_html
    assert 'onclick="inspectDagNode(' not in plan_html
    assert "function escHtml(" in plan_html
    assert "${escHtml(q.prompt)}" in plan_html
    assert "${escHtml(d.title)}" in plan_html

    assert 'addBubble("user", escHtml(q))' in copilot_html
    assert "verified.map(escHtml)" in copilot_html

    assert 'onclick="startInterview(' not in interviews_html
    assert 'data-sid="{{ sid }}" data-job-id="{{ top_job_id }}"' in interviews_html
    assert "{{ student.name | tojson }}" in profile_html


def test_fe02_fe04_fe05_fe06_navigation_error_a11y_fixes() -> None:
    plan_html = (ROOT / "app" / "templates" / "student" / "plan.html").read_text(encoding="utf-8")
    matches_html = (ROOT / "app" / "templates" / "student" / "matches.html").read_text(encoding="utf-8")
    base_html = (ROOT / "app" / "templates" / "base.html").read_text(encoding="utf-8")
    index_html = (ROOT / "app" / "templates" / "index.html").read_text(encoding="utf-8")
    analytics_html = (ROOT / "app" / "templates" / "admin" / "analytics.html").read_text(encoding="utf-8")
    approvals_html = (ROOT / "app" / "templates" / "admin" / "approvals.html").read_text(encoding="utf-8")

    # FE-02: SOP-04 points to plural /interviews
    assert '/students/{{ sid }}/interviews"' in plan_html
    assert '/students/{{ sid }}/interview"' not in plan_html

    # FE-05: res.ok checks before .json() / reload
    assert "if (!res.ok)" in plan_html
    assert "if (!r.ok)" in matches_html

    # FE-05: pageshow reset & PNG/JPG format badges
    assert "pageshow" in index_html
    assert ">PNG<" in index_html and ">JPG<" in index_html

    # FE-05: Blob URL cleanup & base.html fallbacks
    assert "URL.revokeObjectURL" in analytics_html
    assert "encodeURIComponent(id)" in approvals_html
    assert "/admin#scraper" not in base_html
    assert "copilot, home, admin, approvals, audit, know more" in base_html

    # FE-06: invalid <button><input type="checkbox"></button> replaced by <label> preserving IDs
    assert not re.search(r"<button[^>]*id=[\"']filter-fresher-pill[\"']", matches_html)
    assert re.search(r"<label[^>]*id=[\"']filter-fresher-pill[\"'][^>]*for=[\"']filter-fresher-checkbox[\"']", matches_html)
    assert 'id="filter-fresher-checkbox"' in matches_html

    # FE-06: dialog roles + Escape closing
    assert 'id="scrape-modal" role="dialog" aria-modal="true"' in matches_html
    assert 'modal.setAttribute("role", "dialog")' in matches_html
    assert 'overlay.setAttribute("role", "dialog")' in base_html
    assert "confirmModal.remove()" in base_html


def test_fe07_service_worker_precache_paths() -> None:
    sw_js = (ROOT / "app" / "static" / "sw.js").read_text(encoding="utf-8")
    assert '"/manifest.webmanifest"' in sw_js
    assert '"/static/manifest.webmanifest"' not in sw_js
    assert '"/static/theme.css"' in sw_js
    assert '"/static/theme.js"' in sw_js

    # Verify /manifest.webmanifest is actually served at root
    client = _make_client()
    resp = client.get("/manifest.webmanifest")
    assert resp.status_code == 200


def test_fe08_fe09_desktop_and_mobile_hardening() -> None:
    main_js = (ROOT / "desktop" / "main.js").read_text(encoding="utf-8")
    assert "validateHttpUrl" in main_js
    assert "sandbox: true" in main_js
    assert "setWindowOpenHandler" in main_js
    assert 'will-navigate' in main_js
    assert "!app.isPackaged" in main_js

    cap_cfg = json.loads((ROOT / "mobile" / "capacitor.config.json").read_text(encoding="utf-8"))
    assert cap_cfg["android"]["allowMixedContent"] is False
    assert "*" not in cap_cfg["server"]["allowNavigation"]

    mobile_index = (ROOT / "mobile" / "www" / "index.html").read_text(encoding="utf-8")
    assert "validateHttpUrl" in mobile_index
    assert 'parsed.protocol !== "http:" && parsed.protocol !== "https:"' in mobile_index

    manifest_xml = (ROOT / "mobile" / "android" / "app" / "src" / "main" / "AndroidManifest.xml").read_text(encoding="utf-8")
    assert 'android:allowBackup="false"' in manifest_xml

    build_gradle = (ROOT / "mobile" / "android" / "app" / "build.gradle").read_text(encoding="utf-8")
    assert "minifyEnabled true" in build_gradle


def test_ops_and_qa_configs_synced() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = pyproject["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith("live:") for m in markers)

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env\n" in dockerignore or dockerignore.endswith(".env")
    assert ".env.*" in dockerignore

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ghcr.io/astral-sh/uv:latest" not in dockerfile

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    expected_env_keys = {
        k.upper() for k in Settings.model_fields.keys()
    }
    for key in expected_env_keys:
        assert re.search(rf"^#?\s*{re.escape(key)}=", env_example, re.MULTILINE), (
            f".env.example missing {key}"
        )


def test_matches_template_hides_restricted_actions_by_role_and_source() -> None:
    matches_src = (ROOT / "app" / "templates" / "student" / "matches.html").read_text(encoding="utf-8")
    assert 'st !== "nominated" || currentRole === "tpo_admin"' in matches_src
    assert "{% if role != 'faculty_mentor' %}" in matches_src
    assert "{% if not is_live %}" in matches_src

    plan_src = (ROOT / "app" / "templates" / "student" / "plan.html").read_text(encoding="utf-8")
    assert "CSS.escape(String(q.id))" in plan_src

    client = _make_client(with_seed=True)
    r_stu = client.post(
        "/api/students",
        headers={"X-Demo-Role": "tpo_admin"},
        files={"file": ("resume.txt", b"Skills: Python, FastAPI, SQL\n", "text/plain")},
        data={"name": "Role Action Probe", "branch": "CSE", "year": "4", "gpa": "8.5"},
    )
    assert r_stu.status_code == 201
    sid = r_stu.json()["id"]

    # Insert one live job match so we verify Apply / Nominate is hidden on live cards
    from sqlmodel import Session

    from app.models import Job, JobMatch

    with Session(client.app.state.ctx.engine) as db:
        db.add(
            Job(
                id="job-ext-livecheck",
                title="Live Backend Role",
                company="LiveCo",
                description="Python FastAPI",
                required_skills=["python"],
                source="naukri",
                is_live=True,
                owner_student_id=sid,
                job_url="https://example.com/jobs/1",
            )
        )
        db.add(
            JobMatch(
                id="jm-livecheck",
                student_id=sid,
                job_id="job-ext-livecheck",
                score=85,
                band="strong",
                breakdown={
                    "eligible": True,
                    "matched_required": [{"skill": "python", "name": "Python", "evidence": "ev"}],
                    "missing_required": [],
                },
            )
        )
        db.commit()

    fac_html = client.get(
        f"/students/{sid}/matches",
        headers={"X-Demo-Role": "faculty_mentor", "X-Demo-Department": "CSE"},
    )
    assert fac_html.status_code == 200
    assert 'class="btn-favorite-job' not in fac_html.text

    stu_html = client.get(
        f"/students/{sid}/matches",
        headers={"X-Demo-Role": "student", "X-Demo-Student-ID": sid},
    )
    assert stu_html.status_code == 200
    assert 'class="btn-favorite-job' in stu_html.text
    # The live job card block must not expose Apply / Nominate
    live_idx = stu_html.text.find('data-job-id="job-ext-livecheck"')
    assert live_idx != -1
    live_slice = stu_html.text[live_idx : live_idx + 5000]
    assert 'data-action="apply-job"' not in live_slice
