"""Tests for Editable Placement Policies (SOPs), dynamic cascade, grounding sync, and deterministic LLM config."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

import app.campus.policy as campus_policy
import app.rag.policy_rag as policy_rag
from app.agent.verifier import verify_text
from app.analytics.drive_simulator import simulate_placement_drive
from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.bedrock import BedrockLLM
from app.llm.mock import MockLLM
from app.llm.openai import OpenAILLM
from app.matching.engine import band_for, job_tier
from app.models import AuditLog, Interview, Job, Student
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


@pytest.fixture(autouse=True)
def isolated_sops(tmp_path, monkeypatch):
    """Copy placement_sops.json to tmp_path so tests never mutate repo data/placement_sops.json."""
    src = Path(__file__).resolve().parents[1] / "data" / "placement_sops.json"
    dst = tmp_path / "placement_sops.json"
    shutil.copy2(src, dst)
    monkeypatch.setattr(policy_rag, "SOP_FILE", dst)
    monkeypatch.setattr(campus_policy, "SOP_FILE", dst)
    policy_rag.reload_sops()
    yield dst
    policy_rag.reload_sops()


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


def _seed_student_and_mock(client: TestClient, *, name="Policy Probe", branch="CSE",
                           gpa=8.5, backlogs=0, mock_score=70) -> tuple[str, str]:
    r = client.post(
        "/api/students",
        files={"file": ("resume.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n", "text/plain")},
        data={"name": name, "branch": branch, "year": "4", "gpa": str(gpa)},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    with Session(client.app.state.ctx.engine) as db:
        stu = db.get(Student, sid)
        stu.backlogs = backlogs
        db.add(stu)
        job = db.exec(select(Job)).first()
        job.salary_lpa = [10, 12]
        job.min_gpa = 7.0
        job.required_skills = ["python", "fastapi"]
        db.add(job)
        db.commit()
        jid = job.id
        if mock_score is not None:
            iv = Interview(
                id=f"iv-{uuid4().hex[:8]}",
                student_id=sid,
                job_id=jid,
                skill_id="python",
                finished=True,
                score=mock_score,
            )
            db.add(iv)
            db.commit()
    return sid, jid


def test_policy_api_rbac_and_validation():
    client = _make_client()
    with client:
        # Student cannot view or modify admin policies
        assert client.get("/api/admin/policies", headers={"X-Demo-Role": "student"}).status_code == 403
        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 70}},
            headers={"X-Demo-Role": "student"},
        ).status_code == 403

        # Faculty mentor can view read-only, cannot edit or reset
        r_mentor = client.get("/api/admin/policies", headers={"X-Demo-Role": "faculty_mentor"})
        assert r_mentor.status_code == 200
        assert r_mentor.json()["editable"] is False
        assert r_mentor.json()["count"] >= 18

        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 70}},
            headers={"X-Demo-Role": "faculty_mentor"},
        ).status_code == 403
        assert client.post(
            "/api/admin/policies/reset",
            headers={"X-Demo-Role": "faculty_mentor"},
        ).status_code == 403

        # TPO Admin validation guards
        assert client.put(
            "/api/admin/policies/SOP-99",
            json={"parameters": {"min_mock_score": 70}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 404
        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 150}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422
        assert client.put(
            "/api/admin/policies/SOP-03",
            json={"parameters": {"dream_max_backlogs": -1}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422
        assert client.put(
            "/api/admin/policies/SOP-01",
            json={"parameters": {"dream_min_lpa": 20.0, "super_dream_min_lpa": 15.0}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422
        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"applicable_tiers": ["unknown_tier"]}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422
        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"rule_text": "   "},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422


def test_policy_update_writes_audit_log_and_resets():
    client = _make_client()
    with client:
        r = client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 75}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r.status_code == 200, r.text
        assert r.json()["sop"]["parameters"]["min_mock_score"] == 75
        assert policy_rag.load_sop_params("SOP-04")["min_mock_score"] == 75

        with Session(client.app.state.ctx.engine) as db:
            logs = db.exec(
                select(AuditLog).where(AuditLog.action == "policy.updated")
            ).all()
            assert len(logs) == 1
            assert logs[0].entity_type == "sop"
            assert logs[0].entity_id == "SOP-04"
            assert logs[0].actor_role == "tpo_admin"

        r_reset = client.post("/api/admin/policies/reset", headers={"X-Demo-Role": "tpo_admin"})
        assert r_reset.status_code == 200
        assert policy_rag.load_sop_params("SOP-04")["min_mock_score"] == 65


def test_dynamic_cascade_across_eligibility_matching_campus_and_simulator():
    client = _make_client()
    with client:
        sid, jid = _seed_student_and_mock(client, mock_score=70, backlogs=2)
        ctx = client.app.state.ctx

        # 1. SOP-03 default: 2 backlogs exceeds dream limit (1)
        with Session(ctx.engine) as db:
            res = policy_rag.evaluate_student_eligibility(db, ctx, sid, jid)
        assert res["eligible"] is False
        assert any("exceed" in r.lower() for r in res["reasons"])

        # Raise SOP-03 dream_max_backlogs -> 2 via API
        r3 = client.put(
            "/api/admin/policies/SOP-03",
            json={"parameters": {"dream_max_backlogs": 2}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r3.status_code == 200
        b_val, _, _ = campus_policy.backlog_default_for_tier("dream")
        assert b_val == 2

        with Session(ctx.engine) as db:
            res_after_backlog = policy_rag.evaluate_student_eligibility(db, ctx, sid, jid)
        assert res_after_backlog["eligible"] is True

        # 2. Raise SOP-04 min_mock_score 65 -> 75: student with 70 now blocked + simulator updates
        with Session(ctx.engine) as db:
            sim_before = simulate_placement_drive(
                db, ctx, title="Policy Drive", salary_lpa=[10, 12], min_gpa=7.0, required_skills=["python"]
            )
        r4 = client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 75}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r4.status_code == 200

        with Session(ctx.engine) as db:
            res_after_mock = policy_rag.evaluate_student_eligibility(db, ctx, sid, jid)
            sim_after = simulate_placement_drive(
                db, ctx, title="Policy Drive", salary_lpa=[10, 12], min_gpa=7.0, required_skills=["python"]
            )
        assert res_after_mock["eligible"] is False
        assert any("75" in r for r in res_after_mock["reasons"])
        assert sim_after["eligible_now_count"] <= sim_before["eligible_now_count"]

        # Remove dream from applicable_tiers -> student becomes eligible for dream again
        r4_tiers = client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"applicable_tiers": ["super_dream"]}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r4_tiers.status_code == 200
        with Session(ctx.engine) as db:
            res_exempt = policy_rag.evaluate_student_eligibility(db, ctx, sid, jid)
        assert res_exempt["eligible"] is True

        # 3. SOP-01 tier threshold sync across job_tier() and campus_policy.tier_for()
        probe_job = SimpleNamespace(salary_lpa=[10, 16])
        assert job_tier(probe_job) == "super_dream"
        assert campus_policy.tier_for(16.0) == ("super_dream", "SOP-01")

        r1 = client.put(
            "/api/admin/policies/SOP-01",
            json={"parameters": {"dream_min_lpa": 12.0, "super_dream_min_lpa": 20.0}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r1.status_code == 200
        assert job_tier(probe_job) == "dream"
        assert campus_policy.tier_for(16.0) == ("dream", "SOP-01")
        assert campus_policy.tier_for(10.0) == ("standard", "SOP-01")

        # 4. SOP-05 coverage threshold sync across band_for() and eligibility
        assert band_for(45) == "developing"
        r5 = client.put(
            "/api/admin/policies/SOP-05",
            json={"parameters": {"min_coverage_pct": 50}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r5.status_code == 200
        assert band_for(45) == "early"


def test_verifier_grounding_syncs_after_sop_edit():
    client = _make_client()
    with client:
        known = policy_rag.sop_ids()
        # Initially 65 is grounded
        v_initial = verify_text("Per SOP-04, Dream roles require a mock score of 65.", known=known)
        assert v_initial["grounding_score"] == 100

        # Update SOP-04 min_mock_score -> 80 and SOP-05 min_coverage_pct -> 55
        assert client.put(
            "/api/admin/policies/SOP-04",
            json={"parameters": {"min_mock_score": 80}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 200
        assert client.put(
            "/api/admin/policies/SOP-05",
            json={"parameters": {"min_coverage_pct": 55}},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 200

        v_stale = verify_text("Per SOP-04, Dream roles require a mock score of 65.", known=known)
        assert v_stale["grounding_score"] < 100
        assert any("SOP-04:mock_threshold=65!=80" in u for u in v_stale["unverified"])

        v_fresh = verify_text("Per SOP-04, Dream roles require a mock score of 80.", known=known)
        assert v_fresh["grounding_score"] == 100

        v_cov_stale = verify_text("Under SOP-05, 40% skill coverage is required.", known=known)
        assert v_cov_stale["grounding_score"] < 100
        assert any("SOP-05:min_coverage_pct=40!=55" in u for u in v_cov_stale["unverified"])


def test_admin_policies_page_renders_by_role_and_removed_from_analytics():
    client = _make_client()
    with client:
        # Policy editor card removed from /admin analytics page
        analytics_html = client.get("/admin", headers={"X-Demo-Role": "tpo_admin"}).text
        assert 'id="policy-editor-card"' not in analytics_html

        # Student forbidden from /admin/policies
        assert client.get("/admin/policies", headers={"X-Demo-Role": "student"}).status_code == 403

        # TPO Admin sees full Policy Studio controls on /admin/policies
        tpo_resp = client.get("/admin/policies", headers={"X-Demo-Role": "tpo_admin"})
        assert tpo_resp.status_code == 200
        tpo_html = tpo_resp.text
        assert "Institutional Placement Policies (SOPs)" in tpo_html
        assert 'id="open-create-policy-modal"' in tpo_html
        assert 'id="policy-reset-btn"' in tpo_html
        assert 'id="policy-list-container"' in tpo_html
        assert 'id="policy-edit-modal"' in tpo_html
        assert 'id="policy-create-modal"' in tpo_html
        assert 'id="policy-readonly-badge"' not in tpo_html

        # Faculty mentor sees read-only view on /admin/policies
        mentor_resp = client.get("/admin/policies", headers={"X-Demo-Role": "faculty_mentor"})
        assert mentor_resp.status_code == 200
        mentor_html = mentor_resp.text
        assert "Institutional Placement Policies (SOPs)" in mentor_html
        assert 'id="policy-readonly-badge"' in mentor_html
        assert 'id="open-create-policy-modal"' not in mentor_html
        assert 'id="policy-reset-btn"' not in mentor_html


def test_create_and_delete_custom_policy_and_rag_grounding():
    client = _make_client()
    with client:
        # Student & mentor cannot create or delete
        assert client.post(
            "/api/admin/policies",
            json={"title": "Blocked Rule", "text": "Only admins can add policies here."},
            headers={"X-Demo-Role": "student"},
        ).status_code == 403
        assert client.post(
            "/api/admin/policies",
            json={"title": "Blocked Rule", "text": "Only admins can add policies here."},
            headers={"X-Demo-Role": "faculty_mentor"},
        ).status_code == 403

        # Validation: text < 15 chars rejected
        assert client.post(
            "/api/admin/policies",
            json={"title": "Too Short", "text": "Short text"},
            headers={"X-Demo-Role": "tpo_admin"},
        ).status_code == 422

        # Create custom SOP-19 cleanly
        r_create = client.post(
            "/api/admin/policies",
            json={
                "title": "Summer Internship Stipend Floor",
                "category": "Custom Rules",
                "section": "Section 5.1: Stipend Governance",
                "text": "All approved summer internships must offer a minimum stipend of 15000 INR per month.",
                "params": {"min_stipend_inr": 15000, "remote_allowed": True},
            },
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r_create.status_code == 200, r_create.text
        body = r_create.json()
        assert body["ok"] is True
        assert body["policy"]["id"] == "SOP-19"
        assert body["total"] == 19

        # Searchable in RAG & verified by citation checker
        rag = policy_rag.search_sops("SOP-19 summer internship stipend floor")
        assert rag["top_sop_id"] == "SOP-19"
        cit = policy_rag.verify_policy_citations("Approved under SOP-19 stipend floor.")
        assert cit["verified"] == ["SOP-19"]
        assert cit["grounding_score"] == 100

        # Role enforcement & custom SOP deletion
        assert client.delete("/api/admin/policies/SOP-19", headers={"X-Demo-Role": "student"}).status_code == 403
        assert client.delete("/api/admin/policies/SOP-19", headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403
        assert client.delete("/api/admin/policies/SOP-99", headers={"X-Demo-Role": "tpo_admin"}).status_code == 404

        # Delete custom SOP-19 cleanly
        r_del = client.delete("/api/admin/policies/SOP-19", headers={"X-Demo-Role": "tpo_admin"})
        assert r_del.status_code == 200
        assert r_del.json()["deleted_id"] == "SOP-19"
        assert r_del.json()["total"] == 18
        assert policy_rag.get_sop("SOP-19") is None
        assert client.delete("/api/admin/policies/SOP-19", headers={"X-Demo-Role": "tpo_admin"}).status_code == 404

        # Audit log recorded policy.created and policy.deleted, and scrubs raw + masked emails on read
        with Session(client.app.state.ctx.engine) as db:
            db.add(AuditLog(
                id="aud-emailprobe",
                action="auth.denied",
                entity_type="auth",
                entity_id="leak.user@example.com",
                actor_role="anonymous",
                detail={"note": "blocked leak.user@example.com", "actor_ref": "t***@jklu.edu.in"},
            ))
            db.add(AuditLog(
                id="aud-maskedprobe",
                action="auth.login",
                entity_type="auth",
                entity_id="r***@jklu.edu.in",
                actor_role="student",
                detail={"actor_ref": "r***@jklu.edu.in"},
            ))
            db.commit()

        aud_rows = client.get("/api/admin/audit", headers={"X-Demo-Role": "tpo_admin"}).json()
        actions = {r["action"] for r in aud_rows}
        assert "policy.created" in actions
        assert "policy.deleted" in actions
        probe = [r for r in aud_rows if r["id"] == "aud-emailprobe"][0]
        assert "@" not in probe["entity_id"]
        assert probe["entity_id"].startswith("id-")
        assert "@" not in probe["detail"]["note"]
        assert "@" not in probe["detail"]["actor_ref"]
        assert probe["detail"]["actor_ref"].startswith("id-")
        masked = [r for r in aud_rows if r["id"] == "aud-maskedprobe"][0]
        assert "@" not in masked["entity_id"]
        assert masked["entity_id"].startswith("id-")
        assert "@" not in masked["detail"]["actor_ref"]
        assert masked["detail"]["actor_ref"].startswith("id-")


def test_deterministic_temperature_across_llm_backends(monkeypatch):
    # 1. MockLLM determinism & temperature kwarg support
    mock = MockLLM()
    payload = json.dumps({"resume_text": "Jane Doe\nCSE\nCGPA: 8.7\nSkills: Python, FastAPI, Docker"})
    a = mock.complete("TASK:extract_resume", payload, json_mode=True, temperature=0.0)
    b = mock.complete("TASK:extract_resume", payload, json_mode=True, temperature=0.0)
    assert a == b
    streamed = "".join(mock.stream("TASK:extract_resume", payload, json_mode=True, temperature=0.0))
    assert json.loads(streamed) == json.loads(a)

    # 2. BedrockLLM forwards temperature=0.0 in inferenceConfig
    recorded_bedrock = []

    class FakeBedrockClient:
        def converse(self, **kwargs):
            recorded_bedrock.append(("converse", kwargs))
            return {"output": {"message": {"content": [{"text": "ok"}]}}}

        def converse_stream(self, **kwargs):
            recorded_bedrock.append(("converse_stream", kwargs))
            return {"stream": [{"contentBlockDelta": {"delta": {"text": "ok"}}}]}

    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: FakeBedrockClient())
    bedrock = BedrockLLM(region="us-east-1", model="anthropic.test")
    assert bedrock.complete("sys", "usr", temperature=0.0) == "ok"
    assert list(bedrock.stream("sys", "usr", temperature=0.0)) == ["ok"]
    assert recorded_bedrock[0][1]["inferenceConfig"]["temperature"] == 0.0
    assert recorded_bedrock[1][1]["inferenceConfig"]["temperature"] == 0.0

    # 3. OpenAILLM passes temperature=0.0 in stream() and complete()
    recorded_openai = []

    class FakeOpenAIClient:
        class _Chat:
            class _Completions:
                def create(self, **kwargs):
                    recorded_openai.append(kwargs)
                    if kwargs.get("stream"):
                        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="ok"))])]
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

            def __init__(self):
                self.completions = self._Completions()

        def __init__(self):
            self.chat = self._Chat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    oai = OpenAILLM(api_key="test-key", model="gpt-4o-mini")
    monkeypatch.setattr(oai, "_get_client", lambda: FakeOpenAIClient())
    assert oai.complete("sys", "usr", temperature=0.0) == "ok"
    assert list(oai.stream("sys", "usr", temperature=0.0)) == ["ok"]
    assert all(call["temperature"] == 0.0 for call in recorded_openai)


def test_delete_any_core_policy_with_tombstone_and_reset_restores(isolated_sops):
    sop_file_before = policy_rag.SOP_FILE.read_text(encoding="utf-8")

    with _make_client() as client:
        assert client.delete("/api/admin/policies/SOP-01", headers={"X-Demo-Role": "student"}).status_code == 403
        assert client.delete("/api/admin/policies/SOP-01", headers={"X-Demo-Role": "faculty_mentor"}).status_code == 403

        # TPO Admin can remove core SOP-01 cleanly via tombstone without mutating placement_sops.json
        r_del = client.delete("/api/admin/policies/SOP-01", headers={"X-Demo-Role": "tpo_admin"})
        assert r_del.status_code == 200, r_del.text
        assert r_del.json() == {"ok": True, "deleted_id": "SOP-01", "total": 17}
        assert policy_rag.get_sop("SOP-01") is None
        assert len(policy_rag.list_sops()) == 17
        assert policy_rag.SOP_FILE.read_text(encoding="utf-8") == sop_file_before

        # Already deleted core SOP returns 404
        assert client.delete("/api/admin/policies/SOP-01", headers={"X-Demo-Role": "tpo_admin"}).status_code == 404

        # Editing another SOP preserves SOP-01 tombstone and baseline disk copy
        r_upd = client.put(
            "/api/admin/policies/SOP-02",
            json={"parameters": {"dream_min_lpa": 12}},
            headers={"X-Demo-Role": "tpo_admin"},
        )
        assert r_upd.status_code == 200, r_upd.text
        assert policy_rag.get_sop("SOP-01") is None
        assert len(policy_rag.list_sops()) == 17
        disk_ids = {row["id"] for row in json.loads(policy_rag.SOP_FILE.read_text(encoding="utf-8"))}
        assert "SOP-01" in disk_ids

        # Reset Core Defaults strips tombstones and restores all 18 core SOPs
        r_reset = client.post("/api/admin/policies/reset", headers={"X-Demo-Role": "tpo_admin"})
        assert r_reset.status_code == 200, r_reset.text
        assert r_reset.json()["count"] == 18
        assert policy_rag.get_sop("SOP-01") is not None


def test_simulator_and_remove_policy_ui_markup(isolated_sops):
    with _make_client() as client:
        r_pol = client.get("/admin/policies", headers={"X-Demo-Role": "tpo_admin"})
        assert r_pol.status_code == 200
        assert "delete-edit-policy-btn" in r_pol.text
        assert "Remove Policy" in r_pol.text
        assert "deletePolicy(" in r_pol.text

        r_adm = client.get("/admin", headers={"X-Demo-Role": "tpo_admin"})
        assert r_adm.status_code == 200
        assert 'id="sim-branches"' in r_adm.text
        assert "formatTierLabel" in r_adm.text
        assert "Academic CGPA & Backlogs" in r_adm.text
        assert "Unlocked by Intervention" in r_adm.text
        assert "runSim(true)" in r_adm.text
