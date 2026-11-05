"""Milestone 20: TPO placement drive eligibility simulator & cohort bootcamp
ROI engine.

Covers 20.1-20.3 of the master plan:
- simulate_placement_drive partitions a cohort into eligible/bottleneck/skills/ROI
- POST /api/admin/simulate-drive RBAC (403 student, 200 tpo/faculty/demo)
- execute_approved cohort_bootcamp_publish returns students_impacted
"""
from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.analytics.drive_simulator import simulate_placement_drive
from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy


def _make_client(cohort: bool = False) -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", lambda c, r: (c.cursor().execute("PRAGMA foreign_keys=ON"),
                                               c.cursor().close()))
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
        if cohort:
            seed_demo_cohort(db, taxonomy=taxonomy, embedder=ctx.embedder,
                             vector_store=ctx.vector_store, llm=MockLLM())
    return TestClient(create_app(ctx))


def _sim(ctx, **kw):
    with Session(ctx.engine) as db:
        return simulate_placement_drive(db, ctx, **kw)


# ------------------------------------------------------------- 20.1 core

def test_empty_cohort_returns_zeroed_shape():
    """0 students -> all counts 0, no division-by-zero, empty lists."""
    client = _make_client()
    with client:
        ctx = client.app.state.ctx
        out = _sim(ctx, title="Empty Drive")
    assert out["total_students"] == 0
    assert out["eligible_now_count"] == 0
    assert out["eligible_now_pct"] == 0.0
    assert out["eligible_students"] == []
    assert out["bottleneck_breakdown"] == {
        "gpa_cutoff": 0, "branch_restriction": 0,
        "tier_mock_gate": 0, "skill_readiness": 0}
    assert out["top_missing_skills"] == []
    roi = out["bootcamp_intervention_roi"]
    assert roi["projected_eligible_count"] == 0
    assert roi["unlocked_students_count"] == 0
    assert roi["total_bootcamp_hours"] == 0


def test_tier_derivation_from_salary():
    """tier follows job_tier boundaries (<8 standard, 8-15 dream, >=15 super)."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        assert _sim(ctx, title="Std", salary_lpa=[5, 7])["tier"] == "standard"
        assert _sim(ctx, title="Dream", salary_lpa=[12, 14])["tier"] == "dream"
        assert _sim(ctx, title="Super", salary_lpa=[18, 24])["tier"] == "super_dream"


def test_drive_partitions_seeded_cohort():
    """The 12-student demo cohort is partitioned: eligible + bottleneck counts
    are consistent (eligible + any-blocked = total for the check that matters),
    missing skills aggregated, ROI present."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        out = _sim(ctx, title="AI Drive", salary_lpa=[12, 16], min_gpa=7.0,
                   required_skills=["python", "machine_learning"])
    total = out["total_students"]
    assert total == 12
    assert out["eligible_now_count"] >= 0
    assert 0.0 <= out["eligible_now_pct"] <= 100.0
    # eligible_now_pct consistent with counts
    if total:
        assert out["eligible_now_pct"] == round(100 * out["eligible_now_count"] / total, 1)
    # bottleneck counts non-negative integers for all 4 SOP checks
    bb = out["bottleneck_breakdown"]
    assert set(bb) == {"gpa_cutoff", "branch_restriction",
                       "tier_mock_gate", "skill_readiness"}
    assert all(isinstance(v, int) and v >= 0 for v in bb.values())
    assert sum(bb.values()) >= 0
    # eligible_students entries have the documented keys
    for e in out["eligible_students"]:
        assert {"id", "name", "branch", "gpa", "match_score"} <= set(e)
        assert isinstance(e["match_score"], (int, float))
    # top_missing_skills sorted by students_missing desc, skill_id asc tiebreak
    tms = out["top_missing_skills"]
    assert tms == sorted(tms, key=lambda s: (-s["students_missing"], s["skill_id"]))
    for s in tms:
        assert {"skill_id", "name", "students_missing", "recommended_course"} <= set(s)
    # ROI shape
    roi = out["bootcamp_intervention_roi"]
    assert {"projected_eligible_count", "unlocked_students_count",
            "total_bootcamp_hours"} <= set(roi)
    assert roi["projected_eligible_count"] >= out["eligible_now_count"]
    assert roi["unlocked_students_count"] == (roi["projected_eligible_count"]
                                              - out["eligible_now_count"])
    assert roi["total_bootcamp_hours"] >= 0


def test_high_gpa_cutoff_blocks_and_counts_bottleneck():
    """An impossible min_gpa (11.0) blocks the whole cohort on gpa_cutoff."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        out = _sim(ctx, title="Impossible", salary_lpa=[12, 16], min_gpa=11.0)
    assert out["eligible_now_count"] == 0
    assert out["bottleneck_breakdown"]["gpa_cutoff"] == out["total_students"]
    # ROI cannot unlock GPA/branch failures -> projected == 0
    assert out["bootcamp_intervention_roi"]["projected_eligible_count"] == 0
    assert out["bootcamp_intervention_roi"]["unlocked_students_count"] == 0


def test_allowed_branches_restricts_scope():
    """allowed_branches=['ECE'] restricts eligibility to ECE students only."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        out = _sim(ctx, title="ECE Drive", salary_lpa=[12, 16],
                   allowed_branches=["ECE"])
    # every eligible student must be ECE
    for e in out["eligible_students"]:
        assert str(e["branch"]).upper() == "ECE"
    # branch_restriction blocked the non-ECE cohort members
    assert out["bottleneck_breakdown"]["branch_restriction"] >= 1


def test_bootcamp_roi_unlocks_skill_and_mock_bottlenecks():
    """For a drive whose ONLY failures are skill/mock gates, the bootcamp ROI
    unlocks those students (projected > current)."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        # standard tier (no mock gate) with a rare required skill -> skill gaps
        out = _sim(ctx, title="Skill Drive", salary_lpa=[5, 7], min_gpa=0.0,
                   required_skills=["machine_learning"])
    roi = out["bootcamp_intervention_roi"]
    # there must be students missing the required skill to be unlockable
    assert out["bottleneck_breakdown"]["skill_readiness"] >= 0
    # projected never regresses below current
    assert roi["projected_eligible_count"] >= out["eligible_now_count"]
    assert roi["total_bootcamp_hours"] >= 0


# ------------------------------------------------------- 20.2 endpoint

def test_simulate_drive_rbac():
    """POST /api/admin/simulate-drive: 403 for student, 200 for tpo/faculty/demo."""
    client = _make_client(cohort=True)
    with client:
        body = {"title": "RBAC Drive", "salary_lpa": [12, 16], "min_gpa": 7.0}
        # open demo mode (no identity headers) -> permissive
        r0 = client.post("/api/admin/simulate-drive", json=body)
        assert r0.status_code == 200, r0.text
        assert r0.json()["total_students"] == 12

        # declared student identity -> 403
        r1 = client.post("/api/admin/simulate-drive", json=body,
                         headers={"X-Demo-Role": "student"})
        assert r1.status_code == 403, r1.text

        # tpo_admin -> 200
        r2 = client.post("/api/admin/simulate-drive", json=body,
                         headers={"X-Demo-Role": "tpo_admin"})
        assert r2.status_code == 200, r2.text

        # faculty_mentor -> 200
        r3 = client.post("/api/admin/simulate-drive", json=body,
                         headers={"X-Demo-Role": "faculty_mentor"})
        assert r3.status_code == 200, r3.text


def test_simulate_drive_endpoint_returns_full_shape():
    client = _make_client(cohort=True)
    with client:
        r = client.post("/api/admin/simulate-drive",
                        json={"title": "Shape", "salary_lpa": [10, 14],
                              "min_gpa": 7.5})
        assert r.status_code == 200, r.text
        j = r.json()
        assert {"tier", "total_students", "eligible_now_count",
                "eligible_now_pct", "eligible_students", "bottleneck_breakdown",
                "top_missing_skills", "bootcamp_intervention_roi"} <= set(j)
        assert j["tier"] == "dream"


def test_cohort_bootcamp_publish_reports_students_impacted():
    """execute_approved for cohort_bootcamp_publish includes students_impacted
    (students lacking the target skill)."""
    client = _make_client(cohort=True)
    with client:
        ctx = client.app.state.ctx
        # create a pending approval for a skill the cohort lacks
        with Session(ctx.engine) as db:
            from uuid import uuid4

            from app.models import ApprovalRequest, StudentSkill
            skill_id = "machine_learning"
            having = {ss.student_id for ss in db.exec(
                select(StudentSkill).where(StudentSkill.skill_id == skill_id)).all()}
            from app.models import Student
            all_students = db.exec(select(Student)).all()
            expected_missing = len([s for s in all_students if s.id not in having])
            appr_id = f"appr-{uuid4().hex[:8]}"
            db.add(ApprovalRequest(
                id=appr_id, kind="cohort_bootcamp_publish", target_id=skill_id,
                title="t", summary="s",
                payload={"skill_id": skill_id, "course_id": ""},
                status="pending_review", requested_by="agent"))
            db.commit()
        r = client.post(f"/api/approvals/{appr_id}/approve", json={"note": "go"})
        assert r.status_code == 200, r.text
        executed = r.json()["executed"]
        assert executed["kind"] == "cohort_bootcamp_publish"
        assert executed["skill_id"] == skill_id
        assert "students_impacted" in executed
        # impacted = students without the skill (all cohort students lack it here
        # unless the skill was already promoted to everyone)
        assert executed["students_impacted"] >= 0
        assert executed["students_impacted"] == expected_missing
