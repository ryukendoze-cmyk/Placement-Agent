"""Tests for the Readiness Trajectory & Strategy Control Plane (solver, APIs, sanitizer, UI)."""
from __future__ import annotations

import math
import re
from uuid import uuid4

from sqlmodel import Session, select

from app.db import _sanitize_legacy_plans
from app.models import (
    AuditLog,
    Course,
    Job,
    JobMatch,
    PlanItem,
    Student,
    StudentSkill,
    UpskillPlan,
)

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\u2600-\u27BF"
    "]",
    flags=re.UNICODE,
)


def _upload_student(client, *, name="Control Plane Tester", gpa="8.4", backlogs="0", skills="Skills: Python, Linux"):
    up = client.post(
        "/api/students",
        data={"name": name, "branch": "CSE", "year": "4", "cgpa": gpa, "backlogs": backlogs},
        files={"file": ("resume.txt", f"{skills}\n".encode(), "text/plain")},
    )
    assert up.status_code in (200, 201), up.text
    return up.json()["id"]


def test_portfolio_mode_and_bridge_consolidation(client):
    sid = _upload_student(client, skills="Skills: Python, Git")
    m_res = client.get(f"/api/students/{sid}/matches")
    assert m_res.status_code == 200

    res = client.post(
        f"/api/students/{sid}/plans",
        json={
            "target_mode": "portfolio",
            "weekly_cap": 10,
            "horizon_weeks": 6,
            "strategy_mode": "max_roi",
        },
    )
    assert res.status_code == 201, res.text
    plan = res.json()

    strat = plan["strategy"]
    assert strat["solver_tier"] == "Portfolio MILP-DAG"
    assert strat["grounding_score"] == 100
    assert len(strat["portfolio_drives"]) >= 1
    assert strat["readiness_trajectory"]["after_pct"] >= strat["readiness_trajectory"]["before_pct"]

    course_ids = [it["course_id"] for it in plan["items"]]
    assert len(course_ids) == len(set(course_ids)), f"Duplicate bridge courses scheduled: {course_ids}"
    assert all("roi_pts_per_hr" in it and "bkt_mastery_pct" in it for it in plan["items"])


def test_sprint_swimlanes_strictly_obey_weekly_cap(client):
    sid = _upload_student(client, skills="Skills: Git")
    client.get(f"/api/students/{sid}/matches")

    res = client.post(
        f"/api/students/{sid}/plans",
        json={"job_id": "job-001", "weekly_cap": 8, "horizon_weeks": 8, "target_mode": "single"},
    )
    assert res.status_code == 201, res.text
    plan = res.json()
    strat = plan["strategy"]

    # job-001 includes 14h courses (python / postgresql) -> must split across modules so every week <= 8h
    swimlanes = strat["sprint_swimlanes"]
    assert len(swimlanes) >= 2
    assert all(wk["allocated_hours"] <= 8 for wk in swimlanes), swimlanes
    assert all(wk["cap_compliant"] is True for wk in swimlanes)
    assert strat["telemetry"]["sprint_max_week_hours"] <= 8


def test_sop03_unreachable_drive_pivot(client_engine):
    client, eng = client_engine
    sid = _upload_student(client, name="Low CGPA Pivot", gpa="6.1", backlogs="0", skills="Skills: Python, Git")

    with Session(eng) as db:
        high_bar = Job(
            id="job-high-bar",
            title="Exclusive Dream Quant",
            company="AlphaQuant",
            description="Requires strict CGPA >= 8.5",
            required_skills=["python", "sql", "docker"],
            preferred_skills=[],
            min_gpa=8.5,
            max_backlogs=0,
            salary_lpa=[24, 28],
        )
        reachable = Job(
            id="job-reachable",
            title="Accessible Backend Dev",
            company="BuildFast",
            description="Accessible backend engineer role",
            required_skills=["python", "fastapi", "git"],
            preferred_skills=["docker"],
            min_gpa=5.5,
            max_backlogs=2,
            salary_lpa=[7, 9],
        )
        db.merge(high_bar)
        db.merge(reachable)
        db.merge(JobMatch(id=f"jm-{uuid4().hex[:8]}", student_id=sid, job_id=high_bar.id, score=72, band="moderate", breakdown={}))
        db.merge(JobMatch(id=f"jm-{uuid4().hex[:8]}", student_id=sid, job_id=reachable.id, score=68, band="moderate", breakdown={}))
        db.commit()

    res = client.post(
        f"/api/students/{sid}/plans",
        json={"job_id": "job-high-bar", "exclude_unreachable": True, "target_mode": "portfolio"},
    )
    assert res.status_code == 201, res.text
    plan = res.json()
    assert plan["job_id"] != "job-high-bar"
    assert plan["strategy"]["pivot_alert"] is not None
    assert plan["strategy"]["pivot_alert"]["from_job_id"] == "job-high-bar"


def test_cpm_dag_critical_path_and_bkt_mastery(client):
    sid = _upload_student(client, skills="Skills: Git")
    client.get(f"/api/students/{sid}/matches")
    res = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    assert res.status_code == 201, res.text
    plan = res.json()

    dag = plan["strategy"]["dag"]
    assert dag["critical_path_hours"] > 0
    crit_nodes = [n for n in dag["nodes"] if n.get("on_critical_path")]
    assert len(crit_nodes) >= 1
    assert all(n["slack_h"] == 0.0 for n in crit_nodes)

    target_item = plan["items"][0]
    before_bkt = target_item["bkt_mastery_pct"]

    # Fetch quiz and submit passing answers -> verify BKT posterior increases
    q_res = client.get(f"/api/plans/{plan['id']}/items/{target_item['id']}/quiz")
    assert q_res.status_code == 200
    from app.plan.verification import _answers_for_item

    ans = _answers_for_item(target_item["id"], target_item["skill_id"])
    assert len(ans) >= 1

    v_res = client.post(f"/api/plans/{plan['id']}/items/{target_item['id']}/verify", json={"answers": ans})
    assert v_res.status_code == 200
    assert v_res.json()["passed"] is True

    after_plan = client.get(f"/api/plans/{plan['id']}").json()
    updated_item = next(it for it in after_plan["items"] if it["id"] == target_item["id"])
    assert updated_item["bkt_mastery_pct"] > before_bkt


def test_course_swap_updates_plan_and_audit(client_engine):
    client, eng = client_engine
    with Session(eng) as db:
        db.merge(Course(
            id="course-alt-pg",
            title="PostgreSQL Indexing & Query Optimization",
            provider="PostgreSQL University",
            url="https://www.postgresql.org/docs/current/indexes.html",
            skill_ids=["postgresql"],
            level="intermediate",
            hours=8,
            cost="free",
        ))
        db.commit()

    sid = _upload_student(client, skills="Skills: Python, FastAPI, Git")
    client.get(f"/api/students/{sid}/matches")
    plan = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"}).json()

    swappable = next((it for it in plan["items"] if it.get("alternatives")), None)
    assert swappable is not None, "Expected at least one plan item with catalog alternatives"
    alt = swappable["alternatives"][0]

    # Negative check: incompatible skill swap rejected cleanly
    bad = client.post(
        f"/api/plans/{plan['id']}/swap-course",
        json={"item_id": swappable["id"], "course_id": "course-999-missing"},
    )
    assert bad.status_code in (400, 404)

    ok = client.post(
        f"/api/plans/{plan['id']}/swap-course",
        json={"item_id": swappable["id"], "course_id": alt["id"]},
    )
    assert ok.status_code == 200, ok.text
    swapped = ok.json()
    updated = next(it for it in swapped["items"] if it["id"] == swappable["id"])
    assert updated["course_id"] == alt["id"]
    assert any(r["kind"] == "Course Swap" for r in swapped["strategy"]["recovery_ledger"])

    with Session(eng) as db:
        aud = db.exec(select(AuditLog).where(AuditLog.action == "plan.course_swapped", AuditLog.entity_id == swappable["id"])).first()
        assert aud is not None


def test_waiver_request_on_missing_skill_and_tpo_approval(client_engine):
    client, eng = client_engine
    sid = _upload_student(client, skills="Skills: Python, FastAPI")
    client.get(f"/api/students/{sid}/matches")
    plan = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"}).json()
    target = plan["items"][0]

    w_res = client.post(
        f"/api/plans/{plan['id']}/items/{target['id']}/waiver",
        json={"target_level": "intermediate", "reason": "Built capstone CLI with Git CI workflows"},
    )
    assert w_res.status_code == 200, w_res.text
    approval_id = w_res.json()["approval_id"]
    assert w_res.json()["status"] == "pending_review"

    app_res = client.post(f"/api/approvals/{approval_id}/approve", json={"note": "Verified repo history"})
    assert app_res.status_code == 200, app_res.text
    assert app_res.json()["executed"]["new_level"] == "intermediate"

    with Session(eng) as db:
        ss = db.get(StudentSkill, (sid, target["skill_id"]))
        assert ss is not None and ss.level == "intermediate"


def test_counterfactual_whatif_simulator_endpoint(client):
    sid = _upload_student(client, skills="Skills: Python")
    client.get(f"/api/students/{sid}/matches")
    plan = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"}).json()

    sim = client.post(
        f"/api/plans/{plan['id']}/simulate",
        json={"added_skills": ["fastapi", "postgresql", "git"], "weekly_cap": 15, "assume_mock_passed": True},
    )
    assert sim.status_code == 200, sim.text
    body = sim.json()
    assert body["weekly_cap"] == 15
    assert body["projected_weeks"] == math.ceil(body["total_hours"] / 15)
    assert len(body["drives"]) >= 1
    assert any(d["delta"] > 0 for d in body["drives"])


def test_legacy_plan_sanitizer_dedups_bridge_and_repairs_narrative(client_engine):
    _client, eng = client_engine
    with Session(eng) as db:
        stu = Student(id="stu-legacy", name="Legacy Student", branch="CSE", year=4, gpa=8.0, backlogs=0)
        db.add(stu)
        # Simulate stale Coursera 16h course-011 on DB
        c11 = db.get(Course, "course-011")
        if c11 is not None:
            c11.title = "PostgreSQL for Everybody Specialization"
            c11.hours = 16
            c11.cost = "paid"
            c11.provider = "Coursera"
            db.add(c11)
        plan = UpskillPlan(
            id="plan-legacy-1",
            student_id="stu-legacy",
            job_id="job-001",
            total_hours=32,
            narrative="Your plan totals 52 hours; spread across the week (about 13h/wk x 4 weeks).",
        )
        db.add(plan)
        db.flush()
        db.add(PlanItem(id="pi-leg-1", plan_id="plan-legacy-1", order=1, skill_id="sql", course_id="course-011", hours=16, week=1))
        db.add(PlanItem(id="pi-leg-2", plan_id="plan-legacy-1", order=2, skill_id="postgresql", course_id="course-011", hours=16, week=3))
        db.commit()

    _sanitize_legacy_plans(eng)

    with Session(eng) as db:
        repaired_c11 = db.get(Course, "course-011")
        assert repaired_c11 is not None
        assert repaired_c11.cost == "free"
        assert repaired_c11.hours == 14

        items = db.exec(select(PlanItem).where(PlanItem.plan_id == "plan-legacy-1").order_by(PlanItem.order)).all()
        assert len(items) == 1
        assert items[0].skill_id == "postgresql"  # required target preserved over transitive sql
        assert items[0].hours == 14

        p2 = db.get(UpskillPlan, "plan-legacy-1")
        assert p2 is not None
        assert p2.total_hours == 14
        assert "13h x" not in p2.narrative and "13h/wk" not in p2.narrative
        assert "14 hours" in p2.narrative
        assert "sql" in p2.strategy_json["item_enrichment"][items[0].id]["covers_skills"]


def test_rendered_plan_control_plane_zero_emojis_and_4zones(client):
    sid = _upload_student(client, skills="Skills: Python")
    client.get(f"/api/students/{sid}/matches")
    r_plan = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001", "weekly_cap": 8})
    assert r_plan.status_code == 201

    page = client.get(f"/students/{sid}/plan")
    assert page.status_code == 200
    html = page.text

    for expected in (
        "Constraint-Solved Career Trajectory",
        "Portfolio Impact",
        "SOP &amp; Eligibility Gate Radar",
        "Module-Level Weekly Sprints",
        "Critical-Path Dependency DAG",
        "Counterfactual What-If Simulator",
        "Telemetry &amp; Recovery Ledger",
        "plan-items-tbody",
        "plan-bar",
        "plan-waiver-modal",
    ):
        assert expected in html, f"Missing '{expected}' in rendered plan page"

    assert not EMOJI_RE.search(html), "Rendered plan page violated Zero-Emoji Policy!"


def test_include_upgrades_and_intermediate_simulation_do_not_crash(client_engine):
    """Regression: include_upgrades=True with solver modes & intermediate skill upgrades must not 500."""
    from app.models import StudentSkill
    from app.plan.solver import _step_up_level

    assert _step_up_level(None) == "intermediate"
    assert _step_up_level("basic") == "intermediate"
    assert _step_up_level("beginner") == "intermediate"
    assert _step_up_level("intermediate") == "advanced"
    assert _step_up_level("advanced") == "advanced"

    client, eng = client_engine
    sid = _upload_student(
        client,
        name="Upgrade Mode Tester",
        gpa="7.9",
        backlogs="0",
        skills="Skills: Python, FastAPI, Docker, Git, REST API",
    )
    assert client.get(f"/api/students/{sid}/matches").status_code == 200

    # Set one claimed basic skill and one verified intermediate skill to exercise all step-up paths.
    with Session(eng) as db:
        for st_skill in db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all():
            if st_skill.skill_id == "docker":
                st_skill.level = "intermediate"
                st_skill.verification_status = "verified"
            else:
                st_skill.level = "basic"
                st_skill.verification_status = "claimed"
            db.add(st_skill)
        db.commit()

    # Exact control combination from UI bug report: portfolio + deep_mastery + include_upgrades=True
    r_port = client.post(
        f"/api/students/{sid}/plans",
        json={
            "job_id": None,
            "include_upgrades": True,
            "target_mode": "portfolio",
            "weekly_cap": 10,
            "horizon_weeks": 8,
            "cost_filter": "all",
            "strategy_mode": "deep_mastery",
            "exclude_unreachable": False,
        },
    )
    assert r_port.status_code == 201, r_port.text
    plan_data = r_port.json()
    assert plan_data.get("strategy")
    pid = plan_data["id"]

    # Crunch + max_roi + include_upgrades=True
    r_crunch = client.post(
        f"/api/students/{sid}/plans",
        json={
            "job_id": None,
            "include_upgrades": True,
            "target_mode": "crunch",
            "weekly_cap": 5,
            "horizon_weeks": 2,
            "cost_filter": "free",
            "strategy_mode": "max_roi",
            "exclude_unreachable": True,
        },
    )
    assert r_crunch.status_code == 201, r_crunch.text

    # What-If simulator upgrading both already-owned intermediate skill ("docker") and basic skills ("python").
    r_sim = client.post(
        f"/api/plans/{pid}/simulate",
        json={"added_skills": ["python", "docker", "sql"], "weekly_cap": 10, "assume_mock_passed": True},
    )
    assert r_sim.status_code == 200, r_sim.text
    sim_body = r_sim.json()
    assert sim_body["drives"]
    assert all(d["after_score"] >= d["before_score"] for d in sim_body["drives"])

    # Rendered plan page surfaces HTTP status fallback for non-JSON error responses.
    page_res = client.get(f"/students/{sid}/plan")
    assert page_res.status_code == 200
    assert "Could not regenerate plan (HTTP" in page_res.text
