"""Tests for V5 student progress analytics, seed multi-day calibration, and legacy progress repair."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlmodel import Session, select

from app.analytics.student_progress import build_student_progress_report
from app.db import init_db, make_engine
from app.models import (
    Course,
    Interview,
    Job,
    JobApplication,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Student,
    StudentSkill,
    UpskillPlan,
)
from app.seed import _seed_progress
from app.taxonomy.taxonomy import load_taxonomy


def test_build_student_progress_report_full_contract_and_isolation():
    eng = make_engine("sqlite://")
    init_db(eng)
    tax = load_taxonomy()
    now = datetime.now(timezone.utc)

    with Session(eng) as session:
        session.add(Student(id="stu-v5", name="Riya Mehta", branch="CSE", year=4, gpa=8.6, backlogs=0))
        session.add(Student(id="stu-other", name="Other", branch="IT", year=4, gpa=7.5, backlogs=0))

        session.add(StudentSkill(student_id="stu-v5", skill_id="python", level="advanced", verification_status="verified", confidence=0.92, mentions=4, last_verified_score=88))
        session.add(StudentSkill(student_id="stu-v5", skill_id="fastapi", level="intermediate", verification_status="claimed", confidence=0.55, mentions=2))
        session.add(StudentSkill(student_id="stu-v5", skill_id="docker", level="basic", verification_status="needs_reverification", confidence=0.35, mentions=1, last_verified_score=42))

        campus_dream = Job(
            id="job-campus-dream",
            title="Backend Engineer",
            company="JKLU Dream Corp",
            description="Python, FastAPI, PostgreSQL",
            required_skills=["python", "fastapi", "postgresql"],
            salary_lpa=[10, 12],
            min_gpa=7.5,
            allowed_branches=["CSE", "IT"],
            source="local",
            is_live=False,
            owner_student_id=None,
        )
        other_live_job = Job(
            id="job-ext-other",
            title="Hidden Other Student Job",
            company="Private Co",
            description="Secret",
            required_skills=["python"],
            salary_lpa=[20, 24],
            source="linkedin",
            is_live=True,
            owner_student_id="stu-other",
        )
        session.add_all([campus_dream, other_live_job])

        session.add(
            JobMatch(
                id="jm-1",
                student_id="stu-v5",
                job_id="job-campus-dream",
                score=76,
                band="strong",
                breakdown={"matched_required": ["python", "fastapi"], "missing_required": ["postgresql"]},
            )
        )
        session.add(
            JobMatch(
                id="jm-leak",
                student_id="stu-v5",
                job_id="job-ext-other",
                score=99,
                band="strong",
                breakdown={"matched_required": ["python"], "missing_required": []},
            )
        )

        session.add(
            Course(
                id="c-pg",
                title="PostgreSQL Fundamentals",
                provider="PostgreSQL Docs",
                url="https://www.postgresql.org/docs/",
                skill_ids=["postgresql"],
                hours=8,
            )
        )
        session.add(
            Course(
                id="c-dk",
                title="Docker Getting Started",
                provider="Docker Docs",
                url="https://docs.docker.com/get-started/",
                skill_ids=["docker"],
                hours=6,
            )
        )
        session.add(UpskillPlan(id="plan-v5", student_id="stu-v5", job_id="job-campus-dream", total_hours=14))
        session.add(PlanItem(id="pi-1", plan_id="plan-v5", order=1, skill_id="postgresql", course_id="c-pg", hours=8, status="done", week=1))
        session.add(PlanItem(id="pi-2", plan_id="plan-v5", order=2, skill_id="docker", course_id="c-dk", hours=6, status="pending", week=2))

        session.add(
            Interview(
                id="int-v5",
                student_id="stu-v5",
                job_id="job-campus-dream",
                score=72,
                finished=True,
                started_at=now - timedelta(days=2),
                feedback={
                    "rubric_overall": 78,
                    "per_skill": {"python": "verified"},
                    "speech_summary": {"avg_wpm": 138, "filler_rate": 1.2, "avg_star_completeness": 78},
                },
            )
        )
        session.add(JobApplication(id="app-v5", student_id="stu-v5", job_id="job-campus-dream", status="submitted", tier="dream"))

        # Pre-existing genuine multi-day event history
        session.add(ProgressEvent(id="pe-1", student_id="stu-v5", kind="resume_parsed", readiness=38, at=now - timedelta(days=10)))
        session.add(ProgressEvent(id="pe-2", student_id="stu-v5", kind="plan_item_done", ref_id="pi-1", readiness=63, at=now - timedelta(days=3)))
        session.commit()

        before_ev_count = len(session.exec(select(ProgressEvent).where(ProgressEvent.student_id == "stu-v5")).all())
        report = build_student_progress_report(session, "stu-v5", taxonomy=tax)
        after_ev_count = len(session.exec(select(ProgressEvent).where(ProgressEvent.student_id == "stu-v5")).all())

        # Read-only invariant
        assert before_ev_count == after_ev_count == 2

        k = report["kpis"]
        assert k["best_role_job_id"] == "job-campus-dream"
        assert k["best_role_score"] == 76
        assert k["best_role_tier"] == "dream"
        assert k["best_role_tier_label"] == "Dream"
        assert k["total_drives_evaluated"] == 1  # other student's live job excluded
        assert k["plan_done_items"] == 1 and k["plan_total_items"] == 2
        assert k["plan_done_hours"] == 8 and k["plan_total_hours"] == 14
        assert k["plan_completion_pct"] == 50
        # round(0.5*50 + 0.5*76) = 63
        assert k["current_readiness"] == 63
        assert k["baseline_readiness"] == 38
        assert k["readiness_delta"] == 25
        assert k["projected_readiness"] > k["current_readiness"]
        assert k["verified_skills_count"] == 1
        assert k["claimed_skills_count"] == 1
        assert k["flagged_skills_count"] == 1
        assert k["interviews_completed"] == 1
        assert k["best_interview_score"] == 72
        assert k["sop04_dream_ready"] is True
        assert k["sop04_super_dream_ready"] is False
        assert k["avg_wpm"] == 138
        assert k["eligible_drives_count"] == 1
        assert k["active_applications_count"] == 1

        # Target roles never include another student's live job
        assert [r["job_id"] for r in report["target_roles"]] == ["job-campus-dream"]
        assert report["target_roles"][0]["eligible_now"] is True

        # Milestone series ends with projected point & has no HH:MM labels
        ms = report["milestone_series"]
        assert ms[-1]["is_projected"] is True
        assert ms[-1]["readiness"] == k["projected_readiness"]
        assert all(":" not in pt["short_label"] for pt in ms)

        # Weekly plan & plan modules
        assert len(report["weekly_plan"]) == 4
        assert report["weekly_plan"][0]["done_hours"] == 8
        assert report["weekly_plan"][1]["pending_hours"] == 6
        assert len(report["plan_modules"]) == 2
        assert report["enriched_events"][0]["title"] == "Completed: PostgreSQL"


def test_seed_progress_and_recalibrate_legacy_events():
    eng = make_engine("sqlite://")
    init_db(eng)
    now = datetime.now(timezone.utc)

    with Session(eng) as session:
        session.add(Student(id="stu-leg", name="Aarav", branch="CSE", year=4, gpa=8.4))
        session.add(Student(id="stu-new", name="Meera", branch="IT", year=4, gpa=8.2))
        session.add(Job(id="j1", title="SDE", company="Corp", description="Python", source="local"))
        session.add(JobMatch(id="jm-leg", student_id="stu-leg", job_id="j1", score=84, band="strong"))
        session.add(JobMatch(id="jm-new", student_id="stu-new", job_id="j1", score=72, band="moderate"))
        session.add(UpskillPlan(id="p-leg", student_id="stu-leg", job_id="j1", total_hours=10))
        session.add(Course(id="c1", title="Python Docs", provider="Python", url="https://docs.python.org/", hours=10))
        session.add(PlanItem(id="i-leg-1", plan_id="p-leg", order=1, skill_id="python", course_id="c1", hours=10, status="done", week=1))

        # Inject legacy synthetic 30-min [20, 34, 45, 58, 67, 75] ladder
        legacy_vals = [20, 34, 45, 58, 67, 75]
        for idx, val in enumerate(legacy_vals):
            session.add(
                ProgressEvent(
                    id=f"pe-leg-{idx}",
                    student_id="stu-leg",
                    kind="plan_item_done",
                    readiness=val,
                    at=now - timedelta(minutes=30 * (len(legacy_vals) - idx)),
                )
            )
        _seed_progress(session, "stu-new")
        session.commit()

        seeded_new = sorted(
            session.exec(select(ProgressEvent).where(ProgressEvent.student_id == "stu-new")).all(),
            key=lambda e: e.at,
        )
        assert len(seeded_new) >= 5
        assert [e.readiness for e in seeded_new] != legacy_vals
        assert seeded_new[-1].readiness == 72
        assert (seeded_new[-1].at - seeded_new[0].at).days >= 10

    # Re-running init_db triggers _recalibrate_legacy_progress_events automatically
    init_db(eng)

    with Session(eng) as session:
        recal = sorted(
            session.exec(select(ProgressEvent).where(ProgressEvent.student_id == "stu-leg")).all(),
            key=lambda e: e.at,
        )
        assert [e.readiness for e in recal] != [20, 34, 45, 58, 67, 75]
        # round(0.5*100 + 0.5*84) = 92
        assert recal[-1].readiness == 92
        assert all(recal[i].readiness <= recal[i + 1].readiness for i in range(len(recal) - 1))
        assert (recal[-1].at - recal[0].at).days >= 10


def test_recalibrate_legacy_prefix_with_later_events_and_low_readiness():
    eng = make_engine("sqlite://")
    init_db(eng)
    now = datetime.now(timezone.utc)

    with Session(eng) as session:
        session.add(Student(id="stu-mix", name="Kabir", branch="CSE", year=4, gpa=7.5))
        session.add(Job(id="j-low", title="Analyst", company="Beta", description="SQL", source="local"))
        session.add(JobMatch(id="jm-low", student_id="stu-mix", job_id="j-low", score=10, band="early"))
        session.add(UpskillPlan(id="p-mix", student_id="stu-mix", job_id="j-low", total_hours=12))
        session.add(Course(id="c-mix", title="SQL Basics", provider="Docs", url="https://example.org/sql", hours=6))
        session.add(PlanItem(id="pi-mix-1", plan_id="p-mix", order=1, skill_id="sql", course_id="c-mix", hours=6, status="todo", week=1))
        session.add(PlanItem(id="pi-mix-2", plan_id="p-mix", order=2, skill_id="python", course_id="c-mix", hours=6, status="todo", week=2))

        legacy_vals = [20, 34, 45, 58, 67, 75]
        prefix_base = now - timedelta(days=13)
        for idx, val in enumerate(legacy_vals):
            session.add(
                ProgressEvent(
                    id=f"pe-mix-pre-{idx}",
                    student_id="stu-mix",
                    kind="plan_item_done",
                    readiness=val,
                    at=prefix_base + timedelta(minutes=30 * idx),
                )
            )
        later_vals = [36, 51, 66, 78, 93]
        later_base = now - timedelta(days=7)
        for idx, val in enumerate(later_vals):
            session.add(
                ProgressEvent(
                    id=f"pe-mix-post-{idx}",
                    student_id="stu-mix",
                    kind="plan_item_done",
                    ref_id=f"pi-mix-{idx}",
                    readiness=val,
                    at=later_base + timedelta(minutes=15 * idx),
                )
            )
        session.commit()

    # Re-running init_db detects and recalibrates the leading legacy prefix + stale spikes
    init_db(eng)

    with Session(eng) as session:
        recal = sorted(
            session.exec(select(ProgressEvent).where(ProgressEvent.student_id == "stu-mix")).all(),
            key=lambda e: e.at,
        )
        assert len(recal) == 11
        assert [e.readiness for e in recal[:6]] != [20, 34, 45, 58, 67, 75]
        # round(0.5*0 + 0.5*10) = 5
        assert recal[-1].readiness == 5
        assert all(recal[i].readiness <= recal[i + 1].readiness for i in range(len(recal) - 1))
        assert (recal[5].at - recal[0].at).days >= 10

        report = build_student_progress_report(session, "stu-mix")
        k = report["kpis"]
        assert k["current_readiness"] == 5
        assert k["baseline_readiness"] <= k["current_readiness"]
        assert k["readiness_delta"] >= 0
        non_proj = [p["readiness"] for p in report["milestone_series"] if not p["is_projected"]]
        assert all(non_proj[i] <= non_proj[i + 1] for i in range(len(non_proj) - 1))

    from app.db import _recalibrate_legacy_progress_events

    assert _recalibrate_legacy_progress_events(eng) == 0


def test_build_student_progress_report_handles_dict_rubric_overall():
    eng = make_engine("sqlite://")
    init_db(eng)
    now = datetime.now(timezone.utc)

    with Session(eng) as session:
        session.add(Student(id="stu-rub", name="Neha", branch="CSE", year=4, gpa=8.6))
        session.add(Job(id="j-rub", title="Backend", company="Acme", description="Python", source="local"))
        session.add(JobMatch(id="jm-rub", student_id="stu-rub", job_id="j-rub", score=74, band="strong"))
        session.add(
            Interview(
                id="iv-dict-rubric",
                student_id="stu-rub",
                job_id="j-rub",
                score=68,
                finished=True,
                started_at=now - timedelta(days=1),
                feedback={
                    "rubric_overall": {
                        "technical_accuracy": 72,
                        "depth_and_examples": 64,
                        "communication_clarity": 70,
                    },
                    "speech_summary": {"avg_wpm": 132, "filler_rate": 0.9, "avg_star_score": 56},
                },
            )
        )
        session.add(
            Interview(
                id="iv-dict-rubric-fallback",
                student_id="stu-rub",
                job_id="j-rub",
                score=70,
                finished=True,
                started_at=now,
                feedback={
                    "rubric_overall": {
                        "technical_accuracy": 80,
                        "depth_and_examples": 70,
                        "communication_clarity": 75,
                    },
                    "speech_summary": {"avg_wpm": 140, "filler_rate": 0.5},
                },
            )
        )
        session.commit()

        report = build_student_progress_report(session, "stu-rub")
        assert report["kpis"]["interviews_completed"] == 2
        assert report["kpis"]["avg_wpm"] == 136
        assert report["interview_history"][0]["star_score"] == 56
        assert report["interview_history"][1]["star_score"] == 75
        assert report["kpis"]["avg_star_completeness"] == 66
