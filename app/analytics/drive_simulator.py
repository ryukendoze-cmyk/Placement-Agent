"""TPO placement drive eligibility simulator & cohort bootcamp ROI engine (M20).

``simulate_placement_drive`` evaluates the whole cohort against a *hypothetical*
recruiter drive (not persisted). It reuses the production SOP engine
(``evaluate_student_eligibility``) on an ephemeral ``Job`` so the checks stay
byte-for-byte identical to the real application pipeline.
"""
from __future__ import annotations

from uuid import uuid4

from sqlmodel import Session, select

from ..matching.engine import job_tier
from ..models import Job, Skill, Student
from ..plan.planner import _load_courses, pick_course_for_skill
from ..rag.policy_rag import evaluate_student_eligibility

_CHECK_NAMES = ("gpa_cutoff", "branch_restriction", "tier_mock_gate", "skill_readiness")


def _skill_names(session: Session) -> dict[str, str]:
    return {s.id: s.name for s in session.exec(select(Skill)).all()}


def _normalize_salary(salary_lpa: list[int] | None) -> list[int]:
    sal = list(salary_lpa or [0, 0])
    if len(sal) == 0:
        return [0, 0]
    if len(sal) == 1:
        return [sal[0], sal[0]]
    return [sal[0], sal[1]]


def simulate_placement_drive(
    session: Session,
    ctx,
    *,
    title: str,
    company: str = "Prospective Recruiter",
    salary_lpa: list[int] | None = None,
    min_gpa: float = 7.0,
    allowed_branches: list[str] | None = None,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    max_backlogs: int | None = None,
) -> dict:
    """Evaluate every cohort student against a hypothetical drive.

    Returns a JSON-safe dict: tier, total/eligible counts, eligible_students,
    per-check bottleneck_breakdown, top_missing_skills, and a deterministic
    bootcamp_intervention_roi projection.
    """
    salary = _normalize_salary(salary_lpa)
    req_skills = list(required_skills or [])
    pref_skills = list(preferred_skills or [])
    branches = list(allowed_branches or [])

    job = Job(
        id=f"sim-{uuid4().hex[:8]}",
        title=title,
        company=company,
        description=(f"{title} at {company}. "
                     f"Required: {', '.join(req_skills) or 'n/a'}"),
        required_skills=req_skills,
        preferred_skills=pref_skills,
        salary_lpa=salary,
        min_gpa=float(min_gpa or 0.0),
        max_backlogs=max_backlogs,
        allowed_branches=branches,
        job_url="",
        source="local",
        scraper_engine="",
    )
    # Pre-embed the synthetic job so evaluate_student_eligibility hits the
    # in-memory cache and never writes to the vector store.
    try:
        ctx.job_vectors[job.id] = ctx.embedder.embed(job.description)
    except Exception:  # pragma: no cover - defensive
        pass

    all_students = session.exec(select(Student)).all()
    total = len(all_students)
    if total == 0:
        return {
            "title": title, "company": company, "salary_lpa": salary,
            "min_gpa": float(min_gpa or 0.0), "tier": job_tier(job),
            "total_students": 0, "eligible_now_count": 0, "eligible_now_pct": 0.0,
            "eligible_students": [],
            "bottleneck_breakdown": {n: 0 for n in _CHECK_NAMES},
            "top_missing_skills": [],
            "bootcamp_intervention_roi": {
                "projected_eligible_count": 0, "unlocked_students_count": 0,
                "total_bootcamp_hours": 0},
        }

    # Load courses + skill names once (no N+1).
    courses = _load_courses(session)
    names = _skill_names(session)

    bottleneck = {n: 0 for n in _CHECK_NAMES}
    missing_counts: dict[str, int] = {}
    eligible_students: list[dict] = []
    per_student: list[dict] = []

    for s in all_students:
        elig = evaluate_student_eligibility(session, ctx, s.id, job)
        # Per-check non-exclusive counts: a student failing multiple SOPs
        # increments each counter (no evaluation-order masking).
        for c in elig["checks"]:
            if not c["passed"]:
                bottleneck[c["name"]] = bottleneck.get(c["name"], 0) + 1
        for miss in elig.get("missing_required", []):
            if miss:
                missing_counts[miss] = missing_counts.get(miss, 0) + 1
        per_student.append({"student": s, "elig": elig})
        if elig["eligible"]:
            eligible_students.append({
                "id": s.id, "name": s.name, "branch": s.branch,
                "gpa": float(s.gpa or 0.0), "match_score": elig["match_score"],
            })

    eligible_now = len(eligible_students)

    # top missing skills ranked by students_missing desc, skill_id asc tiebreak
    top_missing = [
        {"skill_id": sid, "name": names.get(sid, sid), "students_missing": cnt,
         "recommended_course": _course_payload(pick_course_for_skill(sid, courses))}
        for sid, cnt in missing_counts.items()
    ]
    top_missing.sort(key=lambda x: (-x["students_missing"], x["skill_id"]))
    top_missing = top_missing[:5]

    # ---- deterministic bootcamp ROI projection -------------------------------
    roi = _project_bootcamp_roi(per_student, top_missing, courses, eligible_now,
                                max_backlogs)

    return {
        "title": title, "company": company, "salary_lpa": salary,
        "min_gpa": float(min_gpa or 0.0), "allowed_branches": branches,
        "required_skills": req_skills, "preferred_skills": pref_skills,
        "tier": job_tier(job),
        "total_students": total,
        "eligible_now_count": eligible_now,
        "eligible_now_pct": round(100 * eligible_now / total, 1) if total else 0.0,
        "eligible_students": eligible_students,
        "bottleneck_breakdown": bottleneck,
        "top_missing_skills": top_missing,
        "bootcamp_intervention_roi": roi,
    }


def _course_payload(course: dict | None) -> dict | None:
    if not course:
        return None
    return {"id": course["id"], "title": course["title"],
            "provider": course.get("provider", ""), "hours": course.get("hours", 4)}


def _project_bootcamp_roi(per_student: list[dict], top_missing: list[dict],
                          courses: list[dict], eligible_now: int,
                          max_backlogs: int | None) -> dict:
    """Counterfactual: teach the top 1-2 missing skills to students who lack
    them and run mock interviews for the mock-gate-blocked. Students whose only
    failures are SOP-04 (mock) and/or SOP-05 (skill) become eligible; GPA and
    branch failures are NOT remediable by a bootcamp.

    ``max_backlogs`` is accepted for signature parity with the simulator's
    public API; SOP-03 backlog limits are tier-derived inside the policy engine.
    """
    target_skills = [s["skill_id"] for s in top_missing[:2]
                     if s["students_missing"] > 0]

    # distinct course hours for the targeted skills
    seen_courses: dict[str, int] = {}
    for s in top_missing:
        if s["skill_id"] in target_skills:
            rc = s["recommended_course"]
            if rc:
                seen_courses[rc["id"]] = rc.get("hours", 4)
    total_hours = sum(seen_courses.values())

    projected = 0
    for entry in per_student:
        elig = entry["elig"]
        if elig["eligible"]:
            projected += 1
            continue
        failed = {c["name"] for c in elig["checks"] if not c["passed"]}
        # Hard gates a bootcamp cannot fix:
        if "gpa_cutoff" in failed or "branch_restriction" in failed:
            continue
        # Only remediable failures (mock gate and/or skill readiness)?
        if failed <= {"tier_mock_gate", "skill_readiness"}:
            projected += 1

    return {
        "projected_eligible_count": projected,
        "unlocked_students_count": max(0, projected - eligible_now),
        "total_bootcamp_hours": total_hours,
    }
