"""Upskilling planner: deterministic gap->course selection + LLM narrative.

Reuses score_job's missing_required/missing_preferred breakdown (never re-derives
scores) and rank_jobs only for choosing a default target job when none is given.
"""
import json

from sqlmodel import Session, delete, select

from ..matching.engine import JobScore
from ..models import Course, JobMatch, PlanItem, UpskillPlan
from .prereq import PREREQUISITE_GRAPH

PLAN_NARRATIVE_SYSTEM = (
    "TASK: plan_narrative\n"
    "You are an expert career mentor. Given the student's target job, skill gaps, and total course hours, "
    "write a concise, encouraging, actionable 3-4 sentence upskilling narrative and weekly study strategy. "
    "Respond ONLY with a valid JSON object containing a single key \"narrative\" (string)."
)


def pick_course_for_skill(skill_id: str, courses: list[dict],
                          target_level: str | None = None) -> dict | None:
    """Best course covering skill_id.

    With ``target_level`` in {None, "beginner", "basic"} the original ordering is
    preserved byte-for-byte: beginner-first, then free, then fewer hours.
    "intermediate" prefers intermediate courses; "advanced" prefers advanced then
    intermediate then beginner. Free/fewer-hours remain the tie-breakers, so a
    higher-level target never fails to find a course when only beginner exists.
    """
    candidates = [c for c in courses if skill_id in c.get("skill_ids", [])]
    if not candidates:
        return None
    target = (target_level or "").lower()
    if target == "advanced":
        def level_rank(c: dict) -> int:
            lvl = c.get("level")
            return 0 if lvl == "advanced" else (1 if lvl == "intermediate" else 2)
    elif target == "intermediate":
        def level_rank(c: dict) -> int:
            lvl = c.get("level")
            return 0 if lvl == "intermediate" else (1 if lvl == "beginner" else 2)
    else:
        def level_rank(c: dict) -> int:
            return 0 if c.get("level") == "beginner" else 1
    candidates.sort(key=lambda c: (
        level_rank(c),
        0 if c.get("cost") == "free" else 1,
        c.get("hours", 999),
    ))
    return candidates[0]


def _load_courses(db: Session) -> list[dict]:
    rows = db.exec(select(Course)).all()
    return [
        {"id": c.id, "title": c.title, "provider": c.provider, "url": c.url,
         "skill_ids": c.skill_ids, "level": c.level, "hours": c.hours, "cost": c.cost}
        for c in rows
    ]


TARGET_WEEKLY_HOURS = 10


def apply_plan_item_completion(session, ctx, plan: UpskillPlan, item: PlanItem, *,
                               evidence_note: str = "") -> dict:
    """Shared closed-loop mutation for BOTH POST .../complete and POST .../verify.

    1. Mark PlanItem done (idempotent).
    2. Upsert StudentSkill for item.skill_id: new skill -> basic; existing basic
       -> intermediate; intermediate/advanced -> unchanged (level never regresses).
    3. Re-run rank_jobs across ALL jobs with the updated skills+levels and
       overwrite the student's JobMatch rows in place (score, band, breakdown).
    4. Recompute plan_pct, best_match, readiness = round(0.5*plan_pct + 0.5*best).
    5. Idempotently record/update ProgressEvent(kind="plan_item_done", ref_id=item.id).
    Returns the response payload (skill_id, plan_pct, best_match, readiness, ...).
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from ..models import Job, ProgressEvent, StudentSkill

    already_done = item.status == "done"
    if not already_done:
        item.status = "done"
        session.add(item)

        # 2. skill upsert with evidence - only on the FIRST completion so a
        #    replayed complete/verify never inflates mentions or promotes level
        ss = session.get(StudentSkill, (plan.student_id, item.skill_id))
        course = session.get(Course, item.course_id) if item.course_id else None
        course_label = (course.title if course is not None
                        else (item.course_id or item.skill_id))
        note = evidence_note or f"Completed course {course_label} for {item.skill_id}"
        plan_job = session.get(Job, plan.job_id) if plan.job_id else None
        from ..matching.engine import is_senior_job
        if ss is None:
            session.add(StudentSkill(
                student_id=plan.student_id, skill_id=item.skill_id,
                mentions=1, level="basic", evidence=[note]))
        else:
            if ss.level == "basic":
                ss.level = "intermediate"
            elif ss.level == "intermediate" and is_senior_job(plan_job):
                # senior roles demand advanced depth: an intermediate student who
                # completes the senior-role course is promoted to advanced.
                ss.level = "advanced"
            ss.mentions = (ss.mentions or 0) + 1
            if note not in ss.evidence:
                ss.evidence = list(ss.evidence) + [note]
            session.add(ss)
    session.flush()  # persist skill before re-ranking

    # 3. re-rank all jobs with updated skills + levels
    ss_rows = session.exec(select(StudentSkill).where(
        StudentSkill.student_id == plan.student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    from ..matching.rank import rank_jobs
    sv = ctx.vector_store.get("student", plan.student_id)
    if sv is None:
        from ..models import Student
        student = session.get(Student, plan.student_id)
        sv = ctx.embedder.embed((student.resume_text if student else "") or "")
    jobs = session.exec(select(Job)).all()
    ranked = rank_jobs(jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
                       vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                       top=None, levels=levels)
    from dataclasses import asdict as _asdict
    session.execute(delete(JobMatch).where(JobMatch.student_id == plan.student_id))
    for job, js in ranked:
        session.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=plan.student_id,
                             job_id=job.id, score=js.score, band=js.band,
                             breakdown=_asdict(js)))
    session.flush()

    # Closed-loop application sync: refresh match_score on active JobApplications
    from ..models import JobApplication
    active_apps = session.exec(
        select(JobApplication).where(
            JobApplication.student_id == plan.student_id,
            JobApplication.status.in_(
                ["submitted", "awaiting_mentor_endorsement", "awaiting_tpo_approval"]
            ),
        )
    ).all()
    if active_apps:
        score_by_job = {job.id: float(js.score) for job, js in ranked}
        for app_row in active_apps:
            if app_row.job_id in score_by_job:
                app_row.match_score = score_by_job[app_row.job_id]
                session.add(app_row)

    # 4. recompute metrics
    items = session.exec(select(PlanItem).where(PlanItem.plan_id == plan.id)).all()
    total_hours = plan.total_hours or 1
    done_hours = sum(i.hours for i in items if i.status == "done")
    plan_pct = round(100 * done_hours / total_hours)
    matches = session.exec(select(JobMatch).where(JobMatch.student_id == plan.student_id)).all()
    best_match = max((m.score for m in matches), default=0)
    readiness = round(0.5 * plan_pct + 0.5 * best_match)

    # 5. idempotent progress event
    prior_ev = session.exec(
        select(ProgressEvent).where(
            ProgressEvent.ref_id == item.id, ProgressEvent.kind == "plan_item_done"
        )
    ).first()
    if prior_ev is not None:
        prior_ev.readiness = readiness
        prior_ev.at = datetime.now(timezone.utc)
        session.add(prior_ev)
    else:
        session.add(ProgressEvent(
            id=f"pe-{uuid4().hex[:8]}",
            student_id=plan.student_id,
            kind="plan_item_done",
            ref_id=item.id,
            readiness=readiness,
            at=datetime.now(timezone.utc),
        ))
    session.commit()

    return {"item_id": item.id, "status": item.status, "readiness": readiness,
            "plan_pct": plan_pct, "best_match": best_match,
            "skill_id": item.skill_id}


def build_plan(db: Session, student_id: str, job, job_score: JobScore, *, llm,
               levels: dict[str, str] | None = None) -> tuple[UpskillPlan, list[PlanItem]]:
    """Create UpskillPlan + PlanItems for gaps in job_score breakdown.

    Ordering: prerequisite DAG (app.plan.prereq) puts each prerequisite's course
    before its consumers; within one dependency level, required gaps before
    preferred, then shorter course first. Skills the student already owns are
    never emitted as items (their prerequisite edges still constrain order).
    Items are then packed into weeks of at most TARGET_WEEKLY_HOURS,
    preserving the prerequisite order (a week boundary never hoists a consumer
    above its prerequisite).

    Proficiency upgrades (Milestone 14): when the student owns every required
    skill but at a shallower depth than the role demands (so ``missing_required``
    is empty but ``proficiency_gaps`` are populated), the upgrade skills are
    treated as re-enterable targets -- they are added to the gap set and removed
    from the ``exclude`` set, and courses matching the target depth are chosen
    (e.g. an intermediate course for an intermediate target). This prevents a
    silent 0-item plan for "owns-it-at-basic but role needs intermediate" students
    while leaving the existing missing-required path untouched.

    Returns (plan, items) - SQLModel table models reject arbitrary attribute
    assignment (no field `items`), so items are returned separately.
    """
    from ..models import StudentSkill
    from .prereq import topological_order

    courses = _load_courses(db)
    if levels is None:
        levels = {r.skill_id: r.level for r in db.exec(
            select(StudentSkill).where(StudentSkill.student_id == student_id)).all()}

    # gap tuples: (skill_id, is_required)
    gaps: list[tuple[str, bool]] = []
    for g in job_score.missing_required:
        gaps.append((g["skill"], True))
    for g in job_score.missing_preferred:
        gaps.append((g["skill"], False))
    is_required = dict(gaps)

    # proficiency-upgrade targets: skills the student already owns at a lower
    # depth than the role's target level. Only consulted when nothing is outright
    # missing for a required skill, so existing gap plans stay byte-for-byte equal.
    upgrade_targets: dict[str, str] = {}
    if not job_score.missing_required:
        for g in (job_score.proficiency_gaps or []):
            skill = g.get("skill")
            current = (g.get("current_level") or "").lower()
            target = (g.get("target_level") or "").lower()
            if not skill or current in ("", "none"):
                continue
            if target and target != current:
                upgrade_targets[skill] = target
                is_required.setdefault(skill, True)

    # skills the student already owns - never a plan item, only an ordering edge,
    # EXCEPT proficiency-upgrade targets which must be re-enterable.
    known = {r.skill_id for r in db.exec(
        select(StudentSkill).where(StudentSkill.student_id == student_id)).all()}
    exclude = known - set(upgrade_targets)

    # priority: required (0) before preferred (1), then fewer hours first
    priority = {
        s: (0 if is_required.get(s, False) else 1,
            (pick_course_for_skill(s, courses, upgrade_targets.get(s)) or {}).get("hours", 999))
        for s in is_required
    }

    # 1. prerequisite-first order over the induced gap subgraph
    ordered_skills = topological_order(
        PREREQUISITE_GRAPH, list(is_required), exclude=exclude, priority=priority)

    # 2. resolve each ordered skill to a real course (skip - never fabricate)
    items_spec: list[tuple[str, str, int, bool]] = []  # (skill_id, course_id, hours, is_required)
    for skill_id in ordered_skills:
        course = pick_course_for_skill(skill_id, courses, upgrade_targets.get(skill_id))
        if course is None:
            continue  # no course covers this gap - skip (do not fabricate)
        items_spec.append((skill_id, course["id"], course["hours"], is_required.get(skill_id, False)))

    # 3. stable sort that preserves prerequisite order: required-first only
    #    applies inside the same dependency level, then shorter course first.
    prereq_pos = {s: i for i, s in enumerate(ordered_skills)}
    items_spec.sort(key=lambda t: (prereq_pos.get(t[0], 999), 0 if t[3] else 1, t[2]))

    plan = UpskillPlan(
        id=f"plan-{student_id}-{job.id}",   # deterministic so re-generation overwrites same job
        student_id=student_id,
        job_id=job.id,
        total_hours=sum(t[2] for t in items_spec),
        narrative="",
    )

    # narrative via LLM (single call)
    user_payload = json.dumps({
        "gaps": [t[0] for t in items_spec],
        "total_hours": plan.total_hours,
        "job": getattr(job, "title", ""),
    })
    narrative_raw = llm.complete(system=PLAN_NARRATIVE_SYSTEM, user=user_payload, json_mode=True)
    cleaned_raw = narrative_raw.strip() if isinstance(narrative_raw, str) else ""
    if cleaned_raw.startswith("```"):
        lines = [ln for ln in cleaned_raw.splitlines() if not ln.strip().startswith("```")]
        cleaned_raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned_raw or narrative_raw)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        plan.narrative = parsed.get("narrative") or parsed.get("plan_narrative") or parsed.get("text") or narrative_raw
    else:
        plan.narrative = narrative_raw

    db.add(plan)
    db.flush()  # persist the parent row before children reference it via plan_id FK (Postgres enforces this; SQLite does not)
    # week packing (<= TARGET_WEEKLY_HOURS/week, preserving topological order)
    week, week_hours = 1, 0
    for order, (skill_id, course_id, hours, _is_req) in enumerate(items_spec, start=1):
        if week_hours > 0 and week_hours + hours > TARGET_WEEKLY_HOURS:
            week += 1
            week_hours = 0
        db.add(PlanItem(
            id=f"{plan.id}-i{order}",
            plan_id=plan.id,
            order=order,
            skill_id=skill_id,
            course_id=course_id,
            hours=hours,
            status="pending",
            week=week,
        ))
        week_hours += hours
    db.commit()
    items = db.exec(select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)).all()
    return plan, items
