"""Upskilling planner: deterministic gap->course selection + grounded LLM narrative.

Reuses score_job's missing_required/missing_preferred breakdown (never re-derives
scores) and rank_jobs only for choosing a default target job when none is given.
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlmodel import Session, delete, select

from ..matching.engine import JobScore
from ..models import Course, JobMatch, PlanItem, UpskillPlan
from .prereq import PREREQUISITE_GRAPH

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

PLAN_NARRATIVE_SYSTEM = (
    "TASK: plan_narrative\n"
    "You are an expert career mentor. Given the student's target job, exact week-by-week course schedule, "
    "verified course URLs, and total hours, write a concise, encouraging, actionable 3-4 sentence upskilling "
    "narrative and weekly study strategy. ONLY cite course titles, course IDs, URLs, and hour totals present "
    "verbatim in the provided JSON payload; never invent courses, URLs, or placement rules. "
    "Respond ONLY with a valid JSON object containing a single key \"narrative\" (string)."
)


@lru_cache(maxsize=1)
def _catalog_metadata() -> dict[str, dict]:
    """Load non-table course metadata (cost_tier, url_verified, key_topics, syllabus)."""
    path = DATA_DIR / "courses.json"
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return {r["id"]: r for r in rows if isinstance(r, dict) and r.get("id")}
    except Exception:
        return {}


def pick_course_for_skill(skill_id: str, courses: list[dict],
                          target_level: str | None = None) -> dict | None:
    """Best verified course covering skill_id.

    Prefers verified URLs first, then respects ``target_level`` ordering
    (beginner/basic first by default; intermediate -> beginner for intermediate targets;
    advanced -> intermediate -> beginner for advanced targets), then free cost_tier/cost,
    then fewer hours.
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
        0 if c.get("url_verified", True) else 1,
        level_rank(c),
        0 if (c.get("cost_tier") or c.get("cost")) == "free" else 1,
        c.get("hours", 999),
    ))
    return candidates[0]


def _load_courses(db: Session) -> list[dict]:
    meta = _catalog_metadata()
    rows = db.exec(select(Course)).all()
    out: list[dict] = []
    for c in rows:
        extra = meta.get(c.id, {})
        out.append({
            "id": c.id,
            "title": c.title,
            "provider": c.provider,
            "url": c.url,
            "skill_ids": c.skill_ids,
            "level": c.level,
            "hours": c.hours,
            "cost": c.cost,
            "cost_tier": extra.get("cost_tier") or c.cost,
            "url_verified": bool(extra.get("url_verified", True)),
            "key_topics": list(extra.get("key_topics") or []),
            "syllabus": list(extra.get("syllabus") or []),
        })
    return out


TARGET_WEEKLY_HOURS = 10


def apply_plan_item_completion(
    session,
    ctx,
    plan: UpskillPlan,
    item: PlanItem,
    *,
    evidence_note: str = "",
    verified_via_quiz: bool = False,
    quiz_score: int | None = None,
) -> dict:
    """Shared closed-loop mutation for BOTH POST .../complete and POST .../verify.

    1. Mark PlanItem done (idempotent).
    2. Upsert StudentSkill for item.skill_id: new skill -> basic; existing basic
       -> intermediate; intermediate/advanced -> unchanged unless senior role (advanced).
       Only promotes `verification_status="verified"` when `verified_via_quiz=True`.
    3. Re-run rank_jobs across ALL jobs with the updated skills, levels, student
       eligibility context, and verification statuses, overwriting JobMatch rows.
    4. Recompute plan_pct, best_match, readiness = round(0.5*plan_pct + 0.5*best).
    5. Idempotently record/update ProgressEvent(kind="plan_item_done", ref_id=item.id).
    """
    from datetime import datetime, timezone
    from uuid import uuid4

    from ..matching.engine import is_senior_job
    from ..matching.rank import rank_jobs
    from ..models import AuditLog, Job, JobApplication, ProgressEvent, Student, StudentSkill

    now = datetime.now(timezone.utc)
    already_done = item.status == "done"
    ss = session.get(StudentSkill, (plan.student_id, item.skill_id))
    course = session.get(Course, item.course_id) if item.course_id else None
    course_label = course.title if course is not None else (item.course_id or item.skill_id)
    note = evidence_note or (
        f"Verified via quiz (score={quiz_score}%) for course {course_label} ({item.skill_id})"
        if verified_via_quiz and quiz_score is not None
        else f"Completed course {course_label} for {item.skill_id}"
    )

    if not already_done:
        item.status = "done"
        session.add(item)
        plan_job = session.get(Job, plan.job_id) if plan.job_id else None
        if ss is None:
            ss = StudentSkill(
                student_id=plan.student_id,
                skill_id=item.skill_id,
                mentions=1,
                level="basic",
                evidence=[note],
                verification_status=("verified" if verified_via_quiz else "claimed"),
                confidence=(0.9 if verified_via_quiz else 0.65),
                source=("plan_quiz" if verified_via_quiz else "plan_completion"),
                last_verified_score=(quiz_score if verified_via_quiz else None),
                last_verified_at=(now if verified_via_quiz else None),
            )
            session.add(ss)
        else:
            if ss.level == "basic":
                ss.level = "intermediate"
            elif ss.level == "intermediate" and is_senior_job(plan_job):
                ss.level = "advanced"
            ss.mentions = (ss.mentions or 0) + 1
            if note not in ss.evidence:
                ss.evidence = list(ss.evidence) + [note]
            if verified_via_quiz:
                ss.verification_status = "verified"
                ss.confidence = max(float(getattr(ss, "confidence", 0.5) or 0.5), 0.9)
                ss.source = "plan_quiz"
                ss.last_verified_score = quiz_score
                ss.last_verified_at = now
            elif getattr(ss, "verification_status", "claimed") == "unverified":
                # Completing a course without passing quiz promotes from unverified to needs_reverification.
                ss.verification_status = "needs_reverification"
            session.add(ss)
    elif verified_via_quiz and ss is not None:
        if note not in ss.evidence:
            ss.evidence = list(ss.evidence) + [note]
        ss.verification_status = "verified"
        ss.confidence = max(float(getattr(ss, "confidence", 0.5) or 0.5), 0.9)
        ss.source = "plan_quiz"
        ss.last_verified_score = quiz_score
        ss.last_verified_at = now
        session.add(ss)

    session.flush()  # persist skill before re-ranking

    # 3. re-rank all jobs with updated skills, levels, verification statuses & student context
    ss_rows = session.exec(select(StudentSkill).where(
        StudentSkill.student_id == plan.student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    v_statuses = {
        r.skill_id: (getattr(r, "verification_status", "claimed") or "claimed")
        for r in ss_rows
    }
    student = session.get(Student, plan.student_id)
    sv = ctx.vector_store.get("student", plan.student_id)
    if sv is None:
        sv = ctx.embedder.embed((student.resume_text if student else "") or "")
    jobs = session.exec(select(Job)).all()
    ranked = rank_jobs(
        jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
        vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
        top=None, levels=levels, student=student,
        verification_statuses=v_statuses, session=session,
    )
    from dataclasses import asdict as _asdict
    session.execute(delete(JobMatch).where(JobMatch.student_id == plan.student_id))
    for job, js in ranked:
        session.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=plan.student_id,
                             job_id=job.id, score=js.score, band=js.band,
                             breakdown=_asdict(js)))
    session.flush()

    # Closed-loop application sync: refresh match_score on active JobApplications
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

    # 5. idempotent progress event + audit on non-quiz completion
    prior_ev = session.exec(
        select(ProgressEvent).where(
            ProgressEvent.ref_id == item.id, ProgressEvent.kind == "plan_item_done"
        )
    ).first()
    if prior_ev is not None:
        prior_ev.readiness = readiness
        prior_ev.at = now
        session.add(prior_ev)
    else:
        session.add(ProgressEvent(
            id=f"pe-{uuid4().hex[:8]}",
            student_id=plan.student_id,
            kind="plan_item_done",
            ref_id=item.id,
            readiness=readiness,
            at=now,
        ))
    if not already_done and not verified_via_quiz:
        session.add(AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="plan.item_completed",
            entity_type="plan_item",
            entity_id=item.id,
            actor_role="student",
            detail={
                "plan_id": plan.id,
                "skill_id": item.skill_id,
                "verification_status": (ss.verification_status if ss else "claimed"),
                "readiness": readiness,
            },
        ))
    session.commit()

    return {"item_id": item.id, "status": item.status, "readiness": readiness,
            "plan_pct": plan_pct, "best_match": best_match,
            "skill_id": item.skill_id,
            "verification_status": (ss.verification_status if ss else "claimed")}


def _pack_weeks(
    items_spec: list[tuple[str, str, int, bool]],
    max_weekly_hours: int = TARGET_WEEKLY_HOURS,
) -> tuple[list[tuple[int, str, str, int, bool, int]], int]:
    """Pack ordered course specs into weekly buckets <= max_weekly_hours (preserving order)."""
    packed: list[tuple[int, str, str, int, bool, int]] = []
    week, week_hours = 1, 0
    for order, (skill_id, course_id, hours, is_req) in enumerate(items_spec, start=1):
        if week_hours > 0 and week_hours + hours > max_weekly_hours:
            week += 1
            week_hours = 0
        packed.append((order, skill_id, course_id, hours, is_req, week))
        week_hours += hours
    total_weeks = max((p[5] for p in packed), default=1)
    return packed, total_weeks


def _deterministic_narrative(
    job,
    packed_items: list[tuple[int, str, str, int, bool, int]],
    courses_by_id: dict[str, dict],
    total_hours: int,
    total_weeks: int,
) -> str:
    """Grounded deterministic study strategy citing only verified DB courses and hours."""
    role = getattr(job, "title", None) or "your target role"
    if not packed_items:
        return (
            f"You already meet the required skills for {role}. "
            "Verify any claimed or unverified skills via mock interviews and keep applying to strong matches."
        )
    first = packed_items[0]
    first_course = courses_by_id.get(first[2], {})
    first_label = first_course.get("title") or first[1]
    wk_label = "week" if total_weeks == 1 else "weeks"
    skills_list = ", ".join(p[1] for p in packed_items[:4])
    more = f" (+{len(packed_items) - 4} more)" if len(packed_items) > 4 else ""
    return (
        f"Start with Week 1 ({first_label} for {first[1]}, {first[3]}h) to close the highest-priority "
        f"gaps for {role}. This plan covers {skills_list}{more} across {total_hours}h over "
        f"{total_weeks} {wk_label} (<= {TARGET_WEEKLY_HOURS}h/week). Complete each verified course and "
        "pass the micro-quiz to confirm readiness."
    )


def _is_narrative_grounded(
    text: str,
    db: Session,
    *,
    allowed_urls: set[str],
    allowed_course_ids: set[str],
) -> bool:
    """Validate that an LLM narrative contains no hallucinated URLs, IDs, or SOP rules."""
    if not text or not text.strip():
        return False
    # Guard against any unverified external URLs directly
    for url in re.findall(r"https?://[^\s)\]>'\"]+", text):
        norm = url.rstrip(".,;:!?/")
        if norm not in allowed_urls:
            return False
    try:
        from ..agent.verifier import verify_text
        report = verify_text(
            text,
            known=db,
            allowed_urls=allowed_urls,
            allowed_course_ids=allowed_course_ids,
        )
        if report.get("unverified") or int(report.get("grounding_score", 100)) < 100:
            return False
    except Exception:
        # Never crash plan generation if verifier cannot inspect; enforce direct ID/URL guards below.
        for cid in re.findall(r"\bcourse-\d+\b", text):
            if cid not in allowed_course_ids:
                return False
    return True


def _build_grounded_narrative(
    db: Session,
    job,
    packed_items: list[tuple[int, str, str, int, bool, int]],
    courses_by_id: dict[str, dict],
    total_hours: int,
    total_weeks: int,
    llm: Any,
) -> str:
    """Invoke LLM safely with full grounded schedule context and deterministic fallback."""
    fallback = _deterministic_narrative(job, packed_items, courses_by_id, total_hours, total_weeks)
    if llm is None:
        return fallback

    schedule_ctx = [
        {
            "week": wk,
            "skill": sk,
            "course_id": cid,
            "course_title": (courses_by_id.get(cid, {}).get("title") or cid),
            "url": (courses_by_id.get(cid, {}).get("url") or ""),
            "hours": hrs,
        }
        for _ord, sk, cid, hrs, _req, wk in packed_items
    ]
    allowed_urls = {
        str(row["url"]).strip().rstrip("/")
        for row in schedule_ctx
        if str(row.get("url") or "").strip()
    }
    allowed_course_ids = {cid for _ord, _sk, cid, _hrs, _req, _wk in packed_items}

    user_payload = json.dumps({
        "gaps": [p[1] for p in packed_items],
        "total_hours": total_hours,
        "weeks": total_weeks,
        "total_weeks": total_weeks,
        "weekly_hour_cap": TARGET_WEEKLY_HOURS,
        "job": getattr(job, "title", ""),
        "schedule": schedule_ctx,
    })

    try:
        narrative_raw = llm.complete(system=PLAN_NARRATIVE_SYSTEM, user=user_payload, json_mode=True)
    except Exception:
        logger.warning("LLM plan narrative failed; falling back to deterministic narrative", exc_info=True)
        return fallback

    cleaned_raw = narrative_raw.strip() if isinstance(narrative_raw, str) else ""
    if cleaned_raw.startswith("```"):
        lines = [ln for ln in cleaned_raw.splitlines() if not ln.strip().startswith("```")]
        cleaned_raw = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned_raw or (narrative_raw if isinstance(narrative_raw, str) else ""))
    except (ValueError, TypeError):
        parsed = None

    if isinstance(parsed, dict):
        candidate = parsed.get("narrative") or parsed.get("plan_narrative") or parsed.get("text") or ""
    elif parsed is not None:
        return fallback
    else:
        candidate = cleaned_raw

    if not isinstance(candidate, str) or not candidate.strip():
        return fallback
    if not _is_narrative_grounded(
        candidate,
        db,
        allowed_urls=allowed_urls,
        allowed_course_ids=allowed_course_ids,
    ):
        logger.warning("LLM plan narrative failed grounding check; using deterministic fallback")
        return fallback
    return candidate.strip()


def build_plan(
    db: Session,
    student_id: str,
    job,
    job_score: JobScore,
    *,
    llm,
    levels: dict[str, str] | None = None,
    include_upgrades: bool = False,
) -> tuple[UpskillPlan, list[PlanItem]]:
    """Create UpskillPlan + PlanItems for gaps and remedial/upgrade skills.

    Ordering: prerequisite DAG (app.plan.prereq) puts each prerequisite's course
    before its consumers; within one dependency level, required gaps before
    preferred, then shorter verified course first. Shared courses are deduplicated
    (counted once in total hours). Skills already owned are excluded unless
    `not job_score.missing_required`, `include_upgrades=True`, `verification_status`
    is `unverified` / `needs_reverification`, or upgrading `basic` -> `advanced`.

    Returns (plan, items).
    """
    from ..models import StudentSkill
    from .prereq import topological_order

    courses = _load_courses(db)
    courses_by_id = {c["id"]: c for c in courses}
    ss_rows = db.exec(
        select(StudentSkill).where(StudentSkill.student_id == student_id)
    ).all()
    if levels is None:
        levels = {r.skill_id: r.level for r in ss_rows}
    known = {r.skill_id for r in ss_rows}
    v_statuses = {
        r.skill_id: (getattr(r, "verification_status", "claimed") or "claimed").lower()
        for r in ss_rows
    }

    # gap tuples: (skill_id, is_required)
    gaps: list[tuple[str, bool]] = []
    for g in job_score.missing_required:
        gaps.append((g["skill"], True))
    for g in job_score.missing_preferred:
        gaps.append((g["skill"], False))
    is_required = dict(gaps)

    # Proficiency/verification-upgrade targets:
    # Run when missing_required is empty, OR explicit include_upgrades=True, OR
    # skill is unverified/needs_reverification, OR basic against an advanced senior role.
    upgrade_targets: dict[str, str] = {}
    for g in (job_score.proficiency_gaps or []):
        skill = g.get("skill")
        current = (g.get("current_level") or levels.get(skill or "", "") or "").lower()
        target = (g.get("target_level") or "").lower()
        if not skill or current in ("", "none"):
            continue
        v_stat = v_statuses.get(skill, "claimed")
        should_upgrade = (
            not job_score.missing_required
            or include_upgrades
            or v_stat in ("unverified", "needs_reverification")
            or (target == "advanced" and current == "basic")
        )
        if should_upgrade and target and (target != current or v_stat in ("unverified", "needs_reverification")):
            upgrade_targets[skill] = target
            is_required.setdefault(skill, True)

    job_req = list(getattr(job, "required_skills", None) or [])
    job_pref = list(getattr(job, "preferred_skills", None) or [])
    for sk in job_req + job_pref:
        if sk in known and v_statuses.get(sk, "claimed") in ("unverified", "needs_reverification"):
            upgrade_targets.setdefault(sk, levels.get(sk, "intermediate"))
            is_required.setdefault(sk, sk in job_req)

    # Skills owned cleanly stay excluded as items (still constrain DAG edges).
    exclude = known - set(upgrade_targets)

    # Priority: required (0) before preferred (1), then fewer hours first.
    priority = {
        s: (0 if is_required.get(s, False) else 1,
            (pick_course_for_skill(s, courses, upgrade_targets.get(s)) or {}).get("hours", 999))
        for s in is_required
    }

    # 1. Prerequisite-first order over the induced gap subgraph.
    ordered_skills = topological_order(
        PREREQUISITE_GRAPH, list(is_required), exclude=exclude, priority=priority
    )

    # 2. Resolve each ordered skill to a verified real course, deduplicating course_id
    #    so one multi-skill course is never double-billed in hours. Prefer explicit /
    #    required targets over pure transitive prerequisites for the PlanItem's skill_id.
    items_spec: list[tuple[str, str, int, bool]] = []
    seen_course_idx: dict[str, int] = {}
    prereq_pos = {s: i for i, s in enumerate(ordered_skills)}
    course_pos: dict[str, int] = {}
    for skill_id in ordered_skills:
        course = pick_course_for_skill(skill_id, courses, upgrade_targets.get(skill_id))
        if course is None:
            continue
        cid = course["id"]
        req_flag = is_required.get(skill_id, False)
        course_pos[cid] = min(course_pos.get(cid, 999), prereq_pos.get(skill_id, 999))
        if cid in seen_course_idx:
            idx = seen_course_idx[cid]
            prev_sk, prev_cid, prev_hrs, prev_req = items_spec[idx]
            prefer_new = (req_flag and not prev_req) or (
                req_flag == prev_req
                and skill_id in is_required
                and prev_sk not in is_required
            )
            chosen_sk = skill_id if prefer_new else prev_sk
            items_spec[idx] = (chosen_sk, prev_cid, prev_hrs, prev_req or req_flag)
            continue
        seen_course_idx[cid] = len(items_spec)
        items_spec.append((skill_id, cid, course["hours"], req_flag))

    # 3. Stable sort preserving earliest prerequisite order: required-first inside same depth, shorter first.
    items_spec.sort(key=lambda t: (course_pos.get(t[1], 999), 0 if t[3] else 1, t[2]))

    # 4. Pack weeks FIRST so prompt, UI, and narrative share the exact same weekly schedule.
    packed_items, total_weeks = _pack_weeks(items_spec, TARGET_WEEKLY_HOURS)
    total_hours = sum(t[3] for t in packed_items)

    plan = UpskillPlan(
        id=f"plan-{student_id}-{job.id}",
        student_id=student_id,
        job_id=job.id,
        total_hours=total_hours,
        narrative="",
    )
    plan.narrative = _build_grounded_narrative(
        db, job, packed_items, courses_by_id, total_hours, total_weeks, llm
    )

    db.add(plan)
    db.flush()
    for order, skill_id, course_id, hours, _is_req, week in packed_items:
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
    db.commit()
    items = db.exec(select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)).all()
    return plan, items


def ensure_remedial_plan_items(
    session: Session,
    student_id: str,
    skill_ids: list[str],
    *,
    llm: Any = None,
) -> list[str]:
    """Append or reopen remedial PlanItems for unverified/<70% interview skills.

    Returns the list of PlanItem IDs created or reopened. Never creates fake courses
    or duplicate course hours.
    """
    from uuid import uuid4

    from ..models import Job

    clean_skills = [s.strip() for s in dict.fromkeys(skill_ids or []) if isinstance(s, str) and s.strip()]
    if not clean_skills:
        return []

    courses = _load_courses(session)
    courses_by_id = {c["id"]: c for c in courses}

    plan = session.exec(
        select(UpskillPlan)
        .where(UpskillPlan.student_id == student_id)
        .order_by(UpskillPlan.created_at.desc())
    ).first()

    if plan is None:
        top_match = session.exec(
            select(JobMatch).where(JobMatch.student_id == student_id).order_by(JobMatch.score.desc())
        ).first()
        job = session.get(Job, top_match.job_id) if top_match else session.exec(select(Job)).first()
        job_id = job.id if job is not None else "job-001"
        plan = UpskillPlan(
            id=f"plan-{student_id}-{job_id}",
            student_id=student_id,
            job_id=job_id,
            total_hours=0,
            narrative="",
        )
        session.add(plan)
        session.flush()

    existing_items = session.exec(
        select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)
    ).all()
    by_skill = {it.skill_id: it for it in existing_items}
    existing_courses = {it.course_id for it in existing_items if it.course_id}
    next_order = max((it.order for it in existing_items), default=0) + 1
    current_week = max((it.week for it in existing_items), default=1)
    current_week_hours = sum(it.hours for it in existing_items if it.week == current_week)

    touched_ids: list[str] = []
    for sk in clean_skills:
        existing = by_skill.get(sk)
        if existing is not None:
            if existing.status == "done":
                existing.status = "pending"
                session.add(existing)
                touched_ids.append(existing.id)
            continue

        course = pick_course_for_skill(sk, courses, target_level="intermediate")
        if course is None:
            continue
        cid = course["id"]
        if cid in existing_courses:
            # Reuse existing course coverage without inflating plan hours.
            continue
        hrs = int(course.get("hours") or 4)
        if current_week_hours > 0 and current_week_hours + hrs > TARGET_WEEKLY_HOURS:
            current_week += 1
            current_week_hours = 0
        item = PlanItem(
            id=f"{plan.id}-rem-{uuid4().hex[:6]}",
            plan_id=plan.id,
            order=next_order,
            skill_id=sk,
            course_id=cid,
            hours=hrs,
            status="pending",
            week=current_week,
        )
        session.add(item)
        existing_courses.add(cid)
        next_order += 1
        current_week_hours += hrs
        touched_ids.append(item.id)

    session.flush()
    all_items = session.exec(
        select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)
    ).all()
    plan.total_hours = sum(it.hours for it in all_items)
    if not plan.narrative:
        job = session.get(Job, plan.job_id) if plan.job_id else None
        packed = [(it.order, it.skill_id, it.course_id, it.hours, True, it.week) for it in all_items]
        plan.narrative = _build_grounded_narrative(
            session, job, packed, courses_by_id,
            plan.total_hours, max((it.week for it in all_items), default=1), llm
        )
    session.add(plan)
    session.commit()
    return touched_ids
