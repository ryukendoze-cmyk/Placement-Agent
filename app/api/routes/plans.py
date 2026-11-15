from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import delete
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.matching.rank import rank_jobs
from app.models import Course, Job, JobMatch, PlanItem, Student, StudentSkill, UpskillPlan
from app.plan.planner import _catalog_metadata, build_plan
from app.plan.prereq import PREREQUISITE_GRAPH
from app.security.audit import log_audit
from app.security.rbac import current_role, enforce_student_scope

router = APIRouter()


def _serialize_plan(plan: UpskillPlan, db: Session, ctx=None) -> dict:
    items = db.exec(select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)).all()
    courses = {c.id: c for c in db.exec(select(Course)).all()}
    meta = _catalog_metadata()
    ss_rows = db.exec(select(StudentSkill).where(StudentSkill.student_id == plan.student_id)).all()
    v_statuses = {
        r.skill_id: (getattr(r, "verification_status", "claimed") or "claimed")
        for r in ss_rows
    }
    job = db.get(Job, plan.job_id) if plan.job_id else None
    item_skills = {it.skill_id for it in items}
    total_weeks = max((it.week for it in items), default=1)
    weeks: dict[int, dict] = {}
    for it in items:
        data = weeks.setdefault(it.week, {"total_hours": 0, "items": []})
        data["total_hours"] += it.hours
        data["items"].append(it.id)
    serialized_items = []
    for it in items:
        c = courses.get(it.course_id)
        c_meta = meta.get(it.course_id, {})
        serialized_items.append({
            "id": it.id,
            "order": it.order,
            "week": it.week,
            "skill_id": it.skill_id,
            "skill_verification_status": v_statuses.get(it.skill_id, "missing"),
            "prerequisites": [p for p in PREREQUISITE_GRAPH.get(it.skill_id, []) if p in item_skills],
            "course_id": it.course_id,
            "course_title": c.title if c is not None else "",
            "course_url": c.url if c is not None else "",
            "course_provider": c.provider if c is not None else "",
            "course_cost": c.cost if c is not None else "free",
            "cost_tier": c_meta.get("cost_tier") or (c.cost if c is not None else "free"),
            "url_verified": bool(c_meta.get("url_verified", True)),
            "key_topics": list(c_meta.get("key_topics") or []),
            "syllabus": list(c_meta.get("syllabus") or []),
            "hours": it.hours,
            "status": it.status,
        })
    from app.plan.solver import build_plan_strategy

    strategy = build_plan_strategy(plan, db, ctx=ctx, items=items)
    enrich_map = strategy.get("item_enrichment", {})
    for it_dict in serialized_items:
        e = enrich_map.get(it_dict["id"], {})
        if e:
            it_dict.update({
                "priority_tier": e.get("priority_tier", "Required"),
                "bkt_mastery": e.get("bkt_mastery", 0.12),
                "bkt_mastery_pct": e.get("bkt_mastery_pct", 12),
                "bkt_band": e.get("bkt_band", "Critical Gap"),
                "recommended_modality": e.get("recommended_modality", "Guided Module + Quiz"),
                "roi_pts_per_hr": e.get("roi_pts_per_hr", 0.0),
                "primary_delta_pts": e.get("primary_delta_pts", 0),
                "avg_portfolio_delta_pts": e.get("avg_portfolio_delta_pts", 0.0),
                "drives_helped": e.get("drives_helped", 1),
                "drives_unlocked": e.get("drives_unlocked", 0),
                "covers_skills": e.get("covers_skills", [it_dict["skill_id"]]),
                "alternatives": e.get("alternatives", []),
                "on_critical_path": e.get("on_critical_path", False),
                "slack_h": e.get("slack_h", 0.0),
            })

    return {
        "id": plan.id,
        "student_id": plan.student_id,
        "job_id": plan.job_id,
        "job_title": job.title if job is not None else "",
        "company": job.company if job is not None else "",
        "total_hours": plan.total_hours,
        "total_weeks": total_weeks,
        "narrative": plan.narrative,
        "created_at": plan.created_at.isoformat(),
        "items": serialized_items,
        "weekly_schedule": [
            {"week": wk, "total_hours": data["total_hours"], "items": data["items"]}
            for wk, data in sorted(weeks.items())
        ],
        "strategy": strategy,
    }


@router.post("/api/students/{sid}/plans")
def create_plan(sid: str, request: Request, payload: dict | None = None,
                session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if ctx.llm is None:
        raise HTTPException(503, "LLM not configured")

    job_id = (payload or {}).get("job_id")
    body = payload or {}
    include_upgrades = bool(body.get("include_upgrades", False))
    weekly_cap = max(5, min(25, int(body.get("weekly_cap") or 10)))
    target_mode = str(body.get("target_mode") or "single")
    if target_mode not in ("single", "portfolio", "crunch"):
        target_mode = "single"
    horizon_weeks = max(2, min(12, int(body.get("horizon_weeks") or 4)))
    cost_filter = str(body.get("cost_filter") or "all")
    if cost_filter not in ("all", "free"):
        cost_filter = "all"
    strategy_mode = str(body.get("strategy_mode") or "balanced")
    if strategy_mode not in ("balanced", "max_roi", "fastest_unblock", "deep_mastery"):
        strategy_mode = "balanced"
    exclude_unreachable = bool(body.get("exclude_unreachable", False))
    if job_id:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
    else:
        # default to student's current top match
        matches = session.exec(
            select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
        ).all()
        if not matches:
            raise HTTPException(400, "no matches computed yet; GET /api/students/{sid}/matches first")
        job = session.get(Job, matches[0].job_id)
        if job is None:
            raise HTTPException(404, "job not found")

    # build a JobScore for this (student, job) - reuse rank_jobs, never re-derive
    ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    student_skills = {ss.skill_id: list(ss.evidence) for ss in ss_rows}
    levels = {ss.skill_id: ss.level for ss in ss_rows}
    v_statuses = {
        ss.skill_id: (getattr(ss, "verification_status", "claimed") or "claimed")
        for ss in ss_rows
    }
    sv = ctx.vector_store.get("student", sid)
    if sv is None:
        raise HTTPException(404, "student vector not found; re-upload the resume")
    ranked = rank_jobs(
        [job], sv, student_skills, ctx.embedder, ctx.taxonomy,
        vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors, top=1,
        levels=levels, student=student,
        verification_statuses=v_statuses, session=session,
    )
    if not ranked:
        raise HTTPException(500, "failed to score target job")
    job_score = ranked[0][1]

    # replace any prior plan for this job (deterministic plan id overwrites)
    prior = session.get(UpskillPlan, f"plan-{sid}-{job.id}")
    if prior is not None:
        session.execute(delete(PlanItem).where(PlanItem.plan_id == prior.id))
        session.delete(prior)
        session.commit()

    plan, _items = build_plan(
        session, sid, job, job_score, llm=ctx.llm,
        levels=levels, include_upgrades=include_upgrades,
        weekly_cap=weekly_cap, target_mode=target_mode,
        horizon_weeks=horizon_weeks, cost_filter=cost_filter,
        strategy_mode=strategy_mode, exclude_unreachable=exclude_unreachable,
        ctx=ctx,
    )
    log_audit(session, action="plan.generated", entity_type="plan", entity_id=plan.id,
              actor_role=current_role(request),
              detail={"student_id": sid, "job_id": job.id,
                      "total_hours": plan.total_hours,
                      "proficiency_gaps": len(job_score.proficiency_gaps or [])},
              commit=True)
    return JSONResponse(status_code=201, content=_serialize_plan(plan, session, ctx=ctx))


@router.get("/api/students/{sid}/plans")
def list_plans(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    plans = session.exec(
        select(UpskillPlan).where(UpskillPlan.student_id == sid).order_by(UpskillPlan.created_at.desc())
    ).all()
    ctx = getattr(request.app.state, "ctx", None)
    return [_serialize_plan(p, session, ctx=ctx) for p in plans]


@router.get("/api/plans/{plan_id}")
def get_plan(plan_id: str, request: Request, session: Session = Depends(get_session)):
    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)
    ctx = getattr(request.app.state, "ctx", None)
    return _serialize_plan(plan, session, ctx=ctx)


@router.get("/api/plans/{plan_id}/items/{item_id}/quiz")
def get_quiz(plan_id: str, item_id: str, request: Request,
             session: Session = Depends(get_session)):
    """Return 2 short-answer questions for a plan item. Zero-leak: never
    serializes the grading `keywords` or `explanation` (server-side only)."""
    from app.plan.verification import PASSING_SCORE, QUESTIONS_PER_QUIZ, generate_quiz

    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)
    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan_id:
        raise HTTPException(404, "plan item not found")
    course = session.get(Course, item.course_id)
    meta = _catalog_metadata().get(item.course_id, {})
    questions = generate_quiz(item.skill_id, QUESTIONS_PER_QUIZ)
    return {
        "item_id": item.id,
        "skill_id": item.skill_id,
        "course_title": (course.title if course else ""),
        "key_topics": list(meta.get("key_topics") or []),
        "passing_score": PASSING_SCORE,
        "questions": [
            {
                "id": q["id"],
                "prompt": q.get("question") or q.get("prompt", ""),
                "difficulty": q.get("difficulty", 0),
            }
            for q in questions
        ],
    }


@router.post("/api/plans/{plan_id}/items/{item_id}/verify")
def verify_quiz(plan_id: str, item_id: str, request: Request, payload: dict | None = None,
                session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    """Grade submitted answers. On pass (>= PASSING_SCORE) marks item done,
    promotes StudentSkill verification_status="verified", and triggers shared
    closed-loop re-ranking. On fail returns feedback without mutating state."""
    from app.plan.planner import apply_plan_item_completion
    from app.plan.verification import PASSING_SCORE, check_answers, generate_quiz

    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)
    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan_id:
        raise HTTPException(404, "plan item not found")

    answers = (payload or {}).get("answers") or {}
    if isinstance(answers, list):  # tolerate positional [answer, answer] for ergonomics
        quiz = generate_quiz(item.skill_id)
        answers = {q["id"]: (answers[i] if i < len(answers) else "")
                   for i, q in enumerate(quiz)}
    if not isinstance(answers, dict):
        answers = {}
    passed, score, feedback = check_answers(item.skill_id, answers)

    try:
        from app.interview.feedback_loop import QuestionTelemetryManager
        QuestionTelemetryManager().record_answer_turn(
            question_id=f"plan_quiz:{item.skill_id}",
            score=float(score),
            matched_kw=[item.skill_id] if passed else [],
            expected_kw=[item.skill_id],
            difficulty="intermediate",
        )
    except Exception:
        pass

    if not passed:
        from app.plan.solver import load_catalog_metadata
        cat = load_catalog_metadata().get(item.course_id, {})
        missed = [f["question_id"] for f in feedback if not f["correct"]]
        return {
            "item_id": item.id, "passed": False, "score": score,
            "passing_score": PASSING_SCORE,
            "correct_count": sum(1 for f in feedback if f["correct"]),
            "total_questions": len(feedback),
            "status": item.status, "feedback": feedback,
            "remediation": {
                "missed_question_ids": missed,
                "key_topics": list(cat.get("key_topics") or [])[:4],
                "message": (
                    f"Score {score}% below {PASSING_SCORE}% passing gate. Review the highlighted syllabus "
                    "concepts below before re-taking the verification quiz."
                ),
            },
        }

    result = apply_plan_item_completion(
        session, ctx, plan, item, verified_via_quiz=True, quiz_score=score
    )
    log_audit(session, action="plan.quiz_verified", entity_type="plan_item",
              entity_id=item.id, actor_role=current_role(request),
              detail={"plan_id": plan.id, "skill_id": item.skill_id,
                      "score": score, "readiness": result.get("readiness")},
              commit=True)
    return {
        "item_id": item.id, "passed": True, "score": score,
        "passing_score": PASSING_SCORE,
        "correct_count": sum(1 for f in feedback if f["correct"]),
        "total_questions": len(feedback),
        "status": result["status"], "feedback": feedback,
        "closed_loop": {
            "skill_id": result["skill_id"],
            "verification_status": result.get("verification_status", "verified"),
            "updated_match_score": result["best_match"],
            "readiness": result["readiness"],
            "plan_pct": result["plan_pct"],
        },
    }



@router.post("/api/plans/{plan_id}/swap-course")
def swap_plan_course(
    plan_id: str,
    payload: dict,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    """Replace the course backing a PlanItem with an alternative covering the same skill."""
    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)

    body = payload or {}
    item_id = str(body.get("item_id") or "").strip()
    course_id = str(body.get("course_id") or "").strip()
    if not item_id or not course_id:
        raise HTTPException(400, "item_id and course_id are required")

    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan.id:
        raise HTTPException(404, "plan item not found")

    from app.plan.planner import TARGET_WEEKLY_HOURS, _pack_weeks
    from app.plan.solver import load_catalog_metadata

    catalog = load_catalog_metadata()
    db_course = session.get(Course, course_id)
    cat_entry = catalog.get(course_id)
    if db_course is None and cat_entry is None:
        raise HTTPException(404, f"course '{course_id}' not found")

    allowed_skills = set(getattr(db_course, "skill_ids", None) or [])
    if cat_entry:
        allowed_skills.update(cat_entry.get("skill_ids") or [])
    if item.skill_id not in allowed_skills:
        raise HTTPException(400, f"course '{course_id}' does not cover skill '{item.skill_id}'")

    if db_course is None and cat_entry is not None:
        db_course = Course(
            id=course_id,
            title=str(cat_entry.get("title") or course_id),
            provider=str(cat_entry.get("provider") or "Verified Resource"),
            url=str(cat_entry.get("url") or ""),
            skill_ids=list(cat_entry.get("skill_ids") or [item.skill_id]),
            level=str(cat_entry.get("level") or "beginner"),
            hours=int(cat_entry.get("hours") or 8),
            cost=str(cat_entry.get("cost") or "free"),
        )
        session.add(db_course)
        session.flush()

    old_cid = item.course_id
    new_hours = int((cat_entry or {}).get("hours") or db_course.hours)
    new_title = str((cat_entry or {}).get("title") or db_course.title)
    item.course_id = course_id
    item.hours = new_hours
    session.add(item)
    session.flush()

    all_items = session.exec(
        select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)
    ).all()
    strat = dict(getattr(plan, "strategy_json", None) or {})
    weekly_cap = int((strat.get("constraints") or {}).get("weekly_cap") or TARGET_WEEKLY_HOURS)
    packed, _ = _pack_weeks(
        [(it.skill_id, it.course_id, it.hours, False) for it in all_items],
        max(2, weekly_cap),
    )
    for it, p in zip(all_items, packed):
        it.week = p[5]
        session.add(it)

    plan.total_hours = sum(it.hours for it in all_items)
    ledger = list(strat.get("recovery_ledger") or [])
    ledger.append({
        "kind": "Course Swap",
        "detail": f"Swapped '{item.skill_id}' from {old_cid} to '{new_title}' ({new_hours}h).",
    })
    strat["recovery_ledger"] = ledger
    plan.strategy_json = strat
    session.add(plan)

    log_audit(
        session,
        action="plan.course_swapped",
        entity_type="plan_item",
        entity_id=item.id,
        actor_role=current_role(request),
        detail={"plan_id": plan.id, "skill_id": item.skill_id, "from": old_cid, "to": course_id},
    )
    session.commit()
    session.refresh(plan)
    return _serialize_plan(plan, session, ctx=ctx)


@router.post("/api/plans/{plan_id}/items/{item_id}/waiver")
def request_plan_item_waiver(
    plan_id: str,
    item_id: str,
    payload: dict | None = None,
    request: Request = None,  # type: ignore[assignment]
    session: Session = Depends(get_session),
):
    """Submit a HITL proficiency_waiver ApprovalRequest for a planned skill."""
    from datetime import datetime, timezone
    from uuid import uuid4

    from app.models import ApprovalRequest

    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)

    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan.id:
        raise HTTPException(404, "plan item not found")

    body = payload or {}
    target_level = str(body.get("target_level") or body.get("level") or "intermediate").lower()
    if target_level not in ("beginner", "basic", "intermediate", "advanced"):
        raise HTTPException(400, "invalid target_level")
    if target_level == "beginner":
        target_level = "basic"
    reason = str(body.get("reason") or "Equivalent project or coursework experience submitted for verification.").strip()[:400]

    # Ensure StudentSkill row exists so TPO approval execution never 404s on gap skills
    ss = session.get(StudentSkill, (plan.student_id, item.skill_id))
    if ss is None:
        ss = StudentSkill(
            student_id=plan.student_id,
            skill_id=item.skill_id,
            mentions=0,
            level="basic",
            evidence=[f"waiver_requested:{reason[:80]}"],
            verification_status="unverified",
            source="waiver_request",
        )
        session.add(ss)
        session.flush()

    aid = f"apr-{uuid4().hex[:8]}"
    actor = current_role(request)
    ap = ApprovalRequest(
        id=aid,
        kind="proficiency_waiver",
        target_id=item.skill_id,
        title=f"Proficiency Waiver: {item.skill_id} -> {target_level}",
        summary=reason,
        payload={
            "student_id": plan.student_id,
            "skill_id": item.skill_id,
            "level": target_level,
            "plan_id": plan.id,
            "item_id": item.id,
            "reason": reason,
        },
        status="pending_review",
        requested_by=actor,
        created_at=datetime.now(timezone.utc),
    )
    session.add(ap)
    log_audit(
        session,
        action="plan.waiver_requested",
        entity_type="approval",
        entity_id=aid,
        actor_role=actor,
        detail={"student_id": plan.student_id, "skill_id": item.skill_id, "level": target_level},
    )
    session.commit()
    return {
        "approval_id": aid,
        "status": ap.status,
        "skill_id": item.skill_id,
        "requested_level": target_level,
    }


@router.post("/api/plans/{plan_id}/simulate")
def simulate_plan_endpoint(
    plan_id: str,
    payload: dict | None = None,
    request: Request = None,  # type: ignore[assignment]
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    """Run live counterfactual portfolio simulation for candidate skills & weekly cap."""
    from app.plan.solver import simulate_plan_scenario

    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)

    body = payload or {}
    added_skills = body.get("added_skills")
    if added_skills is not None and not isinstance(added_skills, list):
        raise HTTPException(400, "added_skills must be a list of skill IDs")
    weekly_cap = int(body.get("weekly_cap") or 10)
    assume_mock = bool(body.get("assume_mock_passed", False))

    return simulate_plan_scenario(
        plan,
        session,
        ctx=ctx,
        added_skills=added_skills,
        weekly_cap=weekly_cap,
        assume_mock_passed=assume_mock,
    )
