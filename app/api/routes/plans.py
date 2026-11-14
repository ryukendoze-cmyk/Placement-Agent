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


def _serialize_plan(plan: UpskillPlan, db: Session) -> dict:
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
    include_upgrades = bool((payload or {}).get("include_upgrades", False))
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
    )
    log_audit(session, action="plan.generated", entity_type="plan", entity_id=plan.id,
              actor_role=current_role(request),
              detail={"student_id": sid, "job_id": job.id,
                      "total_hours": plan.total_hours,
                      "proficiency_gaps": len(job_score.proficiency_gaps or [])},
              commit=True)
    return JSONResponse(status_code=201, content=_serialize_plan(plan, session))


@router.get("/api/students/{sid}/plans")
def list_plans(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    plans = session.exec(
        select(UpskillPlan).where(UpskillPlan.student_id == sid).order_by(UpskillPlan.created_at.desc())
    ).all()
    return [_serialize_plan(p, session) for p in plans]


@router.get("/api/plans/{plan_id}")
def get_plan(plan_id: str, request: Request, session: Session = Depends(get_session)):
    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)
    return _serialize_plan(plan, session)


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

    if not passed:
        return {
            "item_id": item.id, "passed": False, "score": score,
            "passing_score": PASSING_SCORE,
            "correct_count": sum(1 for f in feedback if f["correct"]),
            "total_questions": len(feedback),
            "status": item.status, "feedback": feedback,
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
