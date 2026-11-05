from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.api.routes.students import _skill_payload, create_student_core, create_student_core_resume
from app.matching.engine import job_eligibility, job_tier
from app.models import (
    Course,
    Interview,
    Job,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Student,
    StudentSkill,
    UpskillPlan,
)
from app.security.rbac import current_role, enforce_student_scope, get_current_user_optional

router = APIRouter()


def _students_menu(session) -> list:
    return session.exec(select(Student).order_by(Student.created_at.desc())).all()


def _tpl(request: Request, name: str, session: Session, sid: str = "", active: str = "", **ctx):
    students = _students_menu(session)
    role = current_role(request)
    user = get_current_user_optional(request)
    department = (
        request.headers.get("x-demo-department")
        or request.cookies.get("ps06_department")
        or (user.department if user and user.department else None)
        or ctx.get("department")
        or "CSE"
    )
    # §17.2.C.2: a faculty mentor's global switcher shows only their department.
    if user is not None and user.role == "faculty_mentor" and user.department:
        dept = user.department.strip().lower()
        students = [s for s in students if str(s.branch or "").strip().lower() == dept]
    return request.app.state.templates.TemplateResponse(
        request, name,
        {**ctx, "students": students, "sid": sid, "active": active, "role": role,
         "user": user, "current_user": user, "user_name": (user.name if user else ""),
         "department": department,
         "user_department": (user.department if user else None),
         "error": ctx.get("error", "")})


@router.get("/")
def index(request: Request, session: Session = Depends(get_session)):
    students = session.exec(select(Student).order_by(Student.created_at.desc())).all()
    return _tpl(request, "index.html", session, sid="", active="home", students=students, error="")


@router.post("/upload")
def upload(file: UploadFile = File(...), name: str = Form(...), branch: str = Form(""),
           year: int = Form(3), gpa: float = Form(0.0),
           session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    sid, _skills = create_student_core(session, ctx, file, name, branch, year, gpa)
    return RedirectResponse(url=f"/students/{sid}", status_code=303)


def _group_skills(session, ctx, sid):
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    grouped: dict[str, list] = {}
    for ss in rows:
        p = _skill_payload(ss, ctx)
        grouped.setdefault(p["category"], []).append(p)
    for items in grouped.values():
        items.sort(key=lambda x: x["name"])
    return grouped


@router.get("/students/{sid}")
def profile(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    return _tpl(request, "student/profile.html", session, sid=sid, active="profile",
                student=student, skills_by_category=_group_skills(session, ctx, sid), error="")


@router.post("/students/{sid}/resume")
def reupload(sid: str, request: Request, file: UploadFile = File(...),
             session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    create_student_core_resume(session, ctx, sid, file)
    # partial fragment (hx-target): no shell context needed
    return request.app.state.templates.TemplateResponse(
        request, "student/_skill_chips.html",
        {"skills_by_category": _group_skills(session, ctx, sid)})


@router.post("/students/{sid}/delete")
def delete_student_page(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    from app.api.routes.students import delete_student_core
    enforce_student_scope(request, sid)
    delete_student_core(session, ctx, sid)
    return RedirectResponse(url="/", status_code=303)


@router.get("/students/{sid}/matches")
def matches_page(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    evidence: dict[str, str] = {}
    for ss in session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all():
        if ss.evidence:
            evidence[ss.skill_id] = ss.evidence[0]
    rows = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                        .order_by(JobMatch.score.desc())).all()
    # Incremental auto-rank: ensure newly added jobs in DB are scored for students who already ran matching
    existing_job_ids = {m.job_id for m in rows}
    all_jobs = session.exec(select(Job)).all()
    missing_jobs = [j for j in all_jobs if j.id not in existing_job_ids]
    if rows and missing_jobs:
        try:
            from dataclasses import asdict
            from uuid import uuid4

            from app.matching.rank import rank_jobs
            sv = ctx.vector_store.get("student", sid)
            if sv is None and getattr(student, "resume_text", None):
                sv = ctx.embedder.embed(student.resume_text)
                ctx.vector_store.upsert("student", sid, sv)
            if sv is None:
                sv = [0.0] * getattr(ctx.embedder, "dim", 384)
            ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
            student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
            levels = {r.skill_id: r.level for r in ss_rows}
            new_results = rank_jobs(
                missing_jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
                vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                top=None, levels=levels, student=student,
            )
            for job, js in new_results:
                session.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=job.id,
                                     score=js.score, band=js.band, breakdown=asdict(js)))
            session.commit()
            rows = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                                .order_by(JobMatch.score.desc())).all()
        except Exception:
            pass
    cards = []
    for m in rows:
        job = session.get(Job, m.job_id)
        if job is None:
            continue
        b = m.breakdown or {}
        eligible, reasons = job_eligibility(student, job)
        job_url = getattr(job, "job_url", "") or ""
        source = getattr(job, "source", "local") or "local"
        scraper_engine = getattr(job, "scraper_engine", "jobspy") or "jobspy"
        salary_disclosed = bool(getattr(job, "salary_disclosed", False))
        cards.append({
            "id": job.id, "job_id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "salary_lpa": list(job.salary_lpa or [0, 0]),
            "salary_disclosed": salary_disclosed,
            "tier": job_tier(job), "eligible": eligible, "eligibility_reasons": reasons,
            "score": m.score, "band": m.band, "semantic_sim": b.get("semantic_sim", 0),
            "proficiency_bonus": b.get("proficiency_bonus", 0),
            "job_url": job_url, "source": source, "scraper_engine": scraper_engine, "job": job,
            "is_fresher": getattr(job, "is_fresher", True),
            "experience_years": getattr(job, "experience_years", 0) or 0,
            "remote": bool(getattr(job, "remote", False)),
            "created_at": m.created_at.isoformat() if getattr(m, "created_at", None) else "",
            "matched_required": [(e.get("name", e.get("skill", "")),
                                  e.get("evidence") or evidence.get(e.get("skill", ""), ""))
                                 for e in b.get("matched_required", [])],
            "missing_required": [e.get("name", e.get("skill", "")) for e in b.get("missing_required", [])],
            "matched_preferred": [e.get("name", e.get("skill", "")) for e in b.get("matched_preferred", [])],
            "missing_preferred": [e.get("name", e.get("skill", "")) for e in b.get("missing_preferred", [])],
        })
    return _tpl(request, "student/matches.html", session, sid=sid, active="matches",
                cards=cards, has_matches=bool(cards), error="")


@router.get("/students/{sid}/plan")
def plan_page(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    plan = session.exec(select(UpskillPlan).where(UpskillPlan.student_id == sid)
                        .order_by(UpskillPlan.created_at.desc())).first()
    if plan is None:
        return _tpl(request, "student/plan.html", session, sid=sid, active="plan", plan=None,
                    plan_items=[], done_hours=0, total_hours=0, error="")
    items = session.exec(select(PlanItem).where(PlanItem.plan_id == plan.id)
                         .order_by(PlanItem.order)).all()
    courses = {c.id: c for c in session.exec(select(Course)).all()}
    plan_items = [{
        "id": it.id, "order": it.order, "skill_id": it.skill_id, "hours": it.hours, "status": it.status,
        "week": it.week,
        "course_title": courses[it.course_id].title if it.course_id in courses else "",
        "course_url": courses[it.course_id].url if it.course_id in courses else "",
        "course_provider": courses[it.course_id].provider if it.course_id in courses else "",
        "course_cost": courses[it.course_id].cost if it.course_id in courses else "",
    } for it in items]
    done_hours = sum(i["hours"] for i in plan_items if i["status"] == "done")
    total_hours = plan.total_hours or 0
    return _tpl(request, "student/plan.html", session, sid=sid, active="plan", plan=plan,
                plan_items=plan_items, done_hours=done_hours, total_hours=total_hours, error="")


@router.get("/students/{sid}/progress")
def progress_page(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    events = session.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)
                          .order_by(ProgressEvent.at.asc())).all()
    best = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                        .order_by(JobMatch.score.desc())).first()
    return _tpl(request, "student/progress.html", session, sid=sid, active="progress",
                series=[{"at": e.at.isoformat(), "readiness": e.readiness} for e in events],
                events=[{"id": e.id, "kind": e.kind, "readiness": e.readiness, "at": e.at.isoformat()} for e in events],
                best_match=({"job_id": best.job_id, "score": best.score, "band": best.band} if best else None),
                error="")


@router.get("/students/{sid}/interviews")
def interviews_page(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    rows = session.exec(select(Interview).where(Interview.student_id == sid)
                        .order_by(Interview.started_at.desc())).all()
    items = []
    for iv in rows:
        job = session.get(Job, iv.job_id)
        items.append({"id": iv.id, "job_title": job.title if job else iv.job_id,
                      "finished": iv.finished, "score": iv.score,
                      "started_at": iv.started_at.isoformat()})
    top = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                       .order_by(JobMatch.score.desc())).first()
    return _tpl(request, "student/interviews.html", session, sid=sid, active="interviews",
                items=items, top_job_id=(top.job_id if top else None), error="")
