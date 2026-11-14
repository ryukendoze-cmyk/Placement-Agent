from html import escape as _html_escape

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.api.routes.students import (
    _skill_payload,
    consume_resume_reupload_unlock,
    create_student_core,
    create_student_core_resume,
    enforce_resume_reupload_gate,
    is_job_visible_to_student,
)
from app.config import settings as _settings
from app.matching.engine import job_eligibility, job_tier
from app.models import (
    ApprovalRequest,
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
from app.security.rbac import (
    current_role,
    enforce_student_scope,
    get_current_user_optional,
    require_role,
)
from app.taxonomy import branches as _branches_mod
from app.taxonomy.branches import normalize_branch
from app.taxonomy.taxonomy import load_taxonomy

router = APIRouter()

_FALLBACK_BRANCH_CHOICES: list[tuple[str, str]] = [
    ("CSE", "CSE - Computer Science & Engineering"),
    ("AI&DS", "AI&DS - Artificial Intelligence & Data Science"),
    ("ECE", "ECE - Electronics & Communication Engineering"),
    ("IT", "IT - Information Technology"),
    ("EE", "EE - Electrical Engineering"),
    ("ME", "ME - Mechanical Engineering"),
    ("CE", "CE - Civil Engineering"),
]


def _branch_options() -> list[tuple[str, str]]:
    fn = getattr(_branches_mod, "branch_choices", None)
    if callable(fn):
        choices = fn()
        if choices:
            return list(choices)
    return list(_FALLBACK_BRANCH_CHOICES)


def _students_menu(session) -> list:
    rows = session.exec(select(Student)).all()
    rows.sort(key=lambda s: (str(s.name or "").strip().lower(), str(s.id or "")))
    return rows


def _tpl(request: Request, name: str, session: Session, sid: str = "", active: str = "", **ctx):
    all_students = _students_menu(session)
    role = current_role(request)
    user = get_current_user_optional(request)
    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    allow_demo = bool(getattr(_settings, "demo_role_switch", True))
    department = (
        (None if is_strict else request.headers.get("x-demo-department"))
        or (None if is_strict else request.cookies.get("ps06_department"))
        or (user.department if user and user.department else None)
        or ctx.get("department")
        or "CSE"
    )
    if role == "student":
        own_sid = (
            user.student_id
            if user and user.student_id and user.student_id != "demo-student"
            else (sid or "")
        )
        students = [s for s in all_students if own_sid and s.id == own_sid]
        cohort_students = list(students)
    elif role == "faculty_mentor":
        dept_filter = (
            (user.department if user and user.department else None)
            or (None if is_strict else request.headers.get("x-demo-department"))
            or ""
        ).strip()
        if dept_filter:
            dept_norm = (normalize_branch(dept_filter) or dept_filter).strip().lower()
            students = [
                s for s in all_students
                if (normalize_branch(s.branch or "") or (s.branch or "")).strip().lower() == dept_norm
            ]
        else:
            students = list(all_students)
        cohort_students = list(students)
    else:
        students = list(all_students)
        cohort_students = list(all_students)
    return request.app.state.templates.TemplateResponse(
        request, name,
        {**ctx, "students": students, "cohort_students": cohort_students, "sid": sid, "active": active, "role": role,
         "user": user, "current_user": user, "user_name": (user.name if user else ""),
         "department": department,
         "user_department": (user.department if user else None),
         "demo_role_switch": allow_demo,
         "auth_mode": "strict" if is_strict else "demo",
         "taxonomy_count": len(load_taxonomy().all()),
         "branch_options": _branch_options(),
         "normalize_branch": normalize_branch,
         "error": ctx.get("error", "")})


@router.get("/")
def index(request: Request, session: Session = Depends(get_session)):
    role = current_role(request)
    user = get_current_user_optional(request)
    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    if role == "student":
        own_sid = (
            user.student_id
            if user and user.student_id and user.student_id != "demo-student"
            else ""
        )
        if is_strict and own_sid and session.get(Student, own_sid) is not None:
            return RedirectResponse(url=f"/students/{own_sid}", status_code=303)
        return _tpl(request, "index.html", session, sid=own_sid, active="home", error="")
    return _tpl(request, "index.html", session, sid="", active="home", error="")


@router.post("/upload")
def upload(file: UploadFile = File(...), name: str = Form(...), branch: str = Form(""),
           year: int = Form(3), gpa: float = Form(0.0),
           session: Session = Depends(get_session), ctx=Depends(get_ctx),
           _role: str = Depends(require_role("tpo_admin"))):
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
    has_resume = bool((student.resume_text or "").strip())
    reupload_unlocked = bool(getattr(student, "resume_reupload_unlocked", False))
    pending_reupload_req = session.exec(
        select(ApprovalRequest)
        .where(
            ApprovalRequest.kind == "resume_reupload_request",
            ApprovalRequest.target_id == sid,
            ApprovalRequest.status == "pending_review",
        )
        .order_by(ApprovalRequest.created_at.desc())
    ).first()
    return _tpl(
        request,
        "student/profile.html",
        session,
        sid=sid,
        active="profile",
        student=student,
        skills_by_category=_group_skills(session, ctx, sid),
        has_resume=has_resume,
        reupload_unlocked=reupload_unlocked,
        pending_reupload_req=pending_reupload_req,
        error="",
    )


@router.post("/students/{sid}/resume")
def reupload(sid: str, request: Request, file: UploadFile = File(...),
             session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    try:
        consume_unlock = enforce_resume_reupload_gate(request, student)
    except HTTPException as he:
        if he.status_code == 403:
            return HTMLResponse(
                content=(
                    '<div role="alert" class="rounded-xl border border-[#F5D0C8] bg-[#FDF0ED] '
                    'px-4 py-3 text-xs font-semibold text-[#9E2A20]">'
                    f"{_html_escape(str(he.detail))}</div>"
                ),
                status_code=403,
            )
        raise
    create_student_core_resume(session, ctx, sid, file)
    if consume_unlock:
        consume_resume_reupload_unlock(session, session.get(Student, sid), sid, current_role(request))
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
    _role: str = Depends(require_role("tpo_admin")),
):
    from app.api.routes.students import delete_student_core
    enforce_student_scope(request, sid)
    delete_student_core(session, ctx, sid, actor_role=current_role(request))
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
    all_jobs = [
        j for j in session.exec(select(Job)).all()
        if is_job_visible_to_student(j, sid)
    ]
    visible_job_ids = {j.id for j in all_jobs}
    stale_rows = [m for m in rows if m.job_id not in visible_job_ids]
    if stale_rows:
        for sm in stale_rows:
            session.delete(sm)
        session.commit()
        rows = [m for m in rows if m.job_id in visible_job_ids]

    app_rows = session.exec(
        select(JobApplication)
        .where(JobApplication.student_id == sid)
        .order_by(JobApplication.created_at.desc())
    ).all()
    favorite_app_job_ids = {
        a.job_id for a in app_rows if getattr(a, "is_favorite", False)
    }

    # Incremental auto-rank: ensure newly added visible jobs are scored for students who already ran matching
    existing_job_ids = {m.job_id for m in rows}
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
            verification_statuses = {
                r.skill_id: (getattr(r, "verification_status", None) or "claimed") for r in ss_rows
            }
            new_results = rank_jobs(
                missing_jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
                vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                top=None, levels=levels, student=student,
                verification_statuses=verification_statuses, session=session,
            )
            for job, js in new_results:
                session.add(JobMatch(
                    id=f"m-{uuid4().hex[:8]}",
                    student_id=sid,
                    job_id=job.id,
                    score=js.score,
                    band=js.band,
                    breakdown=asdict(js),
                    is_favorite=(job.id in favorite_app_job_ids),
                ))
            session.commit()
            rows = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                                .order_by(JobMatch.score.desc())).all()
        except Exception:
            pass
    cards = []
    for m in rows:
        if getattr(m, "band", None) == "dismissed" or getattr(m, "score", 0) < 0:
            continue
        if m.breakdown and m.breakdown.get("dismissed"):
            continue
        job = session.get(Job, m.job_id)
        if job is None or not is_job_visible_to_student(job, sid):
            continue
        b = m.breakdown or {}
        eligible, reasons = job_eligibility(student, job)
        job_url = getattr(job, "job_url", "") or ""
        source = getattr(job, "source", "local") or "local"
        scraper_engine = getattr(job, "scraper_engine", "") or ""
        salary_disclosed = bool(getattr(job, "salary_disclosed", False))
        cards.append({
            "id": job.id, "job_id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "salary_lpa": list(job.salary_lpa or [0, 0]),
            "salary_disclosed": salary_disclosed,
            "tier": job_tier(job), "eligible": eligible, "eligibility_reasons": reasons,
            "score": m.score, "band": m.band, "semantic_sim": b.get("semantic_sim", 0),
            "proficiency_bonus": b.get("proficiency_bonus", 0),
            "is_favorite": bool(getattr(m, "is_favorite", False)),
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
            "unverified_skills": list(b.get("unverified_skills") or []),
            "needs_reverification_skills": list(b.get("needs_reverification_skills") or []),
            "verification_statuses": dict(b.get("verification_statuses") or {}),
        })
    favorite_matches = [c for c in cards if c.get("is_favorite")]
    student_applications = []
    for app_row in app_rows:
        app_job = session.get(Job, app_row.job_id)
        student_applications.append({
            "id": app_row.id,
            "job_id": app_row.job_id,
            "status": app_row.status,
            "tier": app_row.tier,
            "match_score": app_row.match_score,
            "is_favorite": bool(getattr(app_row, "is_favorite", False)),
            "job_title": app_job.title if app_job else app_row.job_id,
            "job_company": app_job.company if app_job else "",
        })
    return _tpl(
        request, "student/matches.html", session, sid=sid, active="matches",
        student=student, cards=cards, has_matches=bool(cards),
        favorite_matches=favorite_matches, student_applications=student_applications, error="",
    )


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
    from app.analytics.student_progress import build_student_progress_report

    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    events = session.exec(
        select(ProgressEvent).where(ProgressEvent.student_id == sid).order_by(ProgressEvent.at.asc())
    ).all()
    matches = session.exec(
        select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
    ).all()
    best = None
    for m in matches:
        if m.band == "dismissed" or (m.score or 0) < 0:
            continue
        job = session.get(Job, m.job_id)
        if job is not None and is_job_visible_to_student(job, sid):
            best = m
            break
    ctx = getattr(request.app.state, "ctx", None)
    taxonomy = getattr(ctx, "taxonomy", None) if ctx is not None else None
    report = build_student_progress_report(session, sid, taxonomy=taxonomy)
    events_list = [
        {"id": e.id, "kind": e.kind, "readiness": e.readiness, "at": e.at.isoformat()}
        for e in events
    ]
    best_match_dict = (
        {"job_id": best.job_id, "score": best.score, "band": best.band} if best else None
    )
    return _tpl(
        request,
        "student/progress.html",
        session,
        sid=sid,
        active="progress",
        student=student,
        report=report,
        series=report["milestone_series"],
        events=events_list,
        best_match=best_match_dict,
        error="",
    )


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
