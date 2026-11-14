import logging
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import delete
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.api.routes.jobs import is_on_campus_job
from app.config import settings as _settings
from app.embeddings.store import cosine
from app.matching.engine import job_eligibility, job_tier
from app.matching.rank import rank_jobs
from app.matching.simulator import simulate_counterfactual
from app.models import (
    AgentRun,
    ApprovalRequest,
    AuditLog,
    CampusDrive,
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
from app.plan.planner import _load_courses, pick_course_for_skill
from app.resume.parser import parse_resume_bytes
from app.resume.xray import improve_weak_bullets, xray_resume
from app.scraper import ingest_scraped_jobs
from app.scraper.ai_query import LEVEL_RANK as _LEVEL_RANK
from app.scraper.ai_query import suggest_job_queries_for_student
from app.security.guardrails import sanitize_untrusted_text
from app.security.pii import scrub_for_llm
from app.security.rbac import (
    current_role,
    enforce_student_scope,
    get_current_user_optional,
    require_role,
)
from app.skills.extract import extract_resume_with_ai
from app.taxonomy.branches import normalize_branch

logger = logging.getLogger(__name__)
router = APIRouter()

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".text", ".md", ".png", ".jpg", ".jpeg"}
MAX_RESUME_BYTES = 5 * 1024 * 1024   # 5 MB cap on resume uploads


def _read_capped(file: UploadFile, cap: int = MAX_RESUME_BYTES) -> bytes:
    data = file.file.read(cap + 1)
    if len(data) > cap:
        raise HTTPException(413, f"resume exceeds {cap} byte limit")
    return data


def _skill_payload(ss: StudentSkill, ctx) -> dict:
    skill = ctx.taxonomy.get(ss.skill_id)
    return {
        "id": ss.skill_id,
        "name": skill.name,
        "category": skill.category,
        "mentions": ss.mentions,
        "level": ss.level,
        "evidence": ss.evidence,
        "verification_status": getattr(ss, "verification_status", None) or "claimed",
        "confidence": float(getattr(ss, "confidence", 0.5) if getattr(ss, "confidence", None) is not None else 0.5),
        "last_verified_score": getattr(ss, "last_verified_score", None),
        "source": getattr(ss, "source", None) or "resume",
    }


def is_job_visible_to_student(job: Job | None, student_id: str) -> bool:
    """Return True when *job* is owned by *student_id* or is cohort-wide on-campus."""
    if job is None:
        return False
    owner = getattr(job, "owner_student_id", None)
    is_live_job = bool(getattr(job, "is_live", False)) or not is_on_campus_job(job)
    if is_live_job:
        return owner == student_id
    return owner in (None, student_id)


def _match_payload(job, js, student, *, is_favorite: bool = False) -> dict:
    eligible, reasons = job_eligibility(student, job)
    job_url = getattr(job, "job_url", "") or ""
    source = getattr(job, "source", "local") or "local"
    return {
        "id": job.id,
        "job_id": job.id,
        "title": job.title,
        "company": job.company,
        "location": job.location,
        "salary_lpa": list(job.salary_lpa or [0, 0]),
        "tier": job_tier(job),
        "score": js.score,
        "band": js.band,
        "semantic_sim": js.semantic_sim,
        "proficiency_bonus": js.proficiency_bonus,
        "matched_required": js.matched_required,
        "missing_required": js.missing_required,
        "matched_preferred": js.matched_preferred,
        "missing_preferred": js.missing_preferred,
        "proficiency_gaps": js.proficiency_gaps,
        "unverified_skills": list(getattr(js, "unverified_skills", []) or []),
        "needs_reverification_skills": list(getattr(js, "needs_reverification_skills", []) or []),
        "verification_statuses": dict(getattr(js, "verification_statuses", {}) or {}),
        "eligible": eligible,
        "eligibility_reasons": reasons,
        "job_url": job_url,
        "source": source,
        "is_fresher": getattr(job, "is_fresher", True),
        "is_favorite": bool(is_favorite),
        "job": {
            "id": job.id,
            "title": job.title,
            "company": job.company,
            "location": job.location,
            "salary_lpa": list(job.salary_lpa or [0, 0]),
            "job_url": job_url,
            "source": source,
            "is_fresher": getattr(job, "is_fresher", True),
        },
    }


def _rank_and_persist(session, ctx, sid, student, top: int = 10, include_scraped: bool = False) -> list[dict]:
    """Rank visible jobs for a student, preserving dismissed & favorite flags."""
    sv = ctx.vector_store.get("student", sid) if ctx.vector_store else None
    if sv is None and getattr(student, "resume_text", None):
        sv = ctx.embedder.embed(student.resume_text)
        if ctx.vector_store:
            ctx.vector_store.upsert("student", sid, sv)
    if sv is None:
        sv = [0.0] * getattr(ctx.embedder, "dim", 384)
    ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    verification_statuses = {
        r.skill_id: (getattr(r, "verification_status", None) or "claimed") for r in ss_rows
    }

    existing_matches = session.exec(select(JobMatch).where(JobMatch.student_id == sid)).all()
    dismissed_job_ids = {
        m.job_id for m in existing_matches
        if getattr(m, "band", None) == "dismissed"
        or getattr(m, "score", 0) < 0
        or (isinstance(m.breakdown, dict) and m.breakdown.get("dismissed"))
    }
    favorite_job_ids = {
        m.job_id for m in existing_matches if getattr(m, "is_favorite", False)
    } | {
        a.job_id
        for a in session.exec(select(JobApplication).where(JobApplication.student_id == sid)).all()
        if getattr(a, "is_favorite", False)
    }

    all_visible_jobs = [
        j for j in session.exec(select(Job)).all()
        if is_job_visible_to_student(j, sid)
    ]
    visible_job_ids = {j.id for j in all_visible_jobs}
    active_jobs = [j for j in all_visible_jobs if j.id not in dismissed_job_ids]

    results = rank_jobs(
        active_jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
        vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
        top=top, levels=levels, student=student,
        verification_statuses=verification_statuses, session=session,
    )

    owned_or_scraped = [
        j for j in active_jobs
        if getattr(j, "owner_student_id", None) == sid
        or (include_scraped and (bool(getattr(j, "is_live", False)) or not is_on_campus_job(j)))
    ]
    scored_job_ids = {getattr(j, "id", None) for j, _ in results}
    missing_live = [j for j in owned_or_scraped if getattr(j, "id", None) not in scored_job_ids]
    if missing_live:
        ext_results = rank_jobs(
            missing_live, sv, student_skills, ctx.embedder, ctx.taxonomy,
            vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
            top=None, levels=levels, student=student,
            verification_statuses=verification_statuses, session=session,
        )
        results.extend(ext_results)

    session.execute(delete(JobMatch).where(JobMatch.student_id == sid))
    for job, js in results:
        session.add(
            JobMatch(
                id=f"m-{uuid4().hex[:8]}",
                student_id=sid,
                job_id=job.id,
                score=js.score,
                band=js.band,
                breakdown=asdict(js),
                is_favorite=(job.id in favorite_job_ids),
            )
        )
    for jid in dismissed_job_ids & visible_job_ids:
        session.add(
            JobMatch(
                id=f"m-{uuid4().hex[:8]}",
                student_id=sid,
                job_id=jid,
                score=-1,
                band="dismissed",
                breakdown={"dismissed": True},
                is_favorite=(jid in favorite_job_ids),
            )
        )
    session.commit()
    return [
        _match_payload(job, js, student, is_favorite=(job.id in favorite_job_ids))
        for job, js in results
    ]


def create_student_core(session, ctx, file, name, branch, year, gpa):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"unsupported resume format: {suffix or '(none)'}")
    try:
        text = parse_resume_bytes(_read_capped(file), suffix)
    except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:
        logger.exception("resume parse failed for %s", file.filename)
        raise HTTPException(400, f"could not parse resume: {e}") from e
    sid = f"stu-{uuid4().hex[:8]}"
    text, pii_count = scrub_for_llm(text)   # Phase 4: redact PII before store/LLM
    text, inj_flags = sanitize_untrusted_text(text, source="resume")  # M15 injection gate
    parsed = extract_resume_with_ai(text, ctx.taxonomy, getattr(ctx, "llm", None))
    profile = parsed.get("profile") or {}
    # enrich blank/default fields from AI-extracted profile when present
    if not (name or "").strip():
        name = profile.get("name") or "Student"
    if not (branch or "").strip() and profile.get("branch"):
        branch = profile["branch"]
    branch = normalize_branch(branch)
    try:
        gpa_val = float(gpa or 0.0)
    except (TypeError, ValueError):
        gpa_val = 0.0
    if gpa_val == 0.0 and profile.get("gpa") is not None:
        try:
            gpa_val = float(profile["gpa"])
        except (TypeError, ValueError):
            gpa_val = 0.0
    session.add(Student(id=sid, name=name, branch=branch, year=year, gpa=gpa_val,
                        pii_redacted_count=pii_count, resume_text=text))
    for skill_id, record in parsed["skills"].items():
        session.add(StudentSkill(student_id=sid, skill_id=skill_id,
                                 mentions=record["mentions"], level=record["level"],
                                 evidence=list(record["evidence"])))
    from app.security.audit import log_audit
    log_audit(session, action="student.resume_uploaded", entity_type="student",
              entity_id=sid, actor_role="student",
              detail={"branch": branch, "skills": len(parsed["skills"]),
                      "pii_redacted": pii_count})
    if inj_flags:
        log_audit(session, action="security.prompt_injection_blocked",
                  entity_type="resume", entity_id=sid, actor_role="student",
                  detail={"flags": inj_flags, "source": "resume"})
    session.commit()
    ctx.vector_store.upsert("student", sid, ctx.embedder.embed(text))
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    return sid, [_skill_payload(r, ctx) for r in rows]


def delete_student_core(session: Session, ctx, sid: str, actor_role: str = "tpo_admin") -> bool:
    student = session.get(Student, sid)
    if student is None:
        return False
    student_name = student.name
    student_branch = student.branch
    # M19: applications must go first (they FK both student and job).
    session.execute(delete(JobApplication).where(JobApplication.student_id == sid))
    session.execute(delete(JobMatch).where(JobMatch.student_id == sid))
    session.execute(delete(StudentSkill).where(StudentSkill.student_id == sid))
    session.execute(delete(ProgressEvent).where(ProgressEvent.student_id == sid))
    session.execute(delete(Interview).where(Interview.student_id == sid))
    plans = session.exec(select(UpskillPlan).where(UpskillPlan.student_id == sid)).all()
    for p in plans:
        session.execute(delete(PlanItem).where(PlanItem.plan_id == p.id))
    session.execute(delete(UpskillPlan).where(UpskillPlan.student_id == sid))
    for run in session.exec(select(AgentRun).where(AgentRun.student_id == sid)).all():
        run.student_id = None
        session.add(run)
    session.delete(student)
    from app.security.audit import log_audit
    log_audit(
        session,
        action="student.deleted",
        entity_type="student",
        entity_id=sid,
        actor_role=actor_role or "tpo_admin",
        detail={"deleted_student_id": sid, "name": student_name, "branch": student_branch},
    )
    session.commit()
    return True


_TPO_ADMIN_ROLE = require_role("tpo_admin")


@router.delete("/api/students/{sid}")
def delete_student(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    user=Depends(_TPO_ADMIN_ROLE),
):
    enforce_student_scope(request, sid)
    if not delete_student_core(session, ctx, sid, actor_role=current_role(request)):
        raise HTTPException(404, "student not found")
    return {"status": "success", "deleted_student_id": sid}


def create_student_core_resume(session, ctx, sid, file):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"unsupported resume format: {suffix or '(none)'}")
    try:
        text = parse_resume_bytes(_read_capped(file), suffix)
    except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:
        logger.exception("resume reparse failed for %s", file.filename)
        raise HTTPException(400, f"could not parse resume: {e}") from e
    text, pii_count = scrub_for_llm(text)   # Phase 4: redact PII on re-upload too
    text, inj_flags = sanitize_untrusted_text(text, source="resume")  # M15 injection gate
    parsed = extract_resume_with_ai(text, ctx.taxonomy, getattr(ctx, "llm", None))
    profile = parsed.get("profile") or {}
    had_matches = session.exec(select(JobMatch).where(JobMatch.student_id == sid).limit(1)).first() is not None
    session.execute(delete(StudentSkill).where(StudentSkill.student_id == sid))
    for skill_id, record in parsed["skills"].items():
        session.add(StudentSkill(student_id=sid, skill_id=skill_id,
                                 mentions=record["mentions"], level=record["level"],
                                 evidence=list(record["evidence"])))
    student = session.get(Student, sid)
    if student is not None:
        student.resume_text = text
        student.pii_redacted_count = pii_count
        if not (student.branch or "").strip() and profile.get("branch"):
            student.branch = normalize_branch(profile["branch"])
        elif student.branch:
            student.branch = normalize_branch(student.branch)
        try:
            gpa_val = float(student.gpa or 0.0)
        except (TypeError, ValueError):
            gpa_val = 0.0
        if gpa_val == 0.0 and profile.get("gpa") is not None:
            try:
                student.gpa = float(profile["gpa"])
            except (TypeError, ValueError):
                pass
        session.add(student)
    from app.security.audit import log_audit
    log_audit(session, action="student.resume_uploaded", entity_type="student",
              entity_id=sid, actor_role="student",
              detail={"branch": (student.branch if student is not None else ""),
                      "skills": len(parsed["skills"]), "pii_redacted": pii_count,
                      "reupload": True})
    if inj_flags:
        log_audit(session, action="security.prompt_injection_blocked",
                  entity_type="resume", entity_id=sid, actor_role="student",
                  detail={"flags": inj_flags, "source": "resume"})
    session.commit()
    ctx.vector_store.upsert("student", sid, ctx.embedder.embed(text))
    if had_matches and student is not None:
        # refresh persisted JobMatch rows so the dashboard never shows stale cards
        _rank_and_persist(session, ctx, sid, student)
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    return sid, [_skill_payload(r, ctx) for r in rows]


def _parse_and_scrub(file: UploadFile) -> tuple[str, int]:
    """Shared parse + PII-scrub + injection sanitize for /api/resumes/extract."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"unsupported resume format: {suffix or '(none)'}")
    try:
        text = parse_resume_bytes(_read_capped(file), suffix)
    except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:
        logger.exception("resume parse failed for %s", file.filename)
        raise HTTPException(400, f"could not parse resume: {e}") from e
    text, _inj_flags = sanitize_untrusted_text(text, source="resume")  # M15 gate
    return scrub_for_llm(text)


@router.post("/api/resumes/extract")
def extract_resume(
    file: UploadFile = File(...),
    ctx=Depends(get_ctx),
    _role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
):
    """Stateless AI resume parse preview (drives the frontend auto-fill)."""
    text, pii_count = _parse_and_scrub(file)
    parsed = extract_resume_with_ai(text, ctx.taxonomy, getattr(ctx, "llm", None))
    skills = []
    for skill_id, record in parsed["skills"].items():
        skill = ctx.taxonomy.get(skill_id)
        skills.append({
            "id": skill.id,
            "name": skill.name,
            "category": skill.category,
            "level": record["level"],
            "mentions": record["mentions"],
            "evidence": record["evidence"],
        })
    return {
        "profile": parsed["profile"],
        "skills": skills,
        "skill_count": len(skills),
        "pii_redacted_count": pii_count,
        "ai_parsed": parsed["ai_parsed"],
    }


def enforce_resume_reupload_gate(request: Request, student: Student) -> bool:
    """Enforce single-use TPO HITL gate for student resume re-uploads.

    Returns True when a non-TPO caller is consuming an active unlock.
    """
    role = current_role(request)
    has_resume = bool((student.resume_text or "").strip())
    unlocked = bool(getattr(student, "resume_reupload_unlocked", False))
    if role != "tpo_admin" and has_resume and not unlocked:
        raise HTTPException(
            403,
            "Resume re-upload requires TPO Admin permission. Request approval from your Profile page.",
        )
    return role != "tpo_admin" and unlocked


def consume_resume_reupload_unlock(
    session: Session,
    student: Student | None,
    sid: str,
    actor_role: str,
) -> None:
    """Consume the single-use resume re-upload unlock and record an AuditLog row."""
    if student is None:
        return
    student.resume_reupload_unlocked = False
    session.add(student)
    from app.security.audit import log_audit
    log_audit(
        session,
        action="student.resume_reupload_consumed",
        entity_type="student",
        entity_id=sid,
        actor_role=actor_role,
        detail={"student_id": sid, "unlock_consumed": True},
        commit=True,
    )


@router.post("/api/students/{sid}/resume")
def resume_reupload_json(
    sid: str,
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    """JSON counterpart to the HTMX POST /students/{sid}/resume page route."""
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    consume_unlock = enforce_resume_reupload_gate(request, student)
    _sid, skills = create_student_core_resume(session, ctx, sid, file)
    if consume_unlock:
        consume_resume_reupload_unlock(session, session.get(Student, sid), sid, current_role(request))
    return {"id": sid, "skills": skills, "skill_count": len(skills)}


@router.post("/api/students/{sid}/resume-reupload-request", status_code=201)
def request_resume_reupload(
    sid: str,
    request: Request,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("student", "tpo_admin")),
) -> dict[str, Any]:
    """Queue a TPO Admin approval request for single-use student resume re-upload."""
    if role not in ("student", "tpo_admin"):
        raise HTTPException(403, "Only student or TPO admin may request resume re-upload permission")
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")

    existing = session.exec(
        select(ApprovalRequest).where(
            ApprovalRequest.kind == "resume_reupload_request",
            ApprovalRequest.target_id == sid,
            ApprovalRequest.status == "pending_review",
        )
    ).first()
    if existing is not None:
        raise HTTPException(409, "A resume re-upload approval request is already pending review.")

    reason = str((payload or {}).get("reason") or "").strip()
    aid = f"apr-{uuid4().hex[:8]}"
    ap = ApprovalRequest(
        id=aid,
        kind="resume_reupload_request",
        target_id=sid,
        title=f"Resume Re-upload Permission: {student.name}",
        summary=reason or "Student requested permission to re-upload and re-extract resume",
        payload={"student_id": sid, "reason": reason},
        status="pending_review",
        requested_by=role,
        created_at=datetime.now(timezone.utc),
    )
    session.add(ap)
    from app.security.audit import log_audit
    log_audit(
        session,
        action="student.resume_reupload_requested",
        entity_type="student",
        entity_id=sid,
        actor_role=role,
        detail={"approval_id": aid, "student_id": sid, "reason": reason},
    )
    session.commit()
    return {
        "id": ap.id,
        "approval_id": ap.id,
        "kind": ap.kind,
        "target_id": ap.target_id,
        "status": ap.status,
        "title": ap.title,
        "summary": ap.summary,
    }


@router.get("/api/students")
def list_students(
    request: Request,
    q: str | None = None,
    branch: str | None = None,
    min_gpa: float | None = None,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
) -> list[dict[str, Any]]:
    """List students alphabetically by name, RBAC-scoped and filterable by query/branch/GPA."""
    user = get_current_user_optional(request)
    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    rows = session.exec(select(Student)).all()

    if role == "student":
        own_sid = (
            user.student_id
            if user and user.student_id and user.student_id != "demo-student"
            else None
        )
        rows = [s for s in rows if own_sid and s.id == own_sid]
    elif role == "faculty_mentor":
        dept_raw = (
            (user.department if user and user.department else None)
            or (None if is_strict else request.headers.get("x-demo-department"))
            or (None if is_strict else request.cookies.get("ps06_department"))
            or ""
        ).strip()
        if dept_raw:
            dept_norm = (normalize_branch(dept_raw) or dept_raw).strip().lower()
            rows = [
                s for s in rows
                if (normalize_branch(s.branch or "") or (s.branch or "")).strip().lower() == dept_norm
            ]

    q_norm = (q or "").strip().lower()
    b_norm = (normalize_branch(branch) or (branch or "")).strip().lower() if branch and branch.strip() else ""
    filtered: list[Student] = []
    for s in rows:
        canon_branch = normalize_branch(s.branch or "") or (s.branch or "")
        if q_norm:
            hay = f"{s.name or ''} {s.id or ''} {canon_branch}".lower()
            if q_norm not in hay:
                continue
        if b_norm and canon_branch.strip().lower() != b_norm:
            continue
        if min_gpa is not None and float(s.gpa or 0.0) < float(min_gpa):
            continue
        filtered.append(s)

    filtered.sort(key=lambda s: (str(s.name or "").strip().lower(), str(s.id or "")))
    return [
        {
            "id": s.id,
            "name": s.name,
            "branch": normalize_branch(s.branch or "") or s.branch,
            "year": s.year,
            "gpa": s.gpa,
            "has_resume": bool((s.resume_text or "").strip()),
            "resume_reupload_unlocked": bool(getattr(s, "resume_reupload_unlocked", False)),
        }
        for s in filtered
    ]


@router.post("/api/students", status_code=201)
async def create_student(
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    _role: str = Depends(require_role("tpo_admin")),
):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        name = body.get("name") or "Candidate"
        branch = body.get("branch") or ""
        year = int(body.get("year") or 3)
        gpa = float(body.get("gpa") or 0.0)
        norm_branch = normalize_branch(branch) or branch
        sid = f"stu-{uuid4().hex[:8]}"
        student = Student(id=sid, name=name, branch=norm_branch, year=year, gpa=gpa)
        session.add(student)
        session.commit()
        if hasattr(ctx, "vector_store") and hasattr(ctx, "embedder"):
            ctx.vector_store.upsert("student", sid, ctx.embedder.embed(f"{name} {norm_branch}"))
        return {"id": sid, "name": name, "skills": [], "skill_count": 0}

    form = await request.form()
    file = form.get("file")
    if file is None or not hasattr(file, "file"):
        raise HTTPException(422, "file is required for multipart uploads")
    name = str(form.get("name") or "")
    branch = str(form.get("branch") or "")
    try:
        year = int(form.get("year") or 3)
    except (TypeError, ValueError):
        year = 3
    try:
        gpa = float(form.get("gpa") or 0.0)
    except (TypeError, ValueError):
        gpa = 0.0
    sid, skills = create_student_core(session, ctx, file, name, branch, year, gpa)
    student = session.get(Student, sid)
    persisted_name = student.name if student is not None else name
    return {"id": sid, "name": persisted_name, "skills": skills, "skill_count": len(skills)}


@router.get("/api/students/{sid}")
def get_student(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    return {
        "id": student.id,
        "name": student.name,
        "branch": student.branch,
        "year": student.year,
        "gpa": student.gpa,
        "pii_redacted_count": student.pii_redacted_count,
        "skills": [_skill_payload(r, ctx) for r in rows],
        "skill_count": len(rows),
    }


@router.get("/api/students/{sid}/skills")
def get_student_skills(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    return [_skill_payload(r, ctx) for r in rows]


@router.get("/api/students/{sid}/matches")
def get_matches(
    sid: str,
    request: Request,
    include_scraped: bool = False,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if ctx.vector_store.get("student", sid) is None:
        text = student.resume_text or f"{student.name} {student.branch}"
        if text.strip() and hasattr(ctx, "embedder"):
            ctx.vector_store.upsert("student", sid, ctx.embedder.embed(text))
        else:
            raise HTTPException(404, "student vector not found; re-upload the resume")
    return _rank_and_persist(session, ctx, sid, student, top=10, include_scraped=include_scraped)


@router.get("/api/students/{sid}/resume/xray")
def resume_xray(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if not (student.resume_text or "").strip():
        raise HTTPException(404, "no resume stored for this student")
    return xray_resume(student.resume_text, ctx.taxonomy)


@router.post("/api/students/{sid}/resume/improve-bullets")
def improve_resume_bullets(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if ctx.llm is None:
        raise HTTPException(503, "LLM not configured")
    if not (student.resume_text or "").strip():
        raise HTTPException(404, "no resume stored for this student")
    present = [ctx.taxonomy.get(r.skill_id).name
               for r in session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()]
    improvements = improve_weak_bullets(student.resume_text, present, ctx.llm, taxonomy=ctx.taxonomy)
    return {"ats": xray_resume(student.resume_text, ctx.taxonomy), "improvements": improvements}


@router.post("/api/students/{sid}/matches/{job_id}/simulate")
def simulate_match(sid: str, job_id: str, request: Request,
                   payload: dict | None = None,
                   session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    sv = ctx.vector_store.get("student", sid)
    if sv is None:
        raise HTTPException(404, "student vector not found; re-upload the resume")
    ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    jv = ctx.job_vectors.get(job.id) or ctx.vector_store.get("job", job.id)
    if jv is None:
        jv = ctx.embedder.embed(job.description)
        ctx.vector_store.upsert("job", job.id, jv)
        ctx.job_vectors[job.id] = jv
    sim = cosine(sv, jv)
    skills = list(dict.fromkeys((payload or {}).get("skills", [])))
    courses = _load_courses(session)
    courses_for_skill = {s: pick_course_for_skill(s, courses) for s in skills}
    return simulate_counterfactual(student_skills, job, skills, sim, ctx.taxonomy,
                                   levels=levels, courses_for_skill=courses_for_skill)


def _cascade_delete_owned_live_jobs(session: Session, ctx: Any, sid: str) -> tuple[int, int]:
    """Delete live JobMatch rows for *sid* and cascade-delete Job rows owned by *sid*."""
    live_job_ids_for_student = {
        j.id
        for j in session.exec(select(Job)).all()
        if bool(getattr(j, "is_live", False)) or not is_on_campus_job(j)
    }
    owned_live_jobs = [
        j for j in session.exec(select(Job)).all()
        if getattr(j, "owner_student_id", None) == sid
        and (bool(getattr(j, "is_live", False)) or not is_on_campus_job(j))
    ]
    owned_ids = {j.id for j in owned_live_jobs}

    deleted_matches = 0
    for m in session.exec(select(JobMatch).where(JobMatch.student_id == sid)).all():
        if m.job_id in live_job_ids_for_student or m.job_id in owned_ids:
            session.delete(m)
            deleted_matches += 1

    for job in owned_live_jobs:
        jid = job.id
        if hasattr(ctx, "vector_store") and ctx.vector_store:
            try:
                ctx.vector_store.delete("job", jid)
            except Exception:
                pass
        if hasattr(ctx, "job_vectors") and jid in ctx.job_vectors:
            ctx.job_vectors.pop(jid, None)

        plan_ids = list(session.exec(select(UpskillPlan.id).where(UpskillPlan.job_id == jid)).all())
        if plan_ids:
            session.execute(delete(PlanItem).where(PlanItem.plan_id.in_(plan_ids)))
        session.execute(delete(UpskillPlan).where(UpskillPlan.job_id == jid))
        for cd in session.exec(select(CampusDrive).where(CampusDrive.job_id == jid)).all():
            cd.job_id = None
            session.add(cd)
        session.execute(delete(JobApplication).where(JobApplication.job_id == jid))
        session.execute(delete(JobMatch).where(JobMatch.job_id == jid))
        session.execute(delete(Interview).where(Interview.job_id == jid))
        session.delete(job)

    return len(owned_live_jobs), deleted_matches


@router.delete("/api/students/{sid}/matches/live")
@router.delete("/api/students/{sid}/live-jobs")
def student_dismiss_live_jobs_endpoint(
    sid: str,
    request: Request,
    keep_output: bool = False,
    scope: str = "auto",
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    role: str = Depends(require_role("student", "tpo_admin")),
) -> dict[str, Any]:
    """Purge student-owned live jobs + live matches (or global live jobs when TPO Admin)."""
    if role not in ("student", "tpo_admin"):
        raise HTTPException(403, f"Requires exact role 'student' or 'tpo_admin', got '{role}'")
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")

    if role == "tpo_admin" and (scope or "auto").strip().lower() in ("auto", "global"):
        from app.scraper.pipeline import purge_live_jobs

        count = purge_live_jobs(session, ctx=ctx, clear_output=not keep_output)
        try:
            session.add(
                AuditLog(
                    id=f"aud-{uuid4().hex[:8]}",
                    action="jobs.live.purged",
                    entity_type="job",
                    entity_id=f"student:{sid}",
                    actor_role=role,
                    detail={"purged_count": count, "student_id": sid, "keep_output": keep_output},
                )
            )
            session.commit()
        except Exception as exc:
            logger.warning("Failed to record AuditLog for student purge_live_jobs: %s", exc)

        try:
            _rank_and_persist(session, ctx, sid, student, include_scraped=False)
        except Exception as exc:
            logger.warning("Post-purge match refresh failed: %s", exc)

        return {
            "status": "success",
            "purged": count,
            "purged_count": count,
            "message": f"Successfully removed {count} live job(s) from database.",
        }

    # Student role (or student-scoped purge): delete owned live Jobs & live JobMatch rows.
    purged_owned, deleted_matches = _cascade_delete_owned_live_jobs(session, ctx, sid)
    session.add(
        AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="student.live_jobs.purged",
            entity_type="student",
            entity_id=sid,
            actor_role=role,
            detail={
                "purged_count": purged_owned,
                "deleted_matches": deleted_matches,
                "student_id": sid,
            },
        )
    )
    session.commit()
    return {
        "status": "success",
        "purged": purged_owned,
        "purged_count": purged_owned,
        "deleted_matches": deleted_matches,
        "message": f"Successfully removed {purged_owned} live job(s) from your workspace.",
    }


@router.post("/api/students/{sid}/matches/purge-live")
def student_purge_live_jobs_endpoint(
    sid: str,
    request: Request,
    keep_output: bool = False,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    role: str = Depends(require_role("tpo_admin")),
) -> dict[str, Any]:
    """Purge all live scraped jobs from the database and refresh student matches (TPO Admin only)."""
    enforce_student_scope(request, sid)
    from app.scraper.pipeline import purge_live_jobs

    count = purge_live_jobs(session, ctx=ctx, clear_output=not keep_output)

    try:
        session.add(
            AuditLog(
                id=f"aud-{uuid4().hex[:8]}",
                action="jobs.live.purged",
                entity_type="job",
                entity_id=f"student:{sid}",
                actor_role=role,
                detail={"purged_count": count, "student_id": sid, "keep_output": keep_output},
            )
        )
        session.commit()
    except Exception as exc:
        logger.warning("Failed to record AuditLog for student purge_live_jobs: %s", exc)

    st = session.get(Student, sid)
    if st:
        try:
            _rank_and_persist(session, ctx, sid, st, include_scraped=False)
        except Exception as exc:
            logger.warning("Post-purge match refresh failed: %s", exc)

    return {
        "status": "success",
        "purged": count,
        "purged_count": count,
        "message": f"Successfully removed {count} live job(s) from database.",
    }


@router.post("/api/students/{sid}/matches/{job_id}/favorite")
def toggle_favorite_match(
    sid: str,
    job_id: str,
    request: Request,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("student", "tpo_admin")),
) -> dict[str, Any]:
    """Toggle or set favorite status for a job match and sync linked applications."""
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if not is_job_visible_to_student(job, sid):
        raise HTTPException(403, "job is not visible in this student's workspace")

    m = session.exec(
        select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == job_id)
    ).first()
    if payload is not None and "is_favorite" in payload:
        new_val = bool(payload["is_favorite"])
    elif m is not None:
        new_val = not bool(getattr(m, "is_favorite", False))
    else:
        new_val = True

    if m is not None:
        m.is_favorite = new_val
        session.add(m)
    else:
        session.add(
            JobMatch(
                id=f"m-{uuid4().hex[:8]}",
                student_id=sid,
                job_id=job_id,
                score=0,
                band="developing",
                breakdown={},
                is_favorite=new_val,
            )
        )

    for app_row in session.exec(
        select(JobApplication).where(
            JobApplication.student_id == sid, JobApplication.job_id == job_id
        )
    ).all():
        app_row.is_favorite = new_val
        app_row.updated_at = datetime.now(timezone.utc)
        session.add(app_row)

    session.add(
        AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="match.favorited" if new_val else "match.unfavorited",
            entity_type="job_match",
            entity_id=f"{sid}:{job_id}",
            actor_role=role,
            detail={"student_id": sid, "job_id": job_id, "is_favorite": new_val},
        )
    )
    session.commit()
    return {"ok": True, "student_id": sid, "job_id": job_id, "is_favorite": new_val}


@router.delete("/api/students/{sid}/matches/{job_id}")
@router.post("/api/students/{sid}/matches/{job_id}/dismiss")
def dismiss_student_job_match(
    sid: str,
    job_id: str,
    request: Request,
    permanent: bool = False,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(require_role("student", "faculty_mentor", "tpo_admin")),
):
    """Remove/dismiss a job match for a student, or permanently delete the job if requested by TPO Admin."""
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    job = session.get(Job, job_id)

    # Permanent delete is restricted strictly to TPO Admin
    if permanent:
        if role != "tpo_admin":
            raise HTTPException(403, "permanent job deletion requires tpo_admin role")
    elif role != "tpo_admin" and is_on_campus_job(job):
        raise HTTPException(403, "On-campus institutional jobs cannot be dismissed by students.")

    if permanent:
        if role != "tpo_admin":
            raise HTTPException(403, "permanent job deletion requires tpo_admin role")
        if job is not None:
            if hasattr(ctx, "vector_store") and ctx.vector_store:
                try:
                    ctx.vector_store.delete("job", job_id)
                except Exception:
                    pass
            if hasattr(ctx, "job_vectors") and job_id in ctx.job_vectors:
                ctx.job_vectors.pop(job_id, None)

            plan_ids = list(session.exec(select(UpskillPlan.id).where(UpskillPlan.job_id == job_id)).all())
            if plan_ids:
                session.execute(delete(PlanItem).where(PlanItem.plan_id.in_(plan_ids)))
            session.execute(delete(UpskillPlan).where(UpskillPlan.job_id == job_id))
            for cd in session.exec(select(CampusDrive).where(CampusDrive.job_id == job_id)).all():
                cd.job_id = None
                session.add(cd)
            session.execute(delete(JobApplication).where(JobApplication.job_id == job_id))
            session.execute(delete(JobMatch).where(JobMatch.job_id == job_id))
            session.execute(delete(Interview).where(Interview.job_id == job_id))
            session.delete(job)
            session.add(
                AuditLog(
                    id=f"aud-{uuid4().hex[:8]}",
                    action="job.deleted_permanently",
                    entity_type="job",
                    entity_id=job_id,
                    actor_role=role,
                    detail={"deleted_by": role, "sid": sid, "job_title": job.title, "company": job.company},
                )
            )
            session.commit()
            return {"status": "deleted_permanently", "job_id": job_id}

    # Dismiss for this student
    m = session.exec(select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == job_id)).first()
    if m is not None:
        m.band = "dismissed"
        m.score = -1
        b = dict(m.breakdown or {})
        b["dismissed"] = True
        m.breakdown = b
        session.add(m)
    else:
        session.add(
            JobMatch(
                id=f"m-{uuid4().hex[:8]}",
                student_id=sid,
                job_id=job_id,
                score=-1,
                band="dismissed",
                breakdown={"dismissed": True},
            )
        )

    # Withdraw any active application for this job by this student
    active_apps = session.exec(
        select(JobApplication).where(
            JobApplication.student_id == sid,
            JobApplication.job_id == job_id,
            JobApplication.status != "withdrawn",
        )
    ).all()
    for active_app in active_apps:
        active_app.status = "withdrawn"
        active_app.updated_at = datetime.now(timezone.utc)
        active_app.notes = ((active_app.notes or "") + " [Application withdrawn with job dismissal]").strip()
        session.add(active_app)
        if active_app.approval_id:
            ap = session.get(ApprovalRequest, active_app.approval_id)
            if ap and ap.status == "pending_review":
                ap.status = "rejected"
                ap.review_note = "Job opportunity removed / dismissed by candidate"
                ap.reviewed_at = datetime.now(timezone.utc)
                session.add(ap)

    session.add(
        AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="job_match.dismissed",
            entity_type="job_match",
            entity_id=job_id,
            actor_role=role,
            detail={"student_id": sid, "job_id": job_id, "reason": "closed_or_not_wanted"},
        )
    )
    session.commit()
    return {"status": "dismissed", "student_id": sid, "job_id": job_id}



@router.post("/api/plans/{plan_id}/items/{item_id}/complete")
def complete_plan_item(
    plan_id: str,
    item_id: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    """Mark a plan item done and record a readiness snapshot.

    readiness = round(100 * done_hours / total_hours) blended 50/50 with the
    student's current best match score (max JobMatch.score across their matches).
    Formula: final = round(0.5 * plan_completion_pct + 0.5 * best_match_score).
    Delegates to apply_plan_item_completion (shared with POST .../verify) so the
    closed-loop skill upsert + JobMatch re-rank happen identically on both paths.
    """
    from app.plan.planner import apply_plan_item_completion

    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    enforce_student_scope(request, plan.student_id)
    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan_id:
        raise HTTPException(404, "plan item not found")
    return apply_plan_item_completion(session, ctx, plan, item)


@router.get("/api/students/{sid}/progress")
def get_progress(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
):
    from app.analytics.student_progress import build_student_progress_report

    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
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
    report = build_student_progress_report(session, sid, taxonomy=getattr(ctx, "taxonomy", None))
    return {
        "readiness": [{"at": e.at.isoformat(), "readiness": e.readiness} for e in events],
        "events": [
            {"id": e.id, "kind": e.kind, "ref_id": e.ref_id, "readiness": e.readiness, "at": e.at.isoformat()}
            for e in events
        ],
        "best_match": (
            {"job_id": best.job_id, "score": best.score, "band": best.band} if best else None
        ),
        "report": report,
    }


@router.get("/api/students/{sid}/suggest-job-queries")
def suggest_job_queries_endpoint(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
) -> dict[str, Any]:
    """Return skill-targeted search queries (LLM-backed with deterministic fallback).

    ``top_skills`` lists the candidate's top verified skill ids in the same
    advanced > intermediate > basic (then mentions) order the synthesizer uses.
    """
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    skills = sorted(
        rows,
        key=lambda s: (_LEVEL_RANK.get(str(s.level).lower(), 0), int(s.mentions or 0)),
        reverse=True,
    )
    from app.scraper.ai_query import has_diverse_skillset

    is_diverse = has_diverse_skillset(skills)
    queries = suggest_job_queries_for_student(student, skills, getattr(ctx, "llm", None))
    return {
        "student_id": sid,
        "suggested_queries": queries,
        "top_skills": [s.skill_id for s in skills[:5]],
        "is_diverse": is_diverse,
        "query_count": len(queries),
    }


@router.post("/api/students/{sid}/scrape-jobs")
def student_scrape_jobs_endpoint(
    sid: str,
    request: Request,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    _role: str = Depends(require_role("student", "tpo_admin")),
) -> dict[str, Any]:
    """Skill-targeted live scrape: resolve queries, fetch, ingest, re-rank matches.

    When no explicit ``term`` or ``terms`` is supplied, the candidate's verified skills drive
    the search queries via ``suggest_job_queries_for_student``. If the candidate has a diverse
    skillset, multiple targeted keywords are queried. If the candidate lacks diverse skills,
    exactly 1 query is used. The payload accepts ``term`` / ``terms``, ``location`` (default Bengaluru),
    ``limit`` (default 20) and ``sites`` (default naukri, linkedin, indeed).
    """
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    payload = payload or {}
    terms_input = payload.get("terms") or payload.get("queries") or payload.get("search_terms")
    term = payload.get("term") or payload.get("search_term")
    location = payload.get("location") or "Bengaluru"
    limit = int(payload.get("limit") or payload.get("results_wanted") or 20)
    sites = payload.get("sites") or ["naukri", "linkedin", "indeed", "ashby", "greenhouse", "lever"]

    terms: list[str] = []
    if terms_input and isinstance(terms_input, list):
        terms = [str(t).strip() for t in terms_input if str(t).strip()]
    elif term:
        if isinstance(term, list):
            terms = [str(t).strip() for t in term if str(t).strip()]
        elif isinstance(term, str) and "," in term:
            terms = [t.strip() for t in term.split(",") if t.strip()]
        elif isinstance(term, str) and term.strip():
            terms = [term.strip()]

    if not terms:
        rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
        skills = sorted(
            rows,
            key=lambda s: (_LEVEL_RANK.get(str(s.level).lower(), 0), int(s.mentions or 0)),
            reverse=True,
        )
        suggested = suggest_job_queries_for_student(student, skills, getattr(ctx, "llm", None))
        terms = suggested if suggested else ["Software Engineer"]

    from app.scraper.service import CompositeJobScraper

    providers = payload.get("providers") if payload.get("providers") is not None else ["jobspy"]
    scraper = CompositeJobScraper()
    raw_records = scraper.fetch_jobs(
        search_term=terms if len(terms) > 1 else terms[0],
        location=location,
        sites=sites,
        results_wanted=limit,
        providers=providers,
        live=payload.get("live", True),
    )
    res = ingest_scraped_jobs(raw_records, session, ctx, owner_student_id=sid)
    _rank_and_persist(session, ctx, sid, student, include_scraped=True)
    return {
        "status": "success",
        "term_used": terms[0] if len(terms) == 1 else ", ".join(terms),
        "terms_used": terms,
        "fetched": len(raw_records),
        "ingested": res["ingested_count"],
        "job_ids": res["job_ids"],
    }


@router.post("/api/students/{sid}/scrape-jobs/stream")
def student_scrape_jobs_stream_endpoint(
    sid: str,
    request: Request,
    payload: dict[str, Any] | None = None,
    session: Session = Depends(get_session),
    ctx: Any = Depends(get_ctx),
    _role: str = Depends(require_role("student", "tpo_admin")),
):
    """Stream live scrape telemetry for a student workspace, then re-rank matches."""
    import queue
    import threading
    import time

    from fastapi.responses import StreamingResponse

    from app.api.routes.jobs import SSE_HEADERS, _sse_chunk
    from app.scraper.service import CompositeJobScraper

    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    payload = payload or {}
    terms_input = payload.get("terms") or payload.get("queries") or payload.get("search_terms")
    term = payload.get("term") or payload.get("search_term")
    location = payload.get("location") or "Bengaluru"
    limit = int(payload.get("limit") or payload.get("results_wanted") or 20)
    sites = payload.get("sites") or ["naukri", "linkedin", "indeed", "ashby", "greenhouse", "lever"]
    fresher_only = payload.get("fresher_only", True)

    terms: list[str] = []
    if terms_input and isinstance(terms_input, list):
        terms = [str(t).strip() for t in terms_input if str(t).strip()]
    elif term:
        if isinstance(term, list):
            terms = [str(t).strip() for t in term if str(t).strip()]
        elif isinstance(term, str) and "," in term:
            terms = [t.strip() for t in term.split(",") if t.strip()]
        elif isinstance(term, str) and term.strip():
            terms = [term.strip()]

    if not terms:
        rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
        skills = sorted(
            rows,
            key=lambda s: (_LEVEL_RANK.get(str(s.level).lower(), 0), int(s.mentions or 0)),
            reverse=True,
        )
        suggested = suggest_job_queries_for_student(student, skills, getattr(ctx, "llm", None))
        terms = suggested if suggested else ["Software Engineer"]

    if fresher_only:
        fresher_qualifiers = ["Fresher", "Junior", "Intern", "Entry Level"]
        enriched_terms: list[str] = []
        for t in terms:
            if any(q.lower() in t.lower() for q in ("fresher", "junior", "intern", "entry level", "trainee", "associate")):
                enriched_terms.append(t)
            else:
                for q in fresher_qualifiers:
                    enriched_terms.append(f"{t} {q}")
        terms = enriched_terms

    search_term = terms if len(terms) > 1 else terms[0]
    providers = payload.get("providers") if payload.get("providers") is not None else ["jobspy", "stealth_harvester", "ashby_ats"]
    scraper = CompositeJobScraper()

    def event_stream():
        event_q: queue.Queue[tuple[str, Any] | None] = queue.Queue()

        def on_event(name: str, data: Any) -> None:
            if name == "complete":
                return
            event_q.put((name, data))

        records: list[dict[str, Any]] = []
        ingest_result: dict[str, Any] = {}
        started_at = time.perf_counter()

        def run_scrape() -> None:
            try:
                records.extend(
                    scraper.fetch_jobs_streaming(
                        search_term=search_term,
                        location=location,
                        sites=sites,
                        results_wanted=limit,
                        providers=providers,
                        on_event=on_event,
                        live=payload.get("live", True),
                        fresher_only=fresher_only,
                    )
                )
                with Session(ctx.engine) as scrape_session:
                    event_q.put(("ingest_start", {"status": "ingesting", "total": len(records)}))
                    res = ingest_scraped_jobs(records, scrape_session, ctx, owner_student_id=sid)
                    ingest_result.update(res)
                    try:
                        st = scrape_session.get(Student, sid)
                        if st:
                            _rank_and_persist(scrape_session, ctx, sid, st, include_scraped=True)
                    except Exception as err:
                        logger.warning("Post-scrape student match generation failed: %s", err)

                total_jobs = len(records)
                new_ingested = int(ingest_result.get("ingested_count", 0))
                elapsed_total = round(time.perf_counter() - started_at, 3)
                event_q.put((
                    "complete",
                    {
                        "total_jobs": total_jobs,
                        "new_ingested": new_ingested,
                        "elapsed_total_s": elapsed_total,
                    },
                ))
            except Exception as exc:
                event_q.put((
                    "portal_error",
                    {"portal": "stream", "status": "warning", "message": str(exc), "duration_s": 0.0},
                ))
                event_q.put((
                    "complete",
                    {"total_jobs": 0, "new_ingested": 0, "elapsed_total_s": round(time.perf_counter() - started_at, 3)},
                ))
            finally:
                event_q.put(None)

        worker = threading.Thread(target=run_scrape, daemon=True)
        worker.start()

        while True:
            item = event_q.get()
            if item is None:
                break
            ev_name, ev_data = item
            yield _sse_chunk(ev_name, ev_data)

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=SSE_HEADERS)
