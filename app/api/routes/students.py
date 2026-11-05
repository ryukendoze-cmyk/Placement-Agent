import logging
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy import delete
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.embeddings.store import cosine
from app.matching.engine import job_eligibility, job_tier
from app.matching.rank import rank_jobs
from app.matching.simulator import simulate_counterfactual
from app.models import (
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
from app.security.rbac import enforce_student_scope, require_role
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
    }


def _match_payload(job, js, student) -> dict:
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
        "eligible": eligible,
        "eligibility_reasons": reasons,
        "job_url": job_url,
        "source": source,
        "is_fresher": getattr(job, "is_fresher", True),
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
    """Rank all jobs for a student (proficiency-aware), persist JobMatch rows, and
    return the explainable payload. Shared by GET matches and re-upload refresh.

    When include_scraped is True (e.g. after a live market scrape), all external scraped
    jobs are also ranked and persisted so they appear under the Live Market Opportunities view.
    """
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
    jobs = session.exec(select(Job)).all()
    results = rank_jobs(jobs, sv, student_skills, ctx.embedder, ctx.taxonomy,
                        vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                        top=top, levels=levels, student=student)

    if include_scraped:
        external_jobs = [
            j for j in jobs
            if getattr(j, "source", "local") != "local" or getattr(j, "scraper_engine", "local") != "local"
        ]
        scored_job_ids = {getattr(j, "id", None) for j, _ in results}
        missing_external = [j for j in external_jobs if getattr(j, "id", None) not in scored_job_ids]
        if missing_external:
            ext_results = rank_jobs(
                missing_external, sv, student_skills, ctx.embedder, ctx.taxonomy,
                vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                top=None, levels=levels, student=student,
            )
            results.extend(ext_results)

    session.execute(delete(JobMatch).where(JobMatch.student_id == sid))
    for job, js in results:
        session.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=job.id,
                             score=js.score, band=js.band, breakdown=asdict(js)))
    session.commit()
    return [_match_payload(job, js, student) for job, js in results]


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


def delete_student_core(session: Session, ctx, sid: str) -> bool:
    student = session.get(Student, sid)
    if student is None:
        return False
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
    session.delete(student)
    session.commit()
    return True


_STUDENT_OR_ADMIN_ROLE = require_role("student", "faculty_mentor", "tpo_admin")


@router.delete("/api/students/{sid}")
def delete_student(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    user=Depends(_STUDENT_OR_ADMIN_ROLE),
):
    enforce_student_scope(request, sid)
    if not delete_student_core(session, ctx, sid):
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
def extract_resume(file: UploadFile = File(...), ctx=Depends(get_ctx)):
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
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    _sid, skills = create_student_core_resume(session, ctx, sid, file)
    return {"id": sid, "skills": skills, "skill_count": len(skills)}


@router.post("/api/students", status_code=201)
async def create_student(
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
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
def get_progress(sid: str, request: Request, session: Session = Depends(get_session)):
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    events = session.exec(
        select(ProgressEvent).where(ProgressEvent.student_id == sid).order_by(ProgressEvent.at.asc())
    ).all()
    matches = session.exec(
        select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
    ).all()
    best = matches[0] if matches else None
    return {
        "readiness": [{"at": e.at.isoformat(), "readiness": e.readiness} for e in events],
        "events": [
            {"id": e.id, "kind": e.kind, "ref_id": e.ref_id, "readiness": e.readiness, "at": e.at.isoformat()}
            for e in events
        ],
        "best_match": (
            {"job_id": best.job_id, "score": best.score, "band": best.band} if best else None
        ),
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
    sites = payload.get("sites") or ["naukri", "linkedin", "indeed"]

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
    res = ingest_scraped_jobs(raw_records, session, ctx)
    _rank_and_persist(session, ctx, sid, student, include_scraped=True)
    return {
        "status": "success",
        "term_used": terms[0] if len(terms) == 1 else ", ".join(terms),
        "terms_used": terms,
        "fetched": len(raw_records),
        "ingested": res["ingested_count"],
        "job_ids": res["job_ids"],
    }
