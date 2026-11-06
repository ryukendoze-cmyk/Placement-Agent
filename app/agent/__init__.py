"""Shared helpers for the agent package (tools import these)."""
from __future__ import annotations


def _rank_jobs_for(session, ctx, student_id: str, top: int = 10) -> list[dict]:
    """Rank jobs for a student using the existing pipeline; returns explainable
    payloads (does NOT persist JobMatch - tools are read/simulate only)."""
    from ..matching.engine import job_eligibility, job_tier
    from ..matching.rank import rank_jobs
    from ..models import Student
    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    sv = ctx.vector_store.get("student", student_id) if getattr(ctx, "vector_store", None) else None
    if sv is None and getattr(ctx, "embedder", None) and getattr(student, "resume_text", None):
        try:
            sv = ctx.embedder.embed(student.resume_text)
            if getattr(ctx, "vector_store", None):
                ctx.vector_store.upsert("student", student_id, sv)
        except Exception:
            sv = None
    ss_rows = session.exec(
        _select_student_skill(student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    jobs = session.exec(_select_jobs()).all()
    results = rank_jobs(jobs, sv or [], student_skills, ctx.embedder, ctx.taxonomy,
                        vector_store=getattr(ctx, "vector_store", None),
                        job_vec_cache=getattr(ctx, "job_vectors", None),
                        top=int(top), levels=levels)
    out = []
    for job, js in results:
        eligible, reasons = job_eligibility(student, job)
        out.append({
            "job_id": job.id, "title": job.title, "company": getattr(job, "company", "") or "",
            "tier": job_tier(job),
            "score": js.score, "band": js.band, "proficiency_bonus": js.proficiency_bonus,
            "matched_required": js.matched_required, "missing_required": js.missing_required,
            "matched_preferred": js.matched_preferred, "missing_preferred": js.missing_preferred,
            "proficiency_gaps": js.proficiency_gaps,
            "eligible": eligible, "eligibility_reasons": reasons,
        })
    return out


def _select_student_skill(student_id: str):
    from sqlmodel import select

    from ..models import StudentSkill
    return select(StudentSkill).where(StudentSkill.student_id == student_id)


def _select_jobs():
    from sqlmodel import select

    from ..models import Job
    return select(Job)
