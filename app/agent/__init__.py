"""Shared helpers for the agent package (tools import these)."""
from __future__ import annotations


def _rank_jobs_for(session, ctx, student_id: str, top: int = 10) -> list[dict]:
    """Rank visible, non-dismissed jobs for a student using the matches pipeline
    (with student seniority and verification status weighting)."""
    from sqlmodel import select

    from ..matching.engine import is_job_visible_to_student, job_eligibility, job_tier
    from ..matching.rank import rank_jobs
    from ..models import JobApplication, JobMatch, Student

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
    ss_rows = session.exec(_select_student_skill(student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    verification_statuses = {
        r.skill_id: (getattr(r, "verification_status", None) or "claimed")
        for r in ss_rows
    }

    existing_matches = session.exec(
        select(JobMatch).where(JobMatch.student_id == student_id)
    ).all()
    dismissed_job_ids = {
        m.job_id
        for m in existing_matches
        if getattr(m, "band", None) == "dismissed"
        or getattr(m, "score", 0) < 0
        or (isinstance(getattr(m, "breakdown", None), dict) and m.breakdown.get("dismissed"))
    }
    favorite_job_ids = {
        m.job_id for m in existing_matches if getattr(m, "is_favorite", False)
    } | {
        a.job_id
        for a in session.exec(
            select(JobApplication).where(JobApplication.student_id == student_id)
        ).all()
        if getattr(a, "is_favorite", False)
    }

    jobs = [
        j
        for j in session.exec(_select_jobs()).all()
        if is_job_visible_to_student(j, student_id) and j.id not in dismissed_job_ids
    ]
    results = rank_jobs(
        jobs,
        sv or [],
        student_skills,
        ctx.embedder,
        ctx.taxonomy,
        vector_store=getattr(ctx, "vector_store", None),
        job_vec_cache=getattr(ctx, "job_vectors", None),
        top=int(top),
        levels=levels,
        student=student,
        verification_statuses=verification_statuses,
        session=session,
    )
    out = []
    for job, js in results:
        eligible, reasons = job_eligibility(student, job)
        req_names = [
            ctx.taxonomy.get_opt(s).name if getattr(ctx, "taxonomy", None) and ctx.taxonomy.get_opt(s) else s
            for s in (getattr(job, "required_skills", None) or [])
        ]
        out.append({
            "job_id": job.id,
            "title": job.title,
            "company": getattr(job, "company", "") or "",
            "tier": job_tier(job),
            "score": js.score,
            "band": js.band,
            "proficiency_bonus": js.proficiency_bonus,
            "is_favorite": (job.id in favorite_job_ids),
            "required_skills": req_names,
            "matched_required": js.matched_required,
            "missing_required": js.missing_required,
            "matched_preferred": js.matched_preferred,
            "missing_preferred": js.missing_preferred,
            "proficiency_gaps": js.proficiency_gaps,
            "eligible": eligible,
            "eligibility_reasons": reasons,
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
