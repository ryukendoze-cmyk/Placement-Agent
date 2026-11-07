"""HITL approvals (Phase 4): list pending, approve (executes the real mutation),
reject. Every decision writes an immutable AuditLog row.

State-changing kinds handled here:
- job_application_nomination / nomination -> ProgressEvent(kind="nomination")
- cohort_bootcamp_publish -> inject the recommended course into pending cohort plan items
- proficiency_waiver -> bump StudentSkill level + recompute JobMatch
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.models import (
    AgentRun,
    ApprovalRequest,
    AuditLog,
    JobApplication,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Student,
    StudentSkill,
)
from app.security.rbac import require_role

router = APIRouter()


def _audit(session: Session, *, action: str, entity_type: str, entity_id: str,
           actor_role: str, detail: dict) -> None:
    session.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action=action,
                         entity_type=entity_type, entity_id=entity_id,
                         actor_role=actor_role, detail=detail))
    # no commit here: caller commits with the approval mutation (single transaction)


def _payload(ap: ApprovalRequest) -> dict:
    return {
        "id": ap.id, "kind": ap.kind, "target_id": ap.target_id,
        "title": ap.title, "summary": ap.summary, "payload": ap.payload,
        "status": ap.status, "requested_by": ap.requested_by,
        "reviewed_by": ap.reviewed_by, "review_note": ap.review_note,
        "run_id": ap.run_id,
        "created_at": ap.created_at.isoformat() if ap.created_at else None,
        "reviewed_at": ap.reviewed_at.isoformat() if ap.reviewed_at else None,
        "endorsed_by": getattr(ap, "endorsed_by", "") or "",
        "endorsed_at": (ap.endorsed_at.isoformat()
                        if getattr(ap, "endorsed_at", None) else None),
    }


def _linked_application(session: Session, ap: ApprovalRequest) -> JobApplication | None:
    """Resolve the JobApplication linked to an approval (M19).

    Prefers the explicit approval_id link; falls back to (student_id, job_id)
    from the payload for approvals created before the link existed.
    """
    row = session.exec(select(JobApplication)
                       .where(JobApplication.approval_id == ap.id)).first()
    if row is not None:
        return row
    payload = ap.payload or {}
    sid = payload.get("student_id")
    job_id = payload.get("job_id")
    if sid and job_id:
        return session.exec(select(JobApplication)
                            .where(JobApplication.student_id == sid,
                                   JobApplication.job_id == job_id)
                            .order_by(JobApplication.created_at.desc())).first()
    return None


def _recompute_job_match(session: Session, ctx, sid: str) -> None:
    """Re-rank matches for a student after a proficiency waiver (Phase 4)."""
    from dataclasses import asdict as _asdict

    from sqlmodel import delete as _delete

    from app.matching.rank import rank_jobs
    from app.models import Job, Student
    from app.models import StudentSkill as SS
    student = session.get(Student, sid)
    if student is None:
        return
    store = getattr(ctx, "vector_store", None) if ctx is not None else None
    sv = store.get("student", sid) if store is not None else None
    if sv is None and ctx is not None:
        # rebuild from resume if the vector is missing (offline-safe)
        if (student.resume_text or "").strip():
            sv = ctx.embedder.embed(student.resume_text)
            if store is not None:
                store.upsert("student", sid, sv)
        else:
            return
    rows = session.exec(select(SS).where(SS.student_id == sid)).all()
    skills = {r.skill_id: list(r.evidence) for r in rows}
    levels = {r.skill_id: r.level for r in rows}
    jobs = session.exec(select(Job)).all()
    results = rank_jobs(jobs, sv, skills, ctx.embedder, ctx.taxonomy,
                        vector_store=store, job_vec_cache=ctx.job_vectors,
                        top=10, levels=levels)
    session.execute(_delete(JobMatch).where(JobMatch.student_id == sid))
    score_by_job = {}
    for job, js in results:
        score_by_job[job.id] = js.score
        session.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=job.id,
                             score=js.score, band=js.band, breakdown=_asdict(js)))
    # Refresh active JobApplications for this student so TPO sees updated readiness
    active_apps = session.exec(select(JobApplication).where(
        JobApplication.student_id == sid,
        JobApplication.status.in_(("submitted", "awaiting_mentor_endorsement", "awaiting_tpo_approval")),
    )).all()
    for app_row in active_apps:
        if app_row.job_id in score_by_job:
            app_row.match_score = int(score_by_job[app_row.job_id])
            app_row.updated_at = datetime.now(timezone.utc)
            session.add(app_row)


def execute_approved(session: Session, ctx, ap: ApprovalRequest) -> dict:
    """Run the real mutation for an approved request. Returns a summary dict."""
    kind = ap.kind
    payload = ap.payload or {}
    summary: dict = {"kind": kind, "target_id": ap.target_id}

    if kind in ("job_application_nomination", "nomination"):
        sid = payload.get("student_id")
        if not sid and ap.target_id.startswith("stu-"):
            sid = ap.target_id
        if not sid and ap.run_id:
            # provenance: fall back to the student on the originating agent run
            run = session.get(AgentRun, ap.run_id)
            sid = getattr(run, "student_id", None)
        if not sid:
            raise HTTPException(404, "no student resolved for nomination")
        student = session.get(Student, sid)
        if student is None:
            raise HTTPException(404, "student not found for nomination")
        session.add(ProgressEvent(
            id=f"pe-{uuid4().hex[:8]}", student_id=sid, kind="nomination",
            ref_id=payload.get("job_id", ""), readiness=int(payload.get("readiness") or 0),
            at=datetime.now(timezone.utc)))
        summary.update({"student_id": sid, "mutated": "ProgressEvent(nomination)"})
        # M19: advance the linked campus-drive application to "nominated".
        app_row = _linked_application(session, ap)
        if app_row is not None:
            app_row.status = "nominated"
            app_row.approved_by = ap.reviewed_by or ""
            app_row.updated_at = datetime.now(timezone.utc)
            session.add(app_row)
            summary["mutated"] = "ProgressEvent(nomination) + JobApplication(nominated)"

    elif kind == "cohort_bootcamp_publish":
        skill_id = ap.target_id or payload.get("skill_id")
        course_id = payload.get("course_id")
        if not course_id and skill_id:
            # resolve from the course catalog when the agent didn't pass one
            from app.plan.planner import _load_courses, pick_course_for_skill
            course = pick_course_for_skill(skill_id, _load_courses(session))
            course_id = course["id"] if course else None
        updated = 0
        if skill_id and course_id:
            rows = session.exec(select(PlanItem).where(PlanItem.skill_id == skill_id,
                                                       PlanItem.status == "pending")).all()
            for it in rows:
                it.course_id = course_id
                it.status = "published"
                session.add(it)
                updated += 1
        # M20: count cohort students who still lack the bootcamp target skill.
        students_impacted = 0
        if skill_id:
            having = {ss.student_id for ss in session.exec(
                select(StudentSkill).where(StudentSkill.skill_id == skill_id)).all()}
            all_students = session.exec(select(Student)).all()
            students_impacted = len([s for s in all_students
                                     if s.id not in having])
        summary.update({"skill_id": skill_id, "course_id": course_id,
                        "students_impacted": students_impacted,
                        "mutated": f"PlanItem({updated} published)"})

    elif kind == "proficiency_waiver":
        sid = payload.get("student_id")
        skill_id = payload.get("skill_id") or ap.target_id
        new_level = payload.get("level") or payload.get("new_level") or "intermediate"
        ss = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid,
                                                     StudentSkill.skill_id == skill_id)).first()
        if ss is None:
            raise HTTPException(404, "student skill not found for waiver")
        old = ss.level
        ss.level = new_level
        session.add(ss)
        _recompute_job_match(session, ctx, sid)
        summary.update({"student_id": sid, "skill_id": skill_id,
                        "old_level": old, "new_level": new_level,
                        "mutated": "StudentSkill.level + JobMatch recompute"})

    else:
        # unknown kind: record a no-op approval (still audited)
        summary.update({"mutated": "none", "note": "no handler for kind"})

    return summary


@router.get("/api/approvals")
def list_approvals(status: str | None = None,
                   session: Session = Depends(get_session),
                   role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    q = select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
    if status:
        q = q.where(ApprovalRequest.status == status)
    rows = session.exec(q).all()
    return [_payload(a) for a in rows]


@router.post("/api/approvals", status_code=201)
def create_approval(payload: dict,
                    session: Session = Depends(get_session),
                    role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """Queue a new HITL request (e.g. Launch Bootcamp) for review."""
    kind = str((payload or {}).get("kind") or "").strip()
    if not kind:
        raise HTTPException(400, "kind is required")
    target_id = str((payload or {}).get("target_id") or "").strip()
    title = str((payload or {}).get("title") or "").strip() or kind
    summary = str((payload or {}).get("summary") or "").strip()
    body = (payload or {}).get("payload") or {}
    if not isinstance(body, dict):
        body = {}
    aid = f"apr-{uuid4().hex[:8]}"
    ap = ApprovalRequest(id=aid, kind=kind, target_id=target_id, title=title,
                         summary=summary, payload=body, status="pending_review",
                         requested_by=role, created_at=datetime.now(timezone.utc))
    session.add(ap)
    _audit(session, action="approval.requested", entity_type="approval",
           entity_id=aid, actor_role=role,
           detail={"kind": kind, "target_id": target_id, "title": title})
    session.commit()
    return _payload(ap)


@router.post("/api/approvals/{appr_id}/approve")
def approve(appr_id: str, payload: dict | None = None,
            session: Session = Depends(get_session), ctx=Depends(get_ctx),
            role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    if ap.status == "approved":
        return _payload(ap)   # idempotent
    if ap.status == "rejected":
        raise HTTPException(409, "approval already rejected")
    note = str((payload or {}).get("note") or "")
    if ap.status == "pending_review":
        # Separation of duties: faculty mentors endorse Tier-1 applications,
        # while final Tier-2 drive nomination approval or faculty-originated
        # requests require TPO admin sign-off.
        if role == "faculty_mentor" and (
            ap.kind in ("job_application_nomination", "nomination")
            or (ap.endorsed_by or "").strip() == "faculty_mentor"
            or (ap.requested_by or "").strip() == "faculty_mentor"
        ):
            raise HTTPException(
                403,
                "TPO administrator approval is required for final drive nomination or self-requested actions (separation of duties)",
            )
        # M19 two-tier guard: a waiver application must be endorsed by a
        # faculty mentor before the TPO can approve it.
        linked = _linked_application(session, ap)
        if (linked is not None
                and linked.status == "awaiting_mentor_endorsement"
                and not (ap.endorsed_by or "").strip()):
            raise HTTPException(
                400, "faculty mentor endorsement required before TPO approval")
        # reviewed_by must be populated BEFORE execute_approved so the linked
        # JobApplication.approved_by records the deciding role (M19). The
        # status flip happens AFTER the handler so a failed handler's
        # session.rollback() never reverts it (approval still records).
        ap.reviewed_by = role
        ap.review_note = note
        ap.reviewed_at = datetime.now(timezone.utc)
        session.add(ap)
        # Execute the mutation but NEVER let it abort the approval/audit recording:
        # a failed handler is reported in `executed.error` (the approval still flips
        # to approved, and the decision is audited).
        exec_error = ""
        try:
            summary = execute_approved(session, ctx, ap)
        except HTTPException as he:
            session.rollback()
            summary = {"kind": ap.kind, "target_id": ap.target_id, "mutated": "none"}
            exec_error = str(he.detail)
        except Exception as e:  # pragma: no cover - defensive
            session.rollback()
            summary = {"kind": ap.kind, "target_id": ap.target_id, "mutated": "none"}
            exec_error = str(e)
        ap.status = "approved"
        ap.reviewed_by = role
        ap.review_note = note
        ap.reviewed_at = datetime.now(timezone.utc)
        session.add(ap)
        _audit(session, action="approval.approved", entity_type="approval",
               entity_id=ap.id, actor_role=role,
               detail={"kind": ap.kind, "target_id": ap.target_id, "note": note,
                       "mutated": summary.get("mutated", ""),
                       **({"error": exec_error} if exec_error else {})})
        session.commit()
        return {**_payload(ap), "executed": summary, **({"error": exec_error} if exec_error else {})}
    # already approved (idempotent) - just return; no new audit row
    return _payload(ap)


@router.post("/api/approvals/{appr_id}/reject")
def reject(appr_id: str, payload: dict | None = None,
           session: Session = Depends(get_session),
           role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    if ap.status == "approved":
        raise HTTPException(409, "approval already approved")
    note = str((payload or {}).get("note") or "")
    ap.status = "rejected"
    ap.reviewed_by = role
    ap.review_note = note
    ap.reviewed_at = datetime.now(timezone.utc)
    session.add(ap)
    # M19: rejection propagates to the linked campus-drive application.
    app_row = _linked_application(session, ap)
    if app_row is not None and app_row.status not in ("nominated", "withdrawn"):
        app_row.status = "rejected"
        app_row.updated_at = datetime.now(timezone.utc)
        session.add(app_row)
    _audit(session, action="approval.rejected", entity_type="approval",
           entity_id=ap.id, actor_role=role,
           detail={"kind": ap.kind, "target_id": ap.target_id, "note": note})
    session.commit()
    return _payload(ap)


@router.post("/api/approvals/{appr_id}/endorse")
def endorse(appr_id: str, payload: dict | None = None,
            session: Session = Depends(get_session),
            role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """M19 tier-1 decision: faculty mentor endorsement (before TPO approval).

    Sets endorsed_by/endorsed_at on the approval and advances a linked
    JobApplication from awaiting_mentor_endorsement to awaiting_tpo_approval.
    """
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    if ap.status == "approved":
        raise HTTPException(409, "approval already approved")
    if ap.status == "rejected":
        raise HTTPException(409, "approval already rejected")
    note = str((payload or {}).get("note") or "")
    ap.endorsed_by = role
    ap.endorsed_at = datetime.now(timezone.utc)
    session.add(ap)
    app_row = _linked_application(session, ap)
    if app_row is not None and app_row.status == "awaiting_mentor_endorsement":
        app_row.status = "awaiting_tpo_approval"
        app_row.endorsed_by = role
        app_row.updated_at = datetime.now(timezone.utc)
        session.add(app_row)
    _audit(session, action="approval.endorsed", entity_type="approval",
           entity_id=ap.id, actor_role=role,
           detail={"kind": ap.kind, "target_id": ap.target_id, "note": note,
                   "endorsed_by": role})
    session.commit()
    return _payload(ap)
