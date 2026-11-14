"""Campus drive application pipeline (M19).

POST /api/students/{sid}/applications  - apply for an on-campus drive job
GET  /api/students/{sid}/applications  - a student's application history
GET  /api/admin/applications           - cohort ledger (faculty: dept-scoped)

Every state change is mirrored into AuditLog / ProgressEvent and the linked
HITL ApprovalRequest when the tier or SOP gate demands human review.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.matching.engine import job_tier
from app.models import (
    ApprovalRequest,
    AuditLog,
    Job,
    JobApplication,
    JobMatch,
    ProgressEvent,
    Student,
)
from app.rag.policy_rag import evaluate_student_eligibility
from app.security.rbac import (
    CurrentUser,
    enforce_student_scope,
    get_current_user_optional,
    require_role,
)

router = APIRouter()

_STUDENT_OR_ADMIN = require_role("student", "faculty_mentor", "tpo_admin")

# statuses that block a second application for the same (student, job)
_ACTIVE_STATUSES = ("submitted", "awaiting_mentor_endorsement",
                    "awaiting_tpo_approval", "nominated")


def _app_payload(app_row: JobApplication, *, job: Job | None = None,
                 eligibility: dict | None = None,
                 approval: ApprovalRequest | None = None) -> dict:
    return {
        "id": app_row.id,
        "student_id": app_row.student_id,
        "job_id": app_row.job_id,
        "status": app_row.status,
        "tier": app_row.tier,
        "match_score": app_row.match_score,
        "eligible_at_apply": app_row.eligible_at_apply,
        "eligibility_checks": app_row.eligibility_checks or {},
        "approval_id": app_row.approval_id,
        "endorsed_by": app_row.endorsed_by or "",
        "approved_by": app_row.approved_by or "",
        "notes": app_row.notes or "",
        "is_favorite": bool(getattr(app_row, "is_favorite", False)),
        "created_at": app_row.created_at.isoformat() if app_row.created_at else None,
        "updated_at": app_row.updated_at.isoformat() if app_row.updated_at else None,
        "job_title": job.title if job else "",
        "job_company": job.company if job else "",
        "job_salary_lpa": list(job.salary_lpa or [0, 0]) if job else [0, 0],
        "eligibility": eligibility,
        "approval": ({
            "id": approval.id, "status": approval.status,
            "endorsed_by": approval.endorsed_by or "",
        } if approval is not None else None),
    }


def _resolve_match_score(session: Session, sid: str, job_id: str,
                         eligibility: dict) -> int:
    """Prefer the persisted JobMatch score; fall back to skill coverage."""
    row = session.exec(select(JobMatch)
                       .where(JobMatch.student_id == sid,
                              JobMatch.job_id == job_id)).first()
    if row is not None and row.score:
        return int(row.score)
    try:
        return int(eligibility.get("skill_coverage_pct") or 0)
    except (TypeError, ValueError):
        return 0


@router.post("/api/students/{sid}/applications", status_code=201)
def create_application(
    sid: str,
    body: dict,
    request: Request,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
    role: str = Depends(_STUDENT_OR_ADMIN),
):
    """Apply for an on-campus drive job (19.2).

    - eligible + standard tier            -> auto-nominate (no HITL)
    - eligible + dream/super_dream tier   -> ApprovalRequest, awaiting TPO
    - ineligible, no waiver               -> HTTP 422 structured block
    - ineligible, waiver requested        -> ApprovalRequest, awaiting mentor
    """
    enforce_student_scope(request, sid)
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    job_id = str((body or {}).get("job_id") or "").strip()
    if not job_id:
        raise HTTPException(400, "job_id is required")
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    request_waiver = bool((body or {}).get("request_waiver") or False)
    note = str((body or {}).get("note") or "")

    # duplicate active application guard
    existing = session.exec(
        select(JobApplication).where(
            JobApplication.student_id == sid,
            JobApplication.job_id == job_id,
            JobApplication.status.in_(_ACTIVE_STATUSES),
        )).first()
    if existing is not None:
        raise HTTPException(
            409, f"an application for this drive already exists "
                 f"(status: {existing.status})")

    try:
        elig = evaluate_student_eligibility(session, ctx, sid, job_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    tier = job_tier(job)
    match_score = _resolve_match_score(session, sid, job_id, elig)
    eligible = bool(elig.get("eligible"))
    match_row = session.exec(
        select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == job_id)
    ).first()
    initial_favorite = bool(getattr(match_row, "is_favorite", False)) if match_row is not None else False
    now = datetime.now(timezone.utc)
    app_id = f"app-{uuid4().hex[:8]}"

    def _add_audit(action: str, detail: dict) -> None:
        session.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action=action,
                             entity_type="job_application", entity_id=app_id,
                             actor_role=role, detail=detail))

    # ---- branch 1: eligible + standard -> auto-nominate
    if eligible and tier == "standard":
        app_row = JobApplication(
            id=app_id, student_id=sid, job_id=job_id, status="nominated",
            tier=tier, match_score=match_score, eligible_at_apply=True,
            eligibility_checks={"checks": elig.get("checks", []),
                                "cited_sops": elig.get("cited_sops", [])},
            approval_id=None, endorsed_by="", approved_by="", notes=note,
            is_favorite=initial_favorite, created_at=now, updated_at=now)
        session.add(app_row)
        session.add(ProgressEvent(
            id=f"pe-{uuid4().hex[:8]}", student_id=sid, kind="nomination",
            ref_id=job_id, readiness=match_score, at=now))
        _add_audit("application.nominated",
                   {"student_id": sid, "job_id": job_id, "tier": tier,
                    "match_score": match_score})
        session.commit()
        return {"application": _app_payload(app_row, job=job, eligibility=elig),
                "eligibility": elig, "hitl": None}

    # ---- branch 2: eligible + dream/super_dream -> TPO approval required
    if eligible and tier in ("dream", "super_dream"):
        ap = ApprovalRequest(
            id=f"appr-{uuid4().hex[:8]}", kind="job_application_nomination",
            target_id=sid,
            title=f"Drive nomination: {job.title} ({tier})",
            summary=(f"{student.name} is eligible for {tier} tier "
                     f"(readiness {match_score}). TPO approval required "
                     f"[SOP-04 mock gate enforced]."),
            payload={"student_id": sid, "job_id": job_id, "tier": tier,
                     "readiness": match_score},
            status="pending_review", requested_by=role, created_at=now)
        session.add(ap)
        session.flush()  # FK: approval id before child row
        app_row = JobApplication(
            id=app_id, student_id=sid, job_id=job_id,
            status="awaiting_tpo_approval", tier=tier, match_score=match_score,
            eligible_at_apply=True,
            eligibility_checks={"checks": elig.get("checks", []),
                                "cited_sops": elig.get("cited_sops", [])},
            approval_id=ap.id, endorsed_by="", approved_by="", notes=note,
            is_favorite=initial_favorite, created_at=now, updated_at=now)
        session.add(app_row)
        _add_audit("application.submitted",
                   {"student_id": sid, "job_id": job_id, "tier": tier,
                    "approval_id": ap.id, "awaiting": "tpo_approval"})
        session.commit()
        return {"application": _app_payload(app_row, job=job, eligibility=elig,
                                            approval=ap),
                "eligibility": elig,
                "hitl": {"approval_id": ap.id, "awaiting": "tpo_approval"}}

    # ---- branch 3: ineligible without waiver -> structured block
    if not eligible and not request_waiver:
        raise HTTPException(
            422,
            detail={
                "status": "blocked",
                "detail": "Application blocked by placement policy",
                "eligibility": elig,
                "student_id": sid, "job_id": job_id,
                "waiver_hint": "Set request_waiver=true to route this to a "
                               "faculty mentor for endorsement.",
            })

    # ---- branch 4: ineligible + waiver -> mentor endorsement gate
    ap = ApprovalRequest(
        id=f"appr-{uuid4().hex[:8]}", kind="job_application_nomination",
        target_id=sid,
        title=f"Waiver request: {job.title}",
        summary=(f"{student.name} is blocked by SOP checks: "
                 f"{'; '.join(elig.get('reasons', [])[:3]) or 'n/a'}. "
                 f"Faculty mentor endorsement required before TPO review."),
        payload={"student_id": sid, "job_id": job_id, "tier": tier,
                 "waiver_requested": True,
                 "blocked_reasons": elig.get("reasons", []), "note": note},
        status="pending_review", requested_by=role, created_at=now)
    session.add(ap)
    session.flush()  # FK: approval id before child row
    app_row = JobApplication(
        id=app_id, student_id=sid, job_id=job_id,
        status="awaiting_mentor_endorsement", tier=tier,
        match_score=match_score, eligible_at_apply=False,
        eligibility_checks={"checks": elig.get("checks", []),
                            "cited_sops": elig.get("cited_sops", []),
                            "reasons": elig.get("reasons", [])},
        approval_id=ap.id, endorsed_by="", approved_by="", notes=note,
        is_favorite=initial_favorite, created_at=now, updated_at=now)
    session.add(app_row)
    _add_audit("application.waiver_requested",
               {"student_id": sid, "job_id": job_id, "tier": tier,
                "approval_id": ap.id, "blocked_reasons": elig.get("reasons", [])})
    session.commit()
    return {"application": _app_payload(app_row, job=job, eligibility=elig,
                                        approval=ap),
            "eligibility": elig,
            "hitl": {"approval_id": ap.id, "awaiting": "mentor_endorsement"}}


@router.get("/api/students/{sid}/applications")
def list_student_applications(
    sid: str,
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(_STUDENT_OR_ADMIN),
):
    """Application history for one student with job details + HITL status."""
    enforce_student_scope(request, sid)
    rows = session.exec(select(JobApplication)
                        .where(JobApplication.student_id == sid)
                        .order_by(JobApplication.created_at.desc())).all()
    out = []
    for row in rows:
        job = session.get(Job, row.job_id)
        ap = (session.get(ApprovalRequest, row.approval_id)
              if row.approval_id else None)
        out.append(_app_payload(row, job=job, approval=ap))
    return out


@router.get("/api/admin/applications")
def list_admin_applications(
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
):
    """Cohort application ledger: faculty_mentor sees their department only,
    tpo_admin sees everything (19.2)."""
    rows = session.exec(select(JobApplication)
                        .order_by(JobApplication.created_at.desc())).all()
    user: CurrentUser | None = get_current_user_optional(request)
    if user is not None and user.role == "faculty_mentor" and user.department:
        from app.api.routes.admin import _faculty_mentee_ids
        mentee_ids = _faculty_mentee_ids(session, user.department)
        rows = [r for r in rows if r.student_id in mentee_ids]
    out = []
    for row in rows:
        job = session.get(Job, row.job_id)
        student = session.get(Student, row.student_id)
        ap = (session.get(ApprovalRequest, row.approval_id)
              if row.approval_id else None)
        payload = _app_payload(row, job=job, approval=ap)
        payload["student_name"] = student.name if student else ""
        payload["student_branch"] = student.branch if student else ""
        out.append(payload)
    return out


@router.delete("/api/students/{sid}/applications/{app_id}")
@router.post("/api/students/{sid}/applications/{app_id}/withdraw")
def withdraw_student_application(
    sid: str,
    app_id: str,
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(_STUDENT_OR_ADMIN),
):
    """Withdraw or remove an active drive application (accessible to student, faculty mentor, or TPO admin)."""
    enforce_student_scope(request, sid)
    app_row = session.get(JobApplication, app_id)
    if app_row is None or app_row.student_id != sid:
        raise HTTPException(404, "application not found")

    old_status = app_row.status
    app_row.status = "withdrawn"
    app_row.updated_at = datetime.now(timezone.utc)
    app_row.notes = ((app_row.notes or "") + " [Application withdrawn / removed]").strip()
    session.add(app_row)

    # Cancel linked approval request if pending
    if app_row.approval_id:
        ap = session.get(ApprovalRequest, app_row.approval_id)
        if ap and ap.status == "pending_review":
            ap.status = "rejected"
            ap.review_note = "Application withdrawn by candidate or administrator"
            ap.reviewed_at = datetime.now(timezone.utc)
            session.add(ap)

    session.add(
        AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="application.withdrawn",
            entity_type="job_application",
            entity_id=app_id,
            actor_role=role,
            detail={
                "student_id": sid,
                "job_id": app_row.job_id,
                "previous_status": old_status,
                "reason": "closed_or_not_wanted",
            },
        )
    )
    session.commit()
    return {
        "status": "withdrawn",
        "application_id": app_id,
        "student_id": sid,
        "job_id": app_row.job_id,
    }


@router.post("/api/applications/{app_id}/withdraw")
@router.delete("/api/applications/{app_id}")
def withdraw_application_direct(
    app_id: str,
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(_STUDENT_OR_ADMIN),
):
    """Withdraw an active drive application by application ID (with student scope check)."""
    app_row = session.get(JobApplication, app_id)
    if app_row is None:
        raise HTTPException(404, "application not found")
    return withdraw_student_application(
        sid=app_row.student_id,
        app_id=app_id,
        request=request,
        session=session,
        role=role,
    )


@router.delete("/api/admin/applications/{app_id}")
def delete_admin_application(
    app_id: str,
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
):
    """Remove or withdraw an application from the admin ledger."""
    app_row = session.get(JobApplication, app_id)
    if app_row is None:
        raise HTTPException(404, "application not found")

    user = get_current_user_optional(request)
    if user and user.role == "faculty_mentor" and user.department:
        student = session.get(Student, app_row.student_id)
        if student and student.branch and student.branch.upper() != user.department.upper():
            raise HTTPException(403, "not authorized for this student's department")

    old_status = app_row.status
    app_row.status = "withdrawn"
    app_row.updated_at = datetime.now(timezone.utc)
    session.add(app_row)

    if app_row.approval_id:
        ap = session.get(ApprovalRequest, app_row.approval_id)
        if ap and ap.status == "pending_review":
            ap.status = "rejected"
            ap.review_note = "Application withdrawn by faculty or TPO administrator"
            ap.reviewed_at = datetime.now(timezone.utc)
            session.add(ap)

    session.add(
        AuditLog(
            id=f"aud-{uuid4().hex[:8]}",
            action="application.withdrawn",
            entity_type="job_application",
            entity_id=app_id,
            actor_role=role,
            detail={
                "app_id": app_id,
                "student_id": app_row.student_id,
                "job_id": app_row.job_id,
                "previous_status": old_status,
                "reason": "closed_or_not_wanted",
            },
        )
    )
    session.commit()
    return {"status": "withdrawn", "application_id": app_id}

