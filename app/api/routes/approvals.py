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

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.models import (
    AgentRun,
    ApprovalRequest,
    AuditLog,
    Course,
    Job,
    JobApplication,
    JobMatch,
    PlanItem,
    ProgressEvent,
    Student,
    StudentSkill,
    UpskillPlan,
)
from app.security.rbac import get_current_user_optional, require_role

router = APIRouter()


def _audit(session: Session, *, action: str, entity_type: str, entity_id: str,
           actor_role: str, detail: dict) -> None:
    session.add(AuditLog(id=f"aud-{uuid4().hex[:8]}", action=action,
                         entity_type=entity_type, entity_id=entity_id,
                         actor_role=actor_role, detail=detail))
    # no commit here: caller commits with the approval mutation (single transaction)


def _payload(ap: ApprovalRequest, session: Session | None = None) -> dict:
    raw_payload = dict(ap.payload or {})
    if session is not None:
        sid = raw_payload.get("student_id") or (
            ap.target_id if str(ap.target_id or "").startswith("stu-") else None
        )
        if sid:
            stu = session.get(Student, sid)
            if stu is not None:
                raw_payload.setdefault("student_name", stu.name)
                raw_payload.setdefault("student_branch", stu.branch)
        jid = raw_payload.get("job_id")
        if jid:
            job = session.get(Job, jid)
            if job is not None:
                raw_payload.setdefault("job_title", job.title)
                raw_payload.setdefault("job_company", job.company)
    return {
        "id": ap.id, "kind": ap.kind, "target_id": ap.target_id,
        "title": ap.title, "summary": ap.summary, "payload": raw_payload,
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
        resolved_course = None
        if not course_id and skill_id:
            # resolve from the course catalog when the agent didn't pass one
            from app.plan.planner import _load_courses, pick_course_for_skill
            resolved_course = pick_course_for_skill(skill_id, _load_courses(session))
            course_id = resolved_course["id"] if resolved_course else None
        course_hours: int | None = None
        if course_id:
            db_course = session.get(Course, course_id)
            if db_course is not None:
                course_hours = int(db_course.hours)
            elif isinstance(resolved_course, dict) and resolved_course.get("hours") is not None:
                course_hours = int(resolved_course["hours"])
        updated = 0
        affected_plan_ids: set[str] = set()
        if skill_id and course_id:
            rows = session.exec(select(PlanItem).where(PlanItem.skill_id == skill_id,
                                                       PlanItem.status == "pending")).all()
            for it in rows:
                it.course_id = course_id
                if course_hours is not None and course_hours > 0:
                    it.hours = course_hours
                it.status = "published"
                session.add(it)
                affected_plan_ids.add(it.plan_id)
                updated += 1
            for pid in affected_plan_ids:
                plan = session.get(UpskillPlan, pid)
                if plan is not None:
                    p_items = session.exec(select(PlanItem).where(PlanItem.plan_id == pid)).all()
                    plan.total_hours = sum(int(pi.hours or 0) for pi in p_items)
                    session.add(plan)
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

    elif kind == "campus_drive_publish":
        from app.campus.agent import publish_campus_drive
        summary.update(publish_campus_drive(session, ctx, ap))

    elif kind == "resume_reupload_request":
        sid = payload.get("student_id") or ap.target_id
        student = session.get(Student, sid) if sid else None
        if student is None:
            raise HTTPException(404, "student not found for resume re-upload approval")
        student.resume_reupload_unlocked = True
        session.add(student)
        summary.update({"student_id": sid, "mutated": "Student.resume_reupload_unlocked=True"})

    else:
        # unknown kind: record a no-op approval (still audited)
        summary.update({"mutated": "none", "note": "no handler for kind"})

    return summary


def _faculty_approval_allowed(request: Request | None, session: Session, ap: ApprovalRequest) -> bool:
    """Return True when the caller is not a faculty_mentor or the approval matches their department."""
    if request is None:
        return True
    user = get_current_user_optional(request)
    if user is None or user.role != "faculty_mentor":
        return True
    dept = (user.department or "").strip().lower()
    if not dept:
        return True
    payload = ap.payload or {}
    sid = payload.get("student_id")
    if not sid and ap.target_id and ap.target_id.startswith("stu-"):
        sid = ap.target_id
    if sid:
        stu = session.get(Student, sid)
        if stu is not None and (stu.branch or "").strip():
            return stu.branch.strip().lower() == dept
    if ap.kind == "campus_drive_publish":
        cd_payload = payload.get("campus_drive") if isinstance(payload.get("campus_drive"), dict) else payload
        elig = cd_payload.get("eligibility") if isinstance(cd_payload.get("eligibility"), dict) else {}
        branches = elig.get("allowed_branches") or []
        if branches:
            return any(str(b or "").strip().lower() == dept for b in branches)
    return True


def _enforce_faculty_approval_scope(request: Request | None, session: Session, ap: ApprovalRequest) -> None:
    if not _faculty_approval_allowed(request, session, ap):
        user = get_current_user_optional(request) if request is not None else None
        dept = user.department if user else "department"
        raise HTTPException(
            403,
            f"Access denied: Faculty mentor ({dept}) may not act on out-of-department approvals",
        )


@router.get("/api/approvals")
def list_approvals(status: str | None = None,
                   session: Session = Depends(get_session),
                   role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
                   request: Request = None):  # type: ignore[assignment]
    q = select(ApprovalRequest).order_by(ApprovalRequest.created_at.desc())
    if status:
        q = q.where(ApprovalRequest.status == status)
    rows = session.exec(q).all()
    return [_payload(a, session=session) for a in rows if _faculty_approval_allowed(request, session, a)]


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
    return _payload(ap, session=session)


@router.post("/api/approvals/{appr_id}/approve")
def approve(appr_id: str, payload: dict | None = None,
            session: Session = Depends(get_session), ctx=Depends(get_ctx),
            role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
            request: Request = None):  # type: ignore[assignment]
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    _enforce_faculty_approval_scope(request, session, ap)
    if ap.status == "approved":
        return _payload(ap, session=session)   # idempotent
    if ap.status == "rejected":
        raise HTTPException(409, "approval already rejected")
    note = str((payload or {}).get("note") or "")
    if ap.status == "pending_review":
        # Separation of duties: faculty mentors endorse Tier-1 applications,
        # while final Tier-2 drive nomination, campus-wide bootcamp/drive publish,
        # or faculty-originated requests require TPO admin sign-off.
        if role == "faculty_mentor" and (
            ap.kind in (
                "job_application_nomination",
                "nomination",
                "campus_drive_publish",
                "cohort_bootcamp_publish",
                "resume_reupload_request",
            )
            or (ap.endorsed_by or "").strip() == "faculty_mentor"
            or (ap.requested_by or "").strip() == "faculty_mentor"
        ):
            raise HTTPException(
                403,
                "TPO administrator approval is required for final drive nomination, cohort publish, or self-requested actions (separation of duties)",
            )
        # M19 two-tier guard: a waiver application must be endorsed by a
        # faculty mentor before the TPO can approve it.
        linked = _linked_application(session, ap)
        if (linked is not None
                and linked.status == "awaiting_mentor_endorsement"
                and not (ap.endorsed_by or "").strip()):
            raise HTTPException(
                400, "faculty mentor endorsement required before TPO approval")
        # Atomic HITL execution: set reviewed_by so execute_approved can record it
        # on linked rows; if execution fails or raises, rollback completely so
        # status NEVER flips to approved, record approval.approve_failed, and raise 400/409.
        ap.reviewed_by = role
        ap.review_note = note
        ap.reviewed_at = datetime.now(timezone.utc)
        session.add(ap)
        try:
            summary = execute_approved(session, ctx, ap)
        except HTTPException as he:
            session.rollback()
            err_msg = str(he.detail)
            _audit(session, action="approval.approve_failed", entity_type="approval",
                   entity_id=appr_id, actor_role=role,
                   detail={"kind": ap.kind, "target_id": ap.target_id, "note": note, "error": err_msg})
            session.commit()
            raise HTTPException(
                status_code=409 if he.status_code == 409 else 400,
                detail=f"Approval execution failed: {err_msg}",
            ) from he
        except Exception as e:  # pragma: no cover - defensive
            session.rollback()
            err_msg = str(e)
            _audit(session, action="approval.approve_failed", entity_type="approval",
                   entity_id=appr_id, actor_role=role,
                   detail={"kind": ap.kind, "target_id": ap.target_id, "note": note, "error": err_msg})
            session.commit()
            raise HTTPException(status_code=400, detail=f"Approval execution failed: {err_msg}") from e

        if isinstance(summary, dict) and summary.get("error"):
            err_msg = str(summary["error"])
            session.rollback()
            _audit(session, action="approval.approve_failed", entity_type="approval",
                   entity_id=appr_id, actor_role=role,
                   detail={"kind": ap.kind, "target_id": ap.target_id, "note": note, "error": err_msg})
            session.commit()
            raise HTTPException(status_code=400, detail=f"Approval execution failed: {err_msg}")

        ap.status = "approved"
        ap.reviewed_by = role
        ap.review_note = note
        ap.reviewed_at = datetime.now(timezone.utc)
        session.add(ap)
        _audit(session, action="approval.approved", entity_type="approval",
               entity_id=ap.id, actor_role=role,
               detail={"kind": ap.kind, "target_id": ap.target_id, "note": note,
                       "mutated": summary.get("mutated", "")})
        session.commit()
        return {**_payload(ap, session=session), "executed": summary}
    # already approved (idempotent) - just return; no new audit row
    return _payload(ap, session=session)


@router.post("/api/approvals/{appr_id}/reject")
def reject(appr_id: str, payload: dict | None = None,
           session: Session = Depends(get_session),
           role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
           request: Request = None):  # type: ignore[assignment]
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    _enforce_faculty_approval_scope(request, session, ap)
    if role == "faculty_mentor" and ap.kind in (
        "campus_drive_publish",
        "cohort_bootcamp_publish",
        "resume_reupload_request",
    ):
        raise HTTPException(
            403,
            "TPO administrator role is required to reject institutional campus drives, cohort bootcamps, or resume re-upload requests",
        )
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
    if ap.kind == "campus_drive_publish":
        from app.models import CampusDrive
        drive = session.get(CampusDrive, ap.target_id)
        if drive is not None:
            drive.approval_status = "REJECTED"
            drive.reviewed_by = role
            drive.updated_at = datetime.now(timezone.utc)
            session.add(drive)
    _audit(session, action="approval.rejected", entity_type="approval",
           entity_id=ap.id, actor_role=role,
           detail={"kind": ap.kind, "target_id": ap.target_id, "note": note})
    session.commit()
    return _payload(ap, session=session)


@router.post("/api/approvals/{appr_id}/endorse")
def endorse(appr_id: str, payload: dict | None = None,
            session: Session = Depends(get_session),
            role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
            request: Request = None):  # type: ignore[assignment]
    """M19 tier-1 decision: faculty mentor endorsement (before TPO approval).

    Sets endorsed_by/endorsed_at on the approval and advances a linked
    JobApplication from awaiting_mentor_endorsement to awaiting_tpo_approval.
    """
    ap = session.get(ApprovalRequest, appr_id)
    if ap is None:
        raise HTTPException(404, "approval not found")
    _enforce_faculty_approval_scope(request, session, ap)
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
    return _payload(ap, session=session)
