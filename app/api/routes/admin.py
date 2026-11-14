import hashlib
import math
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.analytics.cohort import (
    branch_readiness_breakdown,
    cohort_executive_kpis,
    enriched_skill_gap_matrix,
    programme_recommendations,
    readiness_distribution,
    role_track_readiness,
    skill_gap_report,
    student_readiness_matrix,
    top_requested_courses,
)
from app.analytics.drive_simulator import simulate_placement_drive
from app.api.deps import get_ctx, get_session
from app.config import settings
from app.eval.benchmark import run_evaluation_benchmark
from app.interview.bank import load_bank
from app.models import (
    AgentRun,
    AgentStep,
    ApprovalRequest,
    AuditLog,
    Course,
    Interview,
    Job,
    JobApplication,
    JobMatch,
    Student,
    UpskillPlan,
)
from app.rag.policy_rag import (
    create_sop,
    delete_sop,
    get_sop,
    list_sops,
    load_sops,
    reset_sops,
    update_sop,
)
from app.security.audit import log_audit
from app.security.rbac import current_role, get_current_user_optional, require_role
from app.taxonomy.branches import normalize_branch

router = APIRouter()


def _top_requested_courses_scoped(session: Session, student_ids: set[str] | None, n: int = 5) -> list[dict]:
    """Compute top requested courses optionally restricted to a subset of students."""
    if student_ids is None:
        return top_requested_courses(session, n=n)
    if not student_ids:
        return []
    from collections import Counter

    from app.models import PlanItem

    plans = session.exec(select(UpskillPlan).where(UpskillPlan.student_id.in_(student_ids))).all()
    plan_ids = {p.id for p in plans}
    if not plan_ids:
        return []
    items = session.exec(select(PlanItem).where(PlanItem.plan_id.in_(plan_ids))).all()
    counts = Counter(it.course_id for it in items if getattr(it, "course_id", None))
    courses = {c.id: c for c in session.exec(select(Course)).all()}
    out = []
    for cid, cnt in counts.most_common(n):
        c = courses.get(cid)
        if c is not None:
            out.append({"course_id": c.id, "title": c.title, "provider": c.provider, "count": cnt})
    return out


def _collect(session: Session, student_ids: set[str] | None = None) -> dict:
    """Aggregate analytics, optionally scoped to a set of student ids.

    ``student_ids=None`` keeps the historical full-cohort behaviour; a concrete
    set restricts every metric to those mentees (faculty department scoping,
    ARCHITECTURE §17.2.B.3).
    """
    precomputed_matches = (
        session.exec(select(JobMatch).where(JobMatch.student_id.in_(student_ids))).all()
        if student_ids is not None and student_ids
        else ([] if student_ids is not None else session.exec(select(JobMatch)).all())
    )
    gap_report = skill_gap_report(session, student_ids=student_ids)
    return {
        "skill_gap_report": gap_report,
        "readiness_distribution": readiness_distribution(session, student_ids=student_ids),
        "top_requested_courses": _top_requested_courses_scoped(session, student_ids),
        "programme_recommendations": programme_recommendations(session, gap_report=gap_report),
        "executive_kpis": cohort_executive_kpis(
            session, precomputed_matches, student_ids=student_ids
        ),
        "branch_readiness": branch_readiness_breakdown(
            session, precomputed_matches, student_ids=student_ids
        ),
        "role_track_readiness": role_track_readiness(
            session, precomputed_matches, student_ids=student_ids
        ),
        "student_matrix": student_readiness_matrix(
            session, precomputed_matches, student_ids=student_ids
        ),
        "enriched_skill_gaps": enriched_skill_gap_matrix(
            session, precomputed_matches, student_ids=student_ids
        ),
    }


def _faculty_mentee_ids(session: Session, department: str | None) -> set[str]:
    """Ids of students in the faculty mentor's department (empty set if unknown)."""
    if not department:
        return set()
    dept = (normalize_branch(department) or str(department)).strip().lower()
    return {
        s.id for s in session.exec(select(Student)).all()
        if (normalize_branch(s.branch or "") or str(s.branch or "")).strip().lower() == dept
    }


@router.get("/api/admin/analytics")
def admin_analytics(request: Request, session: Session = Depends(get_session),
                    role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    user = get_current_user_optional(request)
    if user is not None and user.role == "faculty_mentor":
        # Faculty sees only their department's mentees (§17.2.B.3).
        return _collect(session, _faculty_mentee_ids(session, user.department))
    return _collect(session)


class SimulateDriveRequest(BaseModel):
    """Hypothetical placement drive criteria (M20 simulator)."""
    title: str = "Prospective Campus Drive"
    company: str = "Prospective Recruiter"
    salary_lpa: list[int] = [12, 16]
    min_gpa: float = 7.0
    allowed_branches: list[str] | None = None
    required_skills: list[str] | None = None
    preferred_skills: list[str] | None = None
    max_backlogs: int | None = None


@router.post("/api/admin/simulate-drive")
def simulate_drive(body: SimulateDriveRequest | None = None,
                   request: Request = None,
                   session: Session = Depends(get_session),
                   ctx=Depends(get_ctx),
                   role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """M20: evaluate the cohort against a hypothetical drive (TPO tool)."""
    b = body or SimulateDriveRequest()
    return simulate_placement_drive(
        session, ctx,
        title=b.title, company=b.company, salary_lpa=b.salary_lpa,
        min_gpa=b.min_gpa, allowed_branches=b.allowed_branches,
        required_skills=b.required_skills, preferred_skills=b.preferred_skills,
        max_backlogs=b.max_backlogs,
    )


def _task_queue(ctx):
    """Resolve (lazily creating) the process-wide background task queue."""
    if getattr(ctx, "task_queue", None) is None:
        from app.config import settings
        from app.workers.task_queue import TaskQueueManager
        ctx.task_queue = TaskQueueManager.from_settings(settings)
    return ctx.task_queue


@router.get("/api/tasks")
def list_tasks(request: Request, kind: str | None = None, limit: int = 50,
               role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """M22: list background task states (scrape/ranking/OCR jobs), newest first."""
    queue = _task_queue(request.app.state.ctx)
    tasks = queue.list(kind=kind, limit=limit)
    return {"tasks": [t.to_dict() for t in tasks], "count": len(tasks),
            "backend": queue.backend}


@router.get("/api/tasks/{task_id}")
def get_task(task_id: str, request: Request,
             role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """M22: inspect a single background task by id (404 when unknown)."""
    queue = _task_queue(request.app.state.ctx)
    task = queue.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    return task.to_dict()


# ---------------------------------------------------------------------------
# M23: production observability (liveness / readiness / system telemetry)
# ---------------------------------------------------------------------------

@router.get("/healthz")
def healthz():
    """Public liveness probe: fast, no DB, no auth."""
    return {"status": "ok", "service": "ps06-careeros", "version": "3.0.0"}


@router.get("/readyz")
def readyz(request: Request):
    """Readiness probe: 200 when every subsystem check passes, else 503."""
    ctx = request.app.state.ctx
    checks: dict[str, bool] = {}
    try:
        with Session(ctx.engine) as s:
            s.exec(text("SELECT 1")).one()
        checks["database"] = True
    except Exception:
        checks["database"] = False
    checks["taxonomy_skills"] = len(ctx.taxonomy.all()) >= 124
    checks["sops"] = len(load_sops()) >= 18
    try:
        checks["question_bank"] = len(load_bank()) >= 220
    except Exception:
        checks["question_bank"] = False
    try:
        probe = ctx.embedder.embed("readiness probe vector")
        checks["embedder"] = len(probe) == getattr(ctx.embedder, "dim", len(probe))
    except Exception:
        checks["embedder"] = False
    ready = all(checks.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "degraded", "checks": checks})


def _p95(values: list[int | float]):
    """P95 (nearest-rank) of a list; 0.0 when empty."""
    if not values:
        return 0.0
    import math
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _agent_metrics(session: Session) -> dict:
    runs = session.exec(select(AgentRun)).all()
    total = len(runs)
    completed = sum(1 for r in runs if r.status in ("completed", "awaiting_approval"))
    failed = sum(1 for r in runs if r.status == "failed")
    latencies = [r.total_latency_ms for r in runs]
    grounding = [r.citations.get("grounding_score")
                 for r in runs
                 if isinstance(r.citations, dict)
                 and r.citations.get("grounding_score") is not None]
    tool_counts: dict[str, int] = {}
    for name in session.exec(select(AgentStep.tool_name)).all():
        tool_counts[name] = tool_counts.get(name, 0) + 1
    return {
        "total_runs": total,
        "completed_runs": completed,
        "failed_runs": failed,
        "success_rate": round(completed / total, 4) if total else None,
        "avg_latency_ms": round(sum(latencies) / total, 2) if total else 0.0,
        "p95_latency_ms": _p95(latencies),
        "avg_grounding_score": round(sum(grounding) / len(grounding), 2) if grounding else 0.0,
        "fallback_rate": round(sum(1 for r in runs if r.fallback_triggered) / total, 4) if total else 0.0,
        "tool_invocation_counts": tool_counts,
    }


def _entity_counts(session: Session) -> dict:
    jobs = session.exec(select(Job)).all()
    def count(model):
        return session.exec(select(func.count()).select_from(model)).one()
    return {
        "students": count(Student),
        "jobs": len(jobs),
        "local_jobs": sum(1 for j in jobs if j.source == "local"),
        "scraped_jobs": sum(1 for j in jobs if j.source != "local"),
        "matches": count(JobMatch),
        "plans": count(UpskillPlan),
        "interviews": count(Interview),
        "applications": count(JobApplication),
        "approvals": count(ApprovalRequest),
        "agent_runs": count(AgentRun),
        "audit_logs": count(AuditLog),
    }


@router.get("/api/admin/system-telemetry")
def system_telemetry(request: Request, session: Session = Depends(get_session),
                     role: str = Depends(require_role("tpo_admin"))):
    """M23: agent metrics + LLM telemetry + DB entity counts (TPO only)."""
    from app.config import settings

    ctx = request.app.state.ctx
    raw_llm = (ctx.llm_telemetry.stats() if getattr(ctx, "llm_telemetry", None) is not None
               else {"providers": {}, "totals": {"total_calls": 0, "total_tokens": 0,
                                                  "estimated_cost_usd": 0.0,
                                                  "total_errors": 0, "total_fallbacks": 0}})
    totals = raw_llm.get("totals") or {}
    live_llm = getattr(ctx, "llm", None)
    live_provider = (
        getattr(live_llm, "provider", None)
        or (type(live_llm).__name__ if live_llm is not None else None)
        or getattr(settings, "llm_provider", "mock")
        or "mock"
    )
    llm_stats = {
        **raw_llm,
        "provider": str(live_provider),
        "calls_total": totals.get("total_calls", 0),
        "tokens_total": totals.get("total_tokens", 0),
        "estimated_cost_usd": totals.get("estimated_cost_usd", 0.0),
        "fallbacks_triggered": totals.get("total_fallbacks", 0),
    }
    entities = _entity_counts(session)
    return {
        "agent_metrics": _agent_metrics(session),
        "llm": llm_stats,
        "entities": entities,
        "database_counts": entities,
    }


@router.get("/api/admin/eval-benchmark")
def eval_benchmark(request: Request, session: Session = Depends(get_session),
                   ctx=Depends(get_ctx),
                   role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """M23: run the automated RAG / matching / planner / grounding benchmark."""
    return run_evaluation_benchmark(session, ctx)


@router.get("/admin")
def admin_page(request: Request, session: Session = Depends(get_session),
               role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    user = get_current_user_optional(request)
    mentee_ids: set[str] | None = None
    if user is not None and user.role == "faculty_mentor":
        mentee_ids = _faculty_mentee_ids(session, user.department)
    data = _collect(session, mentee_ids)
    students = ([s for s in session.exec(select(Student)).all() if s.id in mentee_ids]
                if mentee_ids is not None
                else session.exec(select(Student)).all())
    jobs = session.exec(select(Job)).all()
    courses = session.exec(select(Course)).all()
    if mentee_ids is not None:
        interviews = (
            session.exec(
                select(Interview).where(Interview.finished == True, Interview.student_id.in_(mentee_ids))  # noqa: E712
            ).all()
            if mentee_ids else []
        )
        matches = (
            session.exec(select(JobMatch).where(JobMatch.student_id.in_(mentee_ids))).all()
            if mentee_ids else []
        )
    else:
        interviews = session.exec(select(Interview).where(Interview.finished == True)).all()  # noqa: E712
        matches = session.exec(select(JobMatch)).all()
    scores = [int(m.score) for m in matches]
    kpis = {
        "students": len(students),
        "jobs": len(jobs),
        "courses": len(courses),
        "interviews_done": len(interviews),
        "mean_match": round(sum(scores) / len(scores)) if scores else 0,
        "open_approvals": len(session.exec(select(ApprovalRequest)
                                           .where(ApprovalRequest.status == "pending_review")).all()),
    }
    return request.app.state.templates.TemplateResponse(
        request, "admin/analytics.html",
        {**data, "analytics": data, "kpis": kpis, "students": students,
         "sid": "", "active": "admin", "role": current_role(request),
         "user": user, "user_name": (user.name if user else ""),
         "user_department": (user.department if user else None),
         "demo_role_switch": bool(settings.demo_role_switch),
         "auth_mode": settings.auth_mode,
         "error": ""})


_AUDIT_EMAIL_RE = re.compile(r"[A-Za-z0-9._%*+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _scrub_email_text(text: str) -> str:
    return _AUDIT_EMAIL_RE.sub(
        lambda m: f"id-{hashlib.sha256(m.group(0).strip().lower().encode('utf-8')).hexdigest()[:8]}",
        text,
    )


def _scrub_audit_value(val: Any) -> Any:
    if isinstance(val, str):
        return _scrub_email_text(val)
    if isinstance(val, dict):
        return {k: _scrub_audit_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_scrub_audit_value(v) for v in val]
    return val


@router.get("/api/admin/audit")
def audit_log(limit: int = 100, session: Session = Depends(get_session),
              role: str = Depends(require_role("tpo_admin"))):
    """Append-only audit trail (read-only; AuditLog rows are immutable)."""
    rows = session.exec(select(AuditLog).order_by(AuditLog.created_at.desc())
                        .limit(max(1, min(int(limit), 500)))).all()
    return [{"id": r.id, "action": r.action, "entity_type": r.entity_type,
             "entity_id": _scrub_email_text(str(r.entity_id or "")),
             "actor_role": _scrub_email_text(str(r.actor_role or "")),
             "detail": _scrub_audit_value(r.detail),
             "created_at": r.created_at.isoformat()}
            for r in rows]


class PolicyUpdateRequest(BaseModel):
    title: str | None = None
    category: str | None = None
    section: str | None = None
    rule_text: str | None = None
    text: str | None = None
    parameters: dict[str, Any] | None = None
    params: dict[str, Any] | None = None


class PolicyCreateRequest(BaseModel):
    id: str | None = None
    title: str
    category: str = "General Governance"
    section: str = "Section 5.0: Institutional Rule"
    text: str | None = None
    rule_text: str | None = None
    params: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None
    keywords: list[str] | None = None
    applies_to: list[str] | None = None


_PCT_PARAM_KEYS = {
    "min_mock_score",
    "min_coverage_pct",
    "min_skill_match_pct",
    "min_attendance_pct",
}
_NON_NEG_INT_KEYS = {
    "super_dream_max_backlogs",
    "dream_max_backlogs",
    "max_offers_per_tier",
    "acceptance_window_hours",
    "retake_cooldown_hours",
    "first_offense_suspension_days",
    "report_within_days",
}
_LPA_PARAM_KEYS = {
    "standard_max_lpa",
    "dream_min_lpa",
    "dream_max_lpa",
    "super_dream_min_lpa",
}
_VALID_TIERS = {"standard", "dream", "super_dream"}


def _validate_policy_params(existing: dict, incoming: dict[str, Any]) -> dict[str, Any]:
    """Validate incoming SOP parameter updates and keep cross-field invariants sound."""
    base = dict(existing.get("parameters") or {})
    cleaned: dict[str, Any] = {}
    for key, val in incoming.items():
        if key in _PCT_PARAM_KEYS:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise HTTPException(422, detail=f"{key} must be a finite number between 0 and 100")
            num = val if isinstance(val, int) else round(float(val), 2)
            if num < 0 or num > 100:
                raise HTTPException(422, detail=f"{key} must be between 0 and 100")
            cleaned[key] = num
        elif key in _NON_NEG_INT_KEYS:
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise HTTPException(422, detail=f"{key} must be a non-negative integer")
            if "backlog" in key and val > 20:
                raise HTTPException(422, detail=f"{key} cannot exceed 20")
            cleaned[key] = val
        elif key in _LPA_PARAM_KEYS:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise HTTPException(422, detail=f"{key} must be a positive finite LPA value")
            fval = round(float(val), 2)
            if fval <= 0 or fval > 200:
                raise HTTPException(422, detail=f"{key} must be in (0, 200] LPA")
            cleaned[key] = fval
        elif key == "applicable_tiers":
            if not isinstance(val, list):
                raise HTTPException(422, detail="applicable_tiers must be a list of placement tiers")
            tiers: list[str] = []
            for item in val:
                t = str(item or "").strip().lower()
                if t not in _VALID_TIERS:
                    raise HTTPException(422, detail=f"Invalid tier '{item}' in applicable_tiers")
                if t not in tiers:
                    tiers.append(t)
            cleaned[key] = tiers
        elif key in ("min_cgpa",):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise HTTPException(422, detail="min_cgpa must be a finite number in [0, 10]")
            fval = round(float(val), 2)
            if fval < 0 or fval > 10:
                raise HTTPException(422, detail="min_cgpa must be between 0.0 and 10.0")
            cleaned[key] = fval
        else:
            cleaned[key] = val

    merged = {**base, **cleaned}
    if str(existing.get("id", "")).upper() == "SOP-01":
        if "dream_min_lpa" in cleaned and "standard_max_lpa" not in cleaned:
            cleaned["standard_max_lpa"] = float(merged["dream_min_lpa"])
            merged["standard_max_lpa"] = cleaned["standard_max_lpa"]
        if "super_dream_min_lpa" in cleaned and "dream_max_lpa" not in cleaned:
            cleaned["dream_max_lpa"] = round(float(merged["super_dream_min_lpa"]) - 0.01, 2)
            merged["dream_max_lpa"] = cleaned["dream_max_lpa"]
        d_min = float(merged.get("dream_min_lpa", 8.0))
        sd_min = float(merged.get("super_dream_min_lpa", 15.0))
        if d_min >= sd_min:
            raise HTTPException(422, detail="dream_min_lpa must be strictly less than super_dream_min_lpa")
    return cleaned


@router.get("/admin/policies")
def policies_page(
    request: Request,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("faculty_mentor", "tpo_admin")),
):
    """Dedicated TPO Admin Policy Studio page (read-only for Faculty Mentor)."""
    students = session.exec(select(Student)).all()
    user = get_current_user_optional(request)
    if user is not None and user.role == "faculty_mentor" and user.department:
        dept = (normalize_branch(user.department) or user.department).strip().lower()
        students = [
            s for s in students
            if (normalize_branch(s.branch or "") or str(s.branch or "")).strip().lower() == dept
        ]
    sops = list_sops()
    return request.app.state.templates.TemplateResponse(
        request,
        "admin/policies.html",
        {
            "students": students,
            "sid": "",
            "active": "policies",
            "policies": sops,
            "editable": role == "tpo_admin",
            "role": current_role(request),
            "user": user,
            "user_name": (user.name if user else ""),
            "user_department": (user.department if user else None),
            "demo_role_switch": bool(settings.demo_role_switch),
            "auth_mode": settings.auth_mode,
            "error": "",
        },
    )


@router.get("/api/admin/policies")
def list_policies(role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    """Return institutional placement SOPs (editable by TPO Admin, read-only for Faculty Mentor)."""
    sops = load_sops()
    return {"sops": sops, "count": len(sops), "editable": role == "tpo_admin"}


@router.post("/api/admin/policies")
def create_policy(
    body: PolicyCreateRequest,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
):
    """Create a custom institutional SOP (SOP-19+) and persist to policies_overrides.json."""
    try:
        created = create_sop(body.model_dump())
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    log_audit(
        session,
        action="policy.created",
        entity_type="sop",
        entity_id=created["id"],
        actor_role=role,
        detail={"sop_id": created["id"], "title": created.get("title", "")},
        commit=True,
    )
    return {"ok": True, "policy": created, "total": len(list_sops())}


@router.put("/api/admin/policies/{sop_id}")
def update_policy(
    sop_id: str,
    body: PolicyUpdateRequest,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
):
    """Update an institutional SOP's metadata, parameters, or rule_text (TPO Admin only)."""
    existing = get_sop(sop_id)
    if existing is None:
        raise HTTPException(404, detail=f"Unknown SOP: {sop_id}")

    raw_rule = body.rule_text if body.rule_text is not None else body.text
    rule_text = raw_rule.strip() if raw_rule is not None else None
    if raw_rule is not None and not rule_text:
        raise HTTPException(422, detail="rule_text must not be empty")

    raw_params = body.parameters if body.parameters is not None else body.params
    validated_params = None
    if raw_params is not None:
        validated_params = _validate_policy_params(existing, raw_params)

    try:
        updated = update_sop(
            sop_id,
            title=body.title,
            category=body.category,
            section=body.section,
            rule_text=rule_text,
            parameters=validated_params,
        )
    except ValueError as e:
        raise HTTPException(422, detail=str(e))

    log_audit(
        session,
        action="policy.updated",
        entity_type="sop",
        entity_id=updated["id"],
        actor_role=role,
        detail={
            "sop_id": updated["id"],
            "title": updated.get("title"),
            "rule_text": rule_text,
            "parameters": validated_params,
        },
        commit=True,
    )
    return {"status": "updated", "ok": True, "sop": updated, "policy": updated}


@router.delete("/api/admin/policies/{sop_id}")
def delete_policy(
    sop_id: str,
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
):
    """Remove any active institutional SOP (SOP-01..SOP-18 via reversible tombstone, or custom SOP-19+)."""
    if get_sop(sop_id) is None:
        raise HTTPException(404, detail=f"Unknown SOP: {sop_id}")
    try:
        deleted = delete_sop(sop_id)
    except (KeyError, ValueError) as e:
        raise HTTPException(404, detail=str(e))
    log_audit(
        session,
        action="policy.deleted",
        entity_type="sop",
        entity_id=deleted["id"],
        actor_role=role,
        detail={"sop_id": deleted["id"], "title": deleted.get("title", "")},
        commit=True,
    )
    return {"ok": True, "deleted_id": deleted["id"], "total": len(list_sops())}


@router.post("/api/admin/policies/reset")
def reset_policies(
    session: Session = Depends(get_session),
    role: str = Depends(require_role("tpo_admin")),
):
    """Restore core SOP-01..SOP-18 definitions to baseline defaults (TPO Admin only)."""
    sops = reset_sops()
    log_audit(
        session,
        action="policy.reset",
        entity_type="sop",
        entity_id="ALL",
        actor_role=role,
        detail={"count": len(sops)},
        commit=True,
    )
    return {"status": "reset", "sops": sops, "count": len(sops)}


@router.get("/admin/approvals")
def approvals_page(request: Request, session: Session = Depends(get_session),
                   role: str = Depends(require_role("faculty_mentor", "tpo_admin"))):
    students = session.exec(select(Student)).all()
    user = get_current_user_optional(request)
    # §17.2.C.2: a faculty mentor's switcher is scoped to their own department.
    if user is not None and user.role == "faculty_mentor" and user.department:
        dept = user.department.strip().lower()
        students = [s for s in students if str(s.branch or "").strip().lower() == dept]
    return request.app.state.templates.TemplateResponse(
        request, "admin/approvals.html",
        {"students": students, "sid": "", "active": "approvals",
         "role": current_role(request), "user": user,
         "user_name": (user.name if user else ""),
         "user_department": (user.department if user else None),
         "demo_role_switch": bool(settings.demo_role_switch),
         "auth_mode": settings.auth_mode,
         "error": ""})


@router.get("/admin/audit")
def audit_page(request: Request, session: Session = Depends(get_session),
               role: str = Depends(require_role("tpo_admin"))):
    students = session.exec(select(Student)).all()
    user = get_current_user_optional(request)
    return request.app.state.templates.TemplateResponse(
        request, "admin/audit.html",
        {"students": students, "sid": "", "active": "audit",
         "role": current_role(request), "user": user,
         "user_name": (user.name if user else ""),
         "user_department": (user.department if user else None),
         "demo_role_switch": bool(settings.demo_role_switch),
         "auth_mode": settings.auth_mode,
         "error": ""})
