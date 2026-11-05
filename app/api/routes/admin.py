from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import func, text
from sqlmodel import Session, select

from app.analytics.cohort import (
    programme_recommendations,
    readiness_distribution,
    skill_gap_report,
    top_requested_courses,
)
from app.analytics.drive_simulator import simulate_placement_drive
from app.api.deps import get_ctx, get_session
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
from app.rag.policy_rag import load_sops
from app.security.rbac import current_role, get_current_user_optional, require_role

router = APIRouter()


def _collect(session: Session, student_ids: set[str] | None = None) -> dict:
    """Aggregate analytics, optionally scoped to a set of student ids.

    ``student_ids=None`` keeps the historical full-cohort behaviour; a concrete
    set restricts every metric to those mentees (faculty department scoping,
    ARCHITECTURE §17.2.B.3).
    """
    gap_report = skill_gap_report(session, student_ids=student_ids)
    return {
        "skill_gap_report": gap_report,
        "readiness_distribution": readiness_distribution(session, student_ids=student_ids),
        "top_requested_courses": top_requested_courses(session),
        "programme_recommendations": programme_recommendations(session, gap_report=gap_report),
    }


def _faculty_mentee_ids(session: Session, department: str | None) -> set[str]:
    """Ids of students in the faculty mentor's department (empty set if unknown)."""
    if not department:
        return set()
    dept = str(department).strip().lower()
    return {s.id for s in session.exec(select(Student)).all()
            if str(s.branch or "").strip().lower() == dept}


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
        from app.workers.task_queue import TaskQueueManager
        ctx.task_queue = TaskQueueManager()
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
    checks["taxonomy_skills"] = len(ctx.taxonomy.all()) == 124
    checks["sops"] = len(load_sops()) == 18
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
        "audit_logs": count(AuditLog),
    }


@router.get("/api/admin/system-telemetry")
def system_telemetry(request: Request, session: Session = Depends(get_session),
                     role: str = Depends(require_role("tpo_admin"))):
    """M23: agent metrics + LLM telemetry + DB entity counts (TPO only)."""
    ctx = request.app.state.ctx
    llm_stats = (ctx.llm_telemetry.stats() if getattr(ctx, "llm_telemetry", None) is not None
                 else {"providers": {}, "totals": {"total_calls": 0, "total_tokens": 0,
                                                    "estimated_cost_usd": 0.0,
                                                    "total_errors": 0, "total_fallbacks": 0}})
    return {"agent_metrics": _agent_metrics(session), "llm": llm_stats,
            "entities": _entity_counts(session)}


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
         "user_department": (user.department if user else None), "error": ""})


@router.get("/api/admin/audit")
def audit_log(limit: int = 100, session: Session = Depends(get_session),
              role: str = Depends(require_role("tpo_admin"))):
    """Append-only audit trail (read-only; AuditLog rows are immutable)."""
    rows = session.exec(select(AuditLog).order_by(AuditLog.created_at.desc())
                        .limit(max(1, min(int(limit), 500)))).all()
    return [{"id": r.id, "action": r.action, "entity_type": r.entity_type,
             "entity_id": r.entity_id, "actor_role": r.actor_role,
             "detail": r.detail, "created_at": r.created_at.isoformat()}
            for r in rows]


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
         "user_department": (user.department if user else None), "error": ""})


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
         "user_department": (user.department if user else None), "error": ""})
