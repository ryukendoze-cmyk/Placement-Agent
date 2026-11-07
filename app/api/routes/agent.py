from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlmodel import Session, select

from app.agent.orchestrator import run_agent, run_agent_stream
from app.api.deps import get_ctx, get_session
from app.models import AgentRun, AgentStep
from app.security.ratelimit import check_rate_limit
from app.security.rbac import enforce_student_scope, get_current_user_optional

router = APIRouter()


def _run_payload(run: AgentRun, steps: list[AgentStep] | None = None) -> dict:
    return {
        "id": run.id,
        "student_id": run.student_id,
        "actor_role": run.actor_role,
        "goal": run.goal,
        "status": run.status,
        "plan": run.plan_json,
        "final_answer": run.final_answer,
        "citations": run.citations or {},
        "total_latency_ms": run.total_latency_ms,
        "provider_used": run.provider_used,
        "fallback_triggered": run.fallback_triggered,
        "created_at": run.created_at.isoformat(),
        "steps": [
            {
                "id": s.id,
                "step_order": s.step_order,
                "thought": s.thought,
                "tool_name": s.tool_name,
                "tool_input": s.tool_input,
                "tool_output": s.tool_output,
                "verification_passed": s.verification_passed,
                "error_recovered": s.error_recovered,
                "latency_ms": s.latency_ms,
            }
            for s in (steps if steps is not None else [])
        ],
    }


@router.post("/api/agent/run", status_code=201)
def start_run(payload: dict, request: Request,
              session: Session = Depends(get_session), ctx=Depends(get_ctx),
              _rl: None = Depends(check_rate_limit("agent_run"))):
    goal = (payload or {}).get("goal") or ""
    if not goal.strip():
        raise HTTPException(400, "goal is required")
    student_id = (payload or {}).get("student_id")
    actor_role = (payload or {}).get("actor_role", "student")
    user = get_current_user_optional(request)
    if user is not None:
        actor_role = user.role
        if user.role == "student":
            if student_id and student_id != user.student_id:
                enforce_student_scope(request, student_id)
            student_id = user.student_id
    run = run_agent(ctx, session, goal=goal, student_id=student_id, actor_role=actor_role)
    steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                         .order_by(AgentStep.step_order)).all()
    return JSONResponse(status_code=201, content=_run_payload(run, steps))


@router.post("/api/agent/run/stream")
def start_run_stream(payload: dict, request: Request,
                     session: Session = Depends(get_session), ctx=Depends(get_ctx),
                     _rl: None = Depends(check_rate_limit("agent_run"))):
    """Run the agent and stream lifecycle events as SSE (Milestone 17).

    Emits ``run_started``, ``plan_created``, ``step_started``/``step_completed``
    (per tool), ``verification_done``, and ``complete``; a clarification pauses
    the run with an ``awaiting_input`` event instead of ``complete``. The run is
    persisted in the DB like the non-streaming path, so ``GET /api/agent/runs/{id}``
    reflects the streamed state after the stream closes.
    """
    from app.api.routes.jobs import SSE_HEADERS, _sse_chunk

    goal = (payload or {}).get("goal") or ""
    if not goal.strip():
        raise HTTPException(400, "goal is required")
    student_id = (payload or {}).get("student_id")
    actor_role = (payload or {}).get("actor_role", "student")
    user = get_current_user_optional(request)
    if user is not None:
        actor_role = user.role
        if user.role == "student":
            if student_id and student_id != user.student_id:
                enforce_student_scope(request, student_id)
            student_id = user.student_id

    def event_stream():
        try:
            for name, data in run_agent_stream(ctx, session, goal=goal,
                                               student_id=student_id,
                                               actor_role=actor_role):
                yield _sse_chunk(name, data)
        except Exception as e:  # pragma: no cover - never break the SSE frame
            yield _sse_chunk("error", {"message": str(e)})
        finally:
            session.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=SSE_HEADERS)


@router.get("/api/agent/runs")
def list_runs(request: Request, session: Session = Depends(get_session)):
    stmt = select(AgentRun)
    user = get_current_user_optional(request)
    if user is not None and user.role == "student":
        stmt = stmt.where(AgentRun.student_id == user.student_id)
    runs = session.exec(stmt.order_by(AgentRun.created_at.desc())).all()
    # the Agent Brain drawer renders from this list, so each run needs its steps
    all_steps = session.exec(select(AgentStep).order_by(AgentStep.step_order)).all()
    steps_by_run: dict[str, list[AgentStep]] = {}
    for s in all_steps:
        steps_by_run.setdefault(s.run_id, []).append(s)
    return [_run_payload(r, steps_by_run.get(r.id, [])) for r in runs]


@router.get("/api/agent/runs/{run_id}")
def get_run(run_id: str, request: Request, session: Session = Depends(get_session)):
    run = session.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(404, "agent run not found")
    user = get_current_user_optional(request)
    if user is not None and user.role == "student" and run.student_id != user.student_id:
        raise HTTPException(
            403,
            "Access denied: Students may only access their own placement records",
        )
    steps = session.exec(select(AgentStep).where(AgentStep.run_id == run_id)
                         .order_by(AgentStep.step_order)).all()
    return _run_payload(run, steps)


@router.post("/api/agent/runs/{run_id}/reply")
def reply_run(run_id: str, payload: dict, request: Request,
              session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    """Resume an ``awaiting_input`` run with the user's clarification (M16)."""
    from app.agent.orchestrator import resume_agent_run

    run = session.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(404, "agent run not found")
    if run.student_id:
        enforce_student_scope(request, run.student_id)
    if run.status != "awaiting_input":
        raise HTTPException(409, f"run is not awaiting input (status={run.status})")
    reply = str((payload or {}).get("reply") or "").strip()
    if not reply:
        raise HTTPException(400, "reply is required")
    try:
        run = resume_agent_run(ctx, session, run_id=run_id, user_reply=reply)
    except ValueError as e:
        raise HTTPException(409, str(e))
    steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                         .order_by(AgentStep.step_order)).all()
    return JSONResponse(status_code=200, content=_run_payload(run, steps))
