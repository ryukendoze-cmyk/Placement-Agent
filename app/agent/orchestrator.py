"""Plan -> Execute -> Verify -> Recover autonomous agent loop.

- Planner: LLM `TASK: plan_steps` returns 2-5 structured steps.
- Executor: runs steps sequentially; resolves `$last.<key>` / `<tool>.<key>` refs from a
  working memory of prior step outputs; short-circuits when a student has no skills.
- Verifier: confirms every cited id exists (DB + run memory); flags unverified ones.
- Recovery: retry once with `TASK: repair_step`, else deterministic fallback; sets
  `fallback_triggered` / `error_recovered`. NEVER raises HTTP 500.

Persists AgentRun + AgentStep rows (session-per-request; flush run before steps).
"""
from __future__ import annotations

import json
import re
import time
from uuid import uuid4

from .tools import TOOL_REGISTRY, call_tool
from .verifier import build_known_entities, verify_text


def _tools_doc() -> str:
    from .tools import TOOLS
    lines = ["Available registered tools (you MUST choose ONLY from these exact tool names):"]
    for t in TOOLS:
        schema_str = ", ".join(f"{k}: {v}" for k, v in t.input_schema.items())
        lines.append(f"- {t.name}({schema_str}): {t.description}")
    return "\n".join(lines)


PLAN_SYSTEM = (
    "TASK: plan_steps\n"
    "You are an autonomous placement agent. Given a goal (and optional student), produce a "
    "2-5 step execution plan using ONLY the registered tools below.\n\n"
    + _tools_doc() + "\n\n"
    "Respond ONLY with JSON: "
    '{"steps": [{"thought": str, "tool": str, "args": {...}}]}.'
)

SYNTH_SYSTEM = (
    "TASK: synthesize\n"
    "You are an autonomous placement agent. Synthesize a structured answer from the goal and "
    "the tool facts gathered. Cite concrete ids (job-, course-, SOP-) where they appear. "
    "Respond ONLY with JSON: {\"summary\": str, \"sections\": [{\"heading\": str, \"body\": str}]}."
)

REPLAN_SYSTEM = (
    "TASK: replan_steps\n"
    "You are an autonomous placement agent re-planning after an observation. "
    "Given the goal and the tool facts so far, produce the NEXT 1-3 steps only. "
    "Respond ONLY with JSON: {\"steps\": [{\"thought\": str, \"tool\": str, \"args\": {...}}]}."
)

#: Max dynamically-injected observation steps (bounds runaway replanning).
MAX_INJECTED_STEPS = 3
#: Hard cap on executed steps per run (M16: dynamic queue guard).
MAX_EXECUTED_STEPS = 8

_REF = re.compile(r"\$(first|last|prev)\.([A-Za-z0-9_]+)|\$([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z0-9_]+)")

#: Goal markers that mean "this run cares about drive eligibility/application".
_ELIGIBILITY_MARKERS = ("eligib", "apply", "nomina", "drive")


def _safe_json(raw):
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")).strip()
    try:
        return json.loads(cleaned or raw)
    except (ValueError, TypeError):
        return None


def _resolve(value, memory: dict):
    """Resolve `$last.key`, `$prev.key`, `$first.key`, `$<tool>.key` recursively."""
    if isinstance(value, str):
        m = _REF.fullmatch(value.strip())
        if not m:
            return value
        src, key, toolname, toolkey = m.group(1), m.group(2), m.group(3), m.group(4)
        if toolkey:
            bucket = memory.get(toolname)
            key = toolkey
        else:
            src = src or "last"
            if src == "first":
                bucket = next(iter(memory.values()), None)
            else:
                # last (or prev) completed tool output
                bucket = list(memory.values())[-1] if memory else None
        if isinstance(bucket, dict) and key in bucket:
            return bucket[key]
        # alias top-level keys that come from a *list* of matches
        if isinstance(bucket, dict) and "matches" in bucket and key == "missing_required":
            return bucket.get("missing_required", [])
        return value
    if isinstance(value, dict):
        return {k: _resolve(v, memory) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, memory) for v in value]
    return value


def _plan(llm, goal, student_id, actor_role) -> list[dict]:
    user = json.dumps({"goal": goal, "student_id": student_id, "actor_role": actor_role})
    raw = llm.complete(system=PLAN_SYSTEM, user=user, json_mode=True) if llm else ""
    parsed = _safe_json(raw)
    steps = parsed.get("steps") if isinstance(parsed, dict) else None
    if not isinstance(steps, list) or not steps:
        raise ValueError("planner returned no steps")
    return steps[:5]


def _synthesize(llm, goal, facts, refs) -> dict:
    user = json.dumps({"goal": goal, "facts": facts, "refs": refs})
    raw = llm.complete(system=SYNTH_SYSTEM, user=user, json_mode=True) if llm else ""
    parsed = _safe_json(raw)
    if not isinstance(parsed, dict):
        parsed = {"summary": f"Goal: {goal}", "sections": []}
    parsed.setdefault("summary", f"Goal: {goal}")
    parsed.setdefault("sections", [])
    return parsed


def _execute_loop(ctx, session, *, run, goal, student_id: str | None,
                  plan_steps: list, memory: dict,
                  started: float, actor_role: str, initial_error_recovered: str,
                  emit=None) -> object:
    """Shared execution engine for ``run_agent`` / ``run_agent_stream``.

    Runs the bounded dynamic-queue loop (M16), then verification + synthesis.
    ``emit`` is an optional ``lambda event, payload: None`` sink invoked at each
    lifecycle phase (Milestone 17 streaming); pass ``None`` for the plain path.
    Returns the finalized ``AgentRun`` (already committed).

    IMPORTANT: any step whose tool output sets ``requires_user_input`` pauses the
    run mid-loop (``status=awaiting_input``), commits, and returns early - the
    caller must handle that state identically in both sync and streaming modes.
    """
    from ..models import AgentStep

    _emit = emit if emit is not None else (lambda _n, _p: None)

    # 2) Execute (M16: dynamic queue with observation-driven re-planning).
    # A step that injects new work APPENDS to `plan_steps`, so the while-condition
    # picks it up; step_order stays contiguous because we iterate by index.
    step_idx = 0
    executed = 0
    injected = 0
    error_recovered = initial_error_recovered
    awaiting_approval = False
    while step_idx < len(plan_steps) and executed < MAX_EXECUTED_STEPS:
        raw_step = plan_steps[step_idx]
        step_idx += 1
        executed += 1
        tool_name = raw_step.get("tool") or ""
        args = _resolve(raw_step.get("args") or {}, memory)
        thought = raw_step.get("thought") or f"Call {tool_name}."
        step_start = time.time()
        _emit("step_started", {"step_order": executed, "tool_name": tool_name,
                               "thought": thought, "tool_input": args})
        step = AgentStep(id=f"step-{uuid4().hex[:8]}", run_id=run.id, step_order=executed,
                         thought=thought, tool_name=tool_name, tool_input=args,
                         tool_output={}, verification_passed=False, error_recovered="",
                         latency_ms=0)
        output: dict = {}
        recovered = ""

        # short-circuit: missing student for a tool that requires one
        if student_id is None and "student_id" in (TOOL_REGISTRY.get(tool_name).required() if tool_name in TOOL_REGISTRY else []):
            recovered = f"skipped {tool_name}: no student in context"
            step.error_recovered = recovered

        if not recovered:
            try:
                output = call_tool(tool_name, ctx, session, **args)
            except Exception as e:
                # retry once with a repair prompt, else deterministic fallback
                try:
                    repair = _safe_json(getattr(ctx, "llm", None).complete(
                        system="TASK: repair_step",
                        user=json.dumps({"tool": tool_name, "args": args, "error": str(e)}),
                        json_mode=True)) if getattr(ctx, "llm", None) else None
                    retry_args = _resolve((repair or {}).get("args", args), memory)
                    output = call_tool(tool_name, ctx, session, **retry_args)
                    recovered = f"recovered after repair: {e}"
                except Exception as e2:
                    run.fallback_triggered = True
                    recovered = f"tool failed after repair: {e2}"
                    output = {"error": str(e2), "tool": tool_name, "fallback": True}
            # sandbox tools must set requires_hitl -> mark run awaiting approval
            if isinstance(output, dict) and output.get("requires_hitl"):
                awaiting_approval = True
                # link the freshly-created approval back to this run for provenance
                appr_id = output.get("approval_id")
                if appr_id:
                    from ..models import ApprovalRequest as _AR
                    _ap = session.get(_AR, appr_id)
                    if _ap is not None:
                        _ap.run_id = run.id
                        session.add(_ap)
            step.tool_output = output
            step.error_recovered = recovered
            # step-level grounding: produced usable output with no error/fallback marker
            step.verification_passed = (
                isinstance(output, dict) and bool(output)
                and not output.get("error") and not output.get("fallback")
            )
            memory[tool_name] = output if isinstance(output, dict) else {}
        session.add(step)
        step.latency_ms = int((time.time() - step_start) * 1000)
        if recovered:
            error_recovered = error_recovered or recovered
        _emit("step_completed", {"step_order": executed, "tool_name": tool_name,
                                 "tool_output": output,
                                 "verification_passed": step.verification_passed,
                                 "error_recovered": step.error_recovered,
                                 "latency_ms": step.latency_ms})

        if not isinstance(output, dict):
            continue

        # --- M16: clarification pause -------------------------------------
        if output.get("requires_user_input"):
            run.status = "awaiting_input"
            run.plan_json = plan_steps
            run.final_answer = str(output.get("question") or "Please clarify your request.")
            run.citations = {"verified": [], "unverified": [],
                             "grounding_score": 100}
            run.total_latency_ms = int((time.time() - started) * 1000)
            session.add(run)
            session.flush()
            _audit_run(session, run, actor_role, len(plan_steps), plan_steps)
            session.commit()
            return run

        # --- M16: observation-driven re-planning (bounded) ----------------
        if injected < MAX_INJECTED_STEPS:
            g = str(goal or "").lower()
            wants_drive = any(mk in g for mk in _ELIGIBILITY_MARKERS)
            queued = {s.get("tool") for s in plan_steps[step_idx:]}

            if (tool_name == "rank_and_explain_jobs" and wants_drive
                    and student_id and output.get("top_job_id")
                    and "check_drive_eligibility" not in queued
                    and executed < MAX_EXECUTED_STEPS):
                plan_steps.append({
                    "thought": ("Observation: top match identified; checking "
                                "institutional SOP eligibility."),
                    "tool": "check_drive_eligibility",
                    "args": {"student_id": student_id,
                             "job_id": "$rank_and_explain_jobs.top_job_id"},
                })
                injected += 1
            elif (tool_name == "check_drive_eligibility"
                    and output.get("eligible") is False
                    and student_id
                    and executed < MAX_EXECUTED_STEPS):
                suggested = " ".join(str(s) for s in (output.get("suggested_actions") or []))
                mock_blocked = "SOP-04" in suggested
                queued_after = {s.get("tool") for s in plan_steps[step_idx:]}
                if mock_blocked and "inspect_interview_history" not in queued_after:
                    plan_steps.append({
                        "thought": ("Observation: blocked by the mock-interview gate "
                                    "(SOP-04); reviewing interview history to close it."),
                        "tool": "inspect_interview_history",
                        "args": {"student_id": student_id},
                    })
                    injected += 1
                elif ("persist_upskilling_plan" not in queued_after
                        and executed < MAX_EXECUTED_STEPS):
                    plan_steps.append({
                        "thought": ("Observation: eligibility blocked; building a "
                                    "proficiency-aware remediation plan."),
                        "tool": "persist_upskilling_plan",
                        "args": {"student_id": student_id,
                                 "job_id": "$check_drive_eligibility.job_id"},
                    })
                    injected += 1

    run.plan_json = plan_steps

    # 3) Verify + 4) Synthesize
    facts = {k: v for k, v in memory.items()}
    refs = []
    for v in memory.values():
        if isinstance(v, dict):
            for key in ("top_job_id", "job_id", "course_id", "approval_id"):
                if v.get(key):
                    refs.append(str(v[key]))
    refs = list(dict.fromkeys(refs))  # dedupe before synthesis
    # Register run-generated (ephemeral) ids so the verifier doesn't flag the
    # agent's own outputs as hallucinations: the run id and any approval ids.
    known = build_known_entities(session)
    known = set(known) | {run.id}
    for v in memory.values():
        if isinstance(v, dict) and v.get("approval_id"):
            known.add(str(v["approval_id"]))
    verify_result = verify_text(json.dumps(facts, default=str), known)
    synthesis = _synthesize(getattr(ctx, "llm", None), goal, facts, refs)
    final_answer = (synthesis.get("summary", "") or "") + "\n\n" + "\n".join(
        f"**{s.get('heading','')}**: {s.get('body','')}" for s in synthesis.get("sections", []))
    final_verified = verify_text(final_answer, known)

    run.final_answer = final_verified["annotated_text"]
    run.citations = {
        "verified": sorted(set(verify_result["verified"]) | set(final_verified["verified"])),
        "unverified": sorted(set(verify_result["unverified"]) | set(final_verified["unverified"])),
        "grounding_score": final_verified["grounding_score"],
    }
    run.total_latency_ms = int((time.time() - started) * 1000)
    if error_recovered:
        run.fallback_triggered = run.fallback_triggered or bool(error_recovered)
    run.status = "awaiting_approval" if awaiting_approval else (
        "failed" if run.fallback_triggered and not memory else "completed")
    session.add(run)
    session.flush()
    _emit("verification_done", {"citations": run.citations})
    _audit_run(session, run, actor_role, len(plan_steps), plan_steps)
    session.commit()
    return run


def run_agent(ctx, session, *, goal: str, student_id: str | None = None,
              actor_role: str = "student") -> object:
    from ..models import AgentRun

    started = time.time()
    student_id = (student_id or "").strip() or None
    provider = getattr(ctx, "llm", None)
    provider_name = type(provider).__name__ if provider is not None else "none"

    # Validate the student reference BEFORE touching the DB so a bogus id can never
    # raise IntegrityError (500) at flush time - we degrade to an unsourced run instead.
    student_ref_error = ""
    if student_id is not None:
        from ..models import Student as _Student
        if session.get(_Student, student_id) is None:
            student_ref_error = f"student '{student_id}' not found"
            student_id = None

    run = AgentRun(id=f"run-{uuid4().hex[:8]}", student_id=student_id, actor_role=actor_role,
                   goal=goal, status="running", plan_json=[], final_answer="",
                   citations={}, total_latency_ms=0, provider_used=provider_name,
                   fallback_triggered=False)
    session.add(run)
    session.flush()  # emit run.id before child FKs

    memory: dict[str, dict] = {}
    plan_steps: list[dict] = []

    # 1) Plan
    try:
        plan_steps = _plan(getattr(ctx, "llm", None), goal, student_id, actor_role)
    except Exception as e:
        run.fallback_triggered = True
        student_ref_error = (student_ref_error + "; " if student_ref_error else "") + \
            f"planner failed: {e}; using deterministic fallback plan"
        plan_steps = [{"thought": "Fallback: inspect the student profile.",
                       "tool": "inspect_student_profile", "args": {"student_id": student_id or ""}}]
    run.plan_json = plan_steps
    if student_ref_error:
        run.fallback_triggered = True

    return _execute_loop(ctx, session, run=run, goal=goal, student_id=student_id,
                         plan_steps=plan_steps, memory=memory, started=started,
                         actor_role=actor_role,
                         initial_error_recovered=student_ref_error)


def run_agent_stream(ctx, session, *, goal: str, student_id: str | None = None,
                     actor_role: str = "student"):
    """Run the agent and yield ``(event_name, payload)`` tuples (Milestone 17).

    Same lifecycle as ``run_agent`` but surfaces each phase as a structured event
    so an SSE endpoint can forward them live:

    1. ``run_started``  - run id created
    2. ``plan_created`` - deterministic/LLM plan committed to ``run.plan_json``
    3. ``step_started`` / ``step_completed`` - per tool call
    4. ``verification_done`` - citations verified
    5. ``complete`` - final ``_run_payload``-shaped dict

    A clarification pause (``awaiting_input``) stops the event stream after the
    ``step_completed`` event that triggered it - no ``complete`` is emitted (the
    caller sees ``status=awaiting_input`` via the terminal ``awaiting_input``
    event, then resumes with ``POST /api/agent/runs/{id}/reply``).
    """
    from sqlmodel import select as _select

    from ..models import AgentRun, AgentStep

    started = time.time()
    student_id = (student_id or "").strip() or None
    provider = getattr(ctx, "llm", None)
    provider_name = type(provider).__name__ if provider is not None else "none"

    student_ref_error = ""
    if student_id is not None:
        from ..models import Student as _Student
        if session.get(_Student, student_id) is None:
            student_ref_error = f"student '{student_id}' not found"
            student_id = None

    run = AgentRun(id=f"run-{uuid4().hex[:8]}", student_id=student_id, actor_role=actor_role,
                   goal=goal, status="running", plan_json=[], final_answer="",
                   citations={}, total_latency_ms=0, provider_used=provider_name,
                   fallback_triggered=False)
    session.add(run)
    session.flush()
    yield ("run_started", {"run_id": run.id, "goal": goal, "student_id": student_id})

    memory: dict[str, dict] = {}
    plan_steps: list[dict] = []

    try:
        plan_steps = _plan(getattr(ctx, "llm", None), goal, student_id, actor_role)
    except Exception as e:
        run.fallback_triggered = True
        student_ref_error = (student_ref_error + "; " if student_ref_error else "") + \
            f"planner failed: {e}; using deterministic fallback plan"
        plan_steps = [{"thought": "Fallback: inspect the student profile.",
                       "tool": "inspect_student_profile", "args": {"student_id": student_id or ""}}]
    run.plan_json = plan_steps
    if student_ref_error:
        run.fallback_triggered = True
    yield ("plan_created", {"run_id": run.id, "plan": plan_steps})

    # The shared engine runs synchronously and pushes lifecycle events into
    # ``events`` via the emit sink; we drain them after it returns (the buffer
    # preserves emission order), then yield the terminal ``complete`` event.
    events: list[tuple[str, dict]] = []

    final_run = _execute_loop(ctx, session, run=run, goal=goal, student_id=student_id,
                              plan_steps=plan_steps, memory=memory, started=started,
                              actor_role=actor_role,
                              initial_error_recovered=student_ref_error,
                              emit=lambda n, p: events.append((n, p)))

    for name, payload in events:
        yield (name, payload)

    if final_run.status == "awaiting_input":
        # Clarification pause: terminal event carries the stored question; no
        # `complete` (the run is not finished) - caller resumes via /reply.
        yield ("awaiting_input", {"run_id": final_run.id,
                                  "question": final_run.final_answer,
                                  "status": final_run.status})
        return

    steps = session.exec(_select(AgentStep).where(AgentStep.run_id == final_run.id)
                         .order_by(AgentStep.step_order)).all()
    from ..api.routes.agent import _run_payload
    yield ("complete", _run_payload(final_run, steps))


def _audit_run(session, run, actor_role, step_count, plan_steps=None) -> None:
    """Append an agent.run_completed audit row; never breaks the run."""
    try:
        from ..security.audit import log_audit
        log_audit(session, action="agent.run_completed", entity_type="agent_run",
                  entity_id=run.id, actor_role=actor_role or "student",
                  detail={"status": run.status, "student_id": run.student_id,
                          "steps": int(step_count),
                          "grounding_score": (run.citations or {}).get("grounding_score"),
                          "clarification": run.status == "awaiting_input",
                          "tools": [s.get("tool") for s in (plan_steps or [])
                                    if isinstance(s, dict)]})
    except Exception:  # pragma: no cover - audit must never break a run
        pass


def resume_agent_run(ctx, session, *, run_id: str, user_reply: str) -> object:
    """Resume a paused ``awaiting_input`` run with the user's clarification (M16).

    Appends the reply to the goal, records an ``AgentStep`` documenting the reply,
    re-plans with the clarification included, executes the new steps, verifies
    citations, and marks the run ``completed`` (or ``awaiting_approval``).
    Raises ``ValueError`` for unknown run ids and for runs not awaiting input
    (callers map these to 404 / 409).
    """
    from sqlmodel import select as _select

    from ..models import AgentRun, AgentStep

    run = session.get(AgentRun, run_id)
    if run is None:
        raise ValueError(f"agent run '{run_id}' not found")
    if run.status != "awaiting_input":
        raise ValueError(f"run '{run_id}' is not awaiting input (status={run.status})")

    reply = str(user_reply or "").strip()
    if not reply:
        raise ValueError("reply is required")

    started = time.time()
    run.goal = f"{run.goal} (User clarification: {reply})"
    run.status = "running"
    session.add(run)
    session.flush()

    # record the user's clarification as its own step so the trace is auditable
    prior_steps = session.exec(
        _select(AgentStep).where(AgentStep.run_id == run.id)
        .order_by(AgentStep.step_order)).all()
    next_order = (prior_steps[-1].step_order if prior_steps else 0) + 1
    session.add(AgentStep(id=f"step-{uuid4().hex[:8]}", run_id=run.id,
                          step_order=next_order,
                          thought="User clarification received.",
                          tool_name="user_reply",
                          tool_input={"reply": reply},
                          tool_output={"reply": reply},
                          verification_passed=True, error_recovered="",
                          latency_ms=0))
    session.flush()

    # Re-plan with the clarified goal and execute only the NEW steps.
    memory: dict[str, dict] = {}
    awaiting_approval = False
    error_recovered = ""
    known = build_known_entities(session)
    existing_tools = {s.get("tool") for s in (run.plan_json or []) if isinstance(s, dict)}

    try:
        new_steps = _plan(getattr(ctx, "llm", None), run.goal, run.student_id,
                          run.actor_role)
        # keep the original trace first; append only genuinely new work
        added = [s for s in new_steps if s.get("tool") not in existing_tools] or new_steps
    except Exception as e:
        run.fallback_triggered = True
        error_recovered = f"replanner failed: {e}; using deterministic fallback"
        added = [{"thought": "Fallback after clarification: inspect the student profile.",
                  "tool": "inspect_student_profile",
                  "args": {"student_id": run.student_id or ""}}]

    prior_plan = list(run.plan_json or [])
    plan_steps = prior_plan + added
    step_order = next_order

    step_idx = 0
    executed = 0
    while step_idx < len(plan_steps) and executed < MAX_EXECUTED_STEPS:
        # only execute steps appended in this resume (skip the completed history)
        if step_idx < len(prior_plan):
            step_idx += 1
            continue
        raw_step = plan_steps[step_idx]
        step_idx += 1
        executed += 1
        step_order += 1
        tool_name = raw_step.get("tool") or ""
        args = _resolve(raw_step.get("args") or {}, memory)
        thought = raw_step.get("thought") or f"Call {tool_name}."
        step_start = time.time()
        step = AgentStep(id=f"step-{uuid4().hex[:8]}", run_id=run.id, step_order=step_order,
                         thought=thought, tool_name=tool_name, tool_input=args,
                         tool_output={}, verification_passed=False, error_recovered="",
                         latency_ms=0)
        output: dict = {}
        recovered = ""
        if run.student_id is None and "student_id" in (
                TOOL_REGISTRY.get(tool_name).required() if tool_name in TOOL_REGISTRY else []):
            recovered = f"skipped {tool_name}: no student in context"
            step.error_recovered = recovered
        if not recovered:
            try:
                output = call_tool(tool_name, ctx, session, **args)
            except Exception as e:
                run.fallback_triggered = True
                recovered = f"tool failed: {e}"
                output = {"error": str(e), "tool": tool_name, "fallback": True}
            if isinstance(output, dict) and output.get("requires_hitl"):
                awaiting_approval = True
                appr_id = output.get("approval_id")
                if appr_id:
                    from ..models import ApprovalRequest as _AR
                    _ap = session.get(_AR, appr_id)
                    if _ap is not None:
                        _ap.run_id = run.id
                        session.add(_ap)
            step.tool_output = output
            step.error_recovered = recovered
            step.verification_passed = (
                isinstance(output, dict) and bool(output)
                and not output.get("error") and not output.get("fallback"))
            memory[tool_name] = output if isinstance(output, dict) else {}
        session.add(step)
        step.latency_ms = int((time.time() - step_start) * 1000)
        if recovered:
            error_recovered = error_recovered or recovered

        if isinstance(output, dict) and output.get("requires_user_input"):
            run.plan_json = plan_steps
            run.status = "awaiting_input"
            run.final_answer = str(output.get("question") or "Please clarify your request.")
            run.total_latency_ms = int((time.time() - started) * 1000)
            session.add(run)
            session.flush()
            _audit_run(session, run, run.actor_role, len(plan_steps), plan_steps)
            session.commit()
            return run

    # Verify + Synthesize over the merged facts
    run.plan_json = plan_steps
    refs = []
    for v in memory.values():
        if isinstance(v, dict):
            for key in ("top_job_id", "job_id", "course_id", "approval_id", "plan_id"):
                if v.get(key):
                    refs.append(str(v[key]))
    refs = list(dict.fromkeys(refs))
    known = set(known) | {run.id}
    for v in memory.values():
        if isinstance(v, dict) and v.get("approval_id"):
            known.add(str(v["approval_id"]))
        if isinstance(v, dict) and v.get("plan_id"):
            known.add(str(v["plan_id"]))
    verify_result = verify_text(json.dumps(memory, default=str), known)
    synthesis = _synthesize(getattr(ctx, "llm", None), run.goal, memory, refs)
    final_answer = (synthesis.get("summary", "") or "") + "\n\n" + "\n".join(
        f"**{s.get('heading','')}**: {s.get('body','')}" for s in synthesis.get("sections", []))
    final_verified = verify_text(final_answer, known)

    run.final_answer = final_verified["annotated_text"]
    prior_citations = run.citations or {}
    run.citations = {
        "verified": sorted(set(prior_citations.get("verified", []))
                           | set(verify_result["verified"])
                           | set(final_verified["verified"])),
        "unverified": sorted(set(prior_citations.get("unverified", []))
                             | set(verify_result["unverified"])
                             | set(final_verified["unverified"])),
        "grounding_score": final_verified["grounding_score"],
    }
    run.total_latency_ms = int((time.time() - started) * 1000)
    if error_recovered:
        run.fallback_triggered = True
    run.status = "awaiting_approval" if awaiting_approval else (
        "failed" if run.fallback_triggered and not memory else "completed")
    session.add(run)
    session.flush()
    _audit_run(session, run, run.actor_role, len(plan_steps), plan_steps)
    session.commit()
    return run
