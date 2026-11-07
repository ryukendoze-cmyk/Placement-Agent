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

REPAIR_SYSTEM = (
    "TASK: repair_step\n"
    "You are an autonomous placement agent repairing a failed tool call. "
    "Given the tool name, input schema, previous args, and error message, "
    "return corrected arguments as JSON: {\"args\": {...}, \"thought\": str}."
)

#: Max dynamically-injected observation steps (bounds runaway replanning).
MAX_INJECTED_STEPS = 3
#: Hard cap on executed steps per run (M16: dynamic queue guard).
MAX_EXECUTED_STEPS = 8

_REF = re.compile(r"\$(first|last|prev)\.([A-Za-z0-9_.]+)|\$([A-Za-z][A-Za-z0-9_]*)\.([A-Za-z0-9_.]+)")

#: Goal markers that mean "this run cares about drive eligibility/application".
_ELIGIBILITY_MARKERS = ("eligib", "apply", "nomina", "drive")


def _sanitize_goal_text(session, raw_text: str, *, entity_id: str, actor_role: str, source: str = "agent_goal") -> str:
    """Scrub PII and neutralize prompt-injection patterns before LLM calls."""
    text = str(raw_text or "")
    try:
        from ..security.pii import scrub_for_llm
        text, _ = scrub_for_llm(text)
    except Exception:
        pass
    try:
        from ..security.prompt_guard import sanitize_untrusted_text
        text, flags = sanitize_untrusted_text(text, source=source)
        if flags and session is not None:
            from ..security.audit import log_audit
            log_audit(session, action="security.prompt_injection_blocked",
                      entity_type="agent_run", entity_id=entity_id,
                      actor_role=actor_role or "student",
                      detail={"flags": flags, "source": source})
    except Exception:
        pass
    return text


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


def _traverse_path(bucket, path: str):
    """Traverse a single key or dotted path (including list indices) inside ``bucket``."""
    if not isinstance(bucket, dict):
        return False, None
    if path in bucket:
        return True, bucket[path]
    if "." in path:
        cur = bucket
        for part in path.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            elif isinstance(cur, list) and part.isdigit() and 0 <= int(part) < len(cur):
                cur = cur[int(part)]
            else:
                return False, None
        return True, cur
    return False, None


def _resolve(value, memory: dict):
    """Resolve `$last.key`, `$prev.key`, `$first.key`, `$<tool>.key`, and nested paths recursively."""
    if isinstance(value, str):
        m = _REF.fullmatch(value.strip())
        if not m:
            return value
        src, key, toolname, toolkey = m.group(1), m.group(2), m.group(3), m.group(4)
        if toolkey:
            bucket = memory.get(toolname) or memory.get(f"_{toolname}")
            key = toolkey
        else:
            src = src or "last"
            non_step_vals = [v for k, v in memory.items() if not str(k).startswith("_")]
            if src == "first":
                bucket = non_step_vals[0] if non_step_vals else None
            else:
                # last (or prev) completed tool output
                bucket = non_step_vals[-1] if non_step_vals else None
        found, resolved_val = _traverse_path(bucket, key)
        if found:
            return resolved_val
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


def _replan_from_observation(llm, goal: str, student_id: str | None,
                             tool_name: str, output: dict,
                             queued_tools: set[str]) -> list[dict]:
    """Use REPLAN_SYSTEM + deterministic SOP guardrails to inject follow-up steps."""
    injected_steps: list[dict] = []
    if llm is not None:
        try:
            user_payload = json.dumps({
                "goal": goal,
                "student_id": student_id,
                "last_tool": tool_name,
                "observation": output,
                "queued_tools": sorted(queued_tools),
            }, default=str)
            raw = llm.complete(system=REPLAN_SYSTEM, user=user_payload, json_mode=True)
            parsed = _safe_json(raw)
            cand = parsed.get("steps") if isinstance(parsed, dict) else None
            if isinstance(cand, list):
                for s in cand:
                    if isinstance(s, dict) and s.get("tool") in TOOL_REGISTRY and s.get("tool") not in queued_tools:
                        injected_steps.append(s)
        except Exception:
            pass

    if injected_steps:
        return injected_steps[:2]

    # Deterministic fallback when LLM returns no steps
    g = str(goal or "").lower()
    wants_drive = any(mk in g for mk in _ELIGIBILITY_MARKERS)
    if (tool_name == "rank_and_explain_jobs" and wants_drive
            and student_id and output.get("top_job_id")
            and "check_drive_eligibility" not in queued_tools):
        return [{
            "thought": "Observation: top match identified; checking institutional SOP eligibility.",
            "tool": "check_drive_eligibility",
            "args": {"student_id": student_id, "job_id": "$rank_and_explain_jobs.top_job_id"},
        }]
    if (tool_name == "check_drive_eligibility"
            and output.get("eligible") is False
            and student_id):
        suggested = " ".join(str(s) for s in (output.get("suggested_actions") or []))
        mock_blocked = "SOP-04" in suggested
        if mock_blocked and "inspect_interview_history" not in queued_tools:
            return [{
                "thought": "Observation: blocked by the mock-interview gate (SOP-04); reviewing interview history to close it.",
                "tool": "inspect_interview_history",
                "args": {"student_id": student_id},
            }]
        if "persist_upskilling_plan" not in queued_tools:
            return [{
                "thought": "Observation: eligibility blocked; building a proficiency-aware remediation plan.",
                "tool": "persist_upskilling_plan",
                "args": {"student_id": student_id, "job_id": "$check_drive_eligibility.job_id"},
            }]
    return []


def _repair_failed_step(llm, tool_name: str, args: dict, student_id: str | None,
                        error_str: str, memory: dict) -> dict:
    """Ask LLM (TASK: repair_step) to repair failed tool arguments."""
    spec = TOOL_REGISTRY.get(tool_name)
    schema = spec.input_schema if spec else {}
    repair = _safe_json(llm.complete(
        system=REPAIR_SYSTEM,
        user=json.dumps({
            "tool": tool_name,
            "schema": schema,
            "args": args,
            "student_id": student_id,
            "error": error_str,
        }, default=str),
        json_mode=True)) if llm is not None else None
    raw_retry_args = None
    if isinstance(repair, dict):
        if isinstance(repair.get("args"), dict):
            raw_retry_args = repair["args"]
        elif isinstance(repair.get("steps"), list) and repair["steps"]:
            first_step = repair["steps"][0]
            if isinstance(first_step, dict) and isinstance(first_step.get("args"), dict):
                raw_retry_args = first_step["args"]
    return _resolve(raw_retry_args if raw_retry_args is not None else args, memory)


def _execute_loop_iter(ctx, session, *, run, goal, student_id: str | None,
                       plan_steps: list, memory: dict,
                       started: float, actor_role: str, initial_error_recovered: str):
    """Real-time generator execution engine for ``run_agent`` and ``run_agent_stream``.

    Yields ``(event_name, payload)`` tuples live at each step transition so SSE
    clients receive ``step_started`` and ``step_completed`` in real time without
    buffering. Mutates and commits ``run`` in place.
    """
    from ..models import AgentStep

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
        yield ("step_started", {"step_order": executed, "tool_name": tool_name,
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
                    retry_args = _repair_failed_step(
                        getattr(ctx, "llm", None), tool_name, args, student_id, str(e), memory
                    )
                    output = call_tool(tool_name, ctx, session, **retry_args)
                    recovered = f"recovered after repair: {e}"
                except Exception as e2:
                    run.fallback_triggered = True
                    recovered = f"tool failed after repair: {e2}"
                    output = {"error": str(e2), "tool": tool_name, "fallback": True}
            # sandbox tools must set requires_hitl -> mark run awaiting approval
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
                and not output.get("error") and not output.get("fallback")
            )
            out_dict = output if isinstance(output, dict) else {}
            if tool_name in memory:
                # Preserve earlier invocation under indexed alias before moving tool_name to tail
                idx = 1
                while f"_{tool_name}_{idx}" in memory:
                    idx += 1
                memory[f"_{tool_name}_{idx}"] = memory.pop(tool_name)
            memory[f"_step_{executed}"] = out_dict
            memory[tool_name] = out_dict
        session.add(step)
        step.latency_ms = int((time.time() - step_start) * 1000)
        if recovered:
            error_recovered = error_recovered or recovered
        yield ("step_completed", {"step_order": executed, "tool_name": tool_name,
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
            return

        # --- M16: observation-driven re-planning (bounded) ----------------
        if injected < MAX_INJECTED_STEPS and executed < MAX_EXECUTED_STEPS:
            queued = {s.get("tool") for s in plan_steps[step_idx:]}
            new_obs_steps = _replan_from_observation(
                getattr(ctx, "llm", None), goal, student_id, tool_name, output, queued
            )
            for ns in new_obs_steps:
                if injected < MAX_INJECTED_STEPS and len(plan_steps) < MAX_EXECUTED_STEPS:
                    plan_steps.append(ns)
                    injected += 1

    run.plan_json = plan_steps

    # 3) Verify + 4) Synthesize
    facts = {k: v for k, v in memory.items()}
    refs = []
    for v in memory.values():
        if isinstance(v, dict):
            for key in ("top_job_id", "job_id", "course_id", "approval_id", "plan_id"):
                if v.get(key):
                    refs.append(str(v[key]))
    refs = list(dict.fromkeys(refs))  # dedupe before synthesis
    known = build_known_entities(session)
    known = set(known) | {run.id}
    for v in memory.values():
        if isinstance(v, dict) and v.get("approval_id"):
            known.add(str(v["approval_id"]))
        if isinstance(v, dict) and v.get("plan_id"):
            known.add(str(v["plan_id"]))
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
    yield ("verification_done", {"citations": run.citations})
    _audit_run(session, run, actor_role, len(plan_steps), plan_steps)
    session.commit()


def _execute_loop(ctx, session, *, run, goal, student_id: str | None,
                  plan_steps: list, memory: dict,
                  started: float, actor_role: str, initial_error_recovered: str,
                  emit=None) -> object:
    """Shared execution engine wrapper for ``run_agent``."""
    _emit = emit if emit is not None else (lambda _n, _p: None)
    for ev_name, ev_payload in _execute_loop_iter(
        ctx, session, run=run, goal=goal, student_id=student_id,
        plan_steps=plan_steps, memory=memory, started=started,
        actor_role=actor_role, initial_error_recovered=initial_error_recovered
    ):
        _emit(ev_name, ev_payload)
    return run


def run_agent(ctx, session, *, goal: str, student_id: str | None = None,
              actor_role: str = "student") -> object:
    from ..models import AgentRun

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

    run_id = f"run-{uuid4().hex[:8]}"
    goal = _sanitize_goal_text(session, goal, entity_id=run_id, actor_role=actor_role, source="agent_goal")

    run = AgentRun(id=run_id, student_id=student_id, actor_role=actor_role,
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
    """Run the agent and yield ``(event_name, payload)`` tuples in real time (Milestone 17)."""
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

    run_id = f"run-{uuid4().hex[:8]}"
    goal = _sanitize_goal_text(session, goal, entity_id=run_id, actor_role=actor_role, source="agent_goal")

    run = AgentRun(id=run_id, student_id=student_id, actor_role=actor_role,
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

    # Stream each step event live as it happens (no buffering)
    for name, payload in _execute_loop_iter(
        ctx, session, run=run, goal=goal, student_id=student_id,
        plan_steps=plan_steps, memory=memory, started=started,
        actor_role=actor_role, initial_error_recovered=student_ref_error
    ):
        yield (name, payload)

    if run.status == "awaiting_input":
        yield ("awaiting_input", {"run_id": run.id,
                                  "question": run.final_answer,
                                  "status": run.status})
        return

    steps = session.exec(_select(AgentStep).where(AgentStep.run_id == run.id)
                         .order_by(AgentStep.step_order)).all()
    from ..api.routes.agent import _run_payload
    yield ("complete", _run_payload(run, steps))


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
    """Resume a paused ``awaiting_input`` run with the user's clarification (M16)."""
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
    reply = _sanitize_goal_text(session, reply, entity_id=run.id, actor_role=run.actor_role, source="agent_reply")

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

    # Restore memory from prior completed tool steps so $first/$last/$<tool> refs resolve!
    memory: dict[str, dict] = {}
    for ps in prior_steps:
        if ps.tool_name not in ("ask_clarifying_question", "user_reply") and isinstance(ps.tool_output, dict) and ps.tool_output:
            memory[ps.tool_name] = ps.tool_output

    awaiting_approval = False
    error_recovered = ""
    known = build_known_entities(session)
    existing_tools = {s.get("tool") for s in (run.plan_json or []) if isinstance(s, dict)}

    try:
        new_steps = _plan(getattr(ctx, "llm", None), run.goal, run.student_id,
                          run.actor_role)
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
    injected = 0
    while step_idx < len(plan_steps) and executed < MAX_EXECUTED_STEPS:
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
                try:
                    retry_args = _repair_failed_step(
                        getattr(ctx, "llm", None), tool_name, args, run.student_id, str(e), memory
                    )
                    output = call_tool(tool_name, ctx, session, **retry_args)
                    recovered = f"recovered after repair: {e}"
                except Exception as e2:
                    run.fallback_triggered = True
                    recovered = f"tool failed after repair: {e2}"
                    output = {"error": str(e2), "tool": tool_name, "fallback": True}
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
            out_dict = output if isinstance(output, dict) else {}
            if tool_name in memory:
                # Preserve earlier invocation under indexed alias before moving tool_name to tail
                idx = 1
                while f"_{tool_name}_{idx}" in memory:
                    idx += 1
                memory[f"_{tool_name}_{idx}"] = memory.pop(tool_name)
            memory[f"_step_{executed}"] = out_dict
            memory[tool_name] = out_dict
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

        if isinstance(output, dict) and injected < MAX_INJECTED_STEPS and executed < MAX_EXECUTED_STEPS:
            queued = {s.get("tool") for s in plan_steps[step_idx:]}
            new_obs_steps = _replan_from_observation(
                getattr(ctx, "llm", None), run.goal, run.student_id, tool_name, output, queued
            )
            for ns in new_obs_steps:
                if injected < MAX_INJECTED_STEPS and executed + len(plan_steps[step_idx:]) < MAX_EXECUTED_STEPS:
                    plan_steps.append(ns)
                    injected += 1

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
