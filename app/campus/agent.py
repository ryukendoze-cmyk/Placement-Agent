"""Agentic Campus JD parser orchestrator: Plan -> Act -> Validate -> Recover."""
from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from sqlmodel import Session

from app.config import settings
from app.llm.mock import MockLLM
from app.security.audit import log_audit
from app.security.guardrails import sanitize_untrusted_text
from app.security.pii import scrub_for_llm
from app.taxonomy.taxonomy import Taxonomy

from .ingest import normalize_jd_text
from .llm_parser import LLMParseError, parse_jd_with_llm, validate_and_ground_llm_json
from .policy import apply_policy_and_defaults
from .static_parser import parse_jd_static


def _merge_hybrid(
    llm_res: dict[str, Any],
    static_res: dict[str, Any],
) -> dict[str, Any]:
    """Field-level hybrid merge: grounded LLM wins; static backfills; conflicts flagged."""
    llm_explicit = set(llm_res.get("_explicit_fields") or [])
    static_explicit = set(static_res.get("_explicit_fields") or [])

    conflicts: list[str] = []
    merged_explicit = llm_explicit | static_explicit

    def _pick_scalar(field: str, llm_val: Any, static_val: Any) -> Any:
        llm_has = field in llm_explicit and llm_val not in (None, "")
        static_has = field in static_explicit and static_val not in (None, "")
        if llm_has and static_has:
            if isinstance(llm_val, float) or isinstance(static_val, float):
                if abs(float(llm_val) - float(static_val)) > 0.05:
                    conflicts.append(f"conflict:{field}")
            elif str(llm_val).strip().lower() != str(static_val).strip().lower():
                conflicts.append(f"conflict:{field}")
            return llm_val
        if llm_has:
            return llm_val
        if static_has:
            return static_val
        return llm_val if llm_val not in (None, "") else static_val

    company = _pick_scalar("company", llm_res.get("company"), static_res.get("company")) or ""
    role_title = _pick_scalar("role_title", llm_res.get("role_title"), static_res.get("role_title")) or ""
    location = _pick_scalar("location", llm_res.get("location"), static_res.get("location")) or ""
    work_mode = _pick_scalar("work_mode", llm_res.get("work_mode"), static_res.get("work_mode"))
    experience_level = _pick_scalar(
        "experience_level", llm_res.get("experience_level"), static_res.get("experience_level")
    )
    ctc_max = _pick_scalar("ctc_lpa_max", llm_res.get("ctc_lpa_max"), static_res.get("ctc_lpa_max"))
    if "ctc_lpa_max" in llm_explicit and llm_res.get("ctc_lpa_max") is not None:
        ctc_min = llm_res.get("ctc_lpa_min")
        salary_disclosed = bool(llm_res.get("salary_disclosed", True))
    elif "ctc_lpa_max" in static_explicit and static_res.get("ctc_lpa_max") is not None:
        ctc_min = static_res.get("ctc_lpa_min")
        salary_disclosed = bool(static_res.get("salary_disclosed", True))
    else:
        ctc_min = None
        salary_disclosed = bool(llm_res.get("salary_disclosed") or static_res.get("salary_disclosed"))

    stipend = (
        llm_res.get("stipend_inr_per_month")
        if llm_res.get("stipend_inr_per_month") is not None
        else static_res.get("stipend_inr_per_month")
    )

    llm_elig = llm_res.get("eligibility") or {}
    static_elig = static_res.get("eligibility") or {}
    min_cgpa = _pick_scalar("min_cgpa", llm_elig.get("min_cgpa"), static_elig.get("min_cgpa"))
    max_backlogs = _pick_scalar(
        "max_active_backlogs",
        llm_elig.get("max_active_backlogs"),
        static_elig.get("max_active_backlogs"),
    )
    grad_year = _pick_scalar(
        "graduation_year",
        llm_elig.get("graduation_year"),
        static_elig.get("graduation_year"),
    )
    llm_branches = list(llm_elig.get("allowed_branches") or [])
    static_branches = list(static_elig.get("allowed_branches") or [])
    if "allowed_branches" in llm_explicit and "allowed_branches" in static_explicit:
        if set(llm_branches) != set(static_branches):
            conflicts.append("conflict:allowed_branches")
        allowed_branches = llm_branches
    elif "allowed_branches" in llm_explicit:
        allowed_branches = llm_branches
    elif "allowed_branches" in static_explicit:
        allowed_branches = static_branches
    else:
        allowed_branches = llm_branches or static_branches

    # Union grounded skills (required beats preferred if present in both)
    req_skills: list[str] = []
    for sid in list(llm_res.get("required_skills") or []) + list(static_res.get("required_skills") or []):
        if sid not in req_skills:
            req_skills.append(sid)
    req_set = set(req_skills)
    pref_skills: list[str] = []
    for sid in list(llm_res.get("preferred_skills") or []) + list(static_res.get("preferred_skills") or []):
        if sid not in req_set and sid not in pref_skills:
            pref_skills.append(sid)

    unmapped = list(static_res.get("unmapped_skill_mentions") or [])
    for m in llm_res.get("unmapped_skill_mentions") or []:
        if m not in unmapped:
            unmapped.append(m)

    # Merge evidence and confidence
    static_meta = static_res.get("parser_metadata") or {}
    llm_meta = llm_res.get("parser_metadata") or {}
    evidence = {**(static_meta.get("evidence") or {}), **(llm_meta.get("evidence") or {})}
    field_conf = dict(static_meta.get("field_confidence") or {})
    for k, v in (llm_meta.get("field_confidence") or {}).items():
        field_conf[k] = max(float(field_conf.get(k, 0.0)), float(v))

    missing = list(static_meta.get("missing_fields_flagged") or [])
    # Only keep non-default flags (apply_policy_and_defaults will re-compute default flags)
    non_default_flags = [
        f for f in missing
        if f in ("percentage_cutoff_unconverted", "salary_undisclosed", "internship_no_sop_tier")
    ] + conflicts

    merged = {
        "company": company,
        "role_title": role_title,
        "location": location,
        "work_mode": work_mode if "work_mode" in merged_explicit else None,
        "ctc_lpa_min": ctc_min,
        "ctc_lpa_max": ctc_max,
        "salary_disclosed": salary_disclosed,
        "stipend_inr_per_month": stipend,
        "policy_tier": None,
        "policy_tier_source": None,
        "experience_level": experience_level if "experience_level" in merged_explicit else None,
        "source_portal": "On-Campus (Local)",
        "source_badge": "On-Campus · JKLU",
        "eligibility": {
            "min_cgpa": min_cgpa,
            "allowed_branches": allowed_branches,
            "max_active_backlogs": max_backlogs if "max_active_backlogs" in merged_explicit else None,
            "graduation_year": grad_year if "graduation_year" in merged_explicit else None,
        },
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "unmapped_skill_mentions": unmapped,
        "raw_jd_text": static_res.get("raw_jd_text") or llm_res.get("raw_jd_text") or "",
        "parser_metadata": {
            "engine_used": "hybrid_merge",
            "fallback_reason": None,
            "attempts": [],
            "confidence_score": 0.0,
            "field_confidence": field_conf,
            "evidence": evidence,
            "missing_fields_flagged": non_default_flags,
            "defaults_applied": [],
        },
    }
    return apply_policy_and_defaults(merged, explicitly_found=merged_explicit)


def parse_campus_jd(
    text: str,
    *,
    mode: str = "auto",
    simulate_failure: str = "none",
    ctx: Any = None,
    trace_id: str | None = None,
    session: Session | None = None,
    actor_role: str = "tpo_admin",
    origin: str = "tpo_smart_jd",
) -> dict[str, Any]:
    """Run Plan -> Act -> Validate -> Recover for campus JD extraction."""
    tid = trace_id or f"tr-{uuid4().hex[:8]}"
    drive_id = f"cd-{uuid4().hex[:8]}"
    mode_norm = (mode or "auto").strip().lower()
    if mode_norm not in ("auto", "llm", "static"):
        mode_norm = "auto"

    sim_norm = (simulate_failure or "none").strip().lower()
    sim_allowed = (
        str(getattr(settings, "auth_mode", "demo")).lower() == "demo"
        and actor_role == "tpo_admin"
    )
    if not sim_allowed:
        sim_norm = "none"

    clean_text = normalize_jd_text(text or "")
    sanitized_text, inj_flags = sanitize_untrusted_text(clean_text, source="campus_jd")
    llm_safe_text, pii_redactions = scrub_for_llm(sanitized_text)

    taxonomy: Taxonomy | None = getattr(ctx, "taxonomy", None) if ctx is not None else None
    if taxonomy is None:
        from app.taxonomy.taxonomy import load_taxonomy
        taxonomy = load_taxonomy()

    if session is not None:
        log_audit(
            session,
            action="campus_drive.jd_extracted",
            entity_type="campus_drive",
            entity_id=drive_id,
            actor_role=actor_role,
            detail={
                "trace_id": tid,
                "char_count": len(sanitized_text),
                "injection_flags": inj_flags,
            },
        )

    attempts: list[dict[str, Any]] = []
    fallback_reason: str | None = None
    result: dict[str, Any] | None = None

    llm = getattr(ctx, "llm", None) if ctx is not None else None
    telemetry = getattr(ctx, "llm_telemetry", None) if ctx is not None else None
    if telemetry is None and llm is not None:
        telemetry = getattr(llm, "tracker", None)
    provider = (
        str(getattr(llm, "provider", "") or getattr(settings, "llm_provider", "mock")).lower()
        if llm is not None
        else str(getattr(settings, "llm_provider", "mock")).lower()
    )

    # Decide execution plan
    if mode_norm == "static":
        t0 = time.perf_counter()
        result = parse_jd_static(sanitized_text, taxonomy)
        ms = round((time.perf_counter() - t0) * 1000)
        attempts.append({"engine": "static", "status": "ok", "latency_ms": ms, "error": None})
        fallback_reason = "forced_static"
        result["parser_metadata"]["engine_used"] = "static_fallback"
        result["parser_metadata"]["fallback_reason"] = fallback_reason
    else:
        # Check simulated failure first (for live demo recovery even when provider is mock)
        if sim_norm in ("timeout", "invalid_json"):
            t0 = time.perf_counter()
            if sim_norm == "timeout":
                err_reason = "llm_timeout"
                err_msg = "Simulated LLM timeout (>6.0s)"
            else:
                try:
                    validate_and_ground_llm_json("{invalid json", sanitized_text, taxonomy)
                except LLMParseError as exc:
                    err_reason = exc.reason
                    err_msg = exc.message
                else:  # pragma: no cover
                    err_reason = "llm_invalid_json"
                    err_msg = "Simulated invalid JSON"
            ms = round((time.perf_counter() - t0) * 1000)
            attempts.append({"engine": "llm", "status": "failed", "latency_ms": ms, "error": err_msg})
            fallback_reason = err_reason
        else:
            is_circuit_open = bool(
                telemetry is not None and telemetry.should_use_fallback(provider)
            )
            is_offline_mock = (
                llm is None
                or isinstance(llm, MockLLM)
                or str(getattr(llm, "provider", "") or "").lower() == "mock"
            )
            if is_circuit_open:
                fallback_reason = "circuit_open"
                attempts.append({
                    "engine": "llm",
                    "status": "failed",
                    "latency_ms": 0,
                    "error": f"Circuit breaker open for provider {provider}",
                })
            elif mode_norm == "auto" and is_offline_mock:
                fallback_reason = "offline_mode"
            else:
                t0 = time.perf_counter()
                try:
                    llm_result = parse_jd_with_llm(llm_safe_text, taxonomy, llm)
                    llm_ms = round((time.perf_counter() - t0) * 1000)
                    attempts.append({"engine": "llm", "status": "ok", "latency_ms": llm_ms, "error": None})

                    t1 = time.perf_counter()
                    static_result = parse_jd_static(sanitized_text, taxonomy)
                    static_ms = round((time.perf_counter() - t1) * 1000)
                    attempts.append({"engine": "static", "status": "ok", "latency_ms": static_ms, "error": None})

                    result = _merge_hybrid(llm_result, static_result)
                except LLMParseError as exc:
                    ms = round((time.perf_counter() - t0) * 1000)
                    attempts.append({"engine": "llm", "status": "failed", "latency_ms": ms, "error": exc.message})
                    fallback_reason = exc.reason
                except Exception as exc:  # pragma: no cover - defensive
                    ms = round((time.perf_counter() - t0) * 1000)
                    attempts.append({"engine": "llm", "status": "failed", "latency_ms": ms, "error": str(exc)})
                    fallback_reason = "llm_error"

        if result is None:
            t_s = time.perf_counter()
            result = parse_jd_static(sanitized_text, taxonomy)
            s_ms = round((time.perf_counter() - t_s) * 1000)
            attempts.append({"engine": "static", "status": "ok", "latency_ms": s_ms, "error": None})
            result["parser_metadata"]["engine_used"] = "static_fallback"
            result["parser_metadata"]["fallback_reason"] = fallback_reason

    result.pop("_explicit_fields", None)
    result["parser_metadata"]["attempts"] = attempts
    result["parser_metadata"]["pii_redactions"] = (
        int(pii_redactions)
        if isinstance(pii_redactions, int)
        else len(pii_redactions or [])
    )

    schema_out = {
        "id": drive_id,
        "trace_id": tid,
        "company": result["company"],
        "role_title": result["role_title"],
        "location": result["location"],
        "work_mode": result["work_mode"],
        "ctc_lpa_min": result["ctc_lpa_min"],
        "ctc_lpa_max": result["ctc_lpa_max"],
        "salary_disclosed": result["salary_disclosed"],
        "stipend_inr_per_month": result["stipend_inr_per_month"],
        "policy_tier": result["policy_tier"],
        "policy_tier_source": result["policy_tier_source"],
        "experience_level": result["experience_level"],
        "source_portal": "On-Campus (Local)",
        "source_badge": "On-Campus · JKLU",
        "origin": origin,
        "eligibility": result["eligibility"],
        "required_skills": result["required_skills"],
        "preferred_skills": result["preferred_skills"],
        "unmapped_skill_mentions": result["unmapped_skill_mentions"],
        "raw_jd_text": result["raw_jd_text"],
        "parser_metadata": result["parser_metadata"],
        "approval_status": "DRAFT",
        "approval_id": None,
        "job_id": None,
        "created_by": actor_role,
        "reviewed_by": None,
    }

    if session is not None:
        log_audit(
            session,
            action="campus_drive.parse_attempt",
            entity_type="campus_drive",
            entity_id=drive_id,
            actor_role=actor_role,
            detail={
                "trace_id": tid,
                "mode": mode_norm,
                "engine_used": schema_out["parser_metadata"]["engine_used"],
                "attempts": attempts,
            },
        )
        if fallback_reason is not None and fallback_reason != "forced_static":
            log_audit(
                session,
                action="campus_drive.fallback_triggered",
                entity_type="campus_drive",
                entity_id=drive_id,
                actor_role=actor_role,
                detail={
                    "trace_id": tid,
                    "fallback_reason": fallback_reason,
                    "attempts": attempts,
                },
            )
        log_audit(
            session,
            action="campus_drive.validated",
            entity_type="campus_drive",
            entity_id=drive_id,
            actor_role=actor_role,
            detail={
                "trace_id": tid,
                "policy_tier": schema_out["policy_tier"],
                "policy_tier_source": schema_out["policy_tier_source"],
                "confidence_score": schema_out["parser_metadata"]["confidence_score"],
                "missing_fields_flagged": schema_out["parser_metadata"]["missing_fields_flagged"],
            },
        )
        log_audit(
            session,
            action="campus_drive.eligibility_inferred",
            entity_type="campus_drive",
            entity_id=drive_id,
            actor_role=actor_role,
            detail={
                "trace_id": tid,
                "eligibility": schema_out["eligibility"],
                "defaults_applied": schema_out["parser_metadata"]["defaults_applied"],
            },
        )
        # Emit a traceable AgentRun + AgentStep episode so Smart JD parses appear in Agent Brain
        try:
            from app.models import AgentRun, AgentStep

            run_id = f"run-{uuid4().hex[:8]}"
            step_plan = [
                {
                    "thought": f"Execute {a.get('engine', 'parser')} JD extraction",
                    "tool": f"smart_jd_{a.get('engine', 'parser')}",
                    "args": {"drive_id": drive_id, "mode": mode_norm},
                }
                for a in attempts
            ] + [
                {
                    "thought": "Validate extracted JD schema and SOP tier boundaries",
                    "tool": "smart_jd_validate_policy",
                    "args": {"drive_id": drive_id, "policy_tier": schema_out["policy_tier"]},
                }
            ]
            run = AgentRun(
                id=run_id,
                student_id=None,
                actor_role=actor_role or "tpo_admin",
                goal=f"Smart JD Parse: {schema_out['company'] or 'Campus'} - {schema_out['role_title'] or 'Drive'}",
                status="completed",
                plan_json=step_plan,
                final_answer=(
                    f"Extracted {schema_out['company'] or 'Campus'} {schema_out['role_title'] or 'Drive'} "
                    f"(Tier: {schema_out['policy_tier']}, [SOP-01], [SOP-03]). "
                    f"Engine: {schema_out['parser_metadata']['engine_used']}, "
                    f"Confidence: {schema_out['parser_metadata'].get('confidence_score')}."
                ),
                citations={
                    "verified": ["SOP-01", "SOP-03"],
                    "unverified": [],
                    "grounding_score": 100,
                    "workflow_kind": "smart_jd",
                    "trace_id": tid,
                },
                total_latency_ms=sum(int(a.get("latency_ms", 0) or 0) for a in attempts),
                provider_used=f"campus_jd:{schema_out['parser_metadata']['engine_used']}",
                fallback_triggered=bool(fallback_reason and fallback_reason != "forced_static"),
            )
            session.add(run)
            session.flush()
            for idx, a in enumerate(attempts, 1):
                ok = a.get("status") == "ok"
                session.add(
                    AgentStep(
                        id=f"step-{uuid4().hex[:8]}",
                        run_id=run_id,
                        step_order=idx,
                        thought=f"Run {a.get('engine')} JD extractor",
                        tool_name=f"smart_jd_{a.get('engine', 'parser')}",
                        tool_input={"drive_id": drive_id, "trace_id": tid, "pii_redactions": len(pii_redactions)},
                        tool_output={"status": a.get("status"), "engine": a.get("engine")},
                        verification_passed=ok,
                        error_recovered=str(a.get("error") or "") if not ok else "",
                        latency_ms=int(a.get("latency_ms", 0) or 0),
                    )
                )
            session.add(
                AgentStep(
                    id=f"step-{uuid4().hex[:8]}",
                    run_id=run_id,
                    step_order=len(attempts) + 1,
                    thought="Validate extracted JD against [SOP-01] CTC tiers and [SOP-03] eligibility.",
                    tool_name="smart_jd_validate_policy",
                    tool_input={"drive_id": drive_id, "required_skills": schema_out["required_skills"]},
                    tool_output={"policy_tier": schema_out["policy_tier"], "confidence": schema_out["parser_metadata"].get("confidence_score")},
                    verification_passed=True,
                    error_recovered=f"fallback:{fallback_reason}" if (fallback_reason and fallback_reason != "forced_static") else "",
                    latency_ms=0,
                )
            )
            session.flush()
        except Exception:
            pass

    return schema_out


def drive_to_dict(drive: Any) -> dict[str, Any]:
    """Serialize a CampusDrive ORM row into the CampusDriveSchema contract dict."""
    return {
        "id": drive.id,
        "trace_id": drive.trace_id,
        "company": drive.company,
        "role_title": drive.role_title,
        "location": drive.location,
        "work_mode": drive.work_mode,
        "ctc_lpa_min": drive.ctc_lpa_min,
        "ctc_lpa_max": drive.ctc_lpa_max,
        "salary_disclosed": bool(drive.salary_disclosed),
        "stipend_inr_per_month": drive.stipend_inr_per_month,
        "policy_tier": drive.policy_tier,
        "policy_tier_source": drive.policy_tier_source,
        "experience_level": drive.experience_level,
        "source_portal": drive.source_portal or "On-Campus (Local)",
        "source_badge": drive.source_badge or "On-Campus · JKLU",
        "origin": drive.origin,
        "eligibility": dict(drive.eligibility or {}),
        "required_skills": list(drive.required_skills or []),
        "preferred_skills": list(drive.preferred_skills or []),
        "unmapped_skill_mentions": list(drive.unmapped_skill_mentions or []),
        "raw_jd_text": drive.raw_jd_text or "",
        "parser_metadata": dict(drive.parser_metadata or {}),
        "cohort_summary": dict(drive.cohort_summary or {}) if getattr(drive, "cohort_summary", None) else {},
        "approval_status": drive.approval_status,
        "approval_id": drive.approval_id,
        "job_id": drive.job_id,
        "created_by": drive.created_by,
        "reviewed_by": drive.reviewed_by,
        "created_at": drive.created_at.isoformat() if getattr(drive, "created_at", None) else None,
        "updated_at": drive.updated_at.isoformat() if getattr(drive, "updated_at", None) else None,
    }


def _resolve_canonical_skills(raw_skills: Any, taxonomy: Taxonomy) -> tuple[list[str], list[str]]:
    """Return (canonical_ids, rejected_unmapped)."""
    kept: list[str] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for item in (raw_skills or []):
        tok = str(item or "").strip()
        if not tok:
            continue
        skill = taxonomy.get_opt(tok)
        if skill is None:
            resolved = taxonomy.resolve(tok)
            skill = taxonomy.get_opt(resolved) if resolved else None
        if skill is None:
            if tok not in rejected:
                rejected.append(tok)
            continue
        if skill.id not in seen:
            seen.add(skill.id)
            kept.append(skill.id)
    return kept, rejected


def validate_drive_payload(
    payload: dict[str, Any],
    taxonomy: Taxonomy,
) -> dict[str, Any]:
    """Validate and ground a CampusDrive payload against taxonomy and SOP-01/SOP-03."""
    from app.taxonomy.branches import normalize_branch

    from .static_parser import CANONICAL_BRANCHES

    raw_elig = dict(payload.get("eligibility") or {})
    req_skills, req_rejected = _resolve_canonical_skills(payload.get("required_skills"), taxonomy)
    pref_raw, pref_rejected = _resolve_canonical_skills(payload.get("preferred_skills"), taxonomy)
    req_set = set(req_skills)
    pref_skills = [s for s in pref_raw if s not in req_set]

    unmapped = [str(x).strip() for x in (payload.get("unmapped_skill_mentions") or []) if str(x).strip()]
    for rej in req_rejected + pref_rejected:
        if rej not in unmapped:
            unmapped.append(rej)

    branches: list[str] = []
    for b in (raw_elig.get("allowed_branches") or []):
        code = normalize_branch(str(b))
        if code in CANONICAL_BRANCHES and code not in branches:
            branches.append(code)

    min_cgpa = raw_elig.get("min_cgpa")
    if min_cgpa is not None and str(min_cgpa).strip() != "":
        min_cgpa = round(float(min_cgpa), 2)
        if min_cgpa <= 0:
            min_cgpa = None
    else:
        min_cgpa = None

    max_backlogs = raw_elig.get("max_active_backlogs")
    if max_backlogs is not None and str(max_backlogs).strip() != "":
        max_backlogs = int(max_backlogs)
    else:
        max_backlogs = None

    grad_year = raw_elig.get("graduation_year")
    if grad_year is not None and str(grad_year).strip() != "":
        grad_year = int(grad_year)
    else:
        grad_year = None

    ctc_min = payload.get("ctc_lpa_min")
    ctc_max = payload.get("ctc_lpa_max")
    ctc_min = float(ctc_min) if ctc_min is not None and str(ctc_min).strip() != "" else None
    ctc_max = float(ctc_max) if ctc_max is not None and str(ctc_max).strip() != "" else None
    salary_disclosed = bool(payload.get("salary_disclosed", ctc_max is not None or ctc_min is not None))

    stipend = payload.get("stipend_inr_per_month")
    stipend = int(stipend) if stipend is not None and str(stipend).strip() != "" else None

    explicit: set[str] = set()
    if max_backlogs is not None:
        explicit.add("max_active_backlogs")
    if grad_year is not None:
        explicit.add("graduation_year")
    if payload.get("work_mode") in ("On-site", "Remote", "Hybrid"):
        explicit.add("work_mode")
    if payload.get("experience_level") in ("Fresher", "Internship"):
        explicit.add("experience_level")
    if branches:
        explicit.add("allowed_branches")

    normalized = {
        "company": str(payload.get("company") or "").strip(),
        "role_title": str(payload.get("role_title") or "").strip(),
        "location": str(payload.get("location") or "").strip(),
        "work_mode": payload.get("work_mode"),
        "ctc_lpa_min": ctc_min,
        "ctc_lpa_max": ctc_max,
        "salary_disclosed": salary_disclosed,
        "stipend_inr_per_month": stipend,
        "experience_level": payload.get("experience_level"),
        "eligibility": {
            "min_cgpa": min_cgpa,
            "allowed_branches": branches,
            "max_active_backlogs": max_backlogs,
            "graduation_year": grad_year,
        },
        "required_skills": req_skills,
        "preferred_skills": pref_skills,
        "unmapped_skill_mentions": unmapped,
        "raw_jd_text": str(payload.get("raw_jd_text") or ""),
        "parser_metadata": dict(payload.get("parser_metadata") or {}),
    }
    return apply_policy_and_defaults(normalized, explicitly_found=explicit)


def run_cohort_rematch(
    ctx: Any,
    *,
    drive_id: str,
    job_id: str,
    trace_id: str,
    actor_role: str = "tpo_admin",
) -> dict[str, Any]:
    """Re-rank all students against the catalog and summarize cohort fit for the new drive."""
    from dataclasses import asdict
    from datetime import datetime, timezone

    from sqlalchemy import delete
    from sqlmodel import select

    from app.matching.engine import job_eligibility
    from app.matching.rank import rank_jobs
    from app.models import CampusDrive, Job, JobMatch, Student, StudentSkill

    with Session(ctx.engine) as s:
        job = s.get(Job, job_id)
        drive = s.get(CampusDrive, drive_id)
        if job is None or drive is None:
            return {
                "students_total": 0,
                "eligible": 0,
                "ineligible_by_reason": {},
                "top_gaps": {},
                "top_matches": [],
            }
        all_jobs = s.exec(select(Job)).all()
        students = s.exec(select(Student)).all()

        eligible_count = 0
        ineligible_by_reason: dict[str, int] = {}
        gap_counts: dict[str, int] = {}
        match_candidates: list[tuple[bool, int, str, dict[str, Any]]] = []

        for student in students:
            sv = ctx.vector_store.get("student", student.id) if ctx.vector_store else None
            if sv is None and (student.resume_text or "").strip():
                sv = ctx.embedder.embed(student.resume_text)
                if ctx.vector_store:
                    ctx.vector_store.upsert("student", student.id, sv)
            if sv is None:
                sv = [0.0] * getattr(ctx.embedder, "dim", 384)

            ss_rows = s.exec(
                select(StudentSkill).where(StudentSkill.student_id == student.id)
            ).all()
            student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
            levels = {r.skill_id: r.level for r in ss_rows}

            ranked = rank_jobs(
                all_jobs,
                sv,
                student_skills,
                ctx.embedder,
                ctx.taxonomy,
                vector_store=ctx.vector_store,
                job_vec_cache=ctx.job_vectors,
                top=None,
                levels=levels,
                student=student,
            )

            # Persist top 10 + ensure the new campus job is always included in JobMatch
            persisted_pairs = list(ranked[:10])
            if not any(j.id == job.id for j, _ in persisted_pairs):
                for j, js in ranked:
                    if j.id == job.id:
                        persisted_pairs.append((j, js))
                        break

            s.execute(delete(JobMatch).where(JobMatch.student_id == student.id))
            for j, js in persisted_pairs:
                s.add(
                    JobMatch(
                        id=f"m-{uuid4().hex[:8]}",
                        student_id=student.id,
                        job_id=j.id,
                        score=js.score,
                        band=js.band,
                        breakdown=asdict(js),
                    )
                )

            js_for_job = next((js for j, js in ranked if j.id == job.id), None)
            elig, reasons = job_eligibility(student, job)
            if elig:
                eligible_count += 1
            else:
                for reason in reasons:
                    ineligible_by_reason[reason] = ineligible_by_reason.get(reason, 0) + 1

            score_val = int(js_for_job.score) if js_for_job is not None else 0
            if js_for_job is not None:
                for miss in js_for_job.missing_required:
                    sk = str(miss.get("skill") or "").strip()
                    if sk:
                        gap_counts[sk] = gap_counts.get(sk, 0) + 1

            match_candidates.append((
                not elig,
                -score_val,
                student.name,
                {
                    "student_id": student.id,
                    "name": student.name,
                    "branch": student.branch,
                    "gpa": float(student.gpa or 0.0),
                    "score": score_val,
                    "eligible": elig,
                },
            ))

        match_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        top_matches = [item[3] for item in match_candidates[:5]]
        top_gaps = dict(sorted(gap_counts.items(), key=lambda kv: (-kv[1], kv[0])))

        cohort_summary = {
            "students_total": len(students),
            "eligible": eligible_count,
            "ineligible_by_reason": ineligible_by_reason,
            "top_gaps": top_gaps,
            "top_matches": top_matches,
        }
        drive.cohort_summary = cohort_summary
        drive.updated_at = datetime.now(timezone.utc)
        s.add(drive)
        log_audit(
            s,
            action="campus_drive.cohort_rematched",
            entity_type="campus_drive",
            entity_id=drive.id,
            actor_role=actor_role,
            detail={
                "trace_id": trace_id,
                "job_id": job_id,
                "cohort_summary": cohort_summary,
            },
        )
        s.commit()
        return cohort_summary


def publish_campus_drive(
    session: Session,
    ctx: Any,
    ap: Any,
) -> dict[str, Any]:
    """Execute HITL approval for kind='campus_drive_publish': upsert Job, embed, rematch cohort."""
    from datetime import datetime, timezone

    from fastapi import HTTPException

    from app.models import CampusDrive, Job

    drive = session.get(CampusDrive, ap.target_id)
    if drive is None:
        raise HTTPException(404, "campus drive not found")

    if not (drive.company or "").strip() or not (drive.role_title or "").strip():
        raise HTTPException(400, "company and role_title are required before publishing")

    validated = validate_drive_payload(drive_to_dict(drive), ctx.taxonomy)

    # Re-apply validated fields onto drive
    drive.policy_tier = validated["policy_tier"]
    drive.policy_tier_source = validated["policy_tier_source"]
    drive.required_skills = validated["required_skills"]
    drive.preferred_skills = validated["preferred_skills"]
    drive.unmapped_skill_mentions = validated["unmapped_skill_mentions"]
    drive.eligibility = validated["eligibility"]
    drive.parser_metadata = validated["parser_metadata"]

    elig = drive.eligibility or {}
    min_cgpa = float(elig.get("min_cgpa") or 0.0)
    allowed_branches = list(elig.get("allowed_branches") or [])
    max_backlogs = elig.get("max_active_backlogs")
    if max_backlogs is not None:
        max_backlogs = int(max_backlogs)

    if drive.salary_disclosed and (drive.ctc_lpa_max is not None or drive.ctc_lpa_min is not None):
        lo = round(float(drive.ctc_lpa_min if drive.ctc_lpa_min is not None else drive.ctc_lpa_max))
        hi = round(float(drive.ctc_lpa_max if drive.ctc_lpa_max is not None else drive.ctc_lpa_min))
        salary_lpa = [lo, hi]
    else:
        salary_lpa = [0, 0]

    job_id = drive.job_id or f"job-campus-{uuid4().hex[:8]}"
    description = (drive.raw_jd_text or "").strip() or f"{drive.role_title} at {drive.company}"

    job = session.get(Job, job_id)
    if job is None:
        job = Job(
            id=job_id,
            title=drive.role_title,
            company=drive.company,
            description=description,
            required_skills=list(drive.required_skills),
            preferred_skills=list(drive.preferred_skills),
            experience_years=0,
            location=drive.location or "",
            remote=(drive.work_mode == "Remote"),
            salary_lpa=salary_lpa,
            salary_disclosed=bool(drive.salary_disclosed),
            min_gpa=min_cgpa,
            allowed_branches=allowed_branches,
            max_backlogs=max_backlogs,
            source="local",
            scraper_engine="campus_drive",
        )
    else:
        job.title = drive.role_title
        job.company = drive.company
        job.description = description
        job.required_skills = list(drive.required_skills)
        job.preferred_skills = list(drive.preferred_skills)
        job.location = drive.location or ""
        job.remote = (drive.work_mode == "Remote")
        job.salary_lpa = salary_lpa
        job.salary_disclosed = bool(drive.salary_disclosed)
        job.min_gpa = min_cgpa
        job.allowed_branches = allowed_branches
        job.max_backlogs = max_backlogs
        job.source = "local"
        job.scraper_engine = "campus_drive"

    session.add(job)

    actor_role = ap.reviewed_by or "tpo_admin"
    drive.job_id = job.id
    drive.approval_status = "PUBLISHED"
    drive.reviewed_by = actor_role
    drive.updated_at = datetime.now(timezone.utc)
    session.add(drive)

    # Embed job description -> vector_store + job_vectors cache
    vec = ctx.embedder.embed(job.description)
    if getattr(ctx, "vector_store", None) is not None:
        ctx.vector_store.upsert("job", job.id, vec)
    ctx.job_vectors[job.id] = vec

    log_audit(
        session,
        action="campus_drive.published",
        entity_type="campus_drive",
        entity_id=drive.id,
        actor_role=actor_role,
        detail={
            "trace_id": drive.trace_id,
            "job_id": job.id,
            "approval_id": ap.id,
            "policy_tier": drive.policy_tier,
        },
    )
    # Commit Job and CampusDrive BEFORE running/submitting cohort rematch in a fresh Session
    session.commit()

    tq = getattr(ctx, "task_queue", None)
    cohort_summary: dict[str, Any] | None = None
    if tq is not None:
        def _rematch_task(payload: dict[str, Any]) -> dict[str, Any]:
            return run_cohort_rematch(
                ctx,
                drive_id=payload["drive_id"],
                job_id=payload["job_id"],
                trace_id=payload["trace_id"],
                actor_role=payload["actor_role"],
            )

        task_id = tq.submit(
            "campus_drive_cohort_rematch",
            _rematch_task,
            {
                "drive_id": drive.id,
                "job_id": job.id,
                "trace_id": drive.trace_id,
                "actor_role": actor_role,
            },
        )
    else:
        cohort_summary = run_cohort_rematch(
            ctx,
            drive_id=drive.id,
            job_id=job.id,
            trace_id=drive.trace_id,
            actor_role=actor_role,
        )
        from app.workers.task_queue import TaskQueueManager, TaskState

        tq = TaskQueueManager.from_settings(settings)
        ctx.task_queue = tq
        task_id = uuid4().hex
        with tq._lock:
            tq._states[task_id] = TaskState(
                task_id=task_id,
                kind="campus_drive_cohort_rematch",
                status="completed",
                progress_pct=100,
                result=cohort_summary,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

    out: dict[str, Any] = {
        "drive_id": drive.id,
        "job_id": job.id,
        "trace_id": drive.trace_id,
        "task_id": task_id,
        "mutated": f"Job({job.id}) + CampusDrive(PUBLISHED) + cohort_rematch({task_id})",
    }
    if cohort_summary is not None:
        out["cohort_summary"] = cohort_summary
    return out
