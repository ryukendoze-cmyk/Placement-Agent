import logging
import re
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.db import sort_students_alphabetically
from app.models import Student
from app.rag.policy_rag import (
    evaluate_student_eligibility,
    search_sops,
    sop_ids,
    verify_policy_citations,
)
from app.security.ratelimit import check_rate_limit
from app.security.rbac import current_role, enforce_student_scope, get_current_user_optional

router = APIRouter()
logger = logging.getLogger(__name__)

QUICK_PROMPTS = [
    "Rank my top job matches",
    "What score do I need for Dream drives?",
    "Am I eligible for Super-Dream offers?",
    "What skills should I prioritize next?",
    "How many backlogs are allowed?",
]

COPILOT_SYSTEM = (
    "TASK: copilot_answer\n"
    "You are the JKLU Placement Copilot, an intelligent AI placement assistant. "
    "You have access to the student's grounded profile (verified competencies & proficiency levels, "
    "explicitly favorited/starred jobs in `student.favorite_jobs`, visible ranked job matches in `student.top_matches`, "
    "upskilling roadmaps, application statuses, GPA, backlogs, and mock interview performance) "
    "as well as retrieved institutional placement policies (SOP clauses).\n\n"
    "CRITICAL PRESENTATION & GROUNDING RULES (STUDENT-FACING):\n"
    "1. WRITE JOB NAMES, NOT IDS: Always refer to jobs naturally using their clear human-readable title and company name "
    "(e.g., 'Backend Engineer (Python) at TechNova'). "
    "NEVER write, mention, or output raw database job IDs (such as 'job-ext-127d15620c', 'job-ext-...', or 'job-001') in your answer. "
    "Normal students do not know what these internal IDs mean—do NOT mention them.\n"
    "2. FAVORITE / SAVED / STARRED JOBS ARE STRICTLY `student.favorite_jobs`: ONLY jobs listed in `student.favorite_jobs` "
    "have been marked as favorites by the student. NEVER treat `student.top_matches` or `student.upskill_plan.target_role` "
    "as favorited jobs unless they also appear in `student.favorite_jobs`. If the student asks about their favorite jobs "
    "and `student.favorite_jobs` is empty, state clearly that they have not marked any jobs as favorites on the Matches page yet.\n"
    "3. ZERO FABRICATION OF SKILLS OR FACTS: ONLY state required skills, preferred skills, missing skills, match scores, tiers, "
    "or eligibility blockers that explicitly appear in `student.favorite_jobs`, `student.top_matches`, `student.skills`, "
    "`student.upskill_plan`, `student.upskill_plans`, `eligibility`, or `sources`. NEVER invent, assume, or guess missing skills "
    "(e.g., TensorFlow, PyTorch, Kubernetes, AWS) from a job title when they are absent from `required_skills`/`missing_required`/`missing_preferred`! "
    "If `missing_required` is empty, explicitly state that no required skills are missing and focus on proficiency levels, verification status, "
    "or SOP blockers (such as SOP-03 GPA/backlogs, SOP-04 mock interview, or SOP-09 branch limits).\n"
    "4. When explaining job ranking, matches, or recommendations: compare compatibility, tier, matched skills, missing skills, and blockers.\n"
    "5. Ground all policy statements in retrieved SOP clauses and cite them explicitly as [SOP-XX] (e.g. [SOP-01], [SOP-04]). "
    "Do NOT cite job IDs.\n\n"
    "Respond ONLY with JSON: "
    '{"answer": str, "cited_sops": list[str]}.'
)


def _skill_display_name(skill_id: str, ctx=None) -> str:
    if ctx is not None and getattr(ctx, "taxonomy", None):
        t_skill = ctx.taxonomy.get_opt(skill_id)
        if t_skill:
            return t_skill.name
    return skill_id


def _format_job_for_copilot(
    *,
    title: str,
    company: str = "",
    tier: str = "standard",
    score: int = 0,
    band: str = "early",
    eligible: bool | None = None,
    eligibility_reasons: list[str] | None = None,
    required_skills: list[str] | None = None,
    matched_required: list[str] | None = None,
    missing_required: list[str] | None = None,
    missing_preferred: list[str] | None = None,
    proficiency_gaps: list[dict] | None = None,
    is_favorite: bool = False,
) -> dict:
    return {
        "title": title,
        "company": company or "",
        "tier": tier,
        "score": int(score or 0),
        "band": band,
        "is_favorite": bool(is_favorite),
        "eligible": eligible,
        "eligibility_reasons": list(eligibility_reasons or []),
        "required_skills": list(required_skills or []),
        "matched_required": list(matched_required or []),
        "missing_required": list(missing_required or []),
        "missing_preferred": list(missing_preferred or []),
        "proficiency_gaps": list(proficiency_gaps or []),
    }


def _student_context(session, student_id: str | None, ctx=None) -> dict:
    if not student_id:
        return {}
    from sqlmodel import select

    from app.matching.engine import is_job_visible_to_student, job_eligibility, job_tier
    from app.matching.rank import rank_jobs
    from app.models import (
        Interview,
        Job,
        JobApplication,
        JobMatch,
        PlanItem,
        Student,
        StudentSkill,
        UpskillPlan,
    )

    student = session.get(Student, student_id)
    if student is None:
        return {}

    # Mock interviews
    rows = session.exec(select(Interview).where(Interview.student_id == student_id,
                                                Interview.finished == True)).all()  # noqa: E712
    scores = [int(r.score) for r in rows if r.score is not None]

    # Candidate Technical Competencies (status reflects StudentSkill.verification_status)
    ss_rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == student_id)).all()
    skills_list = []
    student_skills_map: dict[str, list[str]] = {}
    levels_map: dict[str, str] = {}
    verification_statuses: dict[str, str] = {}
    for ss in ss_rows:
        sk_name = ss.skill_id
        category = "General"
        if ctx is not None and hasattr(ctx, "taxonomy") and ctx.taxonomy:
            t_skill = ctx.taxonomy.get_opt(ss.skill_id)
            if t_skill:
                sk_name = t_skill.name
                category = t_skill.category
        vstatus = getattr(ss, "verification_status", None) or "claimed"
        skills_list.append({
            "id": ss.skill_id,
            "name": sk_name,
            "level": ss.level,
            "category": category,
            "mentions": ss.mentions,
            "verification_status": vstatus,
        })
        student_skills_map[ss.skill_id] = list(ss.evidence or [])
        levels_map[ss.skill_id] = ss.level
        verification_statuses[ss.skill_id] = vstatus

    skills_by_level = {
        "advanced": [s["name"] for s in skills_list if s["level"] == "advanced"],
        "intermediate": [s["name"] for s in skills_list if s["level"] == "intermediate"],
        "basic": [s["name"] for s in skills_list if s["level"] == "basic"],
    }

    # Preload persisted matches & applications to determine exact favorites and dismissed drives
    saved_match_rows = session.exec(
        select(JobMatch).where(JobMatch.student_id == student_id)
    ).all()
    saved_app_rows = session.exec(
        select(JobApplication).where(JobApplication.student_id == student_id)
        .order_by(JobApplication.created_at.desc())
    ).all()

    dismissed_job_ids = {
        m.job_id
        for m in saved_match_rows
        if getattr(m, "band", None) == "dismissed"
        or getattr(m, "score", 0) < 0
        or (isinstance(getattr(m, "breakdown", None), dict) and m.breakdown.get("dismissed"))
    }
    favorite_job_ids = (
        {m.job_id for m in saved_match_rows if getattr(m, "is_favorite", False)}
        | {a.job_id for a in saved_app_rows if getattr(a, "is_favorite", False)}
    ) - dismissed_job_ids

    # Top Ranked Visible Job Matches (titles & companies only, no raw database IDs)
    top_matches = []
    ranked_by_id: dict[str, dict] = {}
    if ctx is not None:
        try:
            from app.agent import _rank_jobs_for
            ranked = _rank_jobs_for(session, ctx, student_id, top=5)
            for r in ranked:
                formatted = _format_job_for_copilot(
                    title=r.get("title") or "Role",
                    company=r.get("company", ""),
                    tier=r.get("tier") or "standard",
                    score=r.get("score") or 0,
                    band=r.get("band") or "early",
                    eligible=r.get("eligible"),
                    eligibility_reasons=r.get("eligibility_reasons", []),
                    required_skills=r.get("required_skills", []),
                    matched_required=[m.get("name", m.get("skill", "")) for m in r.get("matched_required", [])],
                    missing_required=[m.get("name", m.get("skill", "")) for m in r.get("missing_required", [])],
                    missing_preferred=[m.get("name", m.get("skill", "")) for m in r.get("missing_preferred", [])],
                    proficiency_gaps=r.get("proficiency_gaps", []),
                    is_favorite=bool(r.get("is_favorite", False)),
                )
                top_matches.append(formatted)
                if r.get("job_id"):
                    ranked_by_id[r["job_id"]] = formatted
        except Exception:
            pass

    if not top_matches:
        try:
            saved_pairs = session.exec(
                select(JobMatch, Job)
                .where(JobMatch.student_id == student_id, JobMatch.job_id == Job.id)
                .order_by(JobMatch.score.desc())
            ).all()
            for jm, j in saved_pairs:
                if j.id in dismissed_job_ids or not is_job_visible_to_student(j, student_id):
                    continue
                eligible, reasons = job_eligibility(student, j)
                formatted = _format_job_for_copilot(
                    title=j.title,
                    company=j.company,
                    tier=job_tier(j),
                    score=jm.score,
                    band=jm.band,
                    eligible=eligible,
                    eligibility_reasons=reasons,
                    required_skills=[_skill_display_name(s, ctx) for s in (getattr(j, "required_skills", None) or [])],
                    is_favorite=(j.id in favorite_job_ids),
                )
                top_matches.append(formatted)
                ranked_by_id[j.id] = formatted
                if len(top_matches) >= 5:
                    break
        except Exception:
            pass

    # Explicit Favorited Jobs (scored cleanly even when outside top-5 matches)
    favorite_jobs: list[dict] = []
    if favorite_job_ids:
        try:
            missing_fav_jobs: list[Job] = []
            ordered_favs: list[Job] = []
            for jid in sorted(favorite_job_ids):
                job_obj = session.get(Job, jid)
                if job_obj is None or not is_job_visible_to_student(job_obj, student_id):
                    continue
                ordered_favs.append(job_obj)
                if jid not in ranked_by_id:
                    missing_fav_jobs.append(job_obj)

            if missing_fav_jobs and ctx is not None and getattr(ctx, "embedder", None) is not None:
                sv = ctx.vector_store.get("student", student_id) if getattr(ctx, "vector_store", None) else None
                fav_scored = rank_jobs(
                    missing_fav_jobs,
                    sv or [],
                    student_skills_map,
                    ctx.embedder,
                    ctx.taxonomy,
                    vector_store=getattr(ctx, "vector_store", None),
                    job_vec_cache=getattr(ctx, "job_vectors", None),
                    top=None,
                    levels=levels_map,
                    student=student,
                    verification_statuses=verification_statuses,
                    session=session,
                )
                for fj, fjs in fav_scored:
                    eligible, reasons = job_eligibility(student, fj)
                    ranked_by_id[fj.id] = _format_job_for_copilot(
                        title=fj.title,
                        company=getattr(fj, "company", "") or "",
                        tier=job_tier(fj),
                        score=fjs.score,
                        band=fjs.band,
                        eligible=eligible,
                        eligibility_reasons=reasons,
                        required_skills=[_skill_display_name(s, ctx) for s in (getattr(fj, "required_skills", None) or [])],
                        matched_required=[m.get("name", m.get("skill", "")) for m in fjs.matched_required],
                        missing_required=[m.get("name", m.get("skill", "")) for m in fjs.missing_required],
                        missing_preferred=[m.get("name", m.get("skill", "")) for m in fjs.missing_preferred],
                        proficiency_gaps=fjs.proficiency_gaps,
                        is_favorite=True,
                    )

            saved_score_by_jid = {m.job_id: m for m in saved_match_rows}
            for fj in ordered_favs:
                if fj.id in ranked_by_id:
                    entry = dict(ranked_by_id[fj.id])
                    entry["is_favorite"] = True
                    favorite_jobs.append(entry)
                else:
                    jm = saved_score_by_jid.get(fj.id)
                    eligible, reasons = job_eligibility(student, fj)
                    favorite_jobs.append(
                        _format_job_for_copilot(
                            title=fj.title,
                            company=getattr(fj, "company", "") or "",
                            tier=job_tier(fj),
                            score=jm.score if jm else 0,
                            band=jm.band if jm else "early",
                            eligible=eligible,
                            eligibility_reasons=reasons,
                            required_skills=[_skill_display_name(s, ctx) for s in (getattr(fj, "required_skills", None) or [])],
                            is_favorite=True,
                        )
                    )
            favorite_jobs.sort(key=lambda x: x.get("score", 0), reverse=True)
        except Exception:
            pass

    # Upskilling plans (latest overall + distinct job plans)
    plan_data = None
    upskill_plans: list[dict] = []
    try:
        plans = session.exec(
            select(UpskillPlan).where(UpskillPlan.student_id == student_id)
            .order_by(UpskillPlan.created_at.desc())
        ).all()
        seen_plan_jobs: set[str | None] = set()
        for plan in plans:
            if plan.job_id in seen_plan_jobs:
                continue
            seen_plan_jobs.add(plan.job_id)
            items = session.exec(select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)).all()
            target_job = session.get(Job, plan.job_id) if plan.job_id else None
            entry = {
                "target_role": f"{target_job.title} at {target_job.company}" if target_job else "Target Role",
                "total_hours": plan.total_hours,
                "items": [
                    {
                        "skill": _skill_display_name(it.skill_id, ctx),
                        "hours": it.hours,
                        "status": it.status,
                        "week": it.week,
                    }
                    for it in items
                ],
            }
            if plan_data is None:
                plan_data = entry
            upskill_plans.append(entry)
            if len(upskill_plans) >= 4:
                break
    except Exception:
        pass

    # Applications
    app_data = []
    try:
        for a in saved_app_rows:
            job_obj = session.get(Job, a.job_id) if a.job_id else None
            app_data.append({
                "role": f"{job_obj.title} at {job_obj.company}" if job_obj else "Campus Opening",
                "status": a.status,
                "tier": a.tier,
                "match_score": a.match_score,
                "is_favorite": bool(getattr(a, "is_favorite", False)),
            })
    except Exception:
        pass

    return {
        "student_id": student.id,
        "name": student.name,
        "gpa": float(student.gpa or 0.0),
        "branch": student.branch,
        "year": student.year,
        "backlogs": int(getattr(student, "backlogs", 0) or 0),
        "mock_interview_score": max(scores) if scores else None,
        "mock_interview_passed": bool(scores) and max(scores) >= 65,
        "skills": skills_list,
        "skills_by_level": skills_by_level,
        "favorite_jobs": favorite_jobs,
        "favorite_count": len(favorite_jobs),
        "top_matches": top_matches,
        "upskill_plan": plan_data,
        "upskill_plans": upskill_plans,
        "applications": app_data,
        "interview_count": len(rows),
    }


def _clean_job_ids(text: str) -> str:
    """Strip raw internal job IDs (job-ext-... or job-001) so students only see readable role titles."""
    if not text:
        return ""
    cleaned = re.sub(r"\[?\bjob-(?:ext-[0-9a-f]{8,12}|\d{3})\b\]?[:,\s-]*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r",\s*,+", ", ", cleaned)
    cleaned = re.sub(r":\s*,", ": ", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _enforce_copilot_student_scope(user, student_id: str | None, request: Request | None = None) -> str | None:
    """Enforce student & faculty department row-level isolation with explicit 403."""
    if student_id:
        if request is not None:
            from app.security.rbac import enforce_student_scope
            enforce_student_scope(request, str(student_id).strip())
    if user is not None and user.role == "student":
        if student_id and user.student_id and student_id != user.student_id:
            raise HTTPException(
                403,
                "Students may only access their own placement Copilot context.",
            )
        return user.student_id
    return student_id


def _sanitize_copilot_question(session, question: str, *, student_id: str | None,
                               actor_role: str) -> str:
    """Scrub PII, neutralize prompt-injection, and audit blocked injections."""
    from app.security.guardrails import sanitize_untrusted_text
    from app.security.pii import scrub_for_llm

    question, _ = scrub_for_llm(question)
    question, inj_flags = sanitize_untrusted_text(question, source="copilot")
    if inj_flags:
        try:
            from app.security.audit import log_audit
            log_audit(
                session,
                action="security.prompt_injection_blocked",
                entity_type="copilot",
                entity_id=student_id or "general",
                actor_role=actor_role,
                detail={"flags": inj_flags, "source": "copilot"},
                commit=True,
            )
        except Exception:
            pass
    return question


def _record_copilot_agent_run(
    session,
    ctx,
    *,
    question: str,
    student_id: str | None,
    actor_role: str,
    answer: str,
    citations: dict,
    trace: list,
    latency_ms: int = 0,
) -> str | None:
    """Persist a compact AgentRun/AgentStep trace so Copilot queries appear in Agent Brain."""
    if not student_id and len((question or "").strip()) < 10:
        return None
    try:
        from uuid import uuid4

        from app.models import AgentRun, AgentStep

        llm = getattr(ctx, "llm", None) if ctx is not None else None
        provider = str(getattr(llm, "provider", "") or (type(llm).__name__ if llm else "mock"))
        run_id = f"run-{uuid4().hex[:8]}"
        step_plan = [
            {
                "thought": str(t.get("thought") or ""),
                "tool": str(t.get("tool") or "search_placement_policy_rag"),
                "args": {"top_sop": t.get("top_sop")} if t.get("top_sop") else {},
            }
            for t in (trace or [])
            if isinstance(t, dict)
        ]
        run = AgentRun(
            id=run_id,
            student_id=student_id,
            actor_role=actor_role or "student",
            goal=f"Copilot: {question[:140]}",
            status="completed",
            plan_json=step_plan,
            final_answer=answer[:2000],
            citations={
                "verified": list(citations.get("verified") or []),
                "unverified": list(citations.get("unverified") or []),
                "grounding_score": int(citations.get("grounding_score", 100)),
                "workflow_kind": "copilot",
            },
            total_latency_ms=max(0, int(latency_ms)),
            provider_used=f"copilot:{provider}",
            fallback_triggered=False,
        )
        session.add(run)
        session.flush()
        for idx, t in enumerate(trace or [], 1):
            if not isinstance(t, dict):
                continue
            session.add(
                AgentStep(
                    id=f"step-{uuid4().hex[:8]}",
                    run_id=run_id,
                    step_order=idx,
                    thought=str(t.get("thought") or ""),
                    tool_name=str(t.get("tool") or "search_placement_policy_rag"),
                    tool_input={"question": question[:120], "student_id": student_id},
                    tool_output={"top_sop": t.get("top_sop")} if t.get("top_sop") else {"ok": True},
                    verification_passed=True,
                    error_recovered="",
                    latency_ms=0,
                )
            )
        session.commit()
        return run_id
    except Exception:
        return None


def _sanitize_history(raw_history) -> list[dict[str, str]]:
    """Return up to 6 recent sanitized chat turns for multi-turn Copilot context."""
    if not isinstance(raw_history, list):
        return []
    from app.security.guardrails import sanitize_untrusted_text
    from app.security.pii import scrub_for_llm

    cleaned: list[dict[str, str]] = []
    for item in raw_history[-6:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in ("user", "assistant"):
            continue
        text = str(item.get("content") or "").strip()
        if not text:
            continue
        text, _ = scrub_for_llm(text[:800])
        if role == "user":
            text, _ = sanitize_untrusted_text(text, source="copilot_history")
        cleaned.append({"role": role, "content": text})
    return cleaned


def _build_copilot_trace(search: dict, student_context: dict) -> list[dict]:
    """Build deterministic step trace reflecting retrieval, profile, favorites, and matches."""
    trace = [{
        "thought": "Queried institutional placement policy for grounded clauses.",
        "tool": "search_placement_policy_rag",
        "top_sop": search.get("top_sop_id"),
    }]
    if not student_context:
        return trace

    skills = student_context.get("skills", [])
    skill_cnt = len(skills)
    ver_cnt = sum(1 for s in skills if s.get("verification_status") == "verified")
    trace.append({
        "thought": (
            f"Inspected student profile for {student_context.get('name', 'student')} "
            f"({ver_cnt}/{skill_cnt} verified skills, GPA {student_context.get('gpa')})."
        ),
        "tool": "inspect_student_profile",
    })
    favs = student_context.get("favorite_jobs") or []
    if favs:
        fav_summary = ", ".join(
            f"{f.get('title')} at {f.get('company', 'campus partner')}"
            for f in favs[:4]
        )
        trace.append({
            "thought": f"Inspected {len(favs)} favorited job(s): {fav_summary}.",
            "tool": "inspect_favorite_jobs",
        })
    if student_context.get("top_matches"):
        top_m = student_context["top_matches"][0]
        trace.append({
            "thought": (
                f"Evaluated visible job matches (Lead: {top_m.get('title')} at "
                f"{top_m.get('company', 'campus partner')}, score {top_m.get('score')})."
            ),
            "tool": "rank_and_explain_jobs",
        })
    return trace


def _copilot_synthesis(session, ctx, *, question: str, student_id: str | None,
                       job_id: str | None, actor_role: str,
                       sources: list, student_context: dict,
                       eligibility: dict, trace: list,
                       history: list[dict[str, str]] | None = None) -> dict:
    """Run LLM synthesis with grounded placement context and verify SOP + numeric citations."""
    started_at = time.perf_counter()
    user = {"question": question, "student": student_context,
            "sources": sources, "eligibility": eligibility}
    if history:
        user["history"] = history
    import json as _json
    raw = ""
    if getattr(ctx, "llm", None) is not None:
        try:
            logger.info("Copilot asking AI model for student=%s question=%r", student_id, question[:80])
            raw = ctx.llm.complete(system=COPILOT_SYSTEM, user=_json.dumps(user), json_mode=True)
            logger.info("Copilot AI model responded (length=%d)", len(raw))
        except Exception as e:
            logger.warning("Copilot AI model failed (%s); falling back to MockLLM.", e)
            try:
                from app.llm.mock import MockLLM
                raw = MockLLM().complete(system=COPILOT_SYSTEM, user=_json.dumps(user), json_mode=True)
            except Exception:
                raw = ""
    try:
        parsed = _json.loads(raw) if raw else {}
    except Exception:
        parsed = {}
    answer = str(parsed.get("answer") or "").strip()
    if not answer and raw:
        cleaned_raw = raw.strip()
        if cleaned_raw.startswith("```"):
            cleaned_raw = "\n".join(ln for ln in cleaned_raw.splitlines() if not ln.strip().startswith("```")).strip()
        try:
            p2 = _json.loads(cleaned_raw)
            if isinstance(p2, dict) and p2.get("answer"):
                answer = str(p2["answer"]).strip()
        except Exception:
            pass
        if not answer:
            answer = cleaned_raw
    if not answer:
        answer = (f"Based on JKLU placement policy, your inquiry '{question}' is governed by the "
                  f"retrieved sections: {', '.join(s.get('sop_id','') for s in sources)}.")
    answer = _clean_job_ids(answer)
    cited = [str(c) for c in (parsed.get("cited_sops") or [])]

    # Citation + numeric grounding guard (annotate unknown SOP ids and fabricated thresholds)
    from app.agent.verifier import verify_sop_numeric_claims
    from app.rag.policy_rag import sop_ids
    from app.security.audit import log_audit
    known = sop_ids()
    verify = verify_policy_citations(answer, known)
    numeric_issues = verify_sop_numeric_claims(answer)
    annotated = verify["annotated_text"]
    unverified_all = list(verify["unverified"])
    for err in numeric_issues:
        if err not in unverified_all:
            unverified_all.append(err)
        marker = f"[UNVERIFIED: {err}]"
        if marker not in annotated:
            annotated = f"{annotated} {marker}"

    # ensure cited_sops are all known; drop unknown ones from the trusted list
    cited = [c.upper() for c in cited if c.upper() in known]
    verified_all = sorted(set(verify["verified"]) | set(cited))
    denom = len(verified_all) + len(unverified_all)
    grounding_score = round(100 * len(verified_all) / denom) if denom else verify["grounding_score"]
    citations_out = {
        "verified": verified_all,
        "unverified": unverified_all,
        "grounding_score": grounding_score,
    }
    verification_out = {
        **citations_out,
        "numeric_issues": numeric_issues,
        "annotated_text": annotated,
    }
    agent_run_id = _record_copilot_agent_run(
        session,
        ctx,
        question=question,
        student_id=student_id,
        actor_role=actor_role,
        answer=annotated,
        citations=citations_out,
        trace=trace,
        latency_ms=int((time.perf_counter() - started_at) * 1000),
    )
    try:
        log_audit(
            session,
            action="copilot.asked",
            entity_type="copilot",
            entity_id=student_id or "general",
            actor_role=actor_role,
            detail={
                "question": question[:120],
                "grounding_score": grounding_score,
                "verified_sops": verified_all,
                "numeric_issues": numeric_issues,
                "agent_run_id": agent_run_id,
            },
            commit=True,
        )
    except Exception:
        pass

    return {
        "question": question,
        "student_id": student_id,
        "job_id": job_id,
        "answer": annotated,
        "citations": citations_out,
        "verification": verification_out,
        "agent_run_id": agent_run_id,
        "sources": sources,
        "student_context": student_context,
        "eligibility": eligibility or None,
        "trace": trace,
        "actor_role": actor_role,
    }


def _copilot_core(session, ctx, *, question: str, student_id: str | None,
                  job_id: str | None, actor_role: str,
                  history: list[dict[str, str]] | None = None) -> dict:
    """Shared retrieval + synthesis + citation-guard pipeline for ask endpoints."""
    question = _sanitize_copilot_question(
        session, question, student_id=student_id, actor_role=actor_role
    )
    # 1) Retrieval
    search = search_sops(question, top_k=3, embedder=ctx.embedder)
    sources = search.get("results", [])

    # 2) Student context (graceful if missing/invalid)
    student_context = _student_context(session, student_id, ctx=ctx)

    # 3) Optional eligibility summary (if both ids valid)
    eligibility = {}
    if student_id and job_id:
        try:
            eligibility = evaluate_student_eligibility(session, ctx, student_id, job_id)
        except Exception:
            eligibility = {}

    trace = _build_copilot_trace(search, student_context)

    return _copilot_synthesis(session, ctx, question=question,
                              student_id=student_id, job_id=job_id,
                              actor_role=actor_role,
                              sources=sources, student_context=student_context,
                              eligibility=eligibility, trace=trace,
                              history=history)


def _copilot_synthesis_thread(ctx, **kwargs) -> dict:
    """Run copilot synthesis with a dedicated short-lived Session on the worker thread."""
    with Session(ctx.engine) as worker_session:
        return _copilot_synthesis(worker_session, ctx, **kwargs)


@router.post("/api/copilot/ask/stream")
def copilot_ask_stream(payload: dict, request: Request,
                       ctx=Depends(get_ctx),
                       _rl: None = Depends(check_rate_limit("copilot_ask"))):
    """Stream the copilot answer as SSE (Milestone 17).

    Emits ``retrieval`` (sources + student context + eligibility + trace) immediately, then
    ``token`` events carrying the answer synthesized by the LLM path,
    then a final ``complete`` with the citation-verified annotated text.
    """
    from app.api.routes.jobs import SSE_HEADERS, _sse_chunk

    question = str((payload or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    student_id = (payload or {}).get("student_id") or None
    job_id = (payload or {}).get("job_id") or None
    actor_role = (payload or {}).get("actor_role", "student")
    history = _sanitize_history((payload or {}).get("history"))
    user = get_current_user_optional(request)
    if user is not None:
        actor_role = user.role
    student_id = _enforce_copilot_student_scope(user, student_id, request=request)
    with Session(ctx.engine) as pre_session:
        question = _sanitize_copilot_question(
            pre_session, question, student_id=student_id, actor_role=actor_role
        )

    def event_stream():
        try:
            with Session(ctx.engine) as stream_session:
                # 1) Fast retrieval & context
                search = search_sops(question, top_k=3, embedder=ctx.embedder)
                sources = search.get("results", [])
                student_context = _student_context(stream_session, student_id, ctx=ctx)
                eligibility = {}
                if student_id and job_id:
                    try:
                        eligibility = evaluate_student_eligibility(stream_session, ctx, student_id, job_id)
                    except Exception:
                        eligibility = {}

                trace = _build_copilot_trace(search, student_context)

            # Emit retrieval event IMMEDIATELY so the user UI sees Agent Step Trace right away!
            yield _sse_chunk("retrieval", {
                "sources": sources,
                "student_context": student_context,
                "eligibility": eligibility,
                "trace": trace,
            })

            # 2) AI Synthesis (runs with keep-alive pings so connection never drops)
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                _copilot_synthesis_thread, ctx,
                question=question, student_id=student_id,
                job_id=job_id, actor_role=actor_role,
                sources=sources, student_context=student_context,
                eligibility=eligibility, trace=trace,
                history=history,
            )
            while not future.done():
                try:
                    result = future.result(timeout=2.0)
                    break
                except concurrent.futures.TimeoutError:
                    yield ": ping\n\n"
            else:
                result = future.result()
            executor.shutdown(wait=False)

            # 3) Stream the answer text in clean, progressive chunks preserving newlines
            ans = result.get("answer") or ""
            lines = ans.splitlines(keepends=True)
            for line in lines:
                words = re.findall(r"\S+\s*", line)
                if words:
                    for i in range(0, len(words), 3):
                        yield _sse_chunk("token", {"delta": "".join(words[i:i + 3])})
                else:
                    yield _sse_chunk("token", {"delta": line})

            # 4) Complete event
            yield _sse_chunk("complete", {
                "answer": result["answer"],
                "citations": result["citations"],
                "verification": result["verification"],
                "agent_run_id": result.get("agent_run_id"),
                "sources": result["sources"],
                "student_context": result["student_context"],
                "eligibility": result["eligibility"],
                "trace": result["trace"],
                "question": result["question"],
                "student_id": result["student_id"],
                "job_id": result["job_id"],
                "actor_role": result["actor_role"],
            })
        except Exception as e:  # pragma: no cover - never break the SSE frame
            logger.exception("copilot_ask_stream error: %s", e)
            yield _sse_chunk("error", {"message": str(e)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers=SSE_HEADERS)


@router.post("/api/copilot/ask")
def copilot_ask(payload: dict, request: Request,
                session: Session = Depends(get_session), ctx=Depends(get_ctx),
                _rl: None = Depends(check_rate_limit("copilot_ask"))):
    question = str((payload or {}).get("question") or "").strip()
    if not question:
        raise HTTPException(400, "question is required")
    student_id = (payload or {}).get("student_id") or None
    job_id = (payload or {}).get("job_id") or None
    actor_role = (payload or {}).get("actor_role", "student")
    history = _sanitize_history((payload or {}).get("history"))

    user = get_current_user_optional(request)
    if user is not None:
        actor_role = user.role
    student_id = _enforce_copilot_student_scope(user, student_id, request=request)

    return _copilot_core(session, ctx, question=question, student_id=student_id,
                         job_id=job_id, actor_role=actor_role, history=history)


@router.get("/copilot")
def copilot_page(request: Request, session: Session = Depends(get_session)):
    role = current_role(request)
    user = get_current_user_optional(request)
    dept = user.department if user else None

    student_name = ""
    if role == "student":
        sid = user.student_id if user and user.student_id and user.student_id != "demo-student" else None
        st = session.get(Student, sid) if sid else None
        if st is not None:
            students = [st]
            student_name = st.name
            if user:
                user.name = st.name
                user.department = st.branch
                user.student_id = st.id
        else:
            students = []
    elif role == "faculty_mentor":
        if dept:
            from app.taxonomy.branches import normalize_branch
            dept_norm = (normalize_branch(dept) or dept).strip().lower()
            all_s = sort_students_alphabetically(session.exec(select(Student)).all())
            students = [
                s for s in all_s
                if (normalize_branch(s.branch or "") or str(s.branch or "")).strip().lower() == dept_norm
            ]
        else:
            students = []
    else:
        students = sort_students_alphabetically(session.exec(select(Student)).all())

    cohort_students = [] if role == "student" else list(students)

    from app.config import settings as _settings
    is_strict = str(getattr(_settings, "auth_mode", "demo")).strip().lower() == "strict"
    allow_demo = bool(getattr(_settings, "demo_role_switch", True))

    return request.app.state.templates.TemplateResponse(
        request, "copilot.html",
        {"students": students, "cohort_students": cohort_students, "sid": (user.student_id if user and role == "student" else ""), "active": "copilot",
         "role": role, "user": user, "current_user": user,
         "student_name": student_name,
         "department": dept,
         "user_name": (user.name if user else ""),
         "user_department": dept,
         "demo_role_switch": allow_demo,
         "auth_mode": "strict" if is_strict else "demo",
         "quick_prompts": QUICK_PROMPTS,
         "sop_ids": sorted(sop_ids()), "error": ""})


@router.get("/api/students/{sid}/eligibility/{job_id}")
def student_job_eligibility(sid: str, job_id: str, request: Request,
                            session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    from app.models import Job, Student
    enforce_student_scope(request, sid)
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    if session.get(Job, job_id) is None:
        raise HTTPException(404, "job not found")
    try:
        result = evaluate_student_eligibility(session, ctx, sid, job_id)
    except Exception as e:
        raise HTTPException(400, str(e))
    return result
