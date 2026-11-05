"""10-tool registry for the autonomous agent. Each tool is standalone (not
HTTP-coupled): pure inputs -> JSON-safe dict output.

Tools for Phases 3/4/5 (RAG, ApprovalRequest table, prerequisite DAG) are fully
registered here and return structured stubs with a `status`/`note`, so the registry
is complete and honest in Phase 2 and fills in as later phases land.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from uuid import uuid4


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict                 # {"param": "type"} required params listed
    run: Callable[..., dict]

    def required(self) -> list[str]:
        return [k for k in self.input_schema if k.rstrip("?") == k]


def _validate(schema: dict, kwargs: dict) -> dict:
    out = {}
    for key in schema:
        req = not key.endswith("?")
        bare = key.rstrip("?")
        if bare not in kwargs or kwargs[bare] is None:
            if req:
                raise ValueError(f"missing required argument: {bare}")
            continue
        out[bare] = kwargs[bare]
    # pass through extra kwargs (executor may inject resolved refs)
    for k, v in kwargs.items():
        if k not in out:
            out[k] = v
    return out


# --- tool implementations ---------------------------------------------------

def _inspect_student_profile(ctx, session, student_id: str, **_):
    from ..models import Student, StudentSkill
    from ..resume.xray import xray_resume
    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    skills = [
        {"skill": ss.skill_id, "level": ss.level, "mentions": ss.mentions}
        for ss in session.exec(_select(StudentSkill, student_id)).all()
    ]
    xray = xray_resume(student.resume_text or "", ctx.taxonomy) if (student.resume_text or "").strip() else None
    return {
        "student_id": student.id, "name": student.name, "branch": student.branch,
        "year": student.year, "gpa": student.gpa, "skills": skills,
        "skill_ids": [s["skill"] for s in skills],
        "ats": (xray or {}).get("ats_score"),
        "xray": xray,
    }


def _rank_and_explain_jobs(ctx, session, student_id: str, top: int = 3, **_):
    from ..models import Student
    from . import _rank_jobs_for
    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    matches = _rank_jobs_for(session, ctx, student_id, top=int(top))
    top_match = matches[0] if matches else None
    return {
        "student_id": student_id,
        "matches": matches,
        "top_job_id": (top_match or {}).get("job_id"),
        "top_score": (top_match or {}).get("score"),
        "missing_required": [m.get("skill") for m in (top_match or {}).get("missing_required", [])],
        "missing_required_named": [m.get("name") for m in (top_match or {}).get("missing_required", [])],
    }


def _simulate_skill_acquisition(ctx, session, student_id: str, job_id: str, skills: list, **_):
    from ..matching.simulator import simulate_counterfactual
    from ..models import Job, Student, StudentSkill
    from ..plan.planner import _load_courses, pick_course_for_skill
    student = session.get(Student, student_id)
    job = session.get(Job, job_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    if job is None:
        raise ValueError(f"job '{job_id}' not found")
    from ..embeddings.store import cosine
    ss_rows = session.exec(_select(StudentSkill, student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    sv = ctx.vector_store.get("student", student_id)
    jv = ctx.job_vectors.get(job.id) or ctx.vector_store.get("job", job.id)
    if jv is None:
        jv = ctx.embedder.embed(job.description)
        ctx.vector_store.upsert("job", job.id, jv)
        ctx.job_vectors[job.id] = jv
    sim = cosine(sv, jv) if sv else 0.0
    courses = _load_courses(session)
    courses_for_skill = {s: pick_course_for_skill(s, courses) for s in (skills or [])}
    return simulate_counterfactual(student_skills, job, list(skills or []), sim, ctx.taxonomy,
                                   levels=levels, courses_for_skill=courses_for_skill)


def _search_placement_policy_rag(ctx, session, query: str, top_k: int = 3, **_):
    # Phase 3 replaces this with app.rag.policy_rag retrieval; structured stub now.
    from ..data import placement_sops_available
    if not placement_sops_available():
        return {"query": query, "results": [], "count": 0, "status": "stub",
                "note": "Institutional placement SOPs are indexed in Phase 3 RAG."}
    from ..rag.policy_rag import search_sops
    return {"query": query, **search_sops(query, top_k=int(top_k))}


def _search_course_catalog(ctx, session, skill_ids: list, level: str | None = None,
                           max_hours: int | None = None, **_):
    from ..plan.planner import _load_courses
    courses = _load_courses(session)
    picked = []
    for sid in (skill_ids or []):
        from ..plan.planner import pick_course_for_skill
        c = pick_course_for_skill(sid, courses)
        if c is not None:
            picked.append(c)
    if level:
        picked = [c for c in picked if c.get("level") == level] or picked
    if max_hours is not None:
        picked = [c for c in picked if c.get("hours", 0) <= max_hours] or picked
    return {"skill_ids": list(skill_ids or []), "courses": picked, "count": len(picked)}


def _generate_prerequisite_roadmap(ctx, session, student_id: str, target_skills: list, **_):
    from ..models import StudentSkill
    from ..plan.planner import _load_courses, pick_course_for_skill
    from ..plan.prereq import PREREQUISITE_GRAPH, topological_order

    courses = _load_courses(session)
    known = {r.skill_id for r in session.exec(
        _select(StudentSkill, student_id)).all()}
    targets = [t for t in (target_skills or []) if t]
    ordered = topological_order(PREREQUISITE_GRAPH, targets, exclude=known)
    roadmap = []
    for sk in ordered:
        c = pick_course_for_skill(sk, courses)
        if c is None:
            continue
        roadmap.append({
            "skill": sk,
            "prerequisites": [p for p in PREREQUISITE_GRAPH.get(sk, [])
                              if p in set(ordered)],
            "course_id": c["id"],
            "title": c["title"],
            "hours": c["hours"],
            "status": "already_owned" if sk in known else "gap",
        })
    return {"student_id": student_id, "target_skills": list(target_skills or []),
            "roadmap": roadmap, "status": "dag_ordered",
            "owned_excluded": sorted(known & set(targets))}


def _audit_resume_bullets(ctx, session, student_id: str, **_):
    from ..models import Student
    from ..resume.xray import improve_weak_bullets, xray_resume
    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    text = student.resume_text or ""
    present = []
    if ctx.llm is not None and text.strip():
        from ..models import StudentSkill
        present = [ctx.taxonomy.get(r.skill_id).name
                   for r in session.exec(_select(StudentSkill, student_id)).all()]
    ats = xray_resume(text, ctx.taxonomy)
    improvements = improve_weak_bullets(text, present, ctx.llm, taxonomy=ctx.taxonomy) if ctx.llm else []
    return {"student_id": student_id, "ats": ats, "improvements": improvements}


def _query_cohort_bottlenecks(ctx, session, top_n: int = 5, **_):
    from ..analytics.cohort import readiness_distribution, skill_gap_report
    gaps = skill_gap_report(session)[: int(top_n)]
    return {"skill_gaps": gaps, "readiness_distribution": readiness_distribution(session),
            "top_gap_skill": (gaps[0]["skill"] if gaps else None)}


def _draft_cohort_intervention(ctx, session, skill_id: str, **_):
    from ..plan.planner import _load_courses, pick_course_for_skill
    courses = _load_courses(session)
    course = pick_course_for_skill(skill_id, courses)
    if course is None:
        return {"skill_id": skill_id, "workshop": None, "status": "no_course",
                "note": "No catalog course covers this skill."}
    return {
        "skill_id": skill_id,
        "workshop": {
            "title": f"Bridge Bootcamp: {skill_id}",
            "course_id": course["id"], "course_title": course["title"],
            "provider": course["provider"], "hours": course["hours"],
            "outline": [f"Foundations of {skill_id}", "Hands-on lab", "Capstone + review"],
        },
        "status": "ready", "requires_hitl": True,
        "note": "Publication executes after a Phase 4 HITL approval.",
    }


def _propose_hitl_approval(ctx, session, kind: str, target_id: str, title: str,
                           summary: str, payload: dict | None = None, **_):
    from ..models import ApprovalRequest
    approval_id = f"appr-{uuid4().hex[:8]}"
    ap = ApprovalRequest(id=approval_id, kind=kind, target_id=target_id, title=title,
                         summary=summary, payload=payload or {},
                         status="pending_review", requested_by="agent")
    session.add(ap)
    session.flush()  # persist the pending approval row (committed with the run)
    return {"approval_id": approval_id, "kind": kind, "target_id": target_id,
            "title": title, "summary": summary, "payload": payload or {},
            "status": "pending_review", "requires_hitl": True,
            "note": "Stored as a real ApprovalRequest; executes only after approval."}


def _select(Model, sid):
    from sqlmodel import select
    return select(Model).where(Model.student_id == sid)


# --- Milestone 16 tools (11-15) --------------------------------------------

def _check_drive_eligibility(ctx, session, student_id: str, job_id: str, **_):
    """SOP-cited multi-dimension eligibility gate (SOP-03/04/05/09)."""
    from ..rag.policy_rag import evaluate_student_eligibility
    return evaluate_student_eligibility(session, ctx, student_id, job_id)


def _persist_upskilling_plan(ctx, session, student_id: str, job_id: str | None = None, **_):
    """Compute a proficiency-aware JobScore and persist UpskillPlan + PlanItems."""
    from ..matching.rank import rank_jobs
    from ..models import Job, Student, StudentSkill, UpskillPlan
    from ..plan.planner import build_plan

    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    if job_id:
        job = session.get(Job, job_id)
        if job is None:
            raise ValueError(f"job '{job_id}' not found")
    else:
        from . import _rank_jobs_for
        ranked = _rank_jobs_for(session, ctx, student_id, top=1)
        if not ranked:
            raise ValueError("no jobs available to plan against")
        job = session.get(Job, ranked[0]["job_id"])
        if job is None:
            raise ValueError("top matched job no longer exists")

    ss_rows = session.exec(_select(StudentSkill, student_id)).all()
    student_skills = {r.skill_id: list(r.evidence) for r in ss_rows}
    levels = {r.skill_id: r.level for r in ss_rows}
    sv = ctx.vector_store.get("student", student_id)
    if sv is None:
        sv = ctx.embedder.embed(student.resume_text or "")
    ranked = rank_jobs([job], sv, student_skills, ctx.embedder, ctx.taxonomy,
                       vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors,
                       top=1, levels=levels, student=student)
    if not ranked:
        raise ValueError("failed to score target job")
    job_score = ranked[0][1]

    # deterministic plan id -> re-running the tool overwrites instead of duplicating
    plan_id = f"plan-{student_id}-{job.id}"
    from sqlmodel import delete

    from ..models import PlanItem
    prior = session.get(UpskillPlan, plan_id)
    if prior is not None:
        session.execute(delete(PlanItem).where(PlanItem.plan_id == prior.id))
        session.delete(prior)
        session.commit()

    plan, items = build_plan(session, student_id, job, job_score, llm=ctx.llm,
                             levels=levels)
    return {
        "plan_id": plan.id, "job_id": plan.job_id, "total_hours": plan.total_hours,
        "item_count": len(items),
        "items": [{"skill_id": it.skill_id, "course_id": it.course_id,
                   "hours": it.hours, "week": it.week, "status": it.status}
                  for it in items],
        "proficiency_gaps": len(job_score.proficiency_gaps or []),
    }


def _inspect_interview_history(ctx, session, student_id: str, **_):
    """Aggregate completed mock interviews: avg score, rubric axes, weak/strong."""
    from ..models import Interview, Student

    student = session.get(Student, student_id)
    if student is None:
        raise ValueError(f"student '{student_id}' not found")
    rows = session.exec(
        _select(Interview, student_id).where(Interview.finished == True)  # noqa: E712
    ).all()
    if not rows:
        return {"student_id": student_id, "interview_count": 0,
                "average_score": None, "rubric_averages": {},
                "weak_skills": [], "strong_skills": [],
                "latest_scores": [], "status": "no_completed_interviews"}

    scores = [int(r.score) for r in rows if r.score is not None]
    rubric_totals: dict[str, list[float]] = {}
    for r in rows:
        per = (r.feedback or {}).get("rubric_overall") or {}
        if isinstance(per, dict):
            for k, v in per.items():
                try:
                    rubric_totals.setdefault(str(k), []).append(float(v))
                except (TypeError, ValueError):
                    continue
    rubric_avg = {k: round(sum(v) / len(v)) for k, v in rubric_totals.items() if v}

    per_skill_scores: dict[str, list[int]] = {}
    for r in rows:
        per = (r.feedback or {}).get("per_skill") or {}
        if isinstance(per, dict):
            for skill, sc in per.items():
                try:
                    per_skill_scores.setdefault(str(skill), []).append(int(sc))
                except (TypeError, ValueError):
                    continue
    skill_avg = {s: round(sum(v) / len(v)) for s, v in per_skill_scores.items()}
    weak = sorted(s for s, v in skill_avg.items() if v < 50)
    strong = sorted(s for s, v in skill_avg.items() if v >= 80)

    return {
        "student_id": student_id,
        "interview_count": len(rows),
        "average_score": round(sum(scores) / len(scores)) if scores else None,
        "rubric_averages": rubric_avg,
        "weak_skills": weak,
        "strong_skills": strong,
        "latest_scores": scores[-5:],
        "skill_scores": skill_avg,
        "status": "ok",
    }


def _search_live_market_jobs(ctx, session, student_id: str | None = None,
                             query: str | None = None, location: str | None = None,
                             fresher_only: bool | None = None, **_):
    """Search the live (scraped + campus) Job table with optional student ranking."""
    from ..matching.engine import is_fresher_job, job_tier

    jobs = session.exec(_all_jobs()).all()
    q = (query or "").strip().lower()
    loc = (location or "").strip().lower()
    out = []
    for job in jobs:
        hay = f"{job.title} {job.company} {job.description}".lower()
        if q and q not in hay:
            continue
        if loc and loc not in str(job.location or "").lower():
            continue
        if fresher_only and not is_fresher_job(job):
            continue
        out.append(job)

    scores: dict[str, int] = {}
    if student_id and out:
        from . import _rank_jobs_for
        ranked = _rank_jobs_for(session, ctx, student_id, top=len(out))
        scores = {m["job_id"]: m["score"] for m in ranked}

    results = []
    for job in sorted(out, key=lambda j: -scores.get(j.id, 0))[:10]:
        results.append({
            "job_id": job.id, "title": job.title, "company": job.company,
            "location": job.location, "tier": job_tier(job),
            "is_fresher": bool(is_fresher_job(job)),
            "salary_lpa": list(getattr(job, "salary_lpa", None) or [0, 0]),
            "score": scores.get(job.id),
        })
    return {"query": query or "", "location": location or "",
            "fresher_only": bool(fresher_only),
            "count": len(results), "jobs": results}


def _all_jobs():
    from sqlmodel import select

    from ..models import Job
    return select(Job)


def _ask_clarifying_question(ctx, session, question: str, options: list | None = None,
                             missing_fields: list | None = None, **_):
    """Pause the run awaiting user input; the orchestrator flips status."""
    return {"status": "awaiting_input", "requires_user_input": True,
            "question": str(question or "Could you clarify what you need?"),
            "options": [str(o) for o in (options or [])],
            "missing_fields": [str(f) for f in (missing_fields or [])]}



# --- registry ---------------------------------------------------------------

TOOLS = [
    Tool("inspect_student_profile",
         "Fetch a student's skills, GPA, branch, and ATS X-Ray metrics.",
         {"student_id": "str"}, _inspect_student_profile),
    Tool("rank_and_explain_jobs",
         "Run hybrid skill+semantic matching and return explainable top job matches.",
         {"student_id": "str", "top?": "int"}, _rank_and_explain_jobs),
    Tool("simulate_skill_acquisition",
         "Counterfactual: project match score if the student learns given skills.",
         {"student_id": "str", "job_id": "str", "skills": "list[str]"}, _simulate_skill_acquisition),
    Tool("search_placement_policy_rag",
         "Retrieve grounded JKLU placement SOP clauses with citations.",
         {"query": "str", "top_k?": "int"}, _search_placement_policy_rag),
    Tool("search_course_catalog",
         "Find the best courses covering given skills within level/hour budget.",
         {"skill_ids": "list[str]", "level?": "str", "max_hours?": "int"}, _search_course_catalog),
    Tool("generate_prerequisite_roadmap",
         "Build an ordered study roadmap for target skills.",
         {"student_id": "str", "target_skills": "list[str]"}, _generate_prerequisite_roadmap),
    Tool("audit_resume_bullets",
         "Run ATS X-Ray and propose grounded STAR bullet improvements.",
         {"student_id": "str"}, _audit_resume_bullets),
    Tool("query_cohort_bottlenecks",
         "Return cohort-wide skill gaps and readiness distribution for TPOs.",
         {"top_n?": "int"}, _query_cohort_bottlenecks),
    Tool("draft_cohort_intervention",
         "Design a targeted workshop/bootcamp for a top cohort skill gap.",
         {"skill_id": "str"}, _draft_cohort_intervention),
    Tool("propose_hitl_approval",
         "Create a formal HITL approval request for a state-changing action.",
         {"kind": "str", "target_id": "str", "title": "str", "summary": "str", "payload?": "dict"},
         _propose_hitl_approval),
    Tool("check_drive_eligibility",
         "Evaluate SOP-cited drive eligibility (GPA/backlogs, branch, mock gate, skills).",
         {"student_id": "str", "job_id": "str"}, _check_drive_eligibility),
    Tool("persist_upskilling_plan",
         "Compute a proficiency-aware job score and persist an upskilling plan.",
         {"student_id": "str", "job_id?": "str"}, _persist_upskilling_plan),
    Tool("inspect_interview_history",
         "Summarize completed mock interviews: average score, rubric, weak/strong skills.",
         {"student_id": "str"}, _inspect_interview_history),
    Tool("search_live_market_jobs",
         "Search live/campus jobs by query/location/fresher fit, optionally ranked.",
         {"student_id?": "str", "query?": "str", "location?": "str",
          "fresher_only?": "bool"}, _search_live_market_jobs),
    Tool("ask_clarifying_question",
         "Ask the user a clarifying question; pauses the run awaiting their reply.",
         {"question": "str", "options?": "list", "missing_fields?": "list"},
         _ask_clarifying_question),
]

TOOL_REGISTRY: dict[str, Tool] = {t.name: t for t in TOOLS}

#: Common LLM tool name hallucinations mapped to canonical registered tools.
TOOL_ALIASES: dict[str, str] = {
    "get_student_profile": "inspect_student_profile",
    "fetch_student_profile": "inspect_student_profile",
    "student_profile": "inspect_student_profile",
    "search_jobs": "search_live_market_jobs",
    "search_market_jobs": "search_live_market_jobs",
    "rank_jobs": "rank_and_explain_jobs",
    "rank_job_matches": "rank_and_explain_jobs",
    "rank_and_explain": "rank_and_explain_jobs",
    "match_jobs": "rank_and_explain_jobs",
    "generate_plan": "persist_upskilling_plan",
    "build_plan": "persist_upskilling_plan",
    "check_eligibility": "check_drive_eligibility",
    "search_policy": "search_placement_policy_rag",
    "policy_rag": "search_placement_policy_rag",
    "interview_history": "inspect_interview_history",
}


def call_tool(name: str, ctx, session, **kwargs) -> dict:
    canonical = TOOL_ALIASES.get(name, name)
    tool = TOOL_REGISTRY.get(canonical)
    if tool is None:
        raise ValueError(f"unknown tool: {name}")
    # Normalize common parameter aliases from external LLMs
    if canonical == "rank_and_explain_jobs" and "top_n" in kwargs and "top" not in kwargs:
        kwargs["top"] = kwargs.pop("top_n")
    args = _validate(tool.input_schema, kwargs)
    return tool.run(ctx, session, **args)
