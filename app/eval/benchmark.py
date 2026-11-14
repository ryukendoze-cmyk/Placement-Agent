"""Automated agent, matching & RAG evaluation benchmark (Milestone 23).

Four deterministic, offline suites:

- SOP RAG: 10 golden semantic queries against the 18 institutional SOPs
  (recall@1, recall@3, MRR, citation grounding).
- Matching & eligibility: fresher/senior separation, depth-weighted monotonic
  scoring, SOP-03 GPA-cutoff accuracy.
- Planner DAG: prerequisite ordering + acyclicity across 10 multi-skill
  gap profiles.
- Agent grounding: ``verify_text`` must intercept 100% of synthetic
  hallucinated entity ids.
"""
from __future__ import annotations

# (semantic query, expected SOP id) - no SOP ids embedded in the query text,
# so retrieval is measured, not pinned.
GOLDEN_SOP_QUERIES: list[tuple[str, str]] = [
    ("What are the placement tier classification and CTC salary LPA thresholds?", "SOP-01"),
    ("How many offers are allowed per tier and can a student upgrade to a higher tier?", "SOP-02"),
    ("What is the minimum CGPA cutoff and the active backlog limit for placement eligibility?", "SOP-03"),
    ("Is there a mandatory AI mock-interview score gate for dream and super-dream roles?", "SOP-04"),
    ("What skill coverage and readiness band does the course pre-requisite policy require?", "SOP-05"),
    ("What penalty and placement suspension applies when a student no-shows a scheduled interview?", "SOP-06"),
    ("How does a PPO pre-placement offer convert to a full-time offer after the internship?", "SOP-07"),
    ("What does the resume grounding and zero-fabrication integrity policy prohibit?", "SOP-08"),
    ("Can a non-CSE cross-branch student apply for software roles through bridge courses?", "SOP-09"),
    ("What is the 48-hour deadline for placement drive registration and withdrawal?", "SOP-11"),
]

# 10 multi-skill gap profiles; each contains consumer skills together with at
# least one of their prerequisites so ordering constraints are exercised.
DAG_PROFILES: list[list[str]] = [
    ["fastapi", "python"],
    ["microservices", "rest_api", "docker"],
    ["react", "css", "html"],
    ["nextjs", "typescript", "react"],
    ["postgresql", "sql", "redis"],
    ["dbt", "data_modeling", "sql"],
    ["kubernetes", "docker", "linux"],
    ["kafka", "java", "hadoop"],
    ["celery", "python", "redis"],
    ["bootstrap", "tailwind", "css", "html"],
]

HALLUCINATED_IDS: list[str] = ["job-999", "job-ext-deadbeef00", "SOP-99"]


def _sop_rag_benchmark(ctx) -> dict:
    from ..rag.policy_rag import search_sops, sop_ids, verify_policy_citations

    known = {s.upper() for s in sop_ids()}
    recall1 = recall3 = 0
    mrr = 0.0
    grounding = 0.0
    per_query: list[dict] = []
    for query, expected in GOLDEN_SOP_QUERIES:
        res = search_sops(query, top_k=3, embedder=ctx.embedder)
        ids = [r["sop_id"].upper() for r in res["results"]]
        exp = expected.upper()
        rank = next((i + 1 for i, s in enumerate(ids) if s == exp), None)
        recall1 += 1 if rank == 1 else 0
        recall3 += 1 if rank is not None else 0
        mrr += (1.0 / rank) if rank is not None else 0.0
        top1 = res["results"][0] if res["results"] else None
        text = f"Per {expected}, the rule states: {top1['rule_text'] if top1 else ''}"
        grounding += verify_policy_citations(text, known=known)["grounding_score"]
        per_query.append({"query": query, "expected": expected,
                          "top_sop_id": res["top_sop_id"], "rank": rank})
    n = len(GOLDEN_SOP_QUERIES)
    return {"queries": n,
            "recall_at_1": round(recall1 / n, 4),
            "recall_at_3": round(recall3 / n, 4),
            "mean_reciprocal_rank": round(mrr / n, 4),
            "citation_grounding_score": round(grounding / n, 2),
            "per_query": per_query}


def _matching_benchmark(ctx) -> dict:
    from ..matching.engine import job_eligibility, score_job
    from ..models import Job, Student

    fresher = Student(id="eval-stu-mem", name="Eval Fresher", branch="CSE",
                      year=4, gpa=8.8)
    senior_job = Job(id="eval-job-senior", title="Senior Backend Engineer",
                     company="Eval", description="Senior backend engineer role",
                     required_skills=["python"], salary_lpa=[20, 30],
                     experience_years=4)
    fresher_job = Job(id="eval-job-fresh", title="Software Development Engineer",
                      company="Eval", description="Fresher software engineering role",
                      required_skills=["python"], salary_lpa=[6, 8],
                      experience_years=0)
    senior_ok, senior_reasons = job_eligibility(fresher, senior_job)
    fresh_ok, _fresh_reasons = job_eligibility(fresher, fresher_job)

    depth_job = Job(id="eval-job-depth", title="Backend Engineer", company="Eval",
                    description="Backend engineer role",
                    required_skills=["python", "fastapi"],
                    preferred_skills=[], salary_lpa=[8, 12], experience_years=1)
    skills = {"python": ["resume evidence"], "fastapi": ["resume evidence"]}
    depth_scores: dict[str, int] = {}
    for lvl in ("basic", "intermediate", "advanced"):
        levels = {"python": lvl, "fastapi": lvl}
        depth_scores[lvl] = score_job(skills, depth_job, 0.0, ctx.taxonomy,
                                      levels=levels).score

    return {"fresher_senior": {"senior_eligible": senior_ok,
                               "reasons": senior_reasons,
                               "fresher_eligible": fresh_ok},
            "depth_monotonicity": depth_scores}


def _sop_eligibility_benchmark(session, ctx) -> dict:
    """SOP-03 accuracy: two temp students against an 8.5-CGPA standard job.

    Temporary rows live inside a SAVEPOINT that is rolled back in ``finally``
    so the benchmark leaves zero trace even on error paths. Pre-embeds the
    synthetic job into ``ctx.job_vectors`` so ``rank_jobs`` never calls
    ``PgVectorStore.upsert`` (which would commit the outer session inside the
    savepoint).
    """
    from sqlalchemy import delete as sa_delete

    from ..models import Job, Skill, Student, StudentSkill
    from ..rag.policy_rag import evaluate_student_eligibility

    def gpa_passed(result: dict) -> bool:
        return next(c["passed"] for c in result["checks"] if c["name"] == "gpa_cutoff")

    # Clean up any leftover rows if an earlier run ever committed mid-savepoint.
    for stale_sid in ("eval-stu-low", "eval-stu-high"):
        session.execute(
            sa_delete(StudentSkill).where(StudentSkill.student_id == stale_sid)
        )
        stale = session.get(Student, stale_sid)
        if stale is not None:
            session.delete(stale)
    stale_job = session.get(Job, "eval-job-gpa")
    if stale_job is not None:
        session.delete(stale_job)
    session.commit()

    out: dict = {}
    job_desc = "Data engineer role"
    job_vec_cache = getattr(ctx, "job_vectors", None)
    had_cached_vec = isinstance(job_vec_cache, dict) and "eval-job-gpa" in job_vec_cache
    if isinstance(job_vec_cache, dict) and getattr(ctx, "embedder", None) is not None:
        job_vec_cache["eval-job-gpa"] = ctx.embedder.embed(job_desc)

    with session.begin_nested() as savepoint:
        try:
            low = Student(id="eval-stu-low", name="Eval Low", branch="CSE", gpa=8.2)
            high = Student(id="eval-stu-high", name="Eval High", branch="CSE", gpa=9.0)
            job = Job(id="eval-job-gpa", title="Data Engineer", company="Eval",
                      description=job_desc, required_skills=["sql"],
                      min_gpa=8.5, salary_lpa=[6, 7], experience_years=1)
            session.add_all([low, high, job])
            if session.get(Skill, "sql") is None:  # unseeded DB: FK target must exist
                session.add(Skill(id="sql", name="SQL", category="databases"))
            session.add(StudentSkill(student_id="eval-stu-high", skill_id="sql",
                                     mentions=1, level="advanced",
                                     evidence=["resume evidence"]))
            session.flush()
            res_low = evaluate_student_eligibility(session, ctx, "eval-stu-low", "eval-job-gpa")
            res_high = evaluate_student_eligibility(session, ctx, "eval-stu-high", "eval-job-gpa")
            out = {"low_gpa_eligible": res_low["eligible"],
                   "low_gpa_gpa_cutoff_passed": gpa_passed(res_low),
                   "high_gpa_eligible": res_high["eligible"],
                   "high_gpa_gpa_cutoff_passed": gpa_passed(res_high)}
        finally:
            if isinstance(job_vec_cache, dict) and not had_cached_vec:
                job_vec_cache.pop("eval-job-gpa", None)
            try:
                savepoint.rollback()
            except Exception:
                pass
    return out


def _planner_dag_benchmark(session=None) -> dict:
    from ..plan.planner import _load_courses, _pack_weeks, pick_course_for_skill
    from ..plan.prereq import PREREQUISITE_GRAPH, topological_order
    from ..rag.policy_rag import load_sops

    courses = _load_courses(session) if session is not None else []
    violations = 0
    cycles = 0
    duplicate_courses = 0
    unverified_urls = 0
    per_profile: list[dict] = []
    for targets in DAG_PROFILES:
        ordered = topological_order(PREREQUISITE_GRAPH, list(targets))
        pos = {s: i for i, s in enumerate(ordered)}
        profile_violations = 0
        for skill in ordered:
            for prereq in PREREQUISITE_GRAPH.get(skill, []):
                if prereq in pos and pos[prereq] >= pos[skill]:
                    profile_violations += 1
        seen_cids: set[str] = set()
        items_spec: list[tuple[str, str, int, bool]] = []
        if courses:
            for skill in ordered:
                c = pick_course_for_skill(skill, courses)
                if c is None:
                    continue
                cid = str(c.get("id") or "")
                url = str(c.get("url") or "").strip()
                if not url.startswith(("http://", "https://")):
                    unverified_urls += 1
                if cid in seen_cids:
                    continue
                seen_cids.add(cid)
                items_spec.append((skill, cid, int(c.get("hours") or 4), True))
            packed, _ = _pack_weeks(items_spec)
            packed_cids = [p[2] for p in packed]
            duplicate_courses += max(0, len(packed_cids) - len(set(packed_cids)))
        cycles += 1 if _has_cycle(PREREQUISITE_GRAPH, ordered) else 0
        violations += profile_violations
        per_profile.append({
            "targets": list(targets),
            "ordered": ordered,
            "violations": profile_violations,
        })
    is_valid = violations == 0 and cycles == 0 and duplicate_courses == 0
    return {
        "profiles": len(DAG_PROFILES),
        "violations": violations,
        "cycles": cycles,
        "duplicate_courses": duplicate_courses,
        "unverified_urls": unverified_urls,
        "sop_rules_checked": len(load_sops()),
        "valid": is_valid,
        "passed": is_valid,
        "per_profile": per_profile,
    }


def _has_cycle(graph: dict[str, list[str]], nodes: list[str]) -> bool:
    """3-color DFS cycle detection on the induced subgraph of ``nodes``."""
    node_set = set(nodes)
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in nodes}

    def visit(n: str) -> bool:
        color[n] = GRAY
        for p in graph.get(n, []):
            if p not in node_set:
                continue
            if color[p] == GRAY:
                return True
            if color[p] == WHITE and visit(p):
                return True
        color[n] = BLACK
        return False

    return any(color[n] == WHITE and visit(n) for n in nodes)


def _agent_grounding_benchmark(session) -> dict:
    from sqlmodel import select

    from ..agent.verifier import build_known_entities, verify_text
    from ..models import Job

    known = build_known_entities(session)
    real_jobs = session.exec(select(Job.id)).all()[:1]
    text_parts = list(HALLUCINATED_IDS)
    if real_jobs:
        text_parts.append(real_jobs[0] if isinstance(real_jobs[0], str) else real_jobs[0][0])
    text = "Review references: " + ", ".join(text_parts)
    result = verify_text(text, known)
    hallucinated = HALLUCINATED_IDS
    intercepted = [h for h in hallucinated if h in result["unverified"]]

    # Numeric SOP claim & fake URL interception benchmark
    numeric_probe = verify_text(
        "Under SOP-04 requires a score of 40 and Dream tier starts at 99 LPA "
        "and Standard allows 10 backlogs via https://fake.unverified.example/course",
        known,
        allowed_urls={"https://nptel.ac.in/course"},
    )
    numeric_caught = (
        any("SOP-04:mock_threshold=40" in u for u in numeric_probe["unverified"])
        and any("SOP-01:dream_lpa=99" in u for u in numeric_probe["unverified"])
        and any("SOP-03:backlog_limit=10" in u for u in numeric_probe["unverified"])
        and any("https://fake.unverified.example/course" in u for u in numeric_probe["unverified"])
    )
    return {"hallucinated": list(hallucinated),
            "unverified": list(result["unverified"]),
            "verified": list(result["verified"]),
            "grounding_score": result["grounding_score"],
            "interception_rate": round(len(intercepted) / len(hallucinated), 4),
            "numeric_and_url_interception": 1.0 if numeric_caught else 0.0,
            "numeric_probe_unverified": list(numeric_probe["unverified"])}


def _live_agent_metrics_summary(session) -> dict:
    """Compute real persisted AgentRun operational metrics (read-only)."""
    try:
        import math

        from sqlmodel import select

        from ..models import AgentRun

        runs = session.exec(select(AgentRun)).all()
    except Exception:
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "success_rate": None,
            "p95_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "fallback_rate": 0.0,
        }
    total = len(runs)
    if total == 0:
        return {
            "total_runs": 0,
            "completed_runs": 0,
            "failed_runs": 0,
            "success_rate": None,
            "p95_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "fallback_rate": 0.0,
        }
    latencies = sorted(float(r.total_latency_ms or 0) for r in runs)
    p95 = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
    completed = sum(1 for r in runs if r.status in ("completed", "awaiting_approval"))
    failed = sum(1 for r in runs if r.status == "failed")
    fallbacks = sum(1 for r in runs if r.fallback_triggered)
    return {
        "total_runs": total,
        "completed_runs": completed,
        "failed_runs": failed,
        "success_rate": round(completed / total, 4),
        "p95_latency_ms": round(p95, 2),
        "avg_latency_ms": round(sum(latencies) / total, 2),
        "fallback_rate": round(fallbacks / total, 4),
    }


def run_evaluation_benchmark(session, ctx) -> dict:
    """Run all automated benchmark suites plus live agent operational telemetry."""
    sop_rag = _sop_rag_benchmark(ctx)
    matching = _matching_benchmark(ctx)
    matching["sop_eligibility"] = _sop_eligibility_benchmark(session, ctx)
    planner_dag = _planner_dag_benchmark(session)
    agent_grounding = _agent_grounding_benchmark(session)
    agent_metrics = _live_agent_metrics_summary(session)
    return {
        "sop_rag": sop_rag,
        "matching": matching,
        "planner_dag": planner_dag,
        "agent_grounding": agent_grounding,
        "agent_metrics": agent_metrics,
        "summary": {
            "rag_recall_at_3": sop_rag["recall_at_3"],
            "rag_mrr": sop_rag["mean_reciprocal_rank"],
            "dag_valid": planner_dag["valid"],
            "hallucination_interception": agent_grounding["interception_rate"],
            "numeric_and_url_interception": agent_grounding["numeric_and_url_interception"],
            "agent_success_rate": agent_metrics["success_rate"],
            "all_passed": (sop_rag["recall_at_3"] >= 0.9
                           and planner_dag["valid"]
                           and agent_grounding["interception_rate"] == 1.0
                           and agent_grounding["numeric_and_url_interception"] == 1.0),
        },
    }
