"""Phase 2: agent orchestrator + verifier tests (plan section 2.5)."""
import json

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.agent.orchestrator import _resolve, run_agent
from app.agent.verifier import build_known_entities, verify_text
from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import AgentRun, AgentStep, Student
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy

# ---------------- helpers ----------------

def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _session_env(cohort: bool = False):
    """Fresh sqlite DB with seeds + a Session. Returns (session, ctx)."""
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(
        engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(), llm=MockLLM(),
    )
    session = Session(eng)
    seed(session, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    if cohort:
        seed_demo_cohort(session, taxonomy=taxonomy, embedder=ctx.embedder,
                         vector_store=ctx.vector_store, llm=MockLLM())
    return session, ctx


def _make_client(cohort: bool = False) -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
        if cohort:
            seed_demo_cohort(db, taxonomy=taxonomy, embedder=ctx.embedder,
                             vector_store=ctx.vector_store, llm=MockLLM())
    return TestClient(create_app(ctx))


def _seed_student(client, name="Agent Probe", branch="CSE", year=3, gpa=8.0) -> str:
    """Create a student through the API (the pattern every other test uses)."""
    r = client.post(
        "/api/students",
        files={"file": ("r.txt",
                        b"Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n"
                        b"Backend internship building REST APIs\n", "text/plain")},
        data={"name": name, "branch": branch, "year": str(year), "gpa": str(gpa)},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ---------------- verifier ----------------

def test_verify_known_ids_pass_and_unknown_annotated():
    known = {"job-001", "course-027"}
    out = verify_text("Apply to job-001 after course-027 and also job-999", known)
    assert out["verified"] == ["job-001", "course-027"]
    assert out["unverified"] == ["job-999"]
    assert "[UNVERIFIED: job-999]" in out["annotated_text"]
    assert out["grounding_score"] == 67  # 2 of 3 verified


def test_verify_all_known_scores_100():
    out = verify_text("Top job is job-001.", {"job-001"})
    assert out["grounding_score"] == 100
    assert out["unverified"] == []
    assert out["annotated_text"] == "Top job is job-001."


def test_verify_text_handles_empty_and_non_string():
    assert verify_text("", set())["grounding_score"] == 100
    assert verify_text(None, set())["grounding_score"] == 100
    assert verify_text(123, set())["grounding_score"] == 100


def test_build_known_entities_returns_db_ids():
    session, ctx = _session_env()
    try:
        known = build_known_entities(session)
        # base seed() creates jobs/skills/courses but no students; jobs must be known
        assert "job-001" in known
        assert "python" in known
    finally:
        session.close()


def test_build_known_entities_includes_students_when_present():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as db:
            known = build_known_entities(db)
            assert sid in known


# ---------------- arg resolution ($last / $first / $<tool>) ----------------

def test_resolve_last_key_from_memory():
    mem = {"rank_and_explain_jobs": {"top_job_id": "job-004"}}
    assert _resolve("$last.top_job_id", mem) == "job-004"


def test_resolve_first_key_from_memory():
    mem = {"a": {"x": 1}, "b": {"x": 2}}
    assert _resolve("$first.x", mem) == 1


def test_resolve_named_tool_key():
    mem = {"query_cohort_bottlenecks": {"top_gap_skill": "kubernetes"}}
    assert _resolve("$query_cohort_bottlenecks.top_gap_skill", mem) == "kubernetes"


def test_resolve_is_recursive_over_dicts_and_lists():
    mem = {"rank_and_explain_jobs": {"top_job_id": "job-007"}}
    obj = {"job": "$last.top_job_id", "nested": {"again": "$last.top_job_id"},
           "items": ["$last.top_job_id", 3]}
    assert _resolve(obj, mem) == {"job": "job-007", "nested": {"again": "job-007"},
                                  "items": ["job-007", 3]}


def test_resolve_missing_ref_returns_literal():
    assert _resolve("$last.ghost", {}) == "$last.ghost"
    assert _resolve("plain string", {}) == "plain string"


# ---------------- orchestrator: happy path ----------------

def test_multi_step_run_persists_run_and_steps():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs for this student and build a plan",
                            student_id=sid, actor_role="student")
            assert run.status == "completed"
            assert run.fallback_triggered is False
            assert len(run.plan_json) >= 2
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                                 .order_by(AgentStep.step_order)).all()
            assert len(steps) == len(run.plan_json)
            assert [s.step_order for s in steps] == list(range(1, len(steps) + 1))
            assert all(s.thought for s in steps)          # every step carries a thought
            assert all(s.tool_name for s in steps)


def test_tool_chaining_resolves_refs_between_steps():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs and simulate acquiring missing skills",
                            student_id=sid, actor_role="student")
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                                 .order_by(AgentStep.step_order)).all()
            by_tool = {s.tool_name: s for s in steps}
            # inspect ran first with the real student id
            assert by_tool["inspect_student_profile"].tool_input["student_id"] == sid
            # simulate's job_id came from rank's output ($last.top_job_id) - not a literal
            if "simulate_skill_acquisition" in by_tool:
                sim_in = by_tool["simulate_skill_acquisition"].tool_input
                assert str(sim_in.get("job_id", "")).startswith("job-")
                assert sim_in["job_id"] != "$last.top_job_id"
            # rank output fed the pipeline
            assert by_tool["rank_and_explain_jobs"].tool_output.get("top_job_id")


def test_rank_tool_output_explains_tier_and_eligibility():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs and explain the top match",
                            student_id=sid, actor_role="student")
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                                 .order_by(AgentStep.step_order)).all()
            rank = next(s for s in steps if s.tool_name == "rank_and_explain_jobs")
            m = rank.tool_output["matches"][0]
            assert {"job_id", "score", "band", "tier", "eligible", "eligibility_reasons",
                    "proficiency_bonus", "matched_required", "missing_required"} <= set(m)
            assert m["tier"] in {"standard", "dream", "super_dream"}


# ---------------- orchestrator: planner behavior ----------------

def test_planner_goal_intent_drives_plan():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Improve my resume bullets and ATS score",
                            student_id=sid, actor_role="student")
            tools = [s["tool"] for s in run.plan_json]
            assert "audit_resume_bullets" in tools
            assert "rank_and_explain_jobs" not in tools


def test_cohort_goal_plans_bottleneck_then_intervention_with_ref():
    client = _make_client(cohort=True)
    with client:
        with Session(client.app.state.ctx.engine) as session:
            sid = session.exec(select(Student)).first().id
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Find cohort bottlenecks and draft an intervention",
                            student_id=sid, actor_role="tpo_admin")
            tools = [s["tool"] for s in run.plan_json]
            assert tools[0] == "query_cohort_bottlenecks"
            assert "draft_cohort_intervention" in tools
            # draft's skill_id is a $first ref resolved at execute time
            draft = next(s for s in run.plan_json if s["tool"] == "draft_cohort_intervention")
            assert draft["args"]["skill_id"] == "$first.top_gap_skill"
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            executed = next(s for s in steps if s.tool_name == "draft_cohort_intervention")
            assert executed.tool_input["skill_id"] != "$first.top_gap_skill"
            assert executed.tool_input["skill_id"]


def test_run_with_goal_but_no_student_short_circuits_student_tools():
    client = _make_client()
    with client:
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs for a general profile",
                            student_id=None, actor_role="tpo_admin")
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            prof = next(s for s in steps if s.tool_name == "inspect_student_profile")
            # short-circuited instead of crashing / persisting bogus output
            assert "no student in context" in prof.error_recovered
            assert run.status in {"completed", "failed"}


def test_plan_capped_at_five_steps():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Do all the things", student_id=sid)
            assert len(run.plan_json) <= 5


# ---------------- orchestrator: recovery / no-500 ----------------

def test_nomination_plan_resolves_target_ref_and_flags_approval():
    """Goal mentioning 'nominate'/'drive' appends a HITL step; its target_id must
    resolve to a real job id ($rank_and_explain_jobs.top_job_id), not a literal ref."""
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session,
                            goal="Nominate this student for the placement drive",
                            student_id=sid, actor_role="tpo_admin")
            # HITL tool in the plan -> run must land in awaiting_approval
            assert run.status == "awaiting_approval"
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                                 .order_by(AgentStep.step_order)).all()
            hitl = next(s for s in steps if s.tool_name == "propose_hitl_approval")
            assert hitl.tool_input["target_id"].startswith("job-")
            assert "$" not in hitl.tool_input["target_id"]  # ref was resolved
            # the approval id itself is grounded
            appr = hitl.tool_output.get("approval_id", "")
            assert appr.startswith("appr-")
            # the agent's OWN generated approval id must not be flagged as a hallucination
            assert appr not in run.citations["unverified"]
            assert appr in run.citations["verified"]
            assert run.citations["grounding_score"] == 100


def test_step_verification_passed_reflects_tool_success():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs", student_id=sid)
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            # every successful step must be marked verified
            for s in steps:
                if not s.error_recovered:
                    assert s.verification_passed is True, s.tool_name


def test_tool_exception_recovers_without_raising(monkeypatch):
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            from app.agent import tools as tools_mod
            original = tools_mod.TOOL_REGISTRY["inspect_student_profile"].run
            calls = {"n": 0}

            def boom(ctx, session, **kw):
                calls["n"] += 1
                raise RuntimeError("synthetic tool explosion")

            monkeypatch.setattr(tools_mod.TOOL_REGISTRY["inspect_student_profile"], "run", boom)
            # first call raises, orchestrator retries (repair) which raises again -> fallback
            run = run_agent(ctx, session, goal="Inspect the student and rank jobs",
                            student_id=sid, actor_role="student")
            assert calls["n"] >= 1
            assert run.status != "running"
            assert run.fallback_triggered is True
            steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
            prof = next(s for s in steps if s.tool_name == "inspect_student_profile")
            assert prof.error_recovered  # error was captured, not swallowed silently
            monkeypatch.setattr(tools_mod.TOOL_REGISTRY["inspect_student_profile"], "run", original)


def test_malformed_planner_json_uses_fallback_plan():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx

            class BadLLM:
                def complete(self, system, user, json_mode=False):
                    if "plan_steps" in system:
                        return "THIS IS NOT JSON {{{"
                    return json.dumps({"summary": "s", "sections": []})

            ctx.llm = BadLLM()
            run = run_agent(ctx, session, goal="Rank jobs", student_id=sid)
            assert run.fallback_triggered is True
            assert run.plan_json  # fallback plan still produced
            assert run.status == "completed"


def test_planner_returning_empty_steps_falls_back():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx

            class EmptyLLM:
                def complete(self, system, user, json_mode=False):
                    if "plan_steps" in system:
                        return json.dumps({"steps": []})
                    return json.dumps({"summary": "s", "sections": []})

            ctx.llm = EmptyLLM()
            run = run_agent(ctx, session, goal="Rank jobs", student_id=sid)
            assert run.fallback_triggered is True
            assert run.plan_json


# ---------------- orchestrator: citations / grounding ----------------

def test_final_answer_is_grounded_and_citations_recorded():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session, goal="Rank jobs and recommend next steps",
                            student_id=sid, actor_role="student")
            assert isinstance(run.citations, dict)
            assert "grounding_score" in run.citations
            assert 0 <= run.citations["grounding_score"] <= 100
            assert "verified" in run.citations and "unverified" in run.citations
            # synthesizer cites a real job id -> verified
            assert run.citations["verified"], "expected at least one grounded citation"
            # no unverified job ids leaked into the answer unmarked
            assert "[UNVERIFIED" not in run.final_answer or run.citations["unverified"]


# ---------------- API endpoints ----------------

def test_api_agent_run_creates_persisted_trace():
    client = _make_client(cohort=True)
    with client:
        with Session(client.app.state.ctx.engine) as db:
            sid = db.exec(select(Student)).first().id
        r = client.post("/api/agent/run",
                        json={"goal": "Rank jobs and build an upskilling plan",
                              "student_id": sid, "actor_role": "student"})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["id"].startswith("run-")
        assert body["status"] in {"completed", "awaiting_approval", "failed"}
        assert body["plan"] and body["steps"]
        assert all(s["thought"] for s in body["steps"])
        # persisted
        with Session(client.app.state.ctx.engine) as db:
            run = db.get(AgentRun, body["id"])
            assert run is not None and run.status == body["status"]
            n_steps = len(db.exec(select(AgentStep).where(AgentStep.run_id == body["id"])).all())
            assert n_steps == len(body["steps"])


def test_api_agent_run_missing_goal_400():
    client = _make_client()
    with client:
        r = client.post("/api/agent/run", json={"goal": "   "})
        assert r.status_code == 400


def test_api_agent_runs_listing_and_detail():
    client = _make_client(cohort=True)
    with client:
        with Session(client.app.state.ctx.engine) as db:
            sid = db.exec(select(Student)).first().id
        created = client.post("/api/agent/run",
                              json={"goal": "Explain my top match", "student_id": sid}).json()
        listing = client.get("/api/agent/runs")
        assert listing.status_code == 200
        ids = [r["id"] for r in listing.json()]
        assert created["id"] in ids

        detail = client.get(f"/api/agent/runs/{created['id']}")
        assert detail.status_code == 200
        assert detail.json()["final_answer"] == created["final_answer"]
        assert detail.json()["steps"] == created["steps"]


def test_api_agent_run_unknown_student_no_500():
    client = _make_client()
    with client:
        r = client.post("/api/agent/run",
                        json={"goal": "Rank jobs", "student_id": "stu-does-not-exist"})
        # must never 500; degraded run is acceptable
        assert r.status_code in {201, 400, 404}
        if r.status_code == 201:
            assert r.json()["status"] != "running"


def test_api_agent_run_unknown_run_404():
    client = _make_client()
    with client:
        assert client.get("/api/agent/runs/run-none").status_code == 404


def test_resolve_nested_dot_paths_and_step_aliases():
    mem = {
        "_step_1": {"student_id": "stu-123"},
        "rank_and_explain_jobs": {
            "top_job_id": "job-001",
            "matches": [{"job_id": "job-001", "score": 92}, {"job_id": "job-002", "score": 81}],
        },
    }
    assert _resolve("$rank_and_explain_jobs.matches.0.job_id", mem) == "job-001"
    assert _resolve("$last.matches.1.score", mem) == 81
    assert _resolve("$step_1.student_id", mem) == "stu-123"


def test_verify_sop_numeric_claims_flags_hallucinated_thresholds():
    known = {"SOP-04", "SOP-03"}
    bad = verify_text("Under SOP-04 requires a score of 40 to qualify.", known)
    assert any("SOP-04:mock_threshold=40!=65" in u for u in bad["unverified"])
    assert bad["grounding_score"] < 100

    good = verify_text("Under SOP-04 requires a score of 65 to qualify.", known)
    assert good["unverified"] == []
    assert good["grounding_score"] == 100


def test_mock_llm_plans_all_specialized_tools():
    llm = MockLLM()
    for goal, expected_tool in [
        ("Generate a prerequisite DAG roadmap for my missing skills", "generate_prerequisite_roadmap"),
        ("Search live market jobs for backend engineer", "search_live_market_jobs"),
        ("Purge live market jobs from the database", "purge_live_market_jobs"),
        ("Show my past interview history and rubric breakdown", "inspect_interview_history"),
        ("Persist upskilling plan for my top match", "persist_upskilling_plan"),
        ("Submit drive application for my top role", "submit_drive_application"),
    ]:
        raw = llm.complete(
            "TASK: plan_steps",
            json.dumps({"goal": goal, "student_id": "stu-1", "actor_role": "tpo_admin"}),
            json_mode=True,
        )
        steps = json.loads(raw)["steps"]
        assert any(s["tool"] == expected_tool for s in steps), f"Missing {expected_tool} for goal={goal!r}"

