"""Milestone 16: 15-tool registry, iterative ReAct loop, observation-driven
re-planning, and the awaiting_input clarification lifecycle."""
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.agent.orchestrator import resume_agent_run, run_agent
from app.agent.tools import TOOL_REGISTRY, call_tool
from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import AgentRun, AgentStep, Interview, UpskillPlan
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _session_env(cohort: bool = False):
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
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


def _seed_student(client, name="M16 Probe", branch="CSE", year=3, gpa=8.0) -> str:
    r = client.post(
        "/api/students",
        files={"file": ("r.txt",
                        b"Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n"
                        b"Backend internship building REST APIs\n", "text/plain")},
        data={"name": name, "branch": branch, "year": str(year), "gpa": str(gpa)},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# ------------------------------------------------------------ 18-tool registry
def test_tool_registry_has_exactly_15_tools():
    assert len(TOOL_REGISTRY) == 18
    expected = {
        "inspect_student_profile", "rank_and_explain_jobs", "simulate_skill_acquisition",
        "search_placement_policy_rag", "search_course_catalog",
        "generate_prerequisite_roadmap", "audit_resume_bullets",
        "query_cohort_bottlenecks", "draft_cohort_intervention", "propose_hitl_approval",
        # M16 + live market + drive application & simulation tools
        "check_drive_eligibility", "persist_upskilling_plan", "inspect_interview_history",
        "search_live_market_jobs", "purge_live_market_jobs", "ask_clarifying_question",
        "submit_drive_application", "simulate_placement_drive",
    }
    assert set(TOOL_REGISTRY) == expected


def test_tool_check_drive_eligibility_standalone():
    session, ctx = _session_env()
    with session:
        sid = "stu-elig001"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Elig", branch="CSE", year=3, gpa=8.4,
                            resume_text="Skills: Python, FastAPI, Docker, Git\n"))
        for sk in ("python", "fastapi", "docker", "git"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=2,
                                     level="intermediate", evidence=[sk]))
        session.commit()
        out = call_tool("check_drive_eligibility", ctx, session,
                        student_id=sid, job_id="job-001")
        assert out["student_id"] == sid
        assert out["job_id"] == "job-001"
        assert {"eligible", "status", "checks", "cited_sops"} <= set(out)
        assert out["cited_sops"], "must cite SOPs"
        assert any(c["sop_id"] for c in out["checks"])


def test_tool_persist_upskilling_plan_creates_plan_rows():
    session, ctx = _session_env()
    with session:
        sid = "stu-plan001"
        from app.models import PlanItem, Student, StudentSkill
        session.add(Student(id=sid, name="Planner", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python, FastAPI, PostgreSQL, Docker\n"))
        for sk in ("python", "fastapi", "postgresql", "docker"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=2,
                                     level="basic", evidence=[sk]))
        session.commit()
        out = call_tool("persist_upskilling_plan", ctx, session,
                        student_id=sid, job_id="job-001")
        assert out["plan_id"].startswith("plan-")
        assert out["job_id"] == "job-001"
        assert out["total_hours"] >= 0
        assert isinstance(out["items"], list)
        plan = session.get(UpskillPlan, out["plan_id"])
        assert plan is not None and plan.student_id == sid
        items = session.exec(select(PlanItem).where(PlanItem.plan_id == out["plan_id"])).all()
        assert len(items) == out["item_count"]
        # idempotent re-run overwrites, never duplicates
        out2 = call_tool("persist_upskilling_plan", ctx, session,
                         student_id=sid, job_id="job-001")
        assert out2["plan_id"] == out["plan_id"]
        all_plans = session.exec(select(UpskillPlan).where(UpskillPlan.student_id == sid)).all()
        assert len(all_plans) == 1


def test_tool_inspect_interview_history_empty_and_aggregated():
    session, ctx = _session_env()
    with session:
        sid = "stu-int001"
        from app.models import Student
        session.add(Student(id=sid, name="Int", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python\n"))
        session.commit()
        empty = call_tool("inspect_interview_history", ctx, session, student_id=sid)
        assert empty["interview_count"] == 0
        assert empty["status"] == "no_completed_interviews"
        assert empty["weak_skills"] == [] and empty["strong_skills"] == []

        session.add(Interview(id="int-aaaaaaaa", student_id=sid, job_id="job-001",
                              transcript=[], score=85,
                              feedback={"rubric_overall": {"technical_accuracy": 90,
                                                           "depth_and_examples": 80,
                                                           "communication_clarity": 85},
                                        "per_skill": {"python": 88, "git": 40}},
                              finished=True))
        session.add(Interview(id="int-bbbbbbbb", student_id=sid, job_id="job-001",
                              transcript=[], score=45,
                              feedback={"rubric_overall": {"technical_accuracy": 50,
                                                           "depth_and_examples": 40,
                                                           "communication_clarity": 45},
                                        "per_skill": {"python": 60, "redis": 30}},
                              finished=True))
        session.commit()
        out = call_tool("inspect_interview_history", ctx, session, student_id=sid)
        assert out["interview_count"] == 2
        assert out["average_score"] == 65
        assert out["rubric_averages"]["technical_accuracy"] == 70
        assert "redis" in out["weak_skills"] and "git" in out["weak_skills"]
        assert out["latest_scores"] == [85, 45]


def test_tool_search_live_market_jobs_filters_and_ranks():
    session, ctx = _session_env()
    with session:
        sid = "stu-search01"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Searcher", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python, FastAPI, Docker, Git\n"))
        for sk in ("python", "fastapi", "docker", "git"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=2,
                                     level="basic", evidence=[sk]))
        session.commit()
        out = call_tool("search_live_market_jobs", ctx, session,
                        student_id=sid, query="backend")
        assert out["count"] >= 1
        # ranker scores only jobs it actually ranked; unranked-but-matching jobs
        # keep score=None (search filters on title/company/description haystack)
        assert all(j["score"] is None or j["score"] >= 0 for j in out["jobs"])
        assert any(j["score"] is not None for j in out["jobs"]), (
            "at least one returned job should carry a rank score")
        # query filter matched against title+company+description haystack;
        # verify by re-querying with a token that appears in title/company
        out2 = call_tool("search_live_market_jobs", ctx, session,
                         student_id=sid, query="backend engineer")
        assert all("backend engineer" in j["title"].lower() for j in out2["jobs"])
        # fresher_only filters out senior roles
        fresher = call_tool("search_live_market_jobs", ctx, session, fresher_only=True)
        assert all(j["is_fresher"] for j in fresher["jobs"])
        # location filter (empty match is fine, just must not crash)
        loc = call_tool("search_live_market_jobs", ctx, session, location="zzz-nope")
        assert loc["count"] == 0


def test_tool_ask_clarifying_question_shape():
    session, ctx = _session_env()
    with session:
        out = call_tool("ask_clarifying_question", ctx, session,
                        question="Which role?", options=["a", "b"],
                        missing_fields=["role"])
        assert out == {"status": "awaiting_input", "requires_user_input": True,
                       "question": "Which role?", "options": ["a", "b"],
                       "missing_fields": ["role"]}
        minimal = call_tool("ask_clarifying_question", ctx, session, question="Hi")
        assert minimal["options"] == [] and minimal["missing_fields"] == []


# ------------------------------------------- observation-driven re-planning
def test_eligibility_goal_injects_check_drive_eligibility():
    session, ctx = _session_env()
    with session:
        sid = "stu-inj001"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Inj", branch="CSE", year=3, gpa=8.2,
                            resume_text="Skills: Python, FastAPI, Docker, Git, Redis\n"))
        for sk in ("python", "fastapi", "docker", "git", "redis"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=3,
                                     level="intermediate", evidence=[sk]))
        session.commit()
        run = run_agent(ctx, session,
                        goal="Nominate me for the placement drive and apply",
                        student_id=sid, actor_role="student")
        steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)
                             .order_by(AgentStep.step_order)).all()
        step_tools = [s.tool_name for s in steps]
        assert "rank_and_explain_jobs" in step_tools
        assert "check_drive_eligibility" in step_tools, (
            f"observation injection must add check_drive_eligibility; got {step_tools}")
        # contiguous step orders even with injected steps
        assert [s.step_order for s in steps] == list(range(1, len(steps) + 1))
        assert run.status in {"completed", "awaiting_approval"}
        assert len(run.plan_json) >= 3


def test_blocked_eligibility_appends_remediation_plan_tool():
    session, ctx = _session_env()
    with session:
        # student with ONLY git (thin profile -> SOP-05 skill readiness blocked)
        sid = "stu-block01"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Blocked", branch="CSE", year=3, gpa=8.5,
                            resume_text="Skills: Git\n"))
        session.add(StudentSkill(student_id=sid, skill_id="git", mentions=1,
                                 level="basic", evidence=["git"]))
        session.commit()
        run = run_agent(ctx, session,
                        goal="Check my eligibility and apply for the drive",
                        student_id=sid, actor_role="student")
        step_tools = [s.tool_name for s in session.exec(
            select(AgentStep).where(AgentStep.run_id == run.id)
            .order_by(AgentStep.step_order)).all()]
        assert "check_drive_eligibility" in step_tools
        # blocked run must append a remediation tool (plan or interview history)
        assert ("persist_upskilling_plan" in step_tools
                or "inspect_interview_history" in step_tools), step_tools
        # bounded: never exceeds the executed-step cap
        assert len(step_tools) <= 8


def test_non_eligibility_goal_does_not_inject_eligibility_step():
    session, ctx = _session_env()
    with session:
        sid = "stu-noinj01"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Plain", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python, FastAPI, PostgreSQL, Docker, Git\n"))
        for sk in ("python", "fastapi", "postgresql", "docker", "git"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=2,
                                     level="basic", evidence=[sk]))
        session.commit()
        run = run_agent(ctx, session, goal="Rank jobs", student_id=sid,
                        actor_role="student")
        step_tools = [s.tool_name for s in session.exec(
            select(AgentStep).where(AgentStep.run_id == run.id)).all()]
        assert "check_drive_eligibility" not in step_tools


# ------------------------------------------- awaiting_input clarification flow
def test_ambiguous_goal_pauses_awaiting_input():
    session, ctx = _session_env()
    with session:
        sid = "stu-clar001"
        from app.models import Student
        session.add(Student(id=sid, name="Clar", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python\n"))
        session.commit()
        run = run_agent(ctx, session, goal="help me with my career today",
                        student_id=sid, actor_role="student")
        assert run.status == "awaiting_input"
        assert run.final_answer, "clarifying question must be stored as final_answer"
        assert "?" in run.final_answer
        steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
        assert any(s.tool_name == "ask_clarifying_question" for s in steps)
        # step output carries the structured question payload
        clar = next(s for s in steps if s.tool_name == "ask_clarifying_question")
        assert clar.tool_output.get("requires_user_input") is True
        assert clar.tool_output.get("options")


def test_resume_agent_run_completes_after_reply():
    session, ctx = _session_env()
    with session:
        sid = "stu-res0001"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Res", branch="CSE", year=3, gpa=8.1,
                            resume_text="Skills: Python, FastAPI, Docker, Git, Redis\n"))
        for sk in ("python", "fastapi", "docker", "git", "redis"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=3,
                                     level="basic", evidence=[sk]))
        session.commit()
        run = run_agent(ctx, session, goal="help me with my career today",
                        student_id=sid, actor_role="student")
        assert run.status == "awaiting_input"
        run_id = run.id

        run2 = resume_agent_run(ctx, session, run_id=run_id,
                                user_reply="Rank my top job matches")
        assert run2.status in {"completed", "awaiting_approval"}
        assert "User clarification: Rank my top job matches" in run2.goal
        steps = session.exec(select(AgentStep).where(AgentStep.run_id == run_id)
                             .order_by(AgentStep.step_order)).all()
        assert any(s.tool_name == "user_reply" for s in steps), "reply must be traced"
        # new work executed after resume
        assert any(s.tool_name not in ("ask_clarifying_question", "user_reply")
                   for s in steps)
        # orders remain contiguous across both phases
        assert [s.step_order for s in steps] == list(range(1, len(steps) + 1))
        assert run2.final_answer


def test_resume_unknown_or_wrong_status_raises():
    session, ctx = _session_env()
    with session:
        import pytest
        with pytest.raises(ValueError):
            resume_agent_run(ctx, session, run_id="run-none", user_reply="x")
        sid = "stu-res0002"
        from app.models import Student
        session.add(Student(id=sid, name="Res2", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python\n"))
        session.commit()
        run = run_agent(ctx, session, goal="Rank jobs", student_id=sid)
        assert run.status != "awaiting_input"
        with pytest.raises(ValueError):
            resume_agent_run(ctx, session, run_id=run.id, user_reply="x")
        with pytest.raises(ValueError):
            # not awaiting + empty reply must also raise
            resume_agent_run(ctx, session, run_id=run.id, user_reply="   ")


# ------------------------------------------------------------ API lifecycle
def test_api_agent_run_awaiting_then_reply_completed():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        r = client.post("/api/agent/run",
                        json={"goal": "Which company should I target?",
                              "student_id": sid})
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "awaiting_input"
        assert body["final_answer"]
        run_id = body["id"]

        r2 = client.post(f"/api/agent/runs/{run_id}/reply",
                         json={"reply": "Check my eligibility for the drive"})
        assert r2.status_code == 200, r2.text
        done = r2.json()
        assert done["status"] in {"completed", "awaiting_approval"}
        assert "User clarification" in done["goal"]
        orders = [s["step_order"] for s in done["steps"]]
        assert orders == list(range(1, len(orders) + 1))
        # persisted
        with Session(client.app.state.ctx.engine) as db:
            run = db.get(AgentRun, run_id)
            assert run.status == done["status"]


def test_api_reply_bad_requests(client):
    assert client.post("/api/agent/runs/run-none/reply",
                       json={"reply": "hi"}).status_code == 404
    # reply on a run that is not awaiting_input -> 409
    sid = _seed_student(client)
    run = client.post("/api/agent/run",
                      json={"goal": "Rank jobs", "student_id": sid}).json()
    assert client.post(f"/api/agent/runs/{run['id']}/reply",
                       json={"reply": "hi"}).status_code == 409
    # empty reply -> 400
    clar = client.post("/api/agent/run",
                       json={"goal": "help me with my career today",
                             "student_id": sid}).json()
    assert clar["status"] == "awaiting_input"
    assert client.post(f"/api/agent/runs/{clar['id']}/reply",
                       json={"reply": "  "}).status_code == 400


def test_api_reply_respects_student_scope(client):
    client = _make_client()
    with client:
        sid_a = _seed_student(client, "Scope A")
        sid_b = _seed_student(client, "Scope B")
        clar = client.post("/api/agent/run",
                           json={"goal": "help me with my career today",
                                 "student_id": sid_a}).json()
        hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_b}
        r = client.post(f"/api/agent/runs/{clar['id']}/reply",
                        json={"reply": "rank jobs"}, headers=hdr)
        assert r.status_code == 403
        hdr_a = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_a}
        ok = client.post(f"/api/agent/runs/{clar['id']}/reply",
                         json={"reply": "rank jobs"}, headers=hdr_a)
        assert ok.status_code == 200


def test_max_executed_steps_bounded():
    """The dynamic queue must never exceed MAX_EXECUTED_STEPS."""
    from app.agent.orchestrator import MAX_EXECUTED_STEPS
    session, ctx = _session_env()
    with session:
        sid = "stu-cap0001"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Cap", branch="CSE", year=3, gpa=8.2,
                            resume_text="Skills: Python, FastAPI, Docker, Git\n"))
        for sk in ("python", "fastapi", "docker", "git"):
            session.add(StudentSkill(student_id=sid, skill_id=sk, mentions=3,
                                     level="basic", evidence=[sk]))
        session.commit()
        run = run_agent(ctx, session,
                        goal="Nominate and apply for the drive after checking eligibility",
                        student_id=sid, actor_role="student")
        steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
        assert len(steps) <= MAX_EXECUTED_STEPS


def test_tool_aliases_and_prompt_grounding():
    """Verify tool aliases map common external LLM names to canonical tools."""
    from app.agent.orchestrator import PLAN_SYSTEM
    from app.agent.tools import TOOL_ALIASES, call_tool
    client = _make_client()
    with client:
        sid = _seed_student(client, "Alias Test")
        ctx = client.app.state.ctx
        with Session(ctx.engine) as session:
            # 1. get_student_profile -> inspect_student_profile
            assert "get_student_profile" in TOOL_ALIASES
            prof = call_tool("get_student_profile", ctx, session, student_id=sid)
            assert prof["student_id"] == sid
            assert "skills" in prof
            # 2. search_jobs -> search_live_market_jobs
            jobs = call_tool("search_jobs", ctx, session, student_id=sid)
            assert "jobs" in jobs
            # 3. rank_job_matches / rank_jobs -> rank_and_explain_jobs with top_n alias
            ranked = call_tool("rank_job_matches", ctx, session, student_id=sid, top_n=2)
            assert ranked["student_id"] == sid
            assert "matches" in ranked
            # 4. PLAN_SYSTEM documents the exact tools for external LLMs
            assert "inspect_student_profile" in PLAN_SYSTEM
            assert "rank_and_explain_jobs" in PLAN_SYSTEM
            assert "search_live_market_jobs" in PLAN_SYSTEM
            assert "submit_drive_application" in PLAN_SYSTEM
            assert "simulate_placement_drive" in PLAN_SYSTEM


def test_new_tools_submit_drive_application_and_simulate_placement_drive():
    """Verify submit_drive_application and simulate_placement_drive tools in the 18-tool registry."""
    from app.agent.tools import call_tool
    client = _make_client()
    with client:
        sid = _seed_student(client, "Drive Tool Student")
        ctx = client.app.state.ctx
        with Session(ctx.engine) as session:
            sim = call_tool(
                "simulate_placement_drive",
                ctx,
                session,
                title="Prospective AI Drive",
                salary_lpa=[10, 14],
                min_gpa=7.0,
                required_skills=["python"],
            )
            assert "eligible_now_count" in sim
            assert "bootcamp_intervention_roi" in sim

            app_res = call_tool(
                "submit_drive_application",
                ctx,
                session,
                student_id=sid,
                request_waiver=True,
                note="Applying via ReAct agent tool",
            )
            assert "status" in app_res
            assert app_res["job_id"]


def test_step_repair_recovers_missing_student_id():
    """Verify _repair_failed_step uses REPAIR_SYSTEM to fill missing student_id and recover."""
    from app.agent.orchestrator import _repair_failed_step
    from app.agent.tools import call_tool
    session, ctx = _session_env()
    with session:
        sid = "stu-rep0001"
        from app.models import Student, StudentSkill
        session.add(Student(id=sid, name="Repair", branch="CSE", year=3, gpa=8.2,
                            resume_text="Skills: Python, FastAPI\n"))
        session.add(StudentSkill(student_id=sid, skill_id="python", mentions=2,
                                 level="basic", evidence=["python"]))
        session.commit()
        repaired_args = _repair_failed_step(
            ctx.llm,
            tool_name="inspect_student_profile",
            args={},
            student_id=sid,
            error_str="missing required argument: student_id",
            memory={},
        )
        assert repaired_args.get("student_id") == sid
        out = call_tool("inspect_student_profile", ctx, session, **repaired_args)
        assert out.get("student_id") == sid

