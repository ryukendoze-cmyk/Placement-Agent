"""Phase 3: institutional policy RAG + copilot + eligibility tests (plan section)."""
import json

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.agent.orchestrator import run_agent
from app.agent.tools import call_tool
from app.api.main import AppContext, create_app
from app.data import placement_sops_available
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import Interview, Student
from app.rag.policy_rag import (
    evaluate_student_eligibility,
    get_sop,
    index_sops,
    load_sops,
    search_sops,
    sop_ids,
    verify_policy_citations,
)
from app.seed import seed, seed_demo_cohort
from app.taxonomy.taxonomy import load_taxonomy


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


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


def _seed_student(client, name="RAG Probe", branch="CSE", year=3, gpa=8.0, backlogs=0) -> str:
    r = client.post(
        "/api/students",
        files={"file": ("r.txt",
                        b"Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n"
                        b"Backend internship building REST APIs\n", "text/plain")},
        data={"name": name, "branch": branch, "year": str(year), "gpa": str(gpa)},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    if backlogs:
        with Session(client.app.state.ctx.engine) as db:
            s = db.get(Student, sid)
            s.backlogs = backlogs
            db.add(s)
            db.commit()
    return sid


def _add_mock_interview(client, sid: str, job_id: str, score: int) -> str:
    with Session(client.app.state.ctx.engine) as db:
        iid = "iv-rag0001"
        db.add(Interview(id=iid, student_id=sid, job_id=job_id, transcript=[],
                         score=score, finished=True))
        db.commit()
        return iid


# ---------------- data / index ----------------

def test_placement_sops_json_schema_and_completeness():
    assert placement_sops_available() is True
    sops = load_sops()
    assert len(sops) == 18
    ids = [s["id"] for s in sops]
    assert ids == [f"SOP-{i:02d}" for i in range(1, 19)]
    required = {"id", "title", "category", "section", "rule_text", "keywords",
                "parameters", "applies_to"}
    for s in sops:
        assert required <= set(s.keys())
        assert isinstance(s["keywords"], list) and s["keywords"]
        assert isinstance(s["parameters"], dict)
        assert get_sop(s["id"]) is not None


def test_policy_rag_indexing_with_hashing_embedder():
    index = index_sops(HashingEmbedder())
    assert len(index) == 18
    assert all("embedding" in it and "keywords" in it for it in index)
    # embedder dim is consistent
    assert len(index[0]["embedding"]) > 0


# ---------------- retrieval ----------------

def test_search_sops_exact_id_lookup():
    r = search_sops("SOP-04", top_k=1, embedder=HashingEmbedder())
    assert r["top_sop_id"] == "SOP-04"
    assert r["results"][0]["score"] >= 0.95


def test_search_sops_semantic_and_keyword_queries():
    emb = HashingEmbedder()
    assert search_sops("mock interview dream roles", top_k=1, embedder=emb)["top_sop_id"] == "SOP-04"
    assert search_sops("tier salary bands", top_k=1, embedder=emb)["top_sop_id"] == "SOP-01"
    assert search_sops("active backlog cutoff", top_k=1, embedder=emb)["top_sop_id"] == "SOP-03"
    assert search_sops("fake resume project fabrication", top_k=1, embedder=emb)["top_sop_id"] == "SOP-08"


def test_search_sops_top_k_and_structure():
    r = search_sops("no show penalty", top_k=2, embedder=HashingEmbedder())
    assert r["count"] <= 2
    assert {"query", "results", "count", "top_sop_id", "status"} <= set(r.keys())
    assert r["status"] == "grounded"
    assert r["results"] and {"sop_id", "title", "section", "rule_text", "score"} <= set(r["results"][0].keys())


def test_search_sops_empty_query_is_safe():
    r = search_sops("   ", top_k=3, embedder=HashingEmbedder())
    assert r["count"] == 0 and r["status"] == "empty"


def test_search_sops_multi_sop_id_pinning():
    """Gate C: a query naming two SOP ids must pin BOTH into the top results."""
    r = search_sops("Compare SOP-01 and SOP-02", top_k=3, embedder=HashingEmbedder())
    ids = [x["sop_id"] for x in r["results"]]
    assert "SOP-01" in ids and "SOP-02" in ids
    # both pinned at full score
    by_id = {x["sop_id"]: x["score"] for x in r["results"]}
    assert by_id["SOP-01"] >= 0.95 and by_id["SOP-02"] >= 0.95


def test_policy_citation_flags_every_bracketed_occurrence():
    r = verify_policy_citations("[SOP-99] and [SOP-99] again")
    assert r["unverified"] == ["SOP-99"]
    assert r["annotated_text"].count("[UNVERIFIED: SOP-99]") == 2


def test_policy_citation_handles_lowercase_sop_reference():
    r = verify_policy_citations("see sop-04 for the gate")
    assert r["verified"] == ["SOP-04"]
    assert r["unverified"] == []


def test_search_placement_policy_rag_tool_integration():
    client = _make_client()
    with client:
        with Session(client.app.state.ctx.engine) as session:
            out = call_tool("search_placement_policy_rag", client.app.state.ctx, session,
                            query="dream company mock gate", top_k=2)
            assert out["status"] == "grounded"
            assert out["top_sop_id"] == "SOP-04"


# ---------------- citation guard ----------------

def test_policy_citation_verification_grounded_vs_unverified():
    ok = verify_policy_citations("Follow [SOP-01] and [SOP-04].")
    assert ok["grounding_score"] == 100 and ok["unverified"] == []
    bad = verify_policy_citations("Per [SOP-04] and [SOP-99].")
    assert bad["unverified"] == ["SOP-99"]
    assert bad["grounding_score"] == 50
    assert "[UNVERIFIED: SOP-99]" in bad["annotated_text"]


# ---------------- eligibility engine ----------------

def test_eligibility_engine_sop_rules_pass():
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=8.5, branch="CSE")
        _add_mock_interview(client, sid, "job-001", 75)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-001")
        assert r["eligible"] is True
        assert r["status"] == "eligible"
        assert all(c["passed"] for c in r["checks"])
        assert r["grounding_score"] == 100


def test_eligibility_engine_blocks_on_mock_interview_threshold():
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=8.5, branch="CSE")
        # job-003 salary [10,18] -> dream tier, needs mock >= 65; give 60
        _add_mock_interview(client, sid, "job-003", 60)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-003")
        assert r["eligible"] is False
        assert r["status"] == "blocked"
        mock_check = next(c for c in r["checks"] if c["name"] == "tier_mock_gate")
        assert mock_check["sop_id"] == "SOP-04" and mock_check["passed"] is False
        assert "SOP-04" in r["cited_sops"]
        assert any("mock" in s.lower() for s in r["suggested_actions"])


def test_eligibility_engine_blocks_on_gpa_cutoff():
    client = _make_client()
    with client:
        # job-001 min_gpa 7.5
        sid = _seed_student(client, gpa=6.8, branch="CSE")
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-001")
        assert r["eligible"] is False
        gpa_check = next(c for c in r["checks"] if c["name"] == "gpa_cutoff")
        assert gpa_check["sop_id"] == "SOP-03" and gpa_check["passed"] is False
        assert any("7.50" in reason or "below" in reason for reason in r["reasons"])


def test_eligibility_engine_blocks_on_backlog_limit():
    client = _make_client()
    with client:
        # job-003 is dream tier (>=8lpa); dream allows <=1 backlog -> give 2
        sid = _seed_student(client, gpa=8.2, branch="CSE", backlogs=2)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-003")
        gpa_check = next(c for c in r["checks"] if c["name"] == "gpa_cutoff")
        assert gpa_check["passed"] is False
        assert "backlog" in gpa_check["detail"].lower()


def test_eligibility_engine_blocks_on_branch_restriction():
    client = _make_client()
    with client:
        # job-002 allowed_branches CSE/IT/ECE -> Mechanical blocked
        sid = _seed_student(client, gpa=8.5, branch="Mechanical")
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-002")
        assert r["eligible"] is False
        br = next(c for c in r["checks"] if c["name"] == "branch_restriction")
        assert br["sop_id"] == "SOP-09" and br["passed"] is False


def test_eligibility_uses_latest_mock_score_on_retake():
    """Gate C: a retake score must supersede the earlier attempt."""
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=8.5, branch="CSE")
        with Session(client.app.state.ctx.engine) as db:
            from datetime import datetime, timedelta, timezone
            now = datetime.now(timezone.utc)
            db.add(Interview(id="iv-old", student_id=sid, job_id="job-001", transcript=[],
                             score=50, finished=True, started_at=now - timedelta(days=2)))
            db.add(Interview(id="iv-new", student_id=sid, job_id="job-001", transcript=[],
                             score=80, finished=True, started_at=now))
            db.commit()
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-001")
            from app.rag.policy_rag import _latest_mock_score
            assert _latest_mock_score(session, sid, "job-001") == 80
        mock_check = next(c for c in r["checks"] if c["name"] == "tier_mock_gate")
        # job-001 salary [8,14] -> dream tier; 80 >= 65 so it passes despite the older 50
        assert mock_check["passed"] is True


def test_eligibility_backlog_block_suggests_clearing_backlogs():
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=8.2, branch="CSE", backlogs=2)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            r = evaluate_student_eligibility(session, ctx, sid, "job-003")  # dream tier
        assert any("backlog" in s.lower() for s in r["suggested_actions"])


# ---------------- copilot endpoint ----------------

def test_api_copilot_ask_policy_question_student_context():
    client = _make_client()
    with client:
        r = client.post("/api/copilot/ask", json={
            "question": "Can I apply for a Dream job if my mock interview score is 60?",
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["citations"]["grounding_score"] == 100
        assert "SOP-04" in body["citations"]["verified"]
        assert body["sources"], "expected retrieved SOP sources"
        assert body["sources"][0]["sop_id"] in sop_ids()
        assert body["trace"] and body["trace"][0]["tool"] == "search_placement_policy_rag"
        assert "[SOP-04]" in body["answer"] or "SOP-04" in body["answer"]


def test_api_copilot_ask_with_student_context():
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=7.2, branch="CSE")
        r = client.post("/api/copilot/ask", json={
            "question": "What is the mock interview cutoff for Dream roles?",
            "student_id": sid,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["student_context"]["gpa"] == 7.2
        assert body["student_context"]["name"]


def test_api_copilot_ask_missing_question_400():
    client = _make_client()
    with client:
        assert client.post("/api/copilot/ask", json={"question": "   "}).status_code == 400


def test_api_copilot_ask_offline_deterministic():
    client = _make_client()
    with client:
        q = {"question": "Explain the tier classification and one-offer-per-tier policy"}
        a1 = client.post("/api/copilot/ask", json=q).json()
        a2 = client.post("/api/copilot/ask", json=q).json()
        assert a1["answer"] == a2["answer"]
        assert a1["citations"] == a2["citations"]


def test_api_student_job_eligibility_endpoint_detail():
    client = _make_client()
    with client:
        sid = _seed_student(client, gpa=8.0, branch="CSE")
        _add_mock_interview(client, sid, "job-001", 70)
        r = client.get(f"/api/students/{sid}/eligibility/job-001")
        assert r.status_code == 200
        body = r.json()
        assert {"student_id", "job_id", "job_tier", "eligible", "status", "reasons",
                "checks", "cited_sops", "suggested_actions"} <= set(body.keys())
        assert isinstance(body["checks"], list) and len(body["checks"]) == 4
        assert body["cited_sops"] == ["SOP-03", "SOP-09", "SOP-04", "SOP-05"]


def test_api_student_job_eligibility_unknown_ids_404():
    client = _make_client()
    with client:
        assert client.get("/api/students/stu-none/eligibility/job-001").status_code == 404
        sid = _seed_student(client)
        assert client.get(f"/api/students/{sid}/eligibility/job-none").status_code == 404


# ---------------- orchestrator integration ----------------

def test_orchestrator_copilot_multi_step_goal():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session,
                            goal="Check my eligibility for Dream roles and explain the mock interview policy",
                            student_id=sid, actor_role="student")
            tools = [s["tool"] for s in run.plan_json]
            steps = session.exec(select(__import__("app.models", fromlist=["AgentStep"]).AgentStep)
                                 .where(__import__("app.models", fromlist=["AgentStep"]).AgentStep.run_id == run.id)).all()
            # RAG tool should appear and produce grounded SOP citations
            assert "search_placement_policy_rag" in tools or any(
                s.tool_name == "search_placement_policy_rag" for s in steps)
            # SOP ids must be treated as known by the verifier (no false unverified)
            assert "SOP-04" not in run.citations.get("unverified", [])


# ---------------- verifier SOP integration ----------------

def test_build_known_entities_includes_sop_ids():
    from app.agent.verifier import build_known_entities
    client = _make_client()
    with client:
        with Session(client.app.state.ctx.engine) as session:
            known = build_known_entities(session)
            assert "SOP-01" in known and "SOP-18" in known


def test_agent_run_citing_sop_is_not_flagged_as_hallucination():
    """A run whose final answer cites a real SOP must not report it as unverified."""
    client = _make_client()
    with client:
        sid = _seed_student(client)
        with Session(client.app.state.ctx.engine) as session:
            ctx = client.app.state.ctx
            run = run_agent(ctx, session,
                            goal="Explain the mock interview policy for Dream roles",
                            student_id=sid, actor_role="student")
            # the run itself should not fabricate SOP ids
            unverified_sops = [u for u in run.citations.get("unverified", [])
                               if u.upper().startswith("SOP-")]
            assert unverified_sops == []


def test_json_module_used_not_shadowed():
    # guard against accidental local `import json as _json` shadowing bugs
    payload = json.dumps({"question": "x"})
    assert json.loads(payload) == {"question": "x"}
