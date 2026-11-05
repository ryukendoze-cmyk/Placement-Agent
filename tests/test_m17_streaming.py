"""Milestone 17: SSE streaming - LLMClient.stream, agent run stream endpoint,
copilot stream endpoint, and frontend progressive-rendering wiring."""
import json
import re

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.agent.orchestrator import run_agent_stream
from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import AgentRun, Student
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse ``event: <name>\\ndata: <json>\\n\\n`` chunks into (name, data)."""
    events: list[tuple[str, dict]] = []
    name = None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line[len("event: "):]
        elif line.startswith("data: ") and name is not None:
            events.append((name, json.loads(line[len("data: "):])))
            name = None
    return events


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _session_env():
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    session = Session(eng)
    seed(session, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    return session, ctx


def _make_client() -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    return TestClient(create_app(ctx))


def _seed_student(client, name="M17 Probe") -> str:
    r = client.post(
        "/api/students",
        files={"file": ("r.txt",
                        b"Skills: Python, FastAPI, PostgreSQL, Docker, Git, Redis\n"
                        b"Backend internship building REST APIs\n", "text/plain")},
        data={"name": name, "branch": "CSE", "year": "3", "gpa": "8.2"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------------- 17.1 LLMClient streaming
def test_mock_llm_stream_yields_word_chunks_matching_complete():
    m = MockLLM()
    user = json.dumps({"goal": "Rank jobs", "student_id": "s", "actor_role": "student"})
    chunks = list(m.stream("TASK: plan_steps", user, json_mode=True))
    assert chunks, "stream must yield at least one chunk"
    joined = "".join(chunks)
    assert joined == m.complete("TASK: plan_steps", user, json_mode=True)
    # every chunk is 4-6 words (or the final remainder)
    for c in chunks[:-1]:
        assert 4 <= len(c.split()) <= 6, f"chunk size out of band: {c!r}"
    # JSON payload must stay valid when reassembled
    json.loads(joined)


def test_mock_llm_stream_empty_text():
    m = MockLLM()
    out = list(m.stream("TASK: echo", ""))
    assert isinstance(out, list)


def test_llm_protocol_declares_stream():
    from app.llm.base import LLMClient
    assert hasattr(LLMClient, "stream")


# --------------------------------------------- 17.2 run_agent_stream events
def test_run_agent_stream_emits_full_lifecycle():
    session, ctx = _session_env()
    with session:
        sid = "stu-sse001"
        session.add(Student(id=sid, name="Sse", branch="CSE", year=3, gpa=8.2,
                            resume_text="Skills: Python, FastAPI, Docker, Git\n"))
        session.commit()
        events = list(run_agent_stream(ctx, session, goal="Rank my top job matches",
                                       student_id=sid, actor_role="student"))
        names = [n for n, _ in events]
        assert names[0] == "run_started"
        assert "plan_created" in names
        assert "step_started" in names
        assert "step_completed" in names
        assert "verification_done" in names
        assert names[-1] == "complete"
        # step_started/step_completed interleave in order
        started = [p["step_order"] for n, p in events if n == "step_started"]
        completed = [p["step_order"] for n, p in events if n == "step_completed"]
        assert started == completed
        assert started == sorted(started)
        # final complete payload has citations + status
        final = dict(events[-1][1])
        assert final["status"] in {"completed", "awaiting_approval"}
        assert "citations" in final
        # run persisted
        runs = session.exec(select(AgentRun)).all()
        assert any(r.id == final["id"] for r in runs)


def test_run_agent_stream_persists_run_and_steps():
    session, ctx = _session_env()
    with session:
        sid = "stu-sse002"
        session.add(Student(id=sid, name="Sse2", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python\n"))
        session.commit()
        events = list(run_agent_stream(ctx, session, goal="Rank jobs",
                                       student_id=sid, actor_role="student"))
        final = dict(events[-1][1])
        run = session.get(AgentRun, final["id"])
        assert run is not None
        assert run.status == final["status"]
        from app.models import AgentStep
        steps = session.exec(select(AgentStep).where(AgentStep.run_id == run.id)).all()
        assert len(steps) >= 1


def test_run_agent_stream_clarification_emits_awaiting_input_not_complete():
    session, ctx = _session_env()
    with session:
        sid = "stu-sse003"
        session.add(Student(id=sid, name="Clar", branch="CSE", year=3, gpa=8.0,
                            resume_text="Skills: Python\n"))
        session.commit()
        events = list(run_agent_stream(ctx, session,
                                       goal="help me with my career today",
                                       student_id=sid, actor_role="student"))
        names = [n for n, _ in events]
        assert names[-1] == "awaiting_input"
        assert "complete" not in names
        payload = dict(events[-1][1])
        assert payload["question"]
        # persisted as awaiting_input
        runs = session.exec(select(AgentRun)).all()
        assert any(r.status == "awaiting_input" for r in runs)


# ------------------------------------------- 17.2 SSE endpoint (agent run)
def test_api_agent_run_stream_sse_contract(client):
    sid = _seed_student(client)
    resp = client.post("/api/agent/run/stream",
                       json={"goal": "Rank my top job matches", "student_id": sid})
    assert resp.status_code == 200, resp.text[:400]
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"

    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names[0] == "run_started"
    assert names[-1] == "complete"
    for required in ("plan_created", "step_started", "step_completed", "verification_done"):
        assert required in names, f"missing {required}: {names}"
    # complete payload matches _run_payload shape
    final = dict(events[-1][1])
    assert final["id"].startswith("run-")
    assert final["status"] in {"completed", "awaiting_approval"}
    assert isinstance(final["steps"], list) and final["steps"]
    # persisted run readable via GET
    runs = client.get("/api/agent/runs").json()
    assert any(r["id"] == final["id"] for r in runs)


def test_api_agent_run_stream_requires_goal(client):
    assert client.post("/api/agent/run/stream", json={"goal": "  "}).status_code == 400


# ----------------------------------------------- 17.3 copilot stream endpoint
def test_api_copilot_stream_sse_contract(client):
    resp = client.post("/api/copilot/ask/stream",
                       json={"question": "What score do I need for Dream drives?"})
    assert resp.status_code == 200, resp.text[:400]
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names[0] == "retrieval"
    assert names[-1] == "complete"
    assert "token" in names, f"expected token events: {names}"

    retrieval = dict(next(d for n, d in events if n == "retrieval"))
    assert "sources" in retrieval
    assert "student_context" in retrieval
    assert "trace" in retrieval

    # complete carries citation-verified answer with [SOP-XX] citations
    final = dict(events[-1][1])
    assert "answer" in final
    assert "citations" in final
    assert final["citations"]["verified"], "must have verified SOP citations"
    assert all(v.startswith("SOP-") for v in final["citations"]["verified"])
    assert final["citations"]["grounding_score"] >= 0


def test_api_copilot_stream_requires_question(client):
    assert client.post("/api/copilot/ask/stream", json={"question": ""}).status_code == 400


def test_api_copilot_ask_still_works_after_refactor(client):
    r = client.post("/api/copilot/ask", json={"question": "How many backlogs are allowed?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body and "citations" in body
    assert body["citations"]["verified"]


# ------------------------------------------------------- 17.4 frontend wiring
def test_frontend_templates_wire_streaming():
    base = open("app/templates/base.html", encoding="utf-8").read()
    copilot = open("app/templates/copilot.html", encoding="utf-8").read()
    # agent brain drawer streams from run/stream
    assert "/api/agent/run/stream" in base
    assert "runPipelineStreaming" in base
    # copilot streams from ask/stream with JSON fallback
    assert "/api/copilot/ask/stream" in copilot
    assert "/api/copilot/ask" in copilot  # fallback retained
    # preserve required DOM ids
    for dom_id in ("drawer-run", "drawer-trace", "chat-log", "copilot-trace", "copilot-send"):
        assert dom_id in (base + copilot), f"missing DOM id {dom_id}"
    assert ".citation-pill" in copilot
    # zero-emoji: no literal emoji in the streaming additions
    emoji = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
    assert emoji.search(base) is None, "base.html contains literal emoji"
    assert emoji.search(copilot) is None, "copilot.html contains literal emoji"
