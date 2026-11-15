"""Regression tests for strict interview scoring, WPM UI removal, and print Q&A transcript."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session

from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.interview.bank import Question, load_bank
from app.interview.engine import EVALUATE_ANSWER_SYSTEM, InterviewSession, _reconcile_evaluation
from app.interview.speech_analytics import analyze_speech_delivery
from app.llm.mock import MockLLM
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy

ROOT = Path(__file__).resolve().parents[1]


class _LenientStubLLM:
    """Simulates a lenient remote LLM (like mistral-nemo) returning inflated scores."""

    def __init__(self, response_dict: dict):
        self._resp = response_dict

    def complete(self, system: str, user: str, json_mode: bool = True) -> str:
        return json.dumps(self._resp)


def _fk_on(conn, _rec):
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _make_client() -> TestClient:
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(
        engine=eng,
        taxonomy=taxonomy,
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    return TestClient(create_app(ctx))


def _seed_student(client: TestClient) -> str:
    r = client.post(
        "/api/students",
        files={
            "file": (
                "resume.txt",
                b"Skills: Python, SQL, Git, Docker, FastAPI\nBuilt backend APIs.\n",
                "text/plain",
            )
        },
        data={"name": "Print Tester", "branch": "CSE", "year": "4", "gpa": "8.1"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_eval_prompt_enforces_zero_leniency_and_verdict_schema():
    assert "wrong_or_baseless" in EVALUATE_ANSWER_SYSTEM
    assert "vague_superficial" in EVALUATE_ANSWER_SYSTEM
    assert "accurate_claims" in EVALUATE_ANSWER_SYSTEM
    assert "errors_or_bluffs" in EVALUATE_ANSWER_SYSTEM


def test_mock_llm_fails_fabricated_reflog_and_gil_word_salad():
    llm = MockLLM()
    q_reflog = Question(
        id="q-git-reflog",
        role="backend",
        skill_id="git",
        difficulty=2,
        question="What role does the reflog play in recovering lost commits?",
        keywords=("reflog", "reset", "head", "sha"),
        outline=("Tracks local HEAD movements", "Allows recovering unreachable commits after reset"),
    )
    nonsense_reflog = (
        "Reflog plays an important role, it is the repeated flogging that is caused by "
        "code extraction resentment, this means overall reflog would be core architecture process in git."
    )
    res_reflog = json.loads(
        llm.complete(
            system=EVALUATE_ANSWER_SYSTEM,
            user=json.dumps(
                {
                    "question": q_reflog.question,
                    "answer": nonsense_reflog,
                    "keywords": list(q_reflog.keywords),
                    "outline": list(q_reflog.outline),
                }
            ),
            json_mode=True,
        )
    )
    rec_reflog = _reconcile_evaluation(q_reflog, nonsense_reflog, res_reflog)
    assert rec_reflog["score"] <= 22, rec_reflog

    q_gil = Question(
        id="q-py-gil",
        role="backend",
        skill_id="python",
        difficulty=2,
        question="How does the Global Interpreter Lock (GIL) impact the concurrency of threads in CPython?",
        keywords=("gil", "thread", "mutex", "bytecode", "multiprocessing"),
        outline=("Only one thread executes Python bytecode at a time", "Use multiprocessing for CPU-bound work"),
    )
    nonsense_gil = (
        "It heavily impacts and as such if a complex function is above the standardized normal "
        "functions it would actually be an inefficient code and we think in the main file "
        "the mentions should be efficient and gil should not be impacting much"
    )
    res_gil = json.loads(
        llm.complete(
            system=EVALUATE_ANSWER_SYSTEM,
            user=json.dumps(
                {
                    "question": q_gil.question,
                    "answer": nonsense_gil,
                    "keywords": list(q_gil.keywords),
                    "outline": list(q_gil.outline),
                }
            ),
            json_mode=True,
        )
    )
    rec_gil = _reconcile_evaluation(q_gil, nonsense_gil, res_gil)
    assert rec_gil["score"] <= 22, rec_gil


def test_reconcile_caps_lenient_live_llm_scores_on_baseless_answers():
    bank = load_bank()
    q = bank[0]
    nonsense = (
        "It plays a very important role overall and should be efficient in the main process "
        "architecture of modern software systems."
    )

    # Case 1: Live LLM says wrong_or_baseless but still gives inflated 54/100
    lenient_1 = _LenientStubLLM(
        {
            "verdict": "wrong_or_baseless",
            "accurate_claims": [],
            "errors_or_bluffs": ["Fabricated explanation without technical basis"],
            "score": 54,
            "feedback": "Score 54/100 - answer lacks technical accuracy.",
            "rubric": {"technical_accuracy": 50, "depth_and_examples": 40, "communication_clarity": 75},
            "matched_keywords": [],
            "missed_keywords": list(q.keywords),
            "ideal_answer_hint": "Define the core mechanism clearly.",
        }
    )
    s1 = InterviewSession(role="backend", bank=bank, llm=lenient_1, prefer_skills=[q.skill_id], transcript=[])
    s1.next_question()
    t1 = s1.answer(nonsense)
    assert t1.score <= 15, t1

    # Case 2: Live LLM omits verdict, gives 55/100 with empty accurate_claims & negative feedback
    lenient_2 = _LenientStubLLM(
        {
            "accurate_claims": [],
            "errors_or_bluffs": [],
            "score": 55,
            "feedback": "The answer lacks understanding of the underlying mechanism.",
            "rubric": {"technical_accuracy": 45, "depth_and_examples": 30, "communication_clarity": 70},
            "matched_keywords": [],
            "missed_keywords": list(q.keywords),
            "ideal_answer_hint": "Explain the core concept.",
        }
    )
    s2 = InterviewSession(role="backend", bank=bank, llm=lenient_2, prefer_skills=[q.skill_id], transcript=[])
    s2.next_question()
    t2 = s2.answer(nonsense)
    assert t2.score <= 22, t2


def test_genuine_technical_answer_still_scores_high():
    q_gil = Question(
        id="q-py-gil",
        role="backend",
        skill_id="python",
        difficulty=2,
        question="How does the Global Interpreter Lock (GIL) impact the concurrency of threads in CPython?",
        keywords=("gil", "thread", "mutex", "bytecode", "multiprocessing"),
        outline=("Only one thread executes Python bytecode at a time", "Use multiprocessing for CPU-bound work"),
    )
    strong_answer = (
        "In CPython, the GIL is a mutex that allows only one native thread to execute Python bytecode "
        "at a time, which serializes CPU-bound multithreaded workloads on multi-core CPUs. For I/O-bound "
        "tasks, threads release the GIL while waiting, whereas CPU-bound parallelism should use multiprocessing "
        "or C extensions that release the lock."
    )
    llm = MockLLM()
    raw = json.loads(
        llm.complete(
            system=EVALUATE_ANSWER_SYSTEM,
            user=json.dumps(
                {
                    "question": q_gil.question,
                    "answer": strong_answer,
                    "keywords": list(q_gil.keywords),
                    "outline": list(q_gil.outline),
                }
            ),
            json_mode=True,
        )
    )
    rec = _reconcile_evaluation(q_gil, strong_answer, raw)
    assert rec["score"] >= 75, rec


def test_no_wpm_in_coaching_tips_or_ui_templates():
    fast = analyze_speech_delivery("We reduced p99 latency by adding a Redis cache.", duration_seconds=1.2)
    slow = analyze_speech_delivery("I worked on an API endpoint in Python for billing.", duration_seconds=30.0)
    for report in (fast, slow):
        tip = report.get("coaching_tip", "")
        assert "wpm" not in tip.lower(), tip
        assert "slow down" not in tip.lower(), tip
        assert "pick up the pace" not in tip.lower(), tip

    app_js = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "sa.wpm" not in app_js
    assert "ss.avg_wpm" not in app_js
    assert '" WPM"' not in app_js
    assert 'WPM"' not in app_js

    progress_html = (ROOT / "app" / "templates" / "student" / "progress.html").read_text(encoding="utf-8")
    assert "avg_wpm" not in progress_html
    assert "WPM" not in progress_html


def test_finalized_interview_returns_complete_qa_transcript_and_print_dom():
    client = _make_client()
    with client:
        sid = _seed_student(client)
        start = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
        assert start.status_code in (200, 201), start.text
        iid = start.json()["interview_id"]

        final_body = None
        for idx in range(64):
            ans = f"Technical answer turn {idx + 1} covering indexes, yield, latency, and rollback recovery."
            r = client.post(f"/api/interviews/{iid}/answer", json={"answer": ans, "duration_seconds": 12.0})
            assert r.status_code == 200, r.text
            final_body = r.json()
            if final_body["finished"]:
                break

        assert final_body is not None and final_body["finished"] is True
        summary = final_body["summary"]
        assert len(summary["transcript"]) >= 8
        assert summary["coverage"]["complete"]
        assert summary["student_name"] == "Print Tester"
        assert summary["job_title"]
        assert summary["interview_id"] == iid
        for turn in summary["transcript"]:
            assert turn.get("q")
            assert turn.get("a")
            assert "score" in turn
            assert turn.get("feedback")

        get_res = client.get(f"/api/interviews/{iid}")
        assert get_res.status_code == 200
        fetched = get_res.json()
        assert fetched["student_name"] == "Print Tester"
        assert len(fetched["transcript"]) == len(summary["transcript"])
        assert all(t.get("q") and t.get("a") and t.get("feedback") for t in fetched["transcript"])

    interview_html = (ROOT / "app" / "templates" / "student" / "interview.html").read_text(encoding="utf-8")
    assert "qa-transcript-box" in interview_html
    assert "qa-transcript-list" in interview_html
    assert "summary-meta" in interview_html
    assert "@media print" in interview_html
    assert ".qa-turn-card" in interview_html

    app_js = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "renderTranscriptReport" in app_js
    assert "renderSummaryMeta" in app_js
