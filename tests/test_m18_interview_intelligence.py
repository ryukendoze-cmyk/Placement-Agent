"""Milestone 18: adaptive mock interviewer intelligence.

Covers 18.1 dynamic follow-up probing (opt-in), 18.2 resume-grounded project
deep-dive questions, 18.3 voice/acoustic communication telemetry, and 18.4
API integration + backward-compatibility of the legacy 8-turn flow.
"""
import re

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session

from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.interview.bank import load_bank
from app.interview.engine import InterviewSession, generate_followup_probe
from app.interview.resume_probe import generate_resume_project_questions
from app.interview.speech_analytics import analyze_speech_delivery
from app.llm.mock import MockLLM
from app.models import Student, StudentSkill
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


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


def _seed_student(client, name="M18 Probe") -> str:
    r = client.post(
        "/api/students",
        files={"file": ("r.txt",
                        (b"Built a REST API service using Python, FastAPI and PostgreSQL "
                         b"handling 10k requests per second during my internship.\n"
                         b"Skills: Python, FastAPI, PostgreSQL, Docker\n"), "text/plain")},
        data={"name": name, "branch": "CSE", "year": "3", "gpa": "8.4"},
    )
    assert r.status_code == 201, r.text
    return r.json()["id"]


# --------------------------------------------- 18.1 follow-up probing (unit)
def test_followup_default_disabled_preserves_question_ids():
    """Legacy behavior: no follow-up even on a mid/low score with missed keywords."""
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(),
                         prefer_skills=["python"], transcript=[])
    s.next_question()
    r = s.answer("I know a bit of lists and dicts but nothing deep.")
    assert r.followup_probe == ""
    # the next question must be a fresh bank question, not a "-followup" id
    nq = s.next_question()
    assert nq is not None
    assert not s.transcript[-1].get("is_followup")


def test_followup_generated_when_enabled_mid_score_with_missed_keywords():
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(),
                         prefer_skills=["python"], transcript=[], enable_followups=True)
    s.next_question()
    r = s.answer("I know lists and dicts but nothing about generators or joins.")
    # MockLLM mid-band score with missed keywords -> probe generated
    assert r.followup_probe, f"expected followup probe, got score={r.score} missed={r.missed_keywords}"
    assert s.transcript[-1].get("is_followup") is True
    assert s.transcript[-1]["skill_id"] == s.transcript[0]["skill_id"], (
        "follow-up must stay on the same skill")
    # the probe question id follows the original id
    assert s.transcript[-1]["q_id"].endswith("-followup")


def test_followup_probe_not_chained():
    """A follow-up turn must never generate a second follow-up."""
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(),
                         prefer_skills=["python"], transcript=[], enable_followups=True)
    s.next_question()
    s.answer("partial lists and dicts nothing generators joins indexing detail")
    if s.pending and s.pending.get("is_followup"):
        r2 = s.answer("also partial again nothing specific here")
        assert r2.followup_probe == "", "follow-up must not chain"


def test_generate_followup_probe_helper_deterministic():
    probe = generate_followup_probe(
        question="What are Python generators?",
        answer="I know a bit.",
        missed_keywords=["yield", "lazy"],
        skill_id="python",
        llm=MockLLM(),
    )
    assert isinstance(probe, str) and probe.strip()
    assert "yield" in probe or "python" in probe.lower()
    # deterministic for same input with same llm
    probe2 = generate_followup_probe(
        question="What are Python generators?",
        answer="I know a bit.",
        missed_keywords=["yield", "lazy"],
        skill_id="python",
        llm=MockLLM(),
    )
    assert probe == probe2


# ----------------------------------------------- 18.2 resume probes (unit)
def test_generate_resume_project_questions_from_evidence():
    taxonomy = load_taxonomy()
    student = Student(id="stu-rp1", name="Probe", branch="CSE", year=3, gpa=8.4,
                      resume_text=("Built a REST API service using Python, FastAPI "
                                   "and PostgreSQL handling 10k requests per second.\n"))
    skills = [StudentSkill(student_id="stu-rp1", skill_id="python", mentions=2,
                           level="intermediate",
                           evidence=["Built REST API with FastAPI"]),
              StudentSkill(student_id="stu-rp1", skill_id="postgresql", mentions=1,
                           level="basic", evidence=["PostgreSQL schema design"])]
    qs = generate_resume_project_questions(student, skills, taxonomy, llm=MockLLM())
    assert 1 <= len(qs) <= 2
    assert qs[0].id == "q-resume-01"
    if len(qs) > 1:
        assert qs[1].id == "q-resume-02"
    # grounded in real evidence: the question quotes a resume/evidence snippet
    assert "resume" in qs[0].question.lower()
    assert qs[0].keywords, "must include expected keywords"
    assert len(qs[0].outline) >= 4, "4-point evaluation outline required"
    # ids never collide with the static bank
    bank_ids = {q.id for q in load_bank()}
    assert qs[0].id not in bank_ids


def test_generate_resume_project_questions_empty_resume():
    taxonomy = load_taxonomy()
    student = Student(id="stu-rp2", name="Blank", branch="CSE", year=3, gpa=7.0,
                      resume_text="")
    qs = generate_resume_project_questions(student, [], taxonomy, llm=MockLLM())
    assert qs == [], "empty resume with no skills must yield no probes"


def test_interview_start_with_resume_probes_serves_them_first(client_engine=None):
    client = _make_client()
    with client:
        sid = _seed_student(client)
        client.get(f"/api/students/{sid}/matches")
        r = client.post(f"/api/students/{sid}/interviews",
                        json={"job_id": "job-001", "include_resume_probes": True})
        assert r.status_code in (200, 201), r.text
        body = r.json()
        iid = body["interview_id"]
        qid = body["question"]["id"]
        assert qid.startswith("q-resume-"), f"expected resume probe first, got {qid}"
        # Anti-vacuity: the resume probe must be ANSWERABLE (not 400 not-in-bank).
        ans = client.post(f"/api/interviews/{iid}/answers",
                          json={"text": ("The architecture used Python and FastAPI with "
                                         "Redis caching to handle 10k requests per second "
                                         "and I measured p99 latency under load.")})
        assert ans.status_code == 200, ans.text
        assert ans.json()["score"] > 0


def test_interview_default_has_no_resume_probes():
    client = _make_client()
    with client:
        sid = _seed_student(client, "NoProbe")
        client.get(f"/api/students/{sid}/matches")
        r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
        qid = r.json()["question"]["id"]
        assert not qid.startswith("q-resume-"), "default must not inject resume probes"


def test_interviews_html_has_sop08_checkbox():
    html = open("app/templates/student/interviews.html", encoding="utf-8").read()
    assert "Include Resume Project Deep-Dive Questions [SOP-08]" in html
    # zero-emoji + no alert
    emoji = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u231A-\u23F9]")
    assert emoji.search(html) is None


# ------------------------------------------ 18.3 speech analytics (unit)
def test_analyze_speech_delivery_wpm_band():
    text = " ".join(["word"] * 60)  # 60 words
    out = analyze_speech_delivery(text, duration_seconds=30.0)
    assert out["word_count"] == 60
    assert out["wpm"] == 120  # 60 words / 30s * 60 = 120, inside 110-160
    assert 0 <= out["delivery_score"] <= 100
    assert out["coaching_tip"]


def test_analyze_speech_delivery_fillers_and_rate():
    text = "um basically I did the thing you know and actually like it"
    out = analyze_speech_delivery(text, duration_seconds=None)
    assert "um" in out["filler_words"]
    assert "basically" in out["filler_words"]
    assert "you know" in out["filler_words"]
    assert "actually" in out["filler_words"]
    assert out["filler_rate_pct"] > 0
    # no duration -> wpm 0, still scored on STAR
    assert out["wpm"] == 0
    assert 0 <= out["star_score"] <= 100


def test_analyze_speech_delivery_star_components():
    situation = "During my internship our team faced a latency spike in production."
    task = "My task was to reduce p99 latency below 200 milliseconds."
    action = "I introduced a Redis cache layer and rewrote the query path."
    result = "As a result latency dropped 60 percent and throughput doubled."
    text = f"{situation} {task} {action} {result}"
    out = analyze_speech_delivery(text, duration_seconds=40.0)
    assert out["star_components"]["situation"] is True
    assert out["star_components"]["task"] is True
    assert out["star_components"]["action"] is True
    assert out["star_components"]["result"] is True
    assert out["star_score"] == 100


def test_analyze_speech_delivery_partial_star():
    text = "I worked on a project and it was successful with metrics."
    out = analyze_speech_delivery(text, duration_seconds=25.0)
    assert out["star_score"] < 100
    assert isinstance(out["star_components"], dict)


def test_analyze_speech_delivery_coaching_tip_prioritizes_fillers():
    filler_heavy = "um uh like basically actually you know sort of " * 6 + "I did a task with a result."
    out = analyze_speech_delivery(filler_heavy, duration_seconds=30.0)
    tip = out["coaching_tip"].lower()
    assert "filler" in tip


# ------------------------------------ 18.3 + 18.4 API integration (E2E)
def test_answers_accept_duration_seconds_and_attach_speech_analytics():
    client = _make_client()
    with client:
        sid = _seed_student(client, "Speech")
        client.get(f"/api/students/{sid}/matches")
        iid = client.post(f"/api/students/{sid}/interviews",
                          json={"job_id": "job-001"}).json()["interview_id"]
        text = ("During my internship I built a caching layer to fix a latency spike "
                "(task). I rewrote the query path and added a Redis cache (action). "
                "Latency dropped 60 percent and throughput doubled, measured under load "
                "with p99 latency metrics across the fleet of services (result). "
                "I documented the rollout and shared the results with the team.")
        r = client.post(f"/api/interviews/{iid}/answers",
                        json={"text": text, "duration_seconds": 30.0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "speech_analytics" in body, "turn payload must carry speech_analytics"
        sa = body["speech_analytics"]
        assert sa["word_count"] > 0
        assert sa["wpm"] > 0  # duration provided -> WPM computed
        assert 0 <= sa["delivery_score"] <= 100
        # transcript turn now stores speech_analytics too
        g = client.get(f"/api/interviews/{iid}").json()
        assert g["transcript"][0].get("speech_analytics")


def test_finalize_aggregates_speech_summary():
    client = _make_client()
    with client:
        sid = _seed_student(client, "Summ")
        client.get(f"/api/students/{sid}/matches")
        iid = client.post(f"/api/students/{sid}/interviews",
                          json={"job_id": "job-001"}).json()["interview_id"]
        text = ("Situation we faced latency. Task reduce p99. Action add cache. "
                "Result latency dropped 60 percent with throughput doubled plus metrics.")
        finished = False
        for _ in range(8):
            r = client.post(f"/api/interviews/{iid}/answers",
                            json={"text": text, "duration_seconds": 30.0})
            assert r.status_code == 200, r.text
            if r.json().get("finished"):
                finished = True
                break
        assert finished
        g = client.get(f"/api/interviews/{iid}").json()
        assert "speech_summary" in g["feedback"], "finalized feedback must aggregate speech"
        ss = g["feedback"]["speech_summary"]
        assert "avg_wpm" in ss and "total_fillers" in ss
        assert "avg_star_score" in ss and "avg_delivery_score" in ss


def test_speech_analyze_endpoint():
    client = _make_client()
    with client:
        r = client.post("/api/speech/analyze",
                        json={"text": "I built a cache and latency dropped 60 percent.",
                              "duration_seconds": 20.0})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["word_count"] > 0
        assert "wpm" in body and "star_score" in body and "coaching_tip" in body


def test_speech_analyze_endpoint_requires_text():
    client = _make_client()
    with client:
        assert client.post("/api/speech/analyze", json={"text": ""}).status_code == 400


# ------------------------------------------- backward-compat (18.4 guard)
def test_legacy_eight_turn_flow_unchanged():
    """The default flow must still reach 8 turns with bank question ids."""
    client = _make_client()
    with client:
        sid = _seed_student(client, "Legacy")
        client.get(f"/api/students/{sid}/matches")
        iid = client.post(f"/api/students/{sid}/interviews",
                          json={"job_id": "job-001"}).json()["interview_id"]
        finished = False
        for _ in range(8):
            r = client.post(f"/api/interviews/{iid}/answers",
                            json={"text": "I explain lists dicts generators joins indexing here."})
            assert r.status_code == 200, r.text
            if r.json().get("finished"):
                finished = True
        assert finished, "8-turn default flow must finish"
        g = client.get(f"/api/interviews/{iid}").json()
        assert len(g["transcript"]) == 8
        # default (no adaptive_followup) must never inject a follow-up id
        assert not any(t["q_id"].endswith("-followup") for t in g["transcript"])


def test_adaptive_followup_api_flag_enables_probe():
    client = _make_client()
    with client:
        sid = _seed_student(client, "Adapt")
        client.get(f"/api/students/{sid}/matches")
        iid = client.post(f"/api/students/{sid}/interviews",
                          json={"job_id": "job-001"}).json()["interview_id"]
        # submit a partial answer with the adaptive flag -> may produce a probe
        r = client.post(f"/api/interviews/{iid}/answers",
                        json={"text": "partial lists and dicts nothing deep",
                              "adaptive_followup": True})
        assert r.status_code == 200, r.text
        body = r.json()
        # payload must expose the followup_probe key (empty string when not triggered)
        assert "followup_probe" in body
        if body.get("followup_probe"):
            # next question is the follow-up on same skill
            assert body["question"]["id"].endswith("-followup")


# ----------------------------------------------- app.js telemetry (18.3 UI)
def test_app_js_passes_duration_and_renders_telemetry():
    js = open("app/static/app.js", encoding="utf-8").read()
    assert "duration_seconds" in js, "app.js must send elapsed turn duration"
    assert "timerStart" in js
    assert "alert(" not in js, "no browser alert() allowed"
    # telemetry pill / WPM display wiring present
    assert "speech" in js.lower() or "wpm" in js.lower() or "delivery" in js.lower()
