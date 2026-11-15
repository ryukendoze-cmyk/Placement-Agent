"""Complete claim coverage, live authoring and truthful offline assessment."""
import json

import pytest
from sqlmodel import Session, select

from app.api.routes import interviews
from app.interview.bank import load_bank
from app.interview.engine import InterviewSession
from app.interview.generation import author_question
from app.interview.quality import audit_question_bank
from app.llm.mock import MockLLM
from app.llm.telemetry import TelemetryLLM
from app.models import Interview, Skill, Student, StudentSkill


class LiveAI:
    def __init__(self, score=50, fail_generation=False, fail_grading=False):
        self.calls = []
        self.score = score
        self.fail_generation = fail_generation
        self.fail_grading = fail_grading

    def complete(self, system, user, json_mode=False):
        data = json.loads(user)
        self.calls.append((system, data))
        if "TASK: interview_question" in system:
            if self.fail_generation:
                raise TimeoutError("private provider detail")
            skill = data["target"]["skill_id"]
            return json.dumps({
                "question": f"For your {skill} implementation, how did you validate retry behavior and prevent duplicate effects after a timeout?",
                "keywords": ["idempotency", "retry", "timeout"],
                "outline": ["A stable idempotency key deduplicates repeated operations",
                            "Bounded retries use backoff and jitter to reduce contention",
                            "Fault injection simulates a lost response after committing work"],
            })
        if "TASK: evaluate_answer" in system:
            if self.fail_grading:
                raise TimeoutError("private provider detail")
            return json.dumps({"score": self.score, "verdict": "strong" if self.score >= 85 else "partial",
                "feedback": "Explained the mechanism; revisit retry behavior.",
                "accurate_claims": ["Idempotency deduplicates repeated operations"], "errors_or_bluffs": [],
                "rubric": dict.fromkeys(["technical_accuracy", "depth_and_examples", "communication_clarity"], self.score),
                "matched_keywords": ["idempotency"], "missed_keywords": ["retry", "timeout"] if self.score < 85 else [],
                "ideal_answer_hint": "Use stable keys and bounded retries."})
        return json.dumps({"feedback": "Review the assessed concepts."})


ANSWER = "I use an idempotency key to deduplicate repeated operations because a response may be lost after the transaction commits."


def plan(count=25):
    targets = [{"skill_id": f"skill-{i}", "name": f"Skill {i}", "level": "intermediate",
                "verification_status": ["claimed", "unverified", "needs_reverification"][i % 3],
                "evidence": [f"Built project component {i}"]} for i in range(count)]
    return {"version": 1, "targets": targets, "candidate_skills": targets, "context": {"resume": "Built a retry worker"}}


def test_all_25_claims_survive_followups_and_request_reconstruction():
    snapshot = plan()
    llm = LiveAI()
    transcript = []
    asked = []
    for _ in range(60):
        s = InterviewSession("backend", (), llm, [], transcript, enable_followups=True, interview_plan=snapshot)
        if s.finished:
            break
        q = s.next_question()
        asked.append(q.skill_id)
        s.answer(ANSWER)
        transcript = json.loads(json.dumps(s.transcript))
    assert s.finished
    assert s.coverage["assessed"] == s.coverage["total"] == 25
    assert s.turn == 50  # every base question received an additional probe
    assert set(asked) == {t["skill_id"] for t in snapshot["targets"]}
    calls = [d for sys, d in llm.calls if "TASK: interview_question" in sys]
    assert len(calls) == 50
    assert calls[0]["target"]["evidence"] == ["Built project component 0"]
    assert calls[1]["purpose"] == "followup"
    assert calls[1]["previous_answer"]["answer"] == ANSWER
    assert "keywords" not in calls[0] and "outline" not in calls[0]


def test_live_generation_never_reads_static_bank():
    def forbidden_bank():
        pytest.fail("live path must not access the static bank")
    s = InterviewSession("backend", forbidden_bank, LiveAI(), [], [], interview_plan=plan(1))
    q = s.next_question()
    assert q.id.startswith("q-ai-")
    assert s.pending["question_source"] == "live_ai"
    assert s.pending["verification_target"] == "skill-0"


def test_live_general_questions_after_coverage_also_avoid_the_bank():
    def forbidden_bank():
        pytest.fail("general live questions must not read the bank")
    s = InterviewSession("backend", forbidden_bank, LiveAI(score=90), [], [], interview_plan=plan(1))
    for i in range(8):
        s.next_question()
        assert s.pending["question_source"] == "live_ai"
        assert bool(s.pending["verification_target"]) is (i == 0)
        s.answer(ANSWER)
        assert s.finished is (i == 7)
    assert s.coverage["complete"]


@pytest.mark.parametrize("llm", [MockLLM(), None])
def test_offline_exact_skill_and_unsupported_skill_are_never_skipped(llm):
    snapshot = plan(2)
    snapshot["targets"][0]["skill_id"] = "python"
    s = InterviewSession("backend", load_bank(), llm, [], [], interview_plan=snapshot)
    assert s.next_question().skill_id == "python"
    assert s.pending["question_source"] == "offline_bank"
    s.answer("I don't know")
    assert s.next_question().skill_id == "skill-1"
    assert s.pending["question_source"] == "offline_template"
    assert s.pending["assessment_limited"]
    s.answer(ANSWER)
    assert "skill-1" not in s.finish().assessment_per_skill


@pytest.mark.parametrize("bad_response", ["garbage", '{}', '{"question":"short"}'])
def test_invalid_api_question_enters_labeled_offline_mode(bad_response):
    class InvalidAI:
        def complete(self, **kwargs):
            return bad_response
    target = {"skill_id": "python", "verification_status": "claimed"}
    q, source = author_question(llm=InvalidAI(), bank=load_bank(), role="backend", target=target,
                                difficulty=1, qid="test", context={})
    assert q.skill_id == "python"
    assert source == {"question_source": "offline_bank", "generation_error": "ai_generation_failed:ValueError"}


def test_api_outage_fallback_is_labeled_and_later_turn_retries_live():
    llm = LiveAI(fail_generation=True)
    snapshot = plan(2)
    snapshot["targets"][0]["skill_id"] = "python"
    s = InterviewSession("backend", load_bank(), llm, [], [], interview_plan=snapshot)
    s.next_question()
    assert s.pending["question_source"] == "offline_bank"
    assert s.pending["generation_error"] == "ai_generation_failed:TimeoutError"
    s.answer(ANSWER)
    llm.fail_generation = False
    s.next_question()
    assert s.pending["question_source"] == "live_ai"


def test_telemetry_live_only_call_never_uses_mock_when_circuit_is_open():
    wrapped = object.__new__(TelemetryLLM)
    wrapped.provider = "openai"
    class Tracker:
        def should_use_fallback(self, provider):
            return True
    wrapped.tracker = Tracker()
    with pytest.raises(RuntimeError, match="live_provider_circuit_open"):
        wrapped.complete_live(system="test", user="{}")
    _, source = author_question(llm=wrapped, bank=load_bank(), role="backend",
                               target={"skill_id": "python"}, difficulty=1, qid="test", context={})
    assert source == {"question_source": "offline_bank", "generation_error": "ai_generation_failed:RuntimeError"}


def make_candidate(eng, count):
    with Session(eng) as db:
        db.add(Student(id="coverage-student", name="Coverage", resume_text="Built a retry worker with Python"))
        db.commit()
        skills = db.exec(select(Skill).order_by(Skill.id)).all()[:count]
        ids = [sk.id for sk in skills]
        for i, sid in enumerate(ids):
            db.add(StudentSkill(student_id="coverage-student", skill_id=sid, level="intermediate",
                                verification_status=["claimed", "unverified", "needs_reverification"][i % 3],
                                evidence=["Built a worker component and measured retry behavior"]))
        db.commit()
    return ids


def test_api_snapshot_coverage_persistence_and_all_skill_outcomes(client_engine, monkeypatch):
    client, eng = client_engine
    ids = make_candidate(eng, 12)
    client.app.state.ctx.llm = LiveAI(score=90)
    monkeypatch.setattr(interviews, "_rank_and_persist", lambda *a, **kw: None)
    monkeypatch.setattr(interviews, "_load_session_bank", lambda: pytest.fail("no bank on live path"))
    start = client.post("/api/students/coverage-student/interviews", json={"job_id": "job-001"}).json()
    iid = start["id"]
    assert start["coverage"]["total"] == 12
    assert start["question"]["source"] == "live_ai"
    # Editing live profile status must not alter the immutable interview queue.
    with Session(eng) as db:
        row = db.get(StudentSkill, ("coverage-student", ids[-1]))
        row.verification_status = "verified"
        db.add(row)
        db.commit()
    for i in range(12):
        response = client.post(f"/api/interviews/{iid}/answers", json={"text": ANSWER})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["finished"] is (i == 11)
        readback = client.get(f"/api/interviews/{iid}").json()
        assert readback["coverage"]["assessed"] == i + 1
        assert readback["coverage"]["total"] == 12
    assert result["summary"]["coverage"]["complete"]
    assert len(result["summary"]["skill_updates"]) == 12
    with Session(eng) as db:
        row = db.get(Interview, iid)
        assert row.feedback["_verification"]["targets"]
        for sid in ids:
            skill = db.get(StudentSkill, ("coverage-student", sid))
            assert skill.verification_status == "verified"
            assert skill.last_verified_score >= 80


def test_resuming_historical_session_upgrades_remaining_questions(client_engine):
    client, eng = client_engine
    ids = make_candidate(eng, 3)
    old = next(q for q in load_bank() if q.skill_id == ids[0])
    with Session(eng) as db:
        db.add(Interview(id="legacy-coverage", student_id="coverage-student", job_id="job-001",
                         role="backend", feedback={}, adaptive_followup=False,
                         transcript=[{"q_id": old.id, "q": old.question, "skill_id": old.skill_id,
                                      "difficulty": old.difficulty, "a": "", "score": None}]))
        db.commit()
    client.app.state.ctx.llm = LiveAI(score=90)
    response = client.post("/api/interviews/legacy-coverage/answers", json={"text": ANSWER})
    assert response.status_code == 200, response.text
    assert response.json()["coverage"]["total"] == 3
    assert response.json()["question"]["source"] == "live_ai"
    with Session(eng) as db:
        row = db.get(Interview, "legacy-coverage")
        assert row.transcript[0]["q"] == old.question
        assert row.transcript[0]["question_source"] == "legacy_question"
        assert row.feedback["_verification"]["targets"]


def test_early_end_does_not_verify_unanswered_claims(client_engine, monkeypatch):
    client, eng = client_engine
    ids = make_candidate(eng, 3)
    client.app.state.ctx.llm = LiveAI(score=90)
    monkeypatch.setattr(interviews, "_rank_and_persist", lambda *a, **kw: None)
    start = client.post("/api/students/coverage-student/interviews", json={"job_id": "job-001"}).json()
    iid = start["id"]
    first = start["question"]["skill_id"]
    client.post(f"/api/interviews/{iid}/answers", json={"text": ANSWER})
    ended = client.post(f"/api/interviews/{iid}/finish").json()["summary"]
    assert ended["completion_reason"] == "ended_early"
    assert ended["coverage"]["assessed"] == 1
    assert ended["coverage"]["remaining"] == 2
    with Session(eng) as db:
        from app.analytics.student_progress import build_student_progress_report
        from app.rag.policy_rag import _latest_mock_score
        assert _latest_mock_score(db, "coverage-student") is None
        assert build_student_progress_report(db, "coverage-student", client.app.state.ctx.taxonomy)["kpis"]["interviews_completed"] == 0
        for sid in ids:
            skill = db.get(StudentSkill, ("coverage-student", sid))
            assert (skill.verification_status == "verified") is (sid == first)
            assert (skill.last_verified_score is not None) is (sid == first)
    assert client.post(f"/api/interviews/{iid}/finish").status_code == 409


def test_unavailable_grader_never_promotes_from_fallback(client_engine, monkeypatch):
    client, eng = client_engine
    ids = make_candidate(eng, 1)
    client.app.state.ctx.llm = LiveAI(fail_grading=True)
    monkeypatch.setattr(interviews, "_rank_and_persist", lambda *a, **kw: None)
    start = client.post("/api/students/coverage-student/interviews", json={"job_id": "job-001", "adaptive_followup": False}).json()
    iid = start["id"]
    response = client.post(f"/api/interviews/{iid}/answers", json={"text": ANSWER}).json()
    assert response["evaluation_source"] == "offline_fallback"
    assert response["error_recovered"]
    ended = client.post(f"/api/interviews/{iid}/finish").json()["summary"]
    assert ended["skill_updates"] == []
    assert ended["coverage"]["targets"][0]["assessment_limited"]
    with Session(eng) as db:
        assert db.get(StudentSkill, ("coverage-student", ids[0])).verification_status == "claimed"


def test_expanded_offline_bank_has_all_skills_at_all_difficulties():
    from pathlib import Path
    skills = json.loads((Path(__file__).resolve().parents[1] / "data/skills.json").read_text(encoding="utf-8"))
    bank = load_bank()
    assert len(bank) >= 708
    assert len({q.id for q in bank}) == len(bank)
    for skill in skills:
        assert {q.difficulty for q in bank if q.skill_id == skill["id"]} == {0, 1, 2}
    assert audit_question_bank(bank)["rejected_questions"] == 0
