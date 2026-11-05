import pytest
from sqlmodel import Session

from app.interview.bank import Question, get_question, load_bank, select
from app.interview.engine import InterviewSession, InterviewSummary, role_for_job
from app.llm.mock import MockLLM
from app.models import Interview, Job

BANK = load_bank()


def test_load_bank_shape_and_counts():
    assert len(BANK) >= 50
    assert all(isinstance(q, Question) for q in BANK)
    assert all(q.role in {"backend", "frontend", "data", "devops"} for q in BANK)
    assert all(q.difficulty in {0, 1, 2} for q in BANK)
    assert all(q.keywords for q in BANK)
    ids = [q.id for q in BANK]
    assert len(ids) == len(set(ids))


def test_select_respects_difficulty():
    q = select(BANK, role="backend", difficulty=0, exclude_ids=[], prefer_skills=[])
    assert q is not None
    assert q.difficulty == 0
    assert q.role == "backend"


def test_select_excludes_used_ids():
    first = select(BANK, role="backend", difficulty=0, exclude_ids=[], prefer_skills=[])
    second = select(BANK, role="backend", difficulty=0, exclude_ids=[first.id], prefer_skills=[])
    assert second is not None
    assert second.id != first.id


def test_select_prefers_a_given_skill_when_present():
    # backend has a python question; preferring python must surface it
    q = select(BANK, role="backend", difficulty=0, exclude_ids=[], prefer_skills=["python"])
    assert q is not None
    assert q.skill_id == "python"


def test_get_question_by_id():
    q = get_question(BANK, BANK[0].id)
    assert q is not None and q.id == BANK[0].id
    assert get_question(BANK, "does-not-exist") is None


def test_role_for_job():
    assert role_for_job(Job(id="j", title="Backend Engineer (Python)", company="X", description="", required_skills=[], preferred_skills=[])) == "backend"
    assert role_for_job(Job(id="j", title="Data Analyst", company="X", description="", required_skills=[], preferred_skills=[])) == "data"
    assert role_for_job(Job(id="j", title="DevOps Engineer", company="X", description="", required_skills=[], preferred_skills=[])) == "devops"
    assert role_for_job(Job(id="j", title="Frontend Engineer (React)", company="X", description="", required_skills=[], preferred_skills=[])) == "frontend"


def test_session_adapts_up_on_good_answers():
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python"], transcript=[])
    s.next_question()
    r = s.answer("In Python a list is ordered, a dict is key value, and a generator yields lazily.")
    assert 0 <= r.score <= 100
    assert r.next_difficulty == min(2, r.next_difficulty)  # sanity
    s2 = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python"], transcript=[])
    s2.next_question()
    r2 = s2.answer("uh")
    assert r2.score < 70


def test_session_finishes_after_eight_turns():
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python", "sql"], transcript=[])
    for i in range(8):
        s.next_question()
        r = s.answer("I explain lists dicts generators joins and indexing in detail here today.")
        assert r.finished is (i == 7)
    summary = s.finish()
    assert isinstance(summary, InterviewSummary)
    assert 0 <= summary.score <= 100
    assert len([t for t in summary.transcript if t["score"] is not None]) == 8
    assert summary.per_skill  # non-empty
    assert summary.feedback.strip()


def test_interview_api_full_session(client_engine):
    client, eng = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")}, data={"name": "Interviewee"})
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    assert r.status_code in (200, 201)
    body = r.json()
    iid = body["interview_id"]
    assert body["question"] and "question" in body["question"]

    finished = False
    for _ in range(8):
        a = client.post(f"/api/interviews/{iid}/answers", json={"text": "I discuss lists dicts generators joins indexing and transactions in some depth here."})
        assert a.status_code == 200, a.text
        ab = a.json()
        if ab.get("finished"):
            finished = True
            assert "summary" in ab and 0 <= ab["summary"]["score"] <= 100
        else:
            assert "question" in ab
    assert finished
    g = client.get(f"/api/interviews/{iid}")
    assert g.status_code == 200
    gb = g.json()
    assert gb["finished"] is True
    with Session(eng) as db:
        row = db.get(Interview, iid)
        assert row.finished is True
        assert row.score is not None
        assert len(row.transcript) == 8


def _make_student(client, name):
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                    data={"name": name})
    return r.json()["id"]


def test_answer_finished_interview_returns_409(client_engine):
    client, _ = client_engine
    sid = _make_student(client, "Done")
    client.get(f"/api/students/{sid}/matches")
    iid = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"}).json()["interview_id"]
    for _ in range(8):
        assert client.post(f"/api/interviews/{iid}/answers",
                           json={"text": "I explain in reasonable depth here."}).status_code == 200
    assert client.post(f"/api/interviews/{iid}/answers", json={"text": "one more"}).status_code == 409


def test_start_interview_without_matches_returns_400(client_engine):
    client, _ = client_engine
    sid = _make_student(client, "NoMatch")  # no GET /matches -> no JobMatch rows
    assert client.post(f"/api/students/{sid}/interviews", json={}).status_code == 400


def test_interview_endpoints_404_for_unknown(client_engine):
    client, _ = client_engine
    assert client.get("/api/interviews/does-not-exist").status_code == 404
    assert client.post("/api/students/unknown-sid/interviews",
                       json={"job_id": "job-001"}).status_code == 404


def test_interview_html_route_renders(client_engine):
    client, _ = client_engine
    sid = _make_student(client, "Voice UI")
    client.get(f"/api/students/{sid}/matches")
    iid = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"}).json()["interview_id"]
    h = client.get(f"/interviews/{iid}")
    assert h.status_code == 200
    assert "interview" in h.text.lower()
    assert "app.js" in h.text


def test_static_js_served(client_engine):
    client, _ = client_engine
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "speechSynthesis" in r.text or "SpeechRecognition" in r.text


def test_answer_without_question_raises():
    # empty bank, no pending question -> answer() must raise, not crash the route
    s = InterviewSession(role="backend", bank=[], llm=MockLLM(), prefer_skills=[], transcript=[])
    with pytest.raises(ValueError):
        s.answer("hello")


def test_answer_ghost_question_id_raises():
    # a pending q_id that is no longer in the bank -> guard raises ValueError
    ghost = {"q_id": "ghost", "q": "?", "skill_id": "python", "a": "", "difficulty": 1,
             "score": None, "next_difficulty": 1}
    s = InterviewSession(role="backend", bank=[], llm=MockLLM(), prefer_skills=[], transcript=[ghost])
    with pytest.raises(ValueError):
        s.answer("hello")


def test_answer_interview_no_available_question_returns_400(client_engine, monkeypatch):
    import app.api.routes.interviews as iv
    monkeypatch.setattr(iv, "load_bank", lambda: [])
    client, _ = client_engine
    sid = _make_student(client, "Exhaust")
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    assert r.status_code in (200, 201)
    iid = r.json()["interview_id"]
    # no question is pending and none can be supplied -> clean 400, not a 500
    assert client.post(f"/api/interviews/{iid}/answers", json={"text": "hello"}).status_code == 400


def test_reupload_resume_is_idempotent(client_engine):
    client, _ = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                    data={"name": "Reup"})
    sid = r.json()["id"]
    assert client.post(f"/students/{sid}/resume",
                       files={"file": ("r.txt", b"Skills: Python, SQL\n", "text/plain")}).status_code == 200
    assert client.post(f"/students/{sid}/resume",
                       files={"file": ("r.txt", b"Skills: Python, Docker\n", "text/plain")}).status_code == 200
    names = {s["name"] for s in client.get(f"/api/students/{sid}").json()["skills"]}
    assert "Python" in names
    assert "Docker" in names
    assert "FastAPI" not in names  # replaced, not duplicated


# ===========================================================================
# Milestone 11 §19.5 — Interview Verification Feedback Loop
# ===========================================================================
# Charter §7 gate-first, §5 re-attack: this test was written RED (promotion
# loop absent, measured 2026-11-03 by direct read of _finalize). @Coder-Flash
# landed §19.5 the same session; re-attack run PASSED (promotion fires,
# re-rank spy counted >=1, below-threshold skill stayed put) — converted to a
# permanent regression guard. Anti-vacuity: the below-threshold skill stays
# put (a no-op or blanket promotion fails one of the two assertions).
# ===========================================================================


def test_interview_finalization_promotes_skill_level(client_engine):
    """§19.5 — per_skill score >= 80 promotes basic -> intermediate, appends
    verification evidence, and triggers match re-ranking; score < 45 does not promote."""
    from sqlmodel import select

    import app.api.routes.interviews as interviews_mod
    from app.api.routes.interviews import _finalize
    from app.interview.engine import InterviewSummary
    from app.models import Interview, JobMatch, StudentSkill

    client, engine = client_engine
    # POST /api/students already inserts the row (students.py:326-329) — only
    # seed the skills, never a duplicate Student (UNIQUE PK measured 2026-11-03).
    r = client.post("/api/students", json={"name": "Loop Tester", "branch": "CSE",
                                           "year": 3, "gpa": 8.5})
    sid = r.json()["id"]
    assert sid, f"POST /api/students must create the student: {r.status_code} {r.text}"

    with Session(engine) as db:
        db.add(StudentSkill(student_id=sid, skill_id="python", mentions=3, level="basic",
                            evidence=["resume keyword"]))
        db.add(StudentSkill(student_id=sid, skill_id="sql", mentions=1, level="basic",
                            evidence=["resume keyword"]))
        db.commit()

    # Synthetic summary: python 85 (>=80, promote), sql 40 (<45, no promote).
    # per_skill keys are skill TAXONOMY ids (§19.5's own examples:
    # {"machine_learning": 85, ...}) because StudentSkill.skill_id is a taxonomy
    # FK (models.py:21); keys are stringified as the engine's json_mode wire
    # ({k: round(avg)} -> {"85"}), which _finalize stores verbatim into
    # row.feedback["per_skill"] (interviews.py:63-64).
    summary = InterviewSummary(score=85,
                               per_skill={"python": "85", "sql": "40"},
                               feedback="Strong Python, weak SQL.",
                               transcript=[])

    # Precondition: a JobMatch exists so re-ranking is observable.
    with Session(engine) as db:
        db.add(JobMatch(id="jm-loop", student_id=sid, job_id="job-001",
                        score=50, band="developing"))
        db.commit()

    # Capture whether _rank_and_persist is invoked (§19.5.2).
    rank_calls = {"n": 0}
    original_rank = getattr(interviews_mod, "_rank_and_persist", None)
    if original_rank is not None:
        def _spy(session, ctx, s, student, **kw):
            rank_calls["n"] += 1
            return original_rank(session, ctx, s, student, **kw)
        interviews_mod._rank_and_persist = _spy

    try:
        with Session(engine) as db:
            if db.get(Interview, "int-loop") is None:
                db.add(Interview(id="int-loop", student_id=sid, job_id="job-001",
                                 transcript=[], score=None, feedback={}, finished=False))
                db.commit()
            row = db.get(Interview, "int-loop")
            _finalize(db, row, summary)

            skills = {ss.skill_id: ss for ss in db.exec(
                select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
            assert skills["python"].level in ("intermediate", "advanced"), (
                f"python (score 85) must be promoted from basic, got {skills['python'].level}"
            )
            assert any("interview" in str(e).lower()
                       for e in skills["python"].evidence), (
                f"python evidence must record the interview verification, got {skills['python'].evidence}"
            )
            # Anti-vacuity: below-threshold skill stays basic (not blanket-promoted).
            assert skills["sql"].level == "basic", (
                f"sql (score 40) must NOT be promoted, got {skills['sql'].level}"
            )
            # §19.5.2 guarantees a re-rank call; unconditional (charter §4 anti-vacuity).
            assert rank_calls["n"] >= 1, (
                "_finalize did not trigger _rank_and_persist (match re-ranking)"
            )
    finally:
        interviews_mod._rank_and_persist = original_rank
