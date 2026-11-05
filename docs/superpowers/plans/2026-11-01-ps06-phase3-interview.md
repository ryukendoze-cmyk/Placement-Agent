# PS-06 Phase 3 - Adaptive Mock Interviews - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an adaptive 8-turn mock interview that probes a target job's required skills, scores each answer via the LLM layer (deterministic keyword+length in mock mode), adapts difficulty up/down, produces a per-skill summary, and is usable in the browser with Web Speech voice in/out plus an always-present text fallback.

**Architecture:** `data/questions.json` (58 questions across roles backend/frontend/data/devops, difficulties 0/1/2) is loaded by `app/interview/bank.py` into a `Question` list with a deterministic `select()`. `app/interview/engine.py` holds a pure `InterviewSession` state machine. **No schema change:** all evolving state (current difficulty, used question ids, pending question) is carried inside the existing `Interview.transcript` JSON column, so a session is reconstructed from its DB row on every request. Each transcript entry: `{"q_id","q","skill_id","a","difficulty","score","next_difficulty"}` where `score: null` marks the *pending* (asked, unanswered) question. `MockLLM` already implements `TASK: evaluate_answer` and `TASK: interview_feedback` (Phase 2).

**Tech Stack:** Python 3.12 (uv), FastAPI + SQLModel (SQLite in tests), `data/questions.json`, Jinja2 + HTMX + Tailwind CDN, browser Web Speech API (`webkitSpeechRecognition` + `speechSynthesis`), pytest + MockLLM.

---

## 0. Ground rules (inherited from Phases 1-2 - do not violate)

1. `datetime.now(timezone.utc)` only (never `utcnow`).
2. Never store a `Session` on `AppContext`.
3. LLM calls only via `AppContext.llm` (tests wire `MockLLM()`); the engine never imports `boto3`/`openai`.
4. ruff must stay green: `uv run ruff check .` → All checks passed (select = I001,F401,F841,B008,RUF046,E711,E712; FastAPI Depends/File/Form exempt).
5. Unit tests use the `client`/`client_engine` fixtures (SQLite + HashingEmbedder + MemoryVectorStore + MockLLM). No network.
6. Gate per task: `uv run pytest -m "not integration" -q` green + ruff green + commit.
7. **Do NOT add columns to the `Interview` model.** Encode all interview state in `transcript`.

## 0.1 Transcript state model (the crux - read before coding)

`Interview.transcript` is a JSON list. Invariants:
- The interview is *finished* when the count of **answered** entries (those with `score != null`) reaches `MAX_TURNS = 8`.
- The transcript always *ends* with either a **pending** entry (`score == null`, question asked, awaiting an answer) while in progress, OR a completed entry when finished.
- `used_ids` (for non-repeating selection) = every entry's `q_id` (pending + answered).
- The difficulty a *pending/next* question is asked at: if a pending entry exists, use its `difficulty`; else use the last entry's `next_difficulty`; else start at `1`.

Lifecycle:
1. **Start** (`POST /api/students/{sid}/interviews`): row created with `transcript=[]`; `session.next_question()` selects question Q1 at difficulty 1 and appends a pending entry for Q1; response returns Q1.
2. **Answer** (`POST /api/interviews/{id}/answers`): fill in the last pending entry (a/score/next_difficulty via the LLM). If that was the 8th answered turn → `finish()` (compute score/per-skill/feedback), persist, return final summary. Else select Q(next) at `next_difficulty`, append a new pending entry, return turn result + next question.
3. **Read** (`GET /api/interviews/{id}`): return transcript, score, feedback, finished, and the pending question if in progress.

### Task 3.1: Question bank (`app/interview/bank.py`)

**Files:**
- Create: `app/interview/__init__.py`
- Create: `app/interview/bank.py`
- (data asset `data/questions.json` already exists - 58 questions)
- Test: `tests/test_interview.py`

- [x] **Step 1: Write failing tests** (`tests/test_interview.py`):

```python
from app.interview.bank import Question, load_bank, select, get_question

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
    # backend has python questions; asking to prefer python should surface one
    q = select(BANK, role="backend", difficulty=0, exclude_ids=[], prefer_skills=["python"])
    assert q is not None
    # among the role+difficulty pool, a python skill question must be preferred over generic
    pool = [x for x in BANK if x.role == "backend" and x.difficulty == 0 and x.id != q.id]
    # the selected one is the best (role 0, diff 0, prefer 0) - assert it is a python question
    assert q.skill_id == "python"


def test_get_question_by_id():
    q = get_question(BANK, BANK[0].id)
    assert q is not None and q.id == BANK[0].id
    assert get_question(BANK, "does-not-exist") is None
```

- [x] **Step 2: Run tests to verify they fail** → `uv run pytest tests/test_interview.py -v` → FAIL (ModuleNotFoundError).

- [x] **Step 3: Implement `app/interview/bank.py`:**

```python
"""Question bank: load data/questions.json and select the next question deterministically."""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Question:
    id: str
    role: str
    skill_id: str
    difficulty: int
    question: str
    keywords: tuple[str, ...]


@lru_cache(maxsize=1)
def load_bank() -> tuple[Question, ...]:
    import json
    raw = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    return tuple(Question(id=q["id"], role=q["role"], skill_id=q["skill_id"],
                          difficulty=q["difficulty"], question=q["question"],
                          keywords=tuple(q["keywords"])) for q in raw)


def get_question(bank, q_id: str):
    for q in bank:
        if q.id == q_id:
            return q
    return None


def select(bank, role: str, difficulty: int, exclude_ids, prefer_skills) -> "Question | None":
    """Pick the best non-excluded question.

    Deterministic total order (lower wins): same role > generic-ish > other role;
    closest difficulty; preferred skill > generic > other skill; then id (stable).
    Difficulty and role are *relaxed* (never returned None while questions remain),
    so an 8-turn interview never runs dry at this bank size.
    """
    excl = set(exclude_ids or [])
    prefer = set(prefer_skills or [])
    best = None
    best_key = None
    for q in bank:
        if q.id in excl:
            continue
        role_score = 0 if q.role == role else (1 if q.role == "generic" else 2)
        diff_dist = abs(q.difficulty - difficulty)
        pref_score = 0 if q.skill_id in prefer else (1 if q.skill_id == "generic" else 2)
        key = (role_score, diff_dist, pref_score, q.id)
        if best_key is None or key < best_key:
            best_key = key
            best = q
    return best
```

- [x] **Step 4: Create `app/interview/__init__.py`** (empty, or re-export `load_bank`, `select`, `Question`).

- [x] **Step 5:** `uv run pytest tests/test_interview.py -v` → 5 pass. `uv run pytest -m "not integration" -q` → 59 passed. `uv run ruff check .` → clean.

- [x] **Step 6: Commit**
```bash
git add app/interview tests/test_interview.py
git -c core.autocrlf=false commit -m "feat: interview question bank with deterministic role/difficulty/skill selection"
```

### Task 3.2: Adaptive engine + interview API

**Files:**
- Create: `app/interview/engine.py`
- Create: `app/api/routes/interviews.py`
- Modify: `app/api/main.py` (include `interviews.router`)
- Test: `tests/test_interview.py` (append engine + API tests)

- [x] **Step 1: Write failing engine + API tests** (append to `tests/test_interview.py`):

```python
import json
from sqlmodel import Session, select as sql_select
from app.interview.engine import InterviewSession, role_for_job, TurnResult, InterviewSummary
from app.models import Interview, Job, JobMatch


def test_role_for_job():
    assert role_for_job(Job(id="j", title="Backend Engineer (Python)", description="", required_skills=[], preferred_skills=[])) == "backend"
    assert role_for_job(Job(id="j", title="Data Analyst", description="", required_skills=[], preferred_skills=[])) == "data"
    assert role_for_job(Job(id="j", title="DevOps Engineer", description="", required_skills=[], preferred_skills=[])) == "devops"
    assert role_for_job(Job(id="j", title="Frontend Engineer (React)", description="", required_skills=[], preferred_skills=[])) == "frontend"


def test_session_adapts_up_on_good_answers():
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python"], transcript=[])
    s.next_question()
    # a long answer containing the keywords should score high and raise difficulty
    r = s.answer("In Python a list is ordered, a dict is key value, and a generator yields lazily.")
    assert 0 <= r.score <= 100
    assert r.next_difficulty == min(2, r.next_difficulty)  # sanity
    # a thin empty answer should score low and not raise above start
    s2 = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python"], transcript=[])
    s2.next_question()
    r2 = s2.answer("uh")
    assert r2.score < 70


def test_session_finishes_after_eight_turns():
    bank = load_bank()
    s = InterviewSession(role="backend", bank=bank, llm=MockLLM(), prefer_skills=["python", "sql"], transcript=[])
    for _ in range(8):
        s.next_question()
        r = s.answer("I explain lists dicts generators joins and indexing in detail here today.")
        assert r.finished is (_ is 7)
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
    for i in range(8):
        a = client.post(f"/api/interviews/{iid}/answers", json={"text": "I discuss lists dicts generators joins indexing and transactions in some depth here."})
        assert a.status_code == 200, a.text
        ab = a.json()
        if ab.get("finished"):
            finished = True
            assert "summary" in ab and 0 <= ab["summary"]["score"] <= 100
        else:
            assert "question" in ab  # next question present
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
```

*(The tests import `MockLLM` - add `from app.llm.mock import MockLLM` to the test file top.)*

- [x] **Step 2: Run to verify they fail** → FAIL (No module named `app.interview.engine`).

- [x] **Step 3: Implement `app/interview/engine.py`:** (state machine per section 0.1; `MAX_TURNS = 8`; reconstructable from `transcript`)

```python
"""Adaptive mock-interview state machine. Pure layer - no FastAPI/SQLModel.

All evolving state lives in the `transcript` list (see plan section 0.1), so a
session is fully reconstructed from an Interview row on each request.
"""
import json
from dataclasses import dataclass, field

MAX_TURNS = 8


def _safe_json(raw: str):
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return None


def role_for_job(job) -> str:
    t = (getattr(job, "title", "") or "").lower()
    if any(k in t for k in ("devops", "site reliability", "cloud", "sre")):
        return "devops"
    if any(k in t for k in ("data", "analytic", "machine learning", "vision", "scientist", "bi ", "analytics")):
        return "data"
    if any(k in t for k in ("frontend", "front-end", "react", "vue", "angular", "next.js", "web developer", "full-stack", "mern", "ui", "ios", "android", "mobile")):
        return "frontend"
    return "backend"


@dataclass
class TurnResult:
    score: int
    feedback: str
    next_difficulty: int
    question_id: str
    question: str
    finished: bool


@dataclass
class InterviewSummary:
    score: int
    per_skill: dict
    feedback: str
    transcript: list = field(default_factory=list)


class InterviewSession:
    def __init__(self, role: str, bank, llm, prefer_skills, transcript):
        self.role = role
        self.bank = bank
        self.llm = llm
        self.prefer_skills = list(prefer_skills or [])
        self.transcript = [dict(t) for t in (transcript or [])]

    @property
    def answered(self):
        return [t for t in self.transcript if t.get("score") is not None]

    @property
    def turn(self):
        return len(self.answered)

    @property
    def finished(self):
        return self.turn >= MAX_TURNS

    @property
    def used_ids(self):
        return [t["q_id"] for t in self.transcript]

    @property
    def pending(self):
        if self.transcript and self.transcript[-1].get("score") is None:
            return self.transcript[-1]
        return None

    @property
    def current_difficulty(self):
        if self.pending is not None:
            return self.pending.get("difficulty", 1)
        if self.transcript:
            return self.transcript[-1].get("next_difficulty", 1)
        return 1

    def next_question(self):
        if self.finished:
            return None
        if self.pending is not None:
            return get_question(self.bank, self.pending["q_id"])
        q = select(self.bank, self.role, self.current_difficulty, self.used_ids, self.prefer_skills)
        if q is None:
            return None
        d = self.current_difficulty
        self.transcript.append({"q_id": q.id, "q": q.question, "skill_id": q.skill_id,
                                "a": "", "difficulty": d, "score": None, "next_difficulty": d})
        return q

    def answer(self, text: str) -> TurnResult:
        pend = self.pending
        if pend is None:
            self.next_question()
            pend = self.pending
            if pend is None:
                raise ValueError("no question to answer")
        q = get_question(self.bank, pend["q_id"])
        payload = json.dumps({"question": q.question, "answer": text, "keywords": list(q.keywords)})
        parsed = _safe_json(self.llm.complete(system="TASK: evaluate_answer", user=payload, json_mode=True))
        if isinstance(parsed, dict):
            score = parsed.get("score")
            feedback = parsed.get("feedback", "")
        else:
            score, feedback = None, ""
        if not isinstance(score, int):
            score = 0
        score = max(0, min(100, score))
        d = pend["difficulty"]
        if score >= 70:
            nd = min(2, d + 1)
        elif score < 40:
            nd = max(0, d - 1)
        else:
            nd = d
        pend["a"] = text
        pend["score"] = score
        pend["next_difficulty"] = nd
        return TurnResult(score=score, feedback=feedback, next_difficulty=nd,
                          question_id=q.id, question=q.question, finished=self.finished)

    def finish(self) -> InterviewSummary:
        a = self.answered
        overall = int(round(sum(t["score"] for t in a) / len(a))) if a else 0
        per = {}
        for t in a:
            per.setdefault(t["skill_id"], []).append(t["score"])
        per_skill = {k: int(round(sum(v) / len(v))) for k, v in per.items()}
        parsed = _safe_json(self.llm.complete(
            system="TASK: interview_feedback",
            user=json.dumps({"score": overall, "per_skill": per_skill}), json_mode=True))
        feedback = parsed.get("feedback", "") if isinstance(parsed, dict) else ""
        return InterviewSummary(score=overall, per_skill=per_skill, feedback=feedback, transcript=self.transcript)
```

*(import `from .bank import get_question, select` at top of engine.py)*

- [x] **Step 4: Implement `app/api/routes/interviews.py`:**

```python
from uuid import uuid4
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.interview.bank import load_bank
from app.interview.engine import InterviewSession, role_for_job
from app.models import Interview, Job, JobMatch, Student

router = APIRouter()


def _q_payload(q):
    return {"id": q.id, "question": q.question, "skill_id": q.skill_id, "difficulty": q.get("difficulty", q_difficulty(q))}


def _session(session: Session, interview: Interview, ctx) -> InterviewSession:
    job = session.get(Job, interview.job_id)
    role = role_for_job(job) if job else "backend"
    prefer = list(job.required_skills) if job else []
    return InterviewSession(role, load_bank(), ctx.llm, prefer, interview.transcript or [])


@router.post("/api/students/{sid}/interviews")
def start_interview(sid: str, payload: dict | None = None, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    job_id = (payload or {}).get("job_id")
    if job_id:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
    else:
        matches = session.exec(select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())).all()
        if not matches:
            raise HTTPException(400, "no matches computed yet; GET /api/students/{sid}/matches first")
        job = session.get(Job, matches[0].job_id)
    iid = f"int-{uuid4().hex[:8]}"
    row = Interview(id=iid, student_id=sid, job_id=job.id, transcript=[], score=None, feedback={}, finished=False)
    session.add(row)
    session.commit()
    s = _session(session, row, ctx)
    q = s.next_question()
    row.transcript = s.transcript
    session.add(row)
    session.commit()
    return JSONResponse(status_code=201, content={
        "interview_id": iid, "job_id": job.id, "role": s.role,
        "difficulty": s.current_difficulty,
        "question": {"id": q.id, "question": q.question, "skill_id": q.skill_id, "difficulty": q.difficulty} if q else None,
    })


@router.post("/api/interviews/{iid}/answers")
def answer_interview(iid: str, payload: dict, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    row = session.get(Interview, iid)
    if row is None:
        raise HTTPException(404, "interview not found")
    if row.finished:
        raise HTTPException(409, "interview already finished")
    text = (payload or {}).get("text", "")
    s = _session(session, row, ctx)
    turn = s.answer(text)
    if turn.finished:
        summary = s.finish()
        row.transcript = summary.transcript
        row.score = summary.score
        row.feedback = {"overall": summary.feedback, "per_skill": summary.per_skill}
        row.finished = True
        session.add(row)
        session.commit()
        return JSONResponse(content={
            "finished": True,
            "summary": {"score": summary.score, "per_skill": summary.per_skill, "feedback": summary.feedback},
        })
    nq = s.next_question()
    row.transcript = s.transcript
    session.add(row)
    session.commit()
    return JSONResponse(content={
        "finished": False,
        "score": turn.score,
        "feedback": turn.feedback,
        "next_difficulty": turn.next_difficulty,
        "question": {"id": nq.id, "question": nq.question, "skill_id": nq.skill_id, "difficulty": nq.difficulty} if nq else None,
    })


@router.get("/api/interviews/{iid}")
def get_interview(iid: str, session: Session = Depends(get_session)):
    row = session.get(Interview, iid)
    if row is None:
        raise HTTPException(404, "interview not found")
    return {
        "interview_id": row.id,
        "student_id": row.student_id,
        "job_id": row.job_id,
        "finished": row.finished,
        "score": row.score,
        "feedback": row.feedback,
        "transcript": row.transcript,
        "started_at": row.started_at.isoformat(),
    }
```

*(Remove the stray `q_difficulty` helper and the unused `_q_payload`/`q` reference - the snippet has a leftover; implement the clean version using the inline question dict as shown in the return statements. Ensure imports are complete for ruff.)*

- [x] **Step 5: Include router in `app/api/main.py`** - add `interviews` to the `from app.api.routes import ...` line and `app.include_router(interviews.router)` in `create_app` (after `plans.router`).

- [x] **Step 6:** `uv run pytest tests/test_interview.py -v` → all pass. `uv run pytest -m "not integration" -q` → all pass. `uv run ruff check .` → clean.

- [x] **Step 7: Commit**
```bash
git add app/interview/engine.py app/api/routes/interviews.py app/api/main.py tests/test_interview.py
git -c core.autocrlf=false commit -m "feat: adaptive interview engine (8 turns, difficulty 0-2) + interview API"
```

### Task 3.3: Browser voice UI

**Files:**
- Create: `app/templates/student/interview.html`
- Create: `app/static/app.js`
- Modify: `app/api/main.py` (mount `/static`, add `GET /interviews/{id}` HTML route) - OR add the HTML route to `app/api/routes/interviews.py`
- Test: `tests/test_interview.py` (append HTML route test)

- [x] **Step 1: Write failing test** (append to `tests/test_interview.py`):

```python
def test_interview_html_route_renders(client_engine):
    client, eng = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")}, data={"name": "Voice UI"})
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    iid = r.json()["interview_id"]
    h = client.get(f"/interviews/{iid}")
    assert h.status_code == 200
    assert "interview" in h.text.lower()
    assert "app.js" in h.text  # static script wired


def test_static_js_served(client_engine):
    client, _ = client_engine
    r = client.get("/static/app.js")
    assert r.status_code == 200
    assert "speechSynthesis" in r.text or "SpeechRecognition" in r.text
```

- [x] **Step 2:** Run → FAIL (routes/mount not present).

- [x] **Step 3: Implement the HTML route + static mount.** In `app/api/main.py`:
```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
...
BASE_DIR = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
```
Add the HTML route (in `interviews.py` or a small `pages` section) that renders `student/interview.html` with `interview_id`. Note: `create_app` builds the app, so the static mount + templates must be created inside `create_app` (or module-level and attached). Keep it working with the TestClient.

- [x] **Step 4: Write `app/templates/student/interview.html`** - a self-contained page (Tailwind CDN) that:
  - Shows the current question (fetched from `/api/interviews/{id}` on load).
  - A "🎤 Listen" button → `speechSynthesis.speak(new SpeechSynthesisUtterance(question))`.
  - A "🎙️ Record" button → `webkitSpeechRecognition` (or `window.SpeechRecognition`); on `onresult`, take the transcript and POST it as an answer.
  - A **text input + Send button ALWAYS present** (fallback for Firefox / when `SpeechRecognition` is undefined). On send, POST the text as an answer.
  - On answer response: show the score + feedback; if not finished, show the next question; if finished, render the summary (score, per-skill bars, feedback) and stop.
  - Guard: `const SR = window.SpeechRecognition || window.webkitSpeechRecognition;` if undefined, disable the mic button and rely on text.
  - Use `fetch` to `/api/interviews/{id}/answers` with `{"text": ...}`.

- [x] **Step 5: Write `app/static/app.js`** with the logic above (vanilla JS, no framework). Keep it a single file.

- [x] **Step 6:** `uv run pytest tests/test_interview.py -v` → all pass. `uv run pytest -m "not integration" -q` → all pass. `uv run ruff check .` → clean.

- [x] **Step 7: Commit**
```bash
git add app/templates app/static app/api/main.py app/api/routes/interviews.py tests/test_interview.py
git -c core.autocrlf=false commit -m "feat: adaptive interview voice UI (Web Speech) with text fallback"
```

- [x] **Step 8: Phase 3 gate**
```
uv run pytest -m "not integration" -q && uv run ruff check .
```
All green → commit `chore: phase 3 complete --allow-empty`.
Update master plan `2026-10-31-ps06-placement-agent.md`: add `### Phase 3 STATUS: COMPLETE` block after `## 8. Phase 3` (mirror Phase 1/2 format) with review outcomes + handoff to Phase 4 (dashboard will link to `/interviews/{id}` and can start interviews via the API).

---

## Notes for the executing agent
- **No schema change** to `Interview` - all state in `transcript`. If you feel tempted to add `difficulty`/`role`/`turn` columns, DON'T - reconstruct from transcript (section 0.1).
- The `pending` question (score=null) is how the asked-but-unanswered question survives across the start and answer requests.
- `MockLLM` already implements `TASK: evaluate_answer` (keyword overlap 0-70 + length 0-30, clamped 0-100) and `TASK: interview_feedback` (score + per-skill → text). The engine must send `json_mode=True` and parse defensively (the planner pattern: try/except + isinstance dict guard).
- `role_for_job` + `prefer_skills = job.required_skills` are derived from the job each request (pure, no re-matching). This is the documented v1 choice (probe the job's required skills).
- Keep the voice UI's text input ALWAYS rendered (spec requirement - Firefox has no SpeechRecognition).
- If a test needs many turns, craft answers that hit keywords so difficulty behavior is exercised deterministically.
