# PS-06 Phase 2 - LLM Adapters, Upskilling Planner, Progress Tracker - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider-agnostic LLM layer (offline `mock` default, AWS Bedrock free tier, optional OpenAI), a deterministic upskilling planner that turns skill gaps into an ordered course plan with an LLM narrative, and a progress tracker that turns plan completion into a monotonic readiness series.

**Architecture:** `AppContext.llm` (currently `None`) becomes an `LLMClient` resolved once at app startup via `get_llm(settings)` - tests get `MockLLM` (deterministic, offline), production gets `BedrockLLM` when `LLM_PROVIDER=bedrock`. The planner is pure computation (`gaps -> courses -> Plan`) plus one `narrative` LLM call; it **reuses `rank_jobs`** (never re-derives scores) and `score_job` breakdowns for gap extraction. Progress = per-plan-item completion events that compute readiness = `round(100 * done_hours / total_hours)` blended 50/50 with the student's current best match score (formula documented in the endpoint docstring).

**Tech Stack:** Python 3.12 (uv), FastAPI + SQLModel (SQLite in tests / PostgreSQL in dev), boto3 `converse` for Bedrock, `pytest` + `HashingEmbedder` + `MemoryVectorStore` + `MockLLM` for zero-dependency unit tests.

---

## 0. Ground rules (from Phase 1 final review - do not violate)

1. **Reuse, don't re-derive:** planner and progress compute best-match via `app.matching.rank.rank_jobs` + `app.matching.engine.score_job` breakdown. No duplicate weighting/scoring logic.
2. **Never store a `Session` on `AppContext`** - sessions are request-scoped (`get_session` in `app/api/deps.py`); the production `_SessionPerOpPgVectorStore` in `app/api/__main__.py` opens its own sessions per op.
3. **`datetime.now(timezone.utc)` only** - never `datetime.utcnow()` (naive → 500s on API serialization).
4. **All three embed paths (seed, API, CLI) embed `job.description`** - do not change embedding input.
5. **Test seams preserved:** `tests/conftest.py:_build_client()` wires `AppContext(llm=MockLLM())`; `app/cli.py` `_make_engine/_make_embedder/_make_store` monkeypatch seams stay.
6. **Gate per task:** `uv run pytest -m "not integration" -q` green before each commit; `uv run ruff check .` green from Task 2.0 onward.

## 0.1 Test DB policy (unchanged from Phase 1)

Unit tests: in-memory SQLite (`StaticPool`) + `HashingEmbedder` + `MemoryVectorStore` + `MockLLM` - zero network/DB. Integration marker `pytest.mark.integration` hits live Postgres `ps06_test`. CI-style gate: `pytest -m "not integration"`.

---

### Task 2.0: ruff linter baseline (FIRST - fixes the open Phase 1 TODO)

**Files:**
- Modify: `pyproject.toml` (add `[tool.ruff]` section)
- Modify: many (auto-fixed) + hand-fixed

- [x] **Step 1: Write the ruff config** (append to `pyproject.toml`):

```toml
[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["I001", "F401", "F841", "UP017", "B008", "RUF046", "E711", "E712"]
ignore = []

[tool.ruff.lint.flake8-bugbear]
# FastAPI's Depends(...)/File(...)/Form(...) in defaults is idiomatic; exempt it.
extend-immutable-calls = ["fastapi.Depends", "fastapi.File", "fastapi.Form", "fastapi.Query", "fastapi.Path"]
```

- [x] **Step 2: Run autofix** (this resolves I001/F401/UP017/B008 in one pass):

Run: `uv run ruff check . --fix`
Expected: "Fixed N errors" / remaining errors = `BLE001` (2, in `app/cli.py`) + `RUF046` (1, in `app/matching/engine.py`), both hand-fixed below. B008 should be GONE via `extend-immutable-calls`.

- [x] **Step 3: Hand-fix `BLE001` blind-except in `app/cli.py`** (lines ~102 and ~132, the two `except Exception as e:` in `cmd_seed` and `cmd_demo`). Replace both with a targeted tuple covering everything `init_db`/`seed`/`rank_jobs`/parse can raise:

```python
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as e:
        print(f"error: seed failed: {e}", file=sys.stderr)
        return 1
```
(and the same tuple in `cmd_demo`'s outer try with its own message).

- [x] **Step 4: Hand-fix `RUF046` in `app/matching/engine.py`** (line 45, `score = int(round(...))`). ruff's fix is "Remove unnecessary `int` call" - `round(float)` already returns `int` in Python 3. Change to:

```python
    score = round(100 * (WEIGHTS["coverage"] * coverage + WEIGHTS["preferred"] * pref_score + WEIGHTS["semantic"] * sim))
```
Confirm `score` is still typed `int` (JobScore.score is `int`) - `round(x)` with one arg returns `int`, so `JobScore(score=score)` typechecks fine and all existing matching tests still pass.

- [x] **Step 5: Run full suite + lint**

Run: `uv run pytest -m "not integration" -q`
Expected: 31 passed (was 31 before this task - no behavior change).
Run: `uv run ruff check .`
Expected: `All checks passed!`

- [x] **Step 6: Commit**

```bash
git add pyproject.toml app tests
git -c core.autocrlf=false commit -m "chore: add ruff config, autofix imports/unused/UP017, target BLE001/RUF046"
```

---

### Task 2.1: LLM abstraction (`app/llm/`)

**Files:**
- Create: `app/llm/__init__.py`
- Create: `app/llm/base.py`
- Create: `app/llm/mock.py`
- Create: `app/llm/bedrock.py`
- Create: `app/llm/openai.py`
- Modify: `app/config.py` (already has `llm_provider/llm_model/aws_region/aws_access_key_id/aws_secret_access_key/openai_api_key/openai_model` - no change needed)
- Test: `tests/test_llm.py`

- [x] **Step 1: Write failing tests** (`tests/test_llm.py`):

```python
import json

from app.llm.base import LLMClient, get_llm
from app.llm.mock import MockLLM
from app.llm.bedrock import BedrockLLM
from app.llm.openai import OpenAILLM
from app.config import Settings


def _settings(**overrides):
    base = dict(llm_provider="mock", llm_model="m", aws_region="us-east-1",
                aws_access_key_id=None, aws_secret_access_key=None,
                openai_api_key=None, openai_model="gpt-4o-mini")
    base.update(overrides)
    return Settings(**base)


def test_factory_returns_mock_by_default():
    llm = get_llm(_settings())
    assert isinstance(llm, MockLLM)


def test_factory_returns_bedrock_for_bedrock_provider(monkeypatch):
    # avoid boto3 network client construction in __init__ during test
    import app.llm.bedrock as bedrock_mod
    class FakeClient:
        def converse(self, **kw):
            return {"output": {"message": {"content": [{"text": "hi"}]}}}
    monkeypatch.setattr(bedrock_mod.boto3, "client", lambda *a, **k: FakeClient())
    llm = get_llm(_settings(llm_provider="bedrock"))
    assert isinstance(llm, BedrockLLM)


def test_factory_returns_openai_for_openai_provider(monkeypatch):
    llm = get_llm(_settings(llm_provider="openai", openai_api_key="sk-test"))
    assert isinstance(llm, OpenAILLM)


def test_mock_llm_json_mode_returns_parseable_json():
    llm = MockLLM()
    out = llm.complete(system="TASK: plan_narrative", user='{"gaps": ["git"]}', json_mode=True)
    parsed = json.loads(out)
    assert "narrative" in parsed or "text" in parsed


def test_mock_llm_text_mode_returns_text():
    llm = MockLLM()
    out = llm.complete(system="greet", user="world", json_mode=False)
    assert isinstance(out, str) and out


def test_bedrock_complete_builds_converse_request(monkeypatch):
    import app.llm.bedrock as bedrock_mod
    captured = {}

    class FakeClient:
        def converse(self, **kw):
            captured.update(kw)
            return {"output": {"message": {"content": [{"text": "resp"}]}}}

    monkeypatch.setattr(bedrock_mod.boto3, "client", lambda *a, **k: FakeClient())
    llm = BedrockLLM("us-east-1", "amazon.nova-lite-v1:0")
    out = llm.complete(system="sys", user="usr", json_mode=True)
    assert out == "resp"
    assert captured["modelId"] == "amazon.nova-lite-v1:0"
    assert captured["system"][0]["text"] == "sys"
    assert captured["messages"][0]["content"][0]["text"] == "usr"
    assert captured["inferenceConfig"]["maxTokens"] == 2000
    assert captured["additionalModelRequestFields"]["output_format"] == "json"

    captured.clear()
    llm.complete(system="s", user="u", json_mode=False)
    assert "additionalModelRequestFields" not in captured
    assert captured["inferenceConfig"] == {"maxTokens": 2000}


def test_openai_complete_builds_chat_request(monkeypatch):
    import app.llm.openai as openai_mod

    class FakeClient:
        def __init__(self, key):
            self.key = key

        def chat(self):
            return self

        def completions(self):
            return self

        def create(self, **kw):
            captured.update(kw)
            class Msg:
                class Choice:
                    class Msg:
                        content = "ai-reply"
                message = Msg()
            class Choice:
                message = Msg()
            class R:
                choices = [Choice()]
            return R()

    captured = {}
    monkeypatch.setattr(openai_mod, "_client_factory", lambda key: FakeClient(key))
    llm = OpenAILLM("sk-test", "gpt-4o-mini")
    out = llm.complete(system="s", user="u", json_mode=True)
    assert out == "ai-reply"
    assert captured["model"] == "gpt-4o-mini"
```

*(The OpenAI test above is written against the seam `_client_factory` so no real `openai` package import is needed in tests - `app/llm/openai.py` will lazily import `openai` only inside `_client_factory` when the provider is actually used.)*

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.llm'`

- [x] **Step 3: Implement `app/llm/base.py`**

```python
from typing import Protocol, runtime_checkable

from ..config import Settings


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...


def get_llm(settings: Settings) -> LLMClient:
    """Factory on settings.llm_provider. mock is the default offline path."""
    provider = (settings.llm_provider or "mock").lower()
    if provider == "bedrock":
        from .bedrock import BedrockLLM
        return BedrockLLM(settings.aws_region, settings.llm_model,
                          settings.aws_access_key_id, settings.aws_secret_access_key)
    if provider == "openai":
        from .openai import OpenAILLM
        return OpenAILLM(settings.openai_api_key, settings.openai_model)
    from .mock import MockLLM
    return MockLLM()
```

- [x] **Step 4: Implement `app/llm/mock.py`**

`MockLLM` routes on `TASK: <name>` markers in `system` and returns deterministic JSON (json_mode) or text. It must handle the three tasks Phase 2/3 need: `plan_narrative`, `evaluate_answer`, `interview_feedback` (plus a generic fallback).

```python
import json


class MockLLM:
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        tokens = system.split()
        task = ""
        for i, token in enumerate(tokens):
            if not token.startswith("TASK:"):
                continue
            task = token.split(":", 1)[1]
            if not task and i + 1 < len(tokens):
                task = tokens[i + 1]
            break

        if task == "plan_narrative":
            payload = json.loads(user) if user else {}
            gaps = payload.get("gaps", [])
            hours = payload.get("total_hours", 0)
            narrative = (
                f"Focus on {', '.join(gaps) if gaps else 'core fundamentals'} first - "
                f"these are the highest-leverage gaps for your target role. "
                f"Your plan totals {hours} hours; spread across the week "
                f"(about {max(1, round(hours / 4))}h x 4 weeks) keeps it sustainable. "
                "Complete items in order and mark them done to watch your readiness climb."
            )
            if json_mode:
                return json.dumps({"narrative": narrative})
            return narrative

        if task == "evaluate_answer":
            payload = json.loads(user) if user else {}
            answer = (payload.get("answer") or "").lower()
            keywords = [k.lower() for k in payload.get("keywords", [])]
            hits = sum(1 for k in keywords if k in answer)
            denom = max(1, len(keywords))
            keyword_score = round(70 * hits / denom)
            length_bonus = 30 if len(answer.split()) >= 12 else max(0, round(30 * len(answer.split()) / 12))
            score = max(0, min(100, keyword_score + length_bonus))
            feedback = f"Matched {hits}/{denom} key concepts; answer {'is detailed' if length_bonus == 30 else 'could be more detailed'}."
            if json_mode:
                return json.dumps({"score": score, "feedback": feedback})
            return feedback

        if task == "interview_feedback":
            payload = json.loads(user) if user else {}
            score = payload.get("score", 0)
            per_skill = payload.get("per_skill", {})
            strengths = [s for s, v in per_skill.items() if v >= 70]
            weak = [s for s, v in per_skill.items() if v < 70]
            line = (
                f"Overall {score}/100. "
                + (f"Strong on {', '.join(strengths)}. " if strengths else "")
                + (f"Revisit {', '.join(weak)} before applying. " if weak else "")
                + "Keep practicing - consistency beats cramming."
            )
            if json_mode:
                return json.dumps({"feedback": line})
            return line

        # generic fallback
        if json_mode:
            return json.dumps({"echo": user})
        return f"mock response to: {user[:80]}"
```

- [x] **Step 5: Implement `app/llm/bedrock.py`**

```python
import boto3


class BedrockLLM:
    def __init__(self, region: str, model: str, access_key: str | None = None, secret_key: str | None = None):
        kwargs = {"region_name": region}
        if access_key and secret_key:
            kwargs["aws_access_key_id"] = access_key
            kwargs["aws_secret_access_key"] = secret_key
        self._c = boto3.client("bedrock-runtime", **kwargs)
        self._m = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {
            "modelId": self._m,
            "system": [{"text": system}],
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {"maxTokens": 2000},
        }
        if json_mode:
            kwargs["additionalModelRequestFields"] = {"output_format": "json"}
        r = self._c.converse(**kwargs)
        return r["output"]["message"]["content"][0]["text"]
```

*(Note: `boto3` must be imported at module top of `bedrock.py` so tests can monkeypatch `app.llm.bedrock.boto3.client`. `boto3` is already a project dependency.)*

- [x] **Step 6: Implement `app/llm/openai.py`** (lazily imports `openai` only when used; `_client_factory` seam for tests):

```python
def _client_factory(api_key: str):
    import openai  # lazily - only when provider=openai and package installed
    return openai.OpenAI(api_key=api_key)


class OpenAILLM:
    def __init__(self, api_key: str | None, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self._api_key = api_key
        self._client = None
        self._m = model

    def _get_client(self):
        if self._client is None:
            self._client = _client_factory(self._api_key)
        return self._client

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        kwargs = {
            "model": self._m,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._get_client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
```

- [x] **Step 7: Create `app/llm/__init__.py`** (empty or re-export):

```python
from .base import LLMClient, get_llm
from .mock import MockLLM

__all__ = ["LLMClient", "get_llm", "MockLLM"]
```

- [x] **Step 8: Wire `MockLLM` into `tests/conftest.py`** (`_build_client`, replace `llm=None`):

```python
from app.llm.mock import MockLLM
...
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
```

- [x] **Step 9: Run tests**

Run: `uv run pytest tests/test_llm.py -v`
Expected: PASS (all 7 tests).
Run: `uv run pytest -m "not integration" -q`
Expected: 44 passed (32 existing before this task + 12 new LLM tests), 2 deselected.

- [x] **Step 10: Commit**

```bash
git add app/llm tests/test_llm.py tests/conftest.py
git -c core.autocrlf=false commit -m "feat: LLM abstraction with mock/bedrock/openai providers behind get_llm factory"
```

---

### Task 2.2: Upskilling planner

**Files:**
- Create: `app/plan/__init__.py`
- Create: `app/plan/planner.py`
- Create: `app/api/routes/plans.py`
- Modify: `app/api/main.py` (include `plans.router`)
- Modify: `tests/conftest.py` (no change - `MockLLM` already wired)
- Test: `tests/test_planner.py`

- [x] **Step 1: Write failing tests** (`tests/test_planner.py`):

```python
from sqlmodel import Session, select

from app.models import Student, StudentSkill, UpskillPlan, PlanItem, JobMatch
from app.plan.planner import build_plan, pick_course_for_skill


def _seed_student_with_gaps(client_engine):
    """Upload a resume that matches job-001 (Backend Python) but is missing git + redis."""
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Planner Probe"},
    )
    sid = r.json()["id"]
    # trigger matches so JobMatch rows exist
    client.get(f"/api/students/{sid}/matches")
    return client, eng, sid


def test_pick_course_prefers_covers_skill_then_beginner_then_free():
    from app.config import settings  # noqa: F401
    from app.taxonomy.taxonomy import load_taxonomy
    from app.seed import _json
    courses = _json("courses.json")
    # git is covered by course-027 (beginner, free, 3h) only
    c = pick_course_for_skill("git", courses)
    assert c is not None
    assert "git" in c["skill_ids"]
    # sql has 3 courses: course-011 (beginner, paid, 18h), course-013 (intermediate, free, 12h), course-030 (advanced, paid, 16h)
    c_sql = pick_course_for_skill("sql", courses)
    assert c_sql["id"] == "course-011"  # beginner beats free/intermediate


def test_build_plan_orders_required_before_preferred_and_sums_hours(client_engine):
    client, eng, sid = _seed_student_with_gaps(client_engine)
    # build_plan for job-001 against this student
    with Session(eng) as db:
        from app.llm.mock import MockLLM
        from app.taxonomy.taxonomy import load_taxonomy
        from app.embeddings.embedder import HashingEmbedder
        from app.models import Job
        from app.matching.rank import rank_jobs
        job = db.get(Job, "job-001")
        student_skills = {
            ss.skill_id: list(ss.evidence)
            for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
        }
        student_vec = HashingEmbedder().embed("Python FastAPI PostgreSQL Docker")
        ranked = rank_jobs([job], student_vec, student_skills, HashingEmbedder(), load_taxonomy())
        plan, items = build_plan(db, sid, job, ranked[0][1], llm=MockLLM())

    assert plan.total_hours == sum(i.hours for i in items)
    # job-001 missing: required [git], preferred [redis] -> 2 items, required first
    assert len(items) >= 2
    assert items[0].skill_id == "git"
    assert plan.narrative  # MockLLM produced non-empty narrative


def test_plan_api_roundtrip(client_engine):
    client, eng, sid = _seed_student_with_gaps(client_engine)
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    assert r.status_code in (200, 201)
    body = r.json()
    plan_id = body["id"]
    assert body["total_hours"] > 0
    assert body["narrative"]
    # list plans
    r2 = client.get(f"/api/students/{sid}/plans")
    assert any(p["id"] == plan_id for p in r2.json())
    # fetch single
    r3 = client.get(f"/api/plans/{plan_id}")
    assert r3.json()["id"] == plan_id
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_planner.py -v`
Expected: FAIL - `ModuleNotFoundError: No module named 'app.plan'`

- [x] **Step 3: Implement `app/plan/planner.py`**

```python
"""Upskilling planner: deterministic gap->course selection + LLM narrative.

Reuses score_job's missing_required/missing_preferred breakdown (never re-derives
scores) and rank_jobs only for choosing a default target job when none is given.
"""
from sqlmodel import Session, select

from ..matching.engine import JobScore
from ..models import Course, PlanItem, UpskillPlan


def pick_course_for_skill(skill_id: str, courses: list[dict]) -> dict | None:
    """Best course covering skill_id: covers skill, then beginner, then free, then fewer hours."""
    candidates = [c for c in courses if skill_id in c.get("skill_ids", [])]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (
        0 if c.get("level") == "beginner" else 1,
        0 if c.get("cost") == "free" else 1,
        c.get("hours", 999),
    ))
    return candidates[0]


def _load_courses(db: Session) -> list[dict]:
    rows = db.exec(select(Course)).all()
    return [
        {"id": c.id, "title": c.title, "provider": c.provider, "url": c.url,
         "skill_ids": c.skill_ids, "level": c.level, "hours": c.hours, "cost": c.cost}
        for c in rows
    ]


def build_plan(db: Session, student_id: str, job, job_score: JobScore, *, llm) -> tuple[UpskillPlan, list[PlanItem]]:
    """Create UpskillPlan + PlanItems for gaps in job_score breakdown.

    Items ordered: required gaps first, then preferred, each group by hours asc.
    Returns (plan, items) - SQLModel table models reject arbitrary attribute
    assignment (no field `items`), so items are returned separately.
    """
    courses = _load_courses(db)

    # gap tuples: (skill_id, is_required)
    gaps: list[tuple[str, bool]] = []
    for g in job_score.missing_required:
        gaps.append((g["skill"], True))
    for g in job_score.missing_preferred:
        gaps.append((g["skill"], False))

    items_spec: list[tuple[str, str, int, bool]] = []  # (skill_id, course_id, hours, is_required)
    for skill_id, is_required in gaps:
        course = pick_course_for_skill(skill_id, courses)
        if course is None:
            continue  # no course covers this gap - skip (do not fabricate)
        items_spec.append((skill_id, course["id"], course["hours"], is_required))

    # required first, then hours asc within each group
    items_spec.sort(key=lambda t: (0 if t[3] else 1, t[2]))

    plan = UpskillPlan(
        id=f"plan-{student_id}-{job.id}",   # deterministic so re-generation overwrites same job
        student_id=student_id,
        job_id=job.id,
        total_hours=sum(t[2] for t in items_spec),
        narrative="",
    )

    # narrative via LLM (single call, deterministic MockLLM)
    import json
    user_payload = json.dumps({
        "gaps": [t[0] for t in items_spec],
        "total_hours": plan.total_hours,
        "job": getattr(job, "title", ""),
    })
    narrative_raw = llm.complete(system="TASK: plan_narrative", user=user_payload, json_mode=True)
    try:
        parsed = json.loads(narrative_raw)
        plan.narrative = parsed.get("narrative", narrative_raw)
    except (ValueError, TypeError):
        plan.narrative = narrative_raw

    db.add(plan)
    for order, (skill_id, course_id, hours, _is_req) in enumerate(items_spec, start=1):
        db.add(PlanItem(
            id=f"{plan.id}-i{order}",
            plan_id=plan.id,
            order=order,
            skill_id=skill_id,
            course_id=course_id,
            hours=hours,
            status="pending",
        ))
    db.commit()
    items = db.exec(select(PlanItem).where(PlanItem.plan_id == plan.id).order_by(PlanItem.order)).all()
    return plan, items
```

*(Note: returns `(plan, items)` because SQLModel models reject extra attributes - API layer re-queries items via `_serialize_plan`.)*

- [x] **Step 4: Implement `app/api/routes/plans.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import delete
from sqlmodel import Session, select

from app.api.deps import get_ctx, get_session
from app.matching.rank import rank_jobs
from app.models import Course, Job, JobMatch, PlanItem, Student, StudentSkill, UpskillPlan
from app.plan.planner import build_plan

router = APIRouter()


def _serialize_plan(plan: UpskillPlan, db: Session) -> dict:
    items = db.exec(select(PlanItem).where(PlanItem.plan_id == plan.id)).order_by(PlanItem.order).all()
    courses = {c.id: c for c in db.exec(select(Course)).all()}
    return {
        "id": plan.id,
        "student_id": plan.student_id,
        "job_id": plan.job_id,
        "total_hours": plan.total_hours,
        "narrative": plan.narrative,
        "created_at": plan.created_at.isoformat(),
        "items": [
            {
                "id": it.id,
                "order": it.order,
                "skill_id": it.skill_id,
                "course_id": it.course_id,
                "course_title": courses.get(it.course_id).title if it.course_id in courses else "",
                "course_url": courses.get(it.course_id).url if it.course_id in courses else "",
                "hours": it.hours,
                "status": it.status,
            }
            for it in items
        ],
    }


@router.post("/api/students/{sid}/plans")
def create_plan(sid: str, payload: dict | None = None, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
    student = session.get(Student, sid)
    if student is None:
        raise HTTPException(404, "student not found")
    if ctx.llm is None:
        raise HTTPException(503, "LLM not configured")

    job_id = (payload or {}).get("job_id")
    if job_id:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
    else:
        # default to student's current top match
        matches = session.exec(
            select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
        ).all()
        if not matches:
            raise HTTPException(400, "no matches computed yet; GET /api/students/{sid}/matches first")
        job = session.get(Job, matches[0].job_id)
        if job is None:
            raise HTTPException(404, "job not found")

    # build a JobScore for this (student, job) - reuse rank_jobs, never re-derive
    student_skills = {
        ss.skill_id: list(ss.evidence)
        for ss in session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    }
    sv = ctx.vector_store.get("student", sid)
    if sv is None:
        raise HTTPException(404, "student vector not found; re-upload the resume")
    ranked = rank_jobs(
        [job], sv, student_skills, ctx.embedder, ctx.taxonomy,
        vector_store=ctx.vector_store, job_vec_cache=ctx.job_vectors, top=1,
    )
    if not ranked:
        raise HTTPException(500, "failed to score target job")
    job_score = ranked[0][1]

    # replace any prior plan for this job (deterministic plan id overwrites)
    prior = session.get(UpskillPlan, f"plan-{sid}-{job.id}")
    if prior is not None:
        session.execute(delete(PlanItem).where(PlanItem.plan_id == prior.id))
        session.delete(prior)
        session.commit()

    plan, _items = build_plan(session, sid, job, job_score, llm=ctx.llm)
    return JSONResponse(status_code=201, content=_serialize_plan(plan, session))
```


- [x] **Step 5: Add list + get endpoints to `app/api/routes/plans.py`:**

```python
@router.get("/api/students/{sid}/plans")
def list_plans(sid: str, session: Session = Depends(get_session)):
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    plans = session.exec(
        select(UpskillPlan).where(UpskillPlan.student_id == sid).order_by(UpskillPlan.created_at.desc())
    ).all()
    return [_serialize_plan(p, session) for p in plans]


@router.get("/api/plans/{plan_id}")
def get_plan(plan_id: str, session: Session = Depends(get_session)):
    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    return _serialize_plan(plan, session)
```

- [x] **Step 6: Include router in `app/api/main.py`** (add to imports and `create_app`):

```python
from app.api.routes import students, jobs, plans
...
    app.include_router(plans.router)
```

*(Order: include `plans.router` BEFORE `students.router`? No - route paths don't collide (`/api/students/{sid}/plans` vs `/api/students/{sid}` etc.). Include after students for readability.)*

- [x] **Step 7: Run tests**

Run: `uv run pytest tests/test_planner.py -v`
Expected: PASS (3 tests).
Run: `uv run pytest -m "not integration" -q`
Expected: 41 passed (38 + 3).

- [x] **Step 8: Commit**

```bash
git add app/plan app/api/routes/plans.py app/api/main.py tests/test_planner.py
git -c core.autocrlf=false commit -m "feat: upskilling planner with gap->course selection, LLM narrative, plan API"
```

---

### Task 2.3: Progress tracking

**Files:**
- Modify: `app/api/routes/students.py` (add `POST .../complete` + `GET .../progress`)
- Test: `tests/test_progress.py`

- [x] **Step 1: Write failing tests** (`tests/test_progress.py`):

```python
from sqlmodel import Session, select

from app.models import ProgressEvent, UpskillPlan, PlanItem


def _setup_plan(client_engine):
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Progress Probe"},
    )
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    r2 = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    plan = r2.json()
    return client, eng, sid, plan


def test_completing_items_raises_readiness_monotonically(client_engine):
    client, eng, sid, plan = _setup_plan(client_engine)
    items = plan["items"]
    assert len(items) >= 1

    prev = -1
    for it in items:
        r = client.post(f"/api/plans/{plan['id']}/items/{it['id']}/complete")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["readiness"] >= prev
        prev = body["readiness"]
        assert 0 <= body["readiness"] <= 100

    # events persisted
    with Session(eng) as db:
        events = db.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)).all()
        assert len(events) == len(items)
        assert all(e.kind == "plan_item_done" for e in events)


def test_progress_endpoint_returns_ordered_series(client_engine):
    client, eng, sid, plan = _setup_plan(client_engine)
    for it in plan["items"]:
        client.post(f"/api/plans/{plan['id']}/items/{it['id']}/complete")

    r = client.get(f"/api/students/{sid}/progress")
    assert r.status_code == 200
    body = r.json()
    assert "readiness" in body and "events" in body and "best_match" in body
    series = body["readiness"]
    assert len(series) == len(plan["items"])
    # ordered by at ascending
    times = [p["at"] for p in series]
    assert times == sorted(times)
    # best_match present
    assert body["best_match"] is None or "score" in body["best_match"]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_progress.py -v`
Expected: FAIL - `404` (routes not implemented)

- [x] **Step 3: Implement `POST /api/plans/{plan_id}/items/{item_id}/complete` in `app/api/routes/students.py`**

Append at end of file (import `ProgressEvent`, `datetime`, `timezone` at top if not present):

```python
from datetime import datetime, timezone
...
from app.models import JobMatch, PlanItem, ProgressEvent, Student, StudentSkill, UpskillPlan
```

```python
@router.post("/api/plans/{plan_id}/items/{item_id}/complete")
def complete_plan_item(
    plan_id: str,
    item_id: str,
    session: Session = Depends(get_session),
    ctx=Depends(get_ctx),
):
    """Mark a plan item done and record a readiness snapshot.

    readiness = round(100 * done_hours / total_hours) blended 50/50 with the
    student's current best match score (max JobMatch.score across their matches).
    Formula: final = round(0.5 * plan_completion_pct + 0.5 * best_match_score).
    Documented here as the source of truth for the progress chart.
    """
    plan = session.get(UpskillPlan, plan_id)
    if plan is None:
        raise HTTPException(404, "plan not found")
    item = session.get(PlanItem, item_id)
    if item is None or item.plan_id != plan_id:
        raise HTTPException(404, "plan item not found")

    if item.status != "done":
        item.status = "done"
        session.add(item)
        session.commit()

    # plan completion %
    items = session.exec(select(PlanItem).where(PlanItem.plan_id == plan_id)).all()
    total_hours = plan.total_hours or 1
    done_hours = sum(i.hours for i in items if i.status == "done")
    plan_pct = round(100 * done_hours / total_hours)

    # best match score
    matches = session.exec(select(JobMatch).where(JobMatch.student_id == plan.student_id)).all()
    best_match = max((m.score for m in matches), default=0)

    readiness = round(0.5 * plan_pct + 0.5 * best_match)

    # idempotency: plan IDs are deterministic (plan-{sid}-{job_id}), so a
    # regenerated plan has the SAME item ids. If this exact item already
    # produced an event (complete -> regenerate -> complete again), reuse the
    # existing event row's id instead of colliding on the ProgressEvent PK.
    prior_ev = session.exec(
        select(ProgressEvent).where(
            ProgressEvent.ref_id == item.id, ProgressEvent.kind == "plan_item_done"
        )
    ).first()
    ev_id = prior_ev.id if prior_ev else f"pe-{uuid4().hex[:8]}"

    session.add(ProgressEvent(
        id=ev_id,
        student_id=plan.student_id,
        kind="plan_item_done",
        ref_id=item.id,
        readiness=readiness,
        at=datetime.now(timezone.utc),
    ))
    session.commit()

    return {"item_id": item.id, "status": item.status, "readiness": readiness,
            "plan_pct": plan_pct, "best_match": best_match}
```

- [x] **Step 4: Implement `GET /api/students/{sid}/progress` in `app/api/routes/students.py`:**

```python
@router.get("/api/students/{sid}/progress")
def get_progress(sid: str, session: Session = Depends(get_session)):
    if session.get(Student, sid) is None:
        raise HTTPException(404, "student not found")
    events = session.exec(
        select(ProgressEvent).where(ProgressEvent.student_id == sid).order_by(ProgressEvent.at.asc())
    ).all()
    matches = session.exec(select(JobMatch).where(JobMatch.student_id == sid)).order_by(JobMatch.score.desc()).all()
    best = matches[0] if matches else None
    return {
        "readiness": [{"at": e.at.isoformat(), "readiness": e.readiness} for e in events],
        "events": [
            {"id": e.id, "kind": e.kind, "ref_id": e.ref_id, "readiness": e.readiness, "at": e.at.isoformat()}
            for e in events
        ],
        "best_match": (
            {"job_id": best.job_id, "score": best.score, "band": best.band} if best else None
        ),
    }
```

- [x] **Step 5: Run tests**

Run: `uv run pytest tests/test_progress.py -v`
Expected: PASS (2 tests).
Run: `uv run pytest -m "not integration" -q`
Expected: 43 passed.

- [x] **Step 6: Commit**

```bash
git add app/api/routes/students.py tests/test_progress.py
git -c core.autocrlf=false commit -m "feat: plan item completion, readiness blending, progress series endpoint"
```

- [x] **Step 7: Phase 2 gate**

Run: `uv run pytest -m "not integration" -q && uv run ruff check .`
Expected: 43 passed, `All checks passed!`

```bash
git -c core.autocrlf=false commit -m "chore: phase 2 complete" --allow-empty
```

Update the master plan file `docs/superpowers/plans/2026-10-31-ps06-placement-agent.md`: mark Task 2.0/2.1/2.2/2.3 checkboxes `- [x]` and append a `### Phase 2 STATUS: COMPLETE` block after `## 7. Phase 2` header (mirror the Phase 1 STATUS block format).

```bash
git add docs/superpowers/plans/2026-10-31-ps06-placement-agent.md
git -c core.autocrlf=false commit -m "docs: mark phase 2 complete in plan"
```

---

## Notes for the executing agent (subagent or inline)

- **Reuse `rank_jobs` everywhere** for scoring; `score_job`'s `missing_required`/`missing_preferred` is the only gap source. Do not write a second scoring function.
- **`MockLLM` is the only LLM in tests** - never import `boto3.client` or `openai.OpenAI` at module import time in a way that fires network/auth; lazy-import inside `__init__`/`_client_factory`.
- **Plan IDs are deterministic** (`plan-{sid}-{job_id}`) so re-generating a plan for the same job overwrites cleanly - delete prior `PlanItem`s + `UpskillPlan` before insert.
- **`ctx.llm` must be non-`None`** for plan endpoints - `tests/conftest.py` wires `MockLLM()`; if a test builds `AppContext` manually it must pass an LLM or the endpoint 503s (that's intended behavior, tested implicitly).
- **datetime:** always `datetime.now(timezone.utc)`.
- If stuck or a review flags something non-obvious, ask Antigravity (`ask_antigravity`, session_id `ps06-phase2`).
