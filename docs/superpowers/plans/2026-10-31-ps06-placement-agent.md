# PS-06 Intelligent Placement Readiness & Career Matching Agent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an agentic AI system that matches students to jobs with explainable skill-gap analysis, generates personalised upskilling plans, runs adaptive mock interviews, tracks progress over time, and produces cohort-level analytics for the placement cell.

**Architecture:** FastAPI backend (Python 3.12 via `uv`) over PostgreSQL 17 + pgvector. A provider-agnostic LLM layer (offline deterministic `mock` → AWS Bedrock free tier → optional OpenAI) and embedding layer (local `sentence-transformers`, with a deterministic hashing fallback for tests). Job matching = weighted skill coverage + preferred-skill bonus + semantic similarity, with a per-skill explainable breakdown. Server-rendered dashboards (Jinja2 + HTMX + Tailwind CDN + Chart.js — no Node build step). Mock interviews use the browser Web Speech API (free, no keys) for STT/TTS, with an adapter seam for AWS Transcribe/Polly later.

**Tech Stack:** Python 3.12 (uv-managed), FastAPI, SQLModel, psycopg 3, PostgreSQL 17 + pgvector, sentence-transformers (all-MiniLM-L6-v2, 384-dim), boto3 (Bedrock), Jinja2 + HTMX + Tailwind CDN + Chart.js, pytest.

---

## 0. Milestones and how to proceed

| Phase | Deliverable | Maps to spec deliverable | Est. effort |
|---|---|---|---|
| 0 | Env bootstrap (uv project, Postgres+pgvector, git) | — | ~2–3 h |
| 1 | Thin vertical slice: resume → parse → skills → explainable job match (CLI + API + tests) | Job-matching agent, explainable skill-gap analyser | 2–4 days |
| 2 | LLM adapters (mock/Bedrock/OpenAI) + upskilling planner + progress tracker | Personalised upskilling planner, progress tracker | 1–2 days |
| 3 | Adaptive mock interview engine + browser voice UI | Adaptive mock interview module | 1–2 days |
| 4 | Student career dashboard (HTMX pages) | Student career dashboard | 1 day |
| 5 | Placement-cell analytics dashboard | Placement-cell analytics dashboard | 1 day |
| 6 | Polish: README, demo script, seed data, final QA | Expected outcome | half day |

**Execution rule:** Phase 1 below is fully specified at TDD task granularity. Phases 2–5 are specified at the same interface/test level but will be expanded into their own detailed plan (in this directory) when the phase starts — do not start a later phase without that expansion. Each phase ends with a green `pytest -m "not integration"` run and a commit.

## 1. Environment decisions (verified on this machine, 2026-10-31)

1. **Python:** system Python is 3.15.0 (early access; ML wheels unreliable). Pin the project to **Python 3.12** via `uv python pin 3.12` (uv downloads a managed CPython — no installer needed).
2. **Postgres:** no Docker/psql present. Install **PostgreSQL 17 via winget**, then add **pgvector** from its GitHub release binary (merged into the PG install dir). Databases: `ps06` (dev) and `ps06_test` (tests).
3. **Node:** not installed and not needed — dashboards are server-rendered (Jinja2 + HTMX + CDN assets). Zero build step.
4. **git:** not installed — install `Git.Git` via winget in Phase 0.
5. **LLM provider (user decision: AWS free tier):**
   - **`mock` (default):** deterministic template-based LLM. No network, no key. Powers all tests and the offline demo.
   - **`bedrock` (AWS free path):** new AWS accounts get a time-boxed free tier for Bedrock inference (Amazon Titan / Nova Lite — exact units/tokens on the current Bedrock pricing page). Requirements: AWS account with payment method (identity verification only), IAM user credentials, model access enabled in `ap-south-1` (Mumbai), `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` in `.env`. Configure `LLM_PROVIDER=bedrock` once credentials exist — no code change.
   - **`openai` (optional upgrade):** higher-quality interview feedback if budget appears later.
   - **Embeddings never call AWS:** local sentence-transformers (one-time ~90 MB model download) is free, offline, and sufficient for a mock-JD dataset of this scale.
6. **Speech (mock interviews):** browser Web Speech API (`webkitSpeechRecognition` + `speechSynthesis`) in Chrome — free, no keys. `app/speech.py` defines a `SpeechProvider` interface so AWS Transcribe/Polly can be swapped in without UI changes.
7. **Secrets:** `.env` only; `.env` is git-ignored; `.env.example` committed. No key is ever logged or committed.

## 2. Repository layout

```
ps06-placement-agent/
  pyproject.toml
  .python-version            # 3.12 (uv)
  .env.example
  .gitignore
  README.md
  data/
    skills.json              # skill taxonomy: id, name, category, aliases
    jobs.json                # mock job-description dataset (~24 JDs)
    courses.json             # course/resource catalogue (~40 courses)
    questions.json           # interview question bank (Phase 3)
  app/
    __init__.py
    config.py                # pydantic-settings
    db.py                    # engine factory, init_db
    models.py                # SQLModel tables
    schemas.py               # pydantic API schemas
    seed.py                  # load data/*.json into DB
    cli.py                   # python -m app.cli  (seed | match | demo)
    taxonomy/
      __init__.py
      taxonomy.py            # Taxonomy loader + alias resolution
    resume/
      __init__.py
      parser.py              # PDF/DOCX/TXT -> text
    skills/
      __init__.py
      extract.py             # text -> {skill_id: [evidence lines]}
    embeddings/
      __init__.py
      embedder.py            # HashingEmbedder, LocalEmbedder (protocol Embedder)
      store.py               # MemoryVectorStore, PgVectorStore (protocol VectorStore)
    matching/
      __init__.py
      engine.py              # score_job() -> JobScore (explainable)
    llm/
      __init__.py
      base.py                # LLMClient protocol + factory
      mock.py                # deterministic MockLLM
      bedrock.py             # BedrockLLM (boto3 converse)
      openai.py              # OpenAILLM
    plan/
      __init__.py
      planner.py             # gaps -> courses -> Plan + LLM narrative
    interview/
      __init__.py
      bank.py                # load questions.json, selection rules
      engine.py              # adaptive session state machine
    analytics/
      __init__.py
      cohort.py              # cohort aggregations
    api/
      __init__.py
      main.py                # FastAPI app factory + AppContext
      deps.py                # get_session / get_ctx dependencies
      routes/
        __init__.py
        students.py          # upload resume, skills, matches, plan, progress
        jobs.py
        plans.py
        interviews.py
        admin.py             # cohort analytics (Phase 5)
    templates/
      base.html
      student/profile.html, matches.html, plan.html, progress.html, interview.html
      admin/analytics.html
    static/
      app.js                 # Web Speech STT/TTS for interviews (Phase 3)
  tests/
    conftest.py              # sqlite + hashing embedder + memory store + MockLLM
    fixtures/                # tiny sample resumes (txt, pdf, docx)
    test_taxonomy.py, test_extract.py, test_matching.py, test_parser.py,
    test_embed.py, test_seed.py, test_api.py,
    test_planner.py, test_progress.py,
    test_interview.py,
    test_analytics.py,
    integration/
      test_pgvector.py       # marked integration (needs local Postgres)
```

**File responsibility rule:** one clear responsibility per file; matching logic never imports FastAPI; LLM calls only happen inside `app/llm/*` and are injected via `AppContext`.

## 3. Data assets (Phase 1, written before code)

- `data/skills.json` — ~120 skills across categories `languages, web_frontend, web_backend, databases, devops_cloud, data_ml, testing, tools, soft_skills`. Shape:
  ```json
  {"id": "python", "name": "Python", "category": "languages", "aliases": ["python", "python3"]}
  ```
  Aliases must be word-bounded regex-safe strings (no leading `*`; `c`, `c++`, `c#` each get their own alias entry).
- `data/jobs.json` — 24 mock JDs across roles: backend, frontend, fullstack, data analyst, data scientist, ML engineer, DevOps, QA, cloud, mobile. Shape:
  ```json
  {"id": "job-001", "title": "Backend Engineer (Python)", "company": "TechNova",
   "description": "2-4 sentences of realistic role copy used for embedding.",
   "required_skills": ["python", "fastapi", "postgresql", "git"],
   "preferred_skills": ["docker", "aws"], "experience_years": 0,
   "location": "Bengaluru", "remote": true, "salary_lpa": [8, 14]}
  ```
- `data/courses.json` — ~40 courses mapping skills → resources (Coursera/edX/NPTEL/freeCodeCamp/YouTube/Docs). Shape:
  ```json
  {"id": "course-001", "title": "Docker & Kubernetes: The Practical Guide",
   "provider": "Udemy", "url": "https://.../", "skill_ids": ["docker"],
   "level": "intermediate", "hours": 12, "cost": "paid"}
  ```
  Coverage rule: every skill that appears in any `required_skills` must have ≥1 course.

## 4. Domain model and core algorithm

### 4.1 SQLModel tables (`app/models.py`)

```python
class Student(SQLModel, table=True):
    id: str = Field(primary_key=True)          # "stu-<uuid8>"
    name: str
    branch: str = ""
    year: int = 3
    gpa: float = 0.0
    created_at: datetime

class StudentSkill(SQLModel, table=True):
    student_id: str = Field(foreign_key="student.id", primary_key=True)
    skill_id: str = Field(foreign_key="skill.id", primary_key=True)
    mentions: int = 0                           # evidence count from resume
    level: str = "basic"                        # basic|intermediate|advanced

class Skill(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    category: str
    aliases: list[str] = Field(default_factory=list, sa_column=Column(JSON))

class Job(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    company: str
    description: str
    required_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    preferred_skills: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    experience_years: int = 0
    location: str = ""
    remote: bool = False
    salary_lpa: list[int] = Field(default_factory=lambda: [0, 0], sa_column=Column(JSON))

class JobMatch(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    job_id: str = Field(foreign_key="job.id")
    score: int                                  # 0..100
    band: str                                   # strong|moderate|developing|early
    breakdown: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime

class Course(SQLModel, table=True):
    id: str = Field(primary_key=True)
    title: str
    provider: str
    url: str
    skill_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    level: str = "beginner"
    hours: int = 4
    cost: str = "free"

class UpskillPlan(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    job_id: str = Field(foreign_key="job.id")
    total_hours: int
    narrative: str = ""                         # LLM-written motivation/schedule
    created_at: datetime

class PlanItem(SQLModel, table=True):
    id: str = Field(primary_key=True)
    plan_id: str = Field(foreign_key="upskillplan.id")
    order: int
    skill_id: str
    course_id: str = Field(foreign_key="course.id")
    hours: int
    status: str = "pending"                     # pending|in_progress|done

class ProgressEvent(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    kind: str                                   # plan_item_done|match_rescored|interview
    ref_id: str = ""
    readiness: int                              # snapshot 0..100 at this moment
    at: datetime

class Interview(SQLModel, table=True):
    id: str = Field(primary_key=True)
    student_id: str = Field(foreign_key="student.id")
    job_id: str = Field(foreign_key="job.id")
    transcript: list[dict] = Field(default_factory=list, sa_column=Column(JSON))  # [{q, a, difficulty, score}]
    score: int | None = None
    feedback: dict = Field(default_factory=dict, sa_column=Column(JSON))
    finished: bool = False
    started_at: datetime
```

### 4.2 Skill extraction (`app/skills/extract.py`)

Deterministic alias matching against the taxonomy:

```python
import re
from ..taxonomy.taxonomy import Taxonomy

def extract_skills(text: str, taxonomy: Taxonomy) -> dict[str, list[str]]:
    """Return {skill_id: [evidence lines]} for every taxonomy alias found in text."""
    found: dict[str, list[str]] = {}
    for line in text.splitlines():
        low = line.lower()
        for alias, skill_id in taxonomy.alias_map.items():
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", low):
                bucket = found.setdefault(skill_id, [])
                if line.strip() not in bucket:
                    bucket.append(line.strip())
    return found
```

Level heuristic (in `app/skills/extract.py`): `mentions >= 4` or line contains `advanced|expert|proficient` → `advanced`; `mentions >= 2` or `intermediate|experienced` → `intermediate`; else `basic`.

### 4.3 Matching score (`app/matching/engine.py`)

```python
@dataclass
class JobScore:
    job_id: str
    score: int                 # 0..100
    band: str                  # strong|moderate|developing|early
    matched_required: list[dict]   # [{skill, name, category, evidence}]
    missing_required: list[dict]   # [{skill, name, category}]
    matched_preferred: list[dict]
    missing_preferred: list[dict]
    semantic_sim: float

WEIGHTS = ("coverage", "preferred", "semantic")  # 0.6 / 0.2 / 0.2

def score_job(student_skills: dict[str, list[str]], job: Job, sim: float, taxonomy: Taxonomy) -> JobScore:
    req = [taxonomy.get(s) for s in job.required_skills]
    pref = [taxonomy.get(s) for s in job.preferred_skills]
    matched_req = [s for s in req if s.id in student_skills]
    missing_req = [s for s in req if s.id not in student_skills]
    matched_pref = [s for s in pref if s.id in student_skills]
    missing_pref = [s for s in pref if s.id not in student_skills]
    coverage = len(matched_req) / len(req) if req else 1.0
    pref_score = len(matched_pref) / len(pref) if pref else 0.0
    sim = max(0.0, min(1.0, sim))
    score = int(round(100 * (0.6 * coverage + 0.2 * pref_score + 0.2 * sim)))
    return JobScore(
        job_id=job.id, score=score, band=band_for(score),
        matched_required=[_with_evidence(s, student_skills) for s in matched_req],
        missing_required=[_as_dict(s) for s in missing_req],
        matched_preferred=[_as_dict(s) for s in matched_pref],
        missing_preferred=[_as_dict(s) for s in missing_pref],
        semantic_sim=round(sim, 3),
    )

def band_for(score: int) -> str:
    return "strong" if score >= 80 else "moderate" if score >= 60 else "developing" if score >= 40 else "early"
```

`_with_evidence` attaches the first stored resume line proving the skill — this is what makes the gap analysis **explainable** in the UI ("matched because your resume says 'Built REST APIs with FastAPI and PostgreSQL'").

### 4.4 Embeddings and vector store

```python
# app/embeddings/embedder.py
class Embedder(Protocol):
    dim: int
    def embed(self, text: str) -> list[float]: ...

class HashingEmbedder:            # deterministic, zero deps — used by ALL unit tests
    def __init__(self, dim: int = 384):
        self.dim = dim
    def embed(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[a-z0-9+#.]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

class LocalEmbedder:              # real usage: sentence-transformers/all-MiniLM-L6-v2
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.dim = self._m.get_sentence_embedding_dimension()
    def embed(self, text: str) -> list[float]:
        return self._m.encode([text], normalize_embeddings=True)[0].tolist()
```

```python
# app/embeddings/store.py
class VectorStore(Protocol):
    def upsert(self, kind: str, ref_id: str, vector: list[float]) -> None: ...
    def search(self, kind: str, vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def get(self, kind: str, ref_id: str) -> list[float] | None: ...

class MemoryVectorStore: ...      # numpy cosine, O(n) — fine at mock scale

class EmbeddingRow(SQLModel, table=True):
    __tablename__ = "embeddings"
    kind: str = Field(primary_key=True)
    ref_id: str = Field(primary_key=True)
    embedding: list[float] = Field(sa_column=Column(Vector(384)))   # pgvector

class PgVectorStore: ...          # upsert via ON CONFLICT; search via ORDER BY embedding <=> $1 LIMIT k
```

`AppContext` wires: `embedder = LocalEmbedder()` if `settings.embedder == "local"` else `HashingEmbedder()`; `vector_store = PgVectorStore(session)` if Postgres else `MemoryVectorStore()`.

## 5. Phase 0 — Environment bootstrap

### Task 0.1: Project scaffold
**Files:** Create `ps06-placement-agent/` (this directory), `pyproject.toml`, `.python-version`, `.gitignore`, `.env.example`

- [ ] **Step 1:** In `C:\Users\ryuke\Downloads\Agentic AI\ps06-placement-agent`:
  ```powershell
  uv init --name ps06 --no-readme
  uv python pin 3.12
  ```
- [ ] **Step 2:** Write `.gitignore`:
  ```
  .env
  .venv/
  __pycache__/
  *.pyc
  .pytest_cache/
  test_*.db
  ```
- [ ] **Step 3:** Write `.env.example`:
  ```ini
  DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/ps06
  EMBEDDER=local            # local | hashing
  EMBEDDING_DIM=384
  LLM_PROVIDER=mock         # mock | bedrock | openai
  LLM_MODEL=amazon.nova-lite-v1:0
  AWS_REGION=ap-south-1
  # AWS_ACCESS_KEY_ID=
  # AWS_SECRET_ACCESS_KEY=
  OPENAI_API_KEY=
  OPENAI_MODEL=gpt-4o-mini
  ```
- [ ] **Step 4:** Commit: `git init`, `git add -A`, `git commit -m "chore: scaffold uv project for PS-06"`.

### Task 0.2: PostgreSQL 17 via winget
- [ ] **Step 1:** `winget install -e --id PostgreSQL.PostgreSQL.17` (interactive installer — set the **postgres superuser password** when prompted; note it for `.env`).
- [ ] **Step 2:** Verify service: `Get-Service postgresql* | Select-Object Name, Status`
- [ ] **Step 3:** Create databases:
  ```powershell
  $env:PGPASSWORD="your-password"; & "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE ps06; CREATE DATABASE ps06_test;"
  ```
  Expected: two `CREATE DATABASE` lines.

### Task 0.3: pgvector extension
- [ ] **Step 1:** Download the Windows x64 binary for Postgres 17 from https://github.com/pgvector/pgvector/releases (the `pgvector-*-win64.zip` asset matching PG major 17) into the temp dir.
- [ ] **Step 2:** Merge its contents into `C:\Program Files\PostgreSQL\17\` — specifically `pgvector.dll` into `bin\` and `vector.control` + `vector--*.sql` into `share\extension\`.
- [ ] **Step 3:** Restart the service (`Restart-Service postgresql-x64-17`), then in psql: `CREATE EXTENSION vector;` in both `ps06` and `ps06_test`. Expected: `CREATE EXTENSION`.

### Task 0.5: git + baseline test run
- [x] **Step 1:** git installed via winget in Task 0.1.
- [x] **Step 2:** repo-local identity set (repo-local `user.name=ryuke`, `user.email=ryuke@users.noreply.github.com` — real values, not the plan's placeholder).
- [x] **Step 3:** `uv run pytest -q` → `no tests ran` (exit 5) — confirms pytest works.

### Task 0.4: Dependencies
- [x] **Step 1:** deps added via `uv add` (fastapi, uvicorn, sqlmodel, psycopg[binary], pgvector, pydantic-settings, pymupdf, python-docx, sentence-transformers, numpy, boto3) + dev (pytest, httpx). pyproject description set; `.env.example` polished (commented `OPENAI_API_KEY`, `.env*` gitignore with `!.env.example` negation — code-review minors applied).
- [x] **Step 2:** embedder cache warmed: `all-MiniLM-L6-v2` encodes to 384 dims.
- [x] **Step 3:** committed `138947c chore: add dependencies, warm embedder cache, config polish`.

### Task 0.2: PostgreSQL 17 via winget — [DONE, with environment notes]
- [x] winget installed PostgreSQL 17.11 (service `postgresql-x64-17`).
- [x] Databases `ps06` and `ps06_test` created.
- **Environment notes (why this deviated from plan):**
  - The interactive installer's password prompt never completed, leaving an unknown `postgres` password. **Fix:** `pg_hba.conf` loopback entries (`127.0.0.1/32`, `::1/128`) switched from `scram-sha-256` to `trust` (backup at `pg_hba.conf.bak`); local-only, safe for dev. Connection string therefore uses **no password**: `postgresql+psycopg://postgres@127.0.0.1:5432/ps06`.
  - `postgres.exe` is started directly by the setup (service control requires admin; service DACL was extended with `sc sdset` to grant `BU` start/stop/query so normal-user `sc start/stop postgresql-x64-17` works — verified).
  - `.env` (local, git-ignored) carries the real `DATABASE_URL`/`TEST_DATABASE_URL`.

### Task 0.3: pgvector extension — [DONE, with environment notes]
- [x] No prebuilt Windows binaries exist in pgvector releases → **built from source**: cloned pgvector (0.8.7), installed VS BuildTools 2022 (VCTools workload) via winget, `nmake /F Makefile.win` with `PGROOT` set, then copied `vector.dll` → `lib\`, `vector.control` + `vector--0.8.7.sql` → `share\extension\` via one elevated (UAC) helper (no admin token in the normal shell).
- [x] `CREATE EXTENSION vector;` succeeded in both DBs; `'[1,2,3]'::vector <=> '[1,2,4]'::vector` = `0.00853986601633272` from Python (sqlalchemy/psycopg) — proven end-to-end.
- pgvector version **0.8.7**; server PostgreSQL **17.11**.

## 6. Phase 1 — Thin vertical slice (resume → explainable match)

### Phase 1 STATUS: COMPLETE (2026-11-01)
- All tasks 1.1-1.9 executed via subagent-driven development with two-stage review each.
- Final gate: unit 31 passed / 2 deselected, full 33 passed (2 live pgvector integration), 24+ commits.
- Final review verdict: "With fixes" -> **all Important fixes applied**:
  1. JobMatch persistence covered by `test_matches_persist_jobmatch_rows` (+`client_engine` fixture).
  2. Shared ranking extracted to `app/matching/rank.py::rank_jobs` (API + CLI consume it; Phase 2 planner MUST use it too - no third copy).
  3. Upload capped at 5 MB via `_read_capped` -> HTTP 413 (tests included).
  4. README.md stub with 3-command quickstart exists (Phase 6 polishes it).
  5. Linter still open -> FIRST TASK of Phase 2: `uv add --dev ruff` + minimal config + `ruff check .` in the phase gate.
- Extra beyond plan: production ASGI entry `app/api/__main__:app` (the test-only create_app had no module-level app); E2E-verified against dev Postgres (PDF upload -> matches -> 10 JobMatch rows persisted).
- Deferred (minor, tracked for Phase 2): JobMatch unique constraint (student_id, job_id); batched job-vector preload at startup; `app/schemas.py` pydantic response models; module docstrings on remaining modules; `embedding_dim` setting is guard-only (EmbeddingRow fixed at Vector(384)).
- Phase 2 readiness notes (from final review): `AppContext.llm=None` seam ready; NEVER store a Session on AppContext (session-per-op pattern in `__main__.py`); keep `datetime.now(timezone.utc)` (never `utcnow`); planner consumes `JobScore` from `rank_jobs` rather than re-deriving; CLI test seams (`_make_engine/_make_embedder/_make_store`) preserved; all three embed paths must stay on `job.description`.

Each task: write test → see it fail → implement → see it pass → commit.
**Test DB policy:** unit tests run on in-memory SQLite + `HashingEmbedder` + `MemoryVectorStore` (zero external deps). `tests/integration/` is marked `@pytest.mark.integration` and needs the local Postgres; CI-style gate is `pytest -m "not integration"`.

In `pyproject.toml` add:
```toml
[tool.pytest.ini_options]
markers = ["integration: needs local PostgreSQL"]
```

### Task 1.1: Skills taxonomy
**Files:** Create `data/skills.json`, `app/taxonomy/__init__.py`, `app/taxonomy/taxonomy.py`, Test `tests/test_taxonomy.py`

- [ ] **Step 1: Write the failing test**
  ```python
  from app.taxonomy.taxonomy import Taxonomy

  DATA = [
      {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
      {"id": "js", "name": "JavaScript", "category": "web_frontend", "aliases": ["javascript"]},
  ]

  def test_resolve_exact_and_alias():
      t = Taxonomy(DATA)
      assert t.resolve("Python") == "python"
      assert t.resolve("JAVASCRIPT") == "js"
      assert t.resolve("rust") is None

  def test_alias_map_is_flattened():
      t = Taxonomy(DATA)
      assert t.alias_map["javascript"] == "js"
      assert t.alias_map["python"] == "python"
  ```
- [ ] **Step 2:** Run `uv run pytest tests/test_taxonomy.py -q` → Expected: FAIL (module not found).
- [ ] **Step 3: Implement `app/taxonomy/taxonomy.py`**
  ```python
  import json
  from dataclasses import dataclass, field
  from pathlib import Path

  DATA_DIR = Path(__file__).resolve().parents[2] / "data"

  @dataclass(frozen=True)
  class Skill:
      id: str
      name: str
      category: str
      aliases: tuple[str, ...] = field(default=())

  class Taxonomy:
      def __init__(self, skills: list[dict] | list[Skill]):
          self._skills: list[Skill] = [
              s if isinstance(s, Skill) else Skill(s["id"], s["name"], s["category"], tuple(s.get("aliases", [])))
              for s in skills
          ]
          self._by_id = {s.id: s for s in self._skills}
          self.alias_map: dict[str, str] = {}
          for s in self._skills:
              self.alias_map[s.id.lower()] = s.id
              for a in s.aliases:
                  self.alias_map[a.lower()] = s.id

      def resolve(self, token: str) -> str | None:
          return self.alias_map.get(token.lower())

      def get(self, skill_id: str) -> Skill:
          return self._by_id[skill_id]

      def all(self) -> list[Skill]:
          return list(self._skills)

  def load_taxonomy() -> Taxonomy:
      return Taxonomy(json.loads((DATA_DIR / "skills.json").read_text(encoding="utf-8")))
  ```
- [ ] **Step 4:** Write `data/skills.json` (~120 entries, per §3). At minimum include: python, javascript, typescript, java, cplusplus, react, html, css, fastapi, django, nodejs, sql, postgresql, mongodb, redis, docker, kubernetes, aws, git, ci_cd, rest_api, graphql, pandas, numpy, sklearn, machine_learning, deep_learning, tensorflow, pytorch, tabular (powerbi), excel, statistics, communication, teamwork, agile.
- [ ] **Step 5:** Run `uv run pytest tests/test_taxonomy.py -q` → Expected: PASS.
- [ ] **Step 6:** Commit `feat: skill taxonomy loader`.

### Task 1.2: Skill extraction from text
**Files:** Create `app/skills/__init__.py`, `app/skills/extract.py`, Test `tests/test_extract.py`

- [ ] **Step 1: Write the failing test**
  ```python
  from app.taxonomy.taxonomy import Taxonomy
  from app.skills.extract import extract_skills, level_for

  T = Taxonomy([
      {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
      {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
      {"id": "c", "name": "C", "category": "languages", "aliases": ["c"]},
  ])
  TEXT = "Built REST APIs with FastAPI in Python\nAdvanced Python scripting\nUsed C in college"

  def test_finds_skills_with_evidence():
      out = extract_skills(TEXT, T)
      assert set(out) == {"python", "fastapi", "c"}
      assert "Built REST APIs with FastAPI in Python" in out["fastapi"]
      assert len(out["python"]) == 2

  def test_word_boundaries_prevent_false_positives():
      out = extract_skills("I use pythonic style", T)
      assert "python" not in out            # "pythonic" must not match alias "python"

  def test_level_heuristic():
      assert level_for(5) == "advanced"
      assert level_for(2) == "intermediate"
      assert level_for(1) == "basic"
  ```
  (Note: alias matching uses `(?<![a-z0-9])<alias>(?![a-z0-9])`, so `pythonic` fails the trailing boundary and does not match `python`.)
- [ ] **Step 2:** Run → Expected: FAIL.
- [ ] **Step 4: Implement `app/skills/extract.py`**
  ```python
  import re
  from ..taxonomy.taxonomy import Taxonomy

  BOUND = r"(?<![a-z0-9]){alias}(?![a-z0-9])"

  def extract_skills(text: str, taxonomy: Taxonomy) -> dict[str, list[str]]:
      """Return {skill_id: [evidence lines]} for every taxonomy alias found in text."""
      found: dict[str, list[str]] = {}
      for line in text.splitlines():
          low = line.lower()
          for alias, skill_id in taxonomy.alias_map.items():
              if re.search(BOUND.format(alias=re.escape(alias)), low):
                  bucket = found.setdefault(skill_id, [])
                  if line.strip() not in bucket:
                      bucket.append(line.strip())
      return found

  ADVANCED_HINT = ("advanced", "expert", "proficient")
  INTERMEDIATE_HINT = ("intermediate", "experienced", "worked with")

  def level_for(mentions: int, evidence_lines: list[str] | None = None) -> str:
      joined = " ".join(evidence_lines or []).lower()
      if mentions >= 4 or any(h in joined for h in ADVANCED_HINT):
          return "advanced"
      if mentions >= 2 or any(h in joined for h in INTERMEDIATE_HINT):
          return "intermediate"
      return "basic"
  ```
  `level_for(mentions)` (single arg) is valid for the test; the optional second argument is used by the API path in Task 1.7.
- [ ] **Step 4:** Run → Expected: PASS (all 3).
- [ ] **Step 5:** Commit `feat: deterministic skill extraction with evidence lines`.

### Task 1.3: Embedders + vector stores
**Files:** Create `app/embeddings/__init__.py`, `app/embeddings/embedder.py`, `app/embeddings/store.py`, Test `tests/test_embed.py`

- [ ] **Step 1: Write the failing test**
  ```python
  import math
  from app.embeddings.embedder import HashingEmbedder
  from app.embeddings.store import MemoryVectorStore

  def test_hashing_embedder_is_deterministic_normalized_384():
      e = HashingEmbedder()
      a, b = e.embed("python fastapi postgres"), e.embed("python fastapi postgres")
      assert a == b
      assert len(a) == 384
      assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)

  def test_similar_text_scores_higher():
      e = HashingEmbedder()
      near = cosine(e.embed("python fastapi postgresql rest api"), e.embed("python fastapi postgres"))
      far = cosine(e.embed("react css html styling"), e.embed("python fastapi postgres"))
      assert near > far

  def test_memory_store_search_topk():
      e = HashingEmbedder()
      s = MemoryVectorStore()
      s.upsert("job", "j1", e.embed("python fastapi"))
      s.upsert("job", "j2", e.embed("react css"))
      res = s.search("job", e.embed("python fastapi sql"), k=2)
      assert res[0][0] == "j1" and len(res) == 2

  def cosine(a, b):
      return sum(x * y for x, y in zip(a, b))
  ```
- [ ] **Step 2:** Run → Expected: FAIL.
- [ ] **Step 3:** Implement per §4.4 (`MemoryVectorStore` with `get`, `upsert`, `search`; `LocalEmbedder` per spec — do **not** instantiate it in unit tests).
- [ ] **Step 4:** Run → Expected: PASS.
- [ ] **Step 5: Add integration test** `tests/integration/test_pgvector.py`:
  ```python
  import pytest
  from sqlmodel import Session
  from app.db import make_engine
  from app.embeddings.store import PgVectorStore, EmbeddingRow

  pytestmark = pytest.mark.integration

  def test_pgvector_roundtrip():
      eng = make_engine("postgresql+psycopg://postgres:postgres@localhost:5432/ps06_test")
      with Session(eng) as db:
          store = PgVectorStore(db, dim=8)
          store.upsert("job", "x", [1.0, 0, 0, 0, 0, 0, 0, 0])
          store.upsert("job", "y", [0.0, 1, 0, 0, 0, 0, 0, 0])
          res = store.search("job", [0.9, 0.1, 0, 0, 0, 0, 0, 0], k=1)
          assert res[0][0] == "x"
          assert res[0][1] > 0.99
  ```
- [ ] **Step 6:** Run `uv run pytest tests/test_embed.py -q` → PASS; optionally `uv run pytest -m integration -q` → PASS (fix PG password in URL if needed).
- [ ] **Step 7:** Commit `feat: embedding layer with memory and pgvector stores`.

### Task 1.4: Resume parser
**Files:** Create `app/resume/__init__.py`, `app/resume/parser.py`, `tests/fixtures/` (sample `resume.txt`, `resume.pdf`, `resume.docx`), Test `tests/test_parser.py`

- [ ] **Step 1: Create fixtures** (one-time, run in `uv run python`):
  ```python
  # make_fixtures.py — run once, then delete
  from pathlib import Path
  import fitz
  from docx import Document

  p = Path("tests/fixtures"); p.mkdir(parents=True, exist_ok=True)
  (p / "resume.txt").write_text("Bharat Sharma\nSkills: Python, FastAPI, PostgreSQL, Docker\nProjects: Built a job portal REST API\n", encoding="utf-8")
  doc = fitz.open(); page = doc.new_page()
  page.insert_text((72, 100), "Bharat Sharma\nSkills: Python, FastAPI, PostgreSQL, Docker")
  doc.save(p / "resume.pdf"); doc.close()
  d = Document(); d.add_paragraph("Bharat Sharma"); d.add_paragraph("Skills: Python, FastAPI, PostgreSQL, Docker"); d.save(p / "resume.docx")
  ```
- [ ] **Step 2: Write the failing test**
  ```python
  from pathlib import Path
  from app.resume.parser import parse_resume

  FIX = Path(__file__).parent / "fixtures"

  def test_parse_txt():
      assert "FastAPI" in parse_resume(FIX / "resume.txt", ".txt")

  def test_parse_pdf():
      assert "PostgreSQL" in parse_resume(FIX / "resume.pdf", ".pdf")

  def test_parse_docx():
      assert "Bharat Sharma" in parse_resume(FIX / "resume.docx", ".docx")

  def test_unsupported_suffix_raises():
      import pytest
      with pytest.raises(ValueError):
          parse_resume(FIX / "resume.txt", ".png")
  ```
- [ ] **Step 3:** Run → FAIL. **Step 4: Implement** `parse_resume` per §2 (fitz for `.pdf`, python-docx for `.docx`, `read_text` for `.txt`, else `raise ValueError(f"unsupported suffix {suffix}")`).
- [ ] **Step 5:** Run → PASS. **Step 6:** Commit `feat: resume text extraction for pdf/docx/txt`.

### Task 1.5: Matching engine
**Files:** Create `app/matching/__init__.py`, `app/matching/engine.py`, Test `tests/test_matching.py`

- [ ] **Step 1: Write the failing test**
  ```python
  from types import SimpleNamespace
  from app.taxonomy.taxonomy import Taxonomy
  from app.matching.engine import score_job, band_for

  T = Taxonomy([
      {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
      {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
      {"id": "postgresql", "name": "PostgreSQL", "category": "databases", "aliases": ["postgresql"]},
      {"id": "docker", "name": "Docker", "category": "devops_cloud", "aliases": ["docker"]},
      {"id": "aws", "name": "AWS", "category": "devops_cloud", "aliases": ["aws"]},
  ])
  JOB = SimpleNamespace(id="job-1", required_skills=["python", "fastapi", "postgresql"],
                        preferred_skills=["docker", "aws"])

  def test_full_coverage_strong():
      s = score_job({"python": ["l1"], "fastapi": ["l1"], "postgresql": ["l1"], "docker": ["l1"]}, JOB, sim=0.9, taxonomy=T)
      assert s.band == "strong" and s.score >= 85
      assert len(s.missing_required) == 0 and len(s.missing_preferred) == 1
      assert s.matched_required[0]["evidence"] == "l1"

  def test_half_coverage_moderate_or_developing():
      s = score_job({"python": ["l1"]}, JOB, sim=0.5, taxonomy=T)
      assert s.score < 60 and {m["skill"] for m in s.missing_required} == {"fastapi", "postgresql"}

  def test_bands():
      assert band_for(85) == "strong" and band_for(70) == "moderate"
      assert band_for(50) == "developing" and band_for(20) == "early"
  ```
- [ ] **Step 2:** Run → FAIL. **Step 3: Implement** `score_job` per §4.3 (include `_as_dict`, `_with_evidence` helpers: `_as_dict` → `{"skill": s.id, "name": s.name, "category": s.category}`, `_with_evidence` adds `"evidence": student_skills[s.id][0]`).
- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `feat: explainable job matching engine`.

### Task 1.6: Config, DB, models, seed
**Files:** Create `app/config.py`, `app/db.py`, `app/models.py`, `app/seed.py`, `data/jobs.json`, `data/courses.json`, Test `tests/test_seed.py`

- [ ] **Step 1: Write the failing test**
  ```python
  import pytest
  from sqlmodel import Session
  from app.db import make_engine, init_db
  from app.seed import seed
  from app.models import Job, Course, Skill
  from app.taxonomy.taxonomy import load_taxonomy

  def test_seed_populates_all_tables():
      eng = make_engine("sqlite://")
      init_db(eng)
      with Session(eng) as db:
          seed(db, load_taxonomy())
          assert db.query(Skill).count() >= 30
          assert db.query(Job).count() >= 20
          assert db.query(Course).count() >= 20

  def test_every_required_skill_has_a_course():
      jobs = _json("jobs.json"); courses = _json("courses.json")
      covered = {s for c in courses for s in c["skill_ids"]}
      required = {s for j in jobs for s in j["required_skills"]}
      assert not (required - covered), f"missing courses for: {required - covered}"
  ```
  (Implement `init_db(engine: Engine)` taking an explicit engine to keep tests trivial.)
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement**
  - `app/config.py` per §1.5 (pydantic-settings, `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`).
  - `app/db.py`:
    ```python
    from sqlalchemy import Engine
    from sqlalchemy import create_engine
    from sqlmodel import SQLModel
    from .config import settings

    def make_engine(url: str | None = None) -> Engine:
        url = url or settings.database_url
        args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        return create_engine(url, connect_args=args)

    def init_db(engine: Engine) -> None:
        from . import models  # noqa: F401  (register tables)
        SQLModel.metadata.create_all(engine)
    ```
  - `app/models.py` per §4.1.
  - `app/seed.py` (add `_json` helper so both tests and `seed()` share one loader):
    ```python
    import json
    from pathlib import Path
    from sqlmodel import Session
    from .models import Skill, Job, Course
    from .taxonomy.taxonomy import Taxonomy, DATA_DIR

    def _json(name: str):
        return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))

    def seed(db: Session, taxonomy: Taxonomy) -> None:
        for s in taxonomy.all():
            db.merge(Skill(id=s.id, name=s.name, category=s.category, aliases=list(s.aliases)))
        for j in _json("jobs.json"):
            db.merge(Job(**j))
        for c in _json("courses.json"):
            db.merge(Course(**c))
        db.commit()
    ```
  - Write `data/jobs.json` (24 JDs) and `data/courses.json` (40 courses) per §3; validate that every required skill has a course (add a `test_skill_coverage` that asserts this from the JSON files).
- [ ] **Step 4:** Run → PASS. **Step 5:** Commit `feat: models, seed data, config`.

### Task 1.7: API — resume upload + explainable matches
**Files:** Create `app/api/__init__.py`, `app/api/main.py`, `app/api/deps.py`, `app/api/routes/students.py`, `app/api/routes/jobs.py`, `tests/conftest.py`, Test `tests/test_api.py`

- [ ] **Step 1: Write `tests/conftest.py`**
  ```python
  import pytest
  from fastapi.testclient import TestClient
  from sqlmodel import Session
  from app.db import make_engine, init_db
  from app.api.main import create_app, AppContext
  from app.embeddings.embedder import HashingEmbedder
  from app.embeddings.store import MemoryVectorStore
  from app.llm.mock import MockLLM
  from app.seed import seed
  from app.taxonomy.taxonomy import load_taxonomy

  @pytest.fixture()
  def client(tmp_path):
      eng = make_engine("sqlite://")
      init_db(eng)
      ctx = AppContext(engine=eng, taxonomy=load_taxonomy(), embedder=HashingEmbedder(),
                      vector_store=MemoryVectorStore(), llm=MockLLM())
      with Session(eng) as db:
          seed(db, ctx.taxonomy)
      app = create_app(ctx)
      with TestClient(app) as c:
          yield c
  ```
- [ ] **Step 2: Write the failing test**
  ```python
  def test_upload_resume_and_get_matches(client):
      r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker, AWS\nBuilt REST APIs with FastAPI\n", "text/plain")},
                      data={"name": "Bharat Sharma"})
      assert r.status_code == 201, r.text
      body = r.json()
      assert body["id"].startswith("stu-")
      skill_ids = {s["id"] for s in body["skills"]}
      assert {"python", "fastapi", "postgresql"} <= skill_ids

      r2 = client.get(f"/api/students/{body['id']}/matches")
      assert r2.status_code == 200
      matches = r2.json()
      assert matches and 0 <= matches[0]["score"] <= 100
      m = matches[0]
      assert {"job_id", "score", "band", "matched_required", "missing_required", "semantic_sim"} <= set(m)
      assert m["matched_required"] or m["missing_required"]

  def test_jobs_list(client):
      r = client.get("/api/jobs")
      assert r.status_code == 200 and len(r.json()) >= 20
  ```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4: Implement**
  - `AppContext` (in `app/api/main.py`): dataclass holding `engine, taxonomy, embedder, vector_store, llm`; `create_app(ctx)` builds `FastAPI(title="PS-06 Placement Agent")` and mounts routers.
  - `deps.py`: `get_session() -> Iterator[Session]` (session per request over `ctx.engine`), `get_ctx()`.
  - `routes/students.py`:
    ```python
    @router.post("/api/students", status_code=201)
    def upload_resume(file: UploadFile, name: str = Form("Student"), session: Session = Depends(get_session), ctx: AppContext = Depends(get_ctx)):
        suffix = Path(file.filename).suffix.lower()
        text = parse_resume_bytes(file.file.read(), suffix)   # parser accepts bytes too (fitz.open(stream) / Document(io.BytesIO))
        sid = f"stu-{uuid4().hex[:8]}"
        found = extract_skills(text, ctx.taxonomy)
        session.add(Student(id=sid, name=name))
        for skill_id, lines in found.items():
            session.add(StudentSkill(student_id=sid, skill_id=skill_id, mentions=len(lines), level=level_for(len(lines))))
        session.commit()
        vec = ctx.embedder.embed(text)
        ctx.vector_store.upsert("student", sid, vec)
        return StudentOut(id=sid, name=name, skills=[{...}])
    ```
    `parse_resume_bytes(data: bytes, suffix: str) -> str` in `app/resume/parser.py` (PDF via `fitz.open(stream=BytesIO(data))`, DOCX via `Document(BytesIO(data))`, else `data.decode("utf-8", "ignore")`) — keep the `parse_resume(path, suffix)` wrapper for CLI/tests.
  - `GET /api/students/{sid}/matches`: load StudentSkill rows → `student_skills` dict; for each Job: `sim = cosine(student_vec, job_vec)` (job vectors: embed `job.description + " " + " ".join(skill names)` once at seed time and store via `vector_store.upsert("job", job_id, ...)` inside `seed()` when `ctx` is available — for the API path, compute lazily on first request and cache in `ctx._job_vecs` dict). Score all jobs, sort desc, persist top-10 `JobMatch` rows (replace existing for the student), return breakdown list.
  - `routes/jobs.py`: `GET /api/jobs`, `GET /api/jobs/{id}`.
- [ ] **Step 5:** Run `uv run pytest tests/test_api.py -q` → PASS.
- [ ] **Step 6:** Commit `feat: resume upload and explainable match endpoints`.

### Task 1.7b: pgvector seeding path (integration)
**Files:** Modify `app/seed.py`, Test `tests/integration/test_seed_pg.py`
- [ ] **Step 1:** Extend `seed(db, taxonomy, vector_store=None, embedder=None)` to embed + store job vectors when both are provided (Postgres path passes `PgVectorStore`).
- [ ] **Step 2:** Integration test: seed `ps06_test` with `LocalEmbedder` + `PgVectorStore`, assert `SELECT count(*) FROM embeddings WHERE kind='job'` == 24.
- [ ] **Step 3:** Commit `feat: embed job vectors during seed (pgvector)`.

### Task 1.8: CLI
**Files:** Create `app/cli.py`
- [ ] **Step 1:** Implement with argparse:
  ```
  python -m app.cli seed                 # init_db + seed on DATABASE_URL (local embedder + pgvector)
  python -m app.cli demo --file r.pdf --name "Bharat" --top 5
                                         # uploads via in-process call (no HTTP), prints table:
                                         # rank | score | band | title | missing required
  ```
- [ ] **Step 2:** Verify manually: create `r.pdf` fixture, run seed + demo against local Postgres; expected: a ranked table with explainable missing-skill columns.
- [ ] **Step 3:** Commit `feat: cli seed/demo`.

### Task 1.9: Phase 1 gate
- [ ] **Step 1:** `uv run pytest -m "not integration" -q` → all green.
- [ ] **Step 2:** `uv run pytest -m integration -q` → green (local Postgres).
- [ ] **Step 3:** Commit `chore: phase 1 complete — vertical slice`.
**Acceptance:** a stranger runs the 3 README quickstart commands and sees an explainable match for a sample resume.

## 7. Phase 2 — LLM adapters, upskilling planner, progress tracker

> Expand into its own detailed plan (`2026-XX-XX-ps06-phase2-planner.md`) before coding. Interfaces are locked in here.


### Phase 2 STATUS: COMPLETE (2026-11-01)
- Detailed plan: `2026-11-01-ps06-phase2-planner.md`. All tasks (2.0 ruff baseline, 2.1 LLM, 2.2 planner, 2.3 progress) executed via subagent-driven development with two-stage review each.
- Final gate: full 56 passed (54 unit + 2 live pgvector integration), 54 passed/2 deselected for `not integration`, ruff clean. 33+ commits.
- **Review-driven fixes (bugs the plan snippets had, caught by reviews + TDD):**
  1. LLM: Bedrock `json_mode` sent invalid `inferenceConfig.responseFormat` (botocore ParamValidationError) -> `additionalModelRequestFields={"output_format":"json"}`; switched to first-class `system` param.
  2. LLM: MockLLM `TASK:` marker parser fixed for both `TASK:name` and `TASK: name` forms; OpenAILLM defers client construction to first call (openai pkg not installed).
  3. Planner: narrative `json.loads` non-dict (list/number) raised AttributeError -> isinstance dict guard; 5 extra tests (llm-503, 400-no-matches, default-top-match, non-dict narrative, skip-uncovered-gap).
  4. Progress: idempotent re-completion crashed with IntegrityError PK collision on plan regen -> update-in-place for prior event; fixed order_by-on-session-exec bug.
- **ruff baseline (Task 2.0):** select = I001,F401,F841,B008,RUF046,E711,E712 (UP017 deliberately excluded to preserve `datetime.now(timezone.utc)`); FastAPI Depends/File/Form exempted via extend-immutable-calls; 2 BLE001 narrowed in cli.py (SQLAlchemyError added for cmd_seed DB failures).
- **Follow-ups for Phase 4 (progress chart) - MUST address before building the chart:**
  1. Orphan ProgressEvents on plan regeneration with fewer items (regeneration deletes PlanItems, not events) -> clean up or scope events by plan.
  2. Update-in-place re-stamps `at`, so a regenerated + re-completed item moves to end of the chronological series (can dip). Document/adjust series semantics.
- **Handoff to Phase 3:** `AppContext.llm` = MockLLM in tests (Bedrock/OpenAI in prod via `get_llm(settings)`); interview engine uses `TASK: evaluate_answer` + `TASK: interview_feedback` (already in MockLLM, tested); deterministic per-skill keyword scoring available offline.

### Task 2.1: LLM abstraction
**Files:** Create `app/llm/{base,mock,bedrock,openai}.py`, Test `tests/test_llm.py`
- [ ] `base.py`:
  ```python
  class LLMClient(Protocol):
      def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...

  def get_llm(settings) -> LLMClient:   # factory on settings.llm_provider
      if settings.llm_provider == "bedrock": from .bedrock import BedrockLLM; return BedrockLLM(settings.aws_region, settings.llm_model)
      if settings.llm_provider == "openai": from .openai import OpenAILLM; return OpenAILLM(settings.openai_api_key, settings.openai_model)
      from .mock import MockLLM; return MockLLM()
  ```
- [ ] `bedrock.py` (boto3 `converse`, no extra deps):
  ```python
  import json
  import boto3

  class BedrockLLM:
      def __init__(self, region: str, model: str):
          self._c = boto3.client("bedrock-runtime", region_name=region)
          self._m = model
      def complete(self, system: str, user: str, json_mode: bool = False) -> str:
          kwargs = {"modelId": self._m,
                    "system": [{"text": system}],
                    "messages": [{"role": "user", "content": [{"text": user}]}],
                    "inferenceConfig": {"maxTokens": 2000}}
          if json_mode:
              kwargs["additionalModelRequestFields"] = {"output_format": "json"}
          r = self._c.converse(**kwargs)
          return r["output"]["message"]["content"][0]["text"]
  ```
  (Amazon Nova models: `additionalModelRequestFields={"output_format": "json"}` requests JSON output. `inferenceConfig` only accepts `maxTokens/temperature/topP/stopSequences` - do NOT put `responseFormat` there, botocore's service model rejects it with ParamValidationError.)
- [ ] `mock.py`: `MockLLM.complete()` routes on markers in `system` (e.g. `"TASK: plan_narrative"`, `"TASK: evaluate_answer"`, `"TASK: interview_feedback"`) and returns deterministic JSON/text built from the `user` payload. This keeps every downstream feature testable offline.
- [ ] Tests: factory returns right class per provider value; `MockLLM` returns parseable JSON for `json_mode` tasks; `BedrockLLM.complete` against `boto3.client` mocked with `monkeypatch` (no network in tests).
- [ ] Commit per task.

### Task 2.2: Upskilling planner
**Files:** Create `app/plan/planner.py`, Modify `app/api/routes/plans.py`, Test `tests/test_planner.py`
- [ ] Algorithm (deterministic core, LLM only for narrative):
  1. Input: student, target job (default: their top match).
  2. `gaps = missing_required + missing_preferred` from `score_job` breakdown (required first).
  3. For each gap skill: pick best course from `courses.json` by `(covers skill, level==beginner first, free first, fewer hours first)`.
  4. `total_hours = sum(course.hours)`; items ordered by required→preferred, then hours asc.
  5. `narrative = llm.complete(system="TASK: plan_narrative...")` — 3-sentence motivation + suggested weekly split; MockLLM produces the same shape from templates.
- [ ] API: `POST /api/students/{sid}/plans {job_id?}` → creates `UpskillPlan` + `PlanItem`s (replaces previous open plan for that job); `GET /api/students/{sid}/plans`, `GET /api/plans/{id}`.
- [ ] Test: student missing 2 required + 1 preferred → plan has ≥3 items, required items ordered before preferred, `total_hours` equals sum, narrative non-empty (MockLLM).
- [ ] Commit `feat: upskilling planner + API`.

### Task 2.3: Progress tracking
**Files:** Modify `app/models.py` (already has `ProgressEvent`), `app/api/routes/students.py`, Test `tests/test_progress.py`
- [ ] `POST /api/plans/{plan_id}/items/{item_id}/complete` → sets `status="done"`, computes current readiness = `round(100 * done_hours / total_hours)` blended 50/50 with the student's best current match score (document the formula in the endpoint docstring), appends `ProgressEvent(kind="plan_item_done", readiness=..., at=now)`.
- [ ] `GET /api/students/{sid}/progress` → `{"readiness": [...{at, readiness}], "events": [...], "best_match": {...}}`.
- [ ] Tests: completing items raises readiness monotonically; progress endpoint returns ordered series.
- [ ] Commit `feat: progress events and readiness series`.
- [ ] **Gate:** full test suite green; commit `chore: phase 2 complete`.

## 8. Phase 3 — Adaptive mock interviews

> Expand into its own detailed plan before coding.


### Phase 3 STATUS: COMPLETE (2026-11-01)
- Detailed plan: `2026-11-01-ps06-phase3-interview.md`. All tasks (3.1 bank, 3.2 engine+API, 3.3 voice UI) executed via subagent-driven development with spec + quality review.
- Final gate: full 70 passed (68 unit + 2 live pgvector integration), 68 passed/2 deselected for `not integration`, ruff clean. 14 tests in `tests/test_interview.py`.
- **Architecture (locked):** NO schema change to `Interview` - all evolving state (current difficulty, used ids, pending question) carried in the `transcript` JSON column; each entry `{"q_id","q","skill_id","a","difficulty","score","next_difficulty"}` with `score: null` marking the pending/asked-unanswered question. `InterviewSession` is reconstructed from the DB row on every request (no server-side state). `role_for_job` + `prefer_skills = job.required_skills` derived per request (v1: probe the job's required skills).
- **Review-driven fixes (caught by review + TDD + live-PG smoke):**
  1. Plan bug: `test_role_for_job` built `Job(...)` without the required `company` field -> added `company="X"`.
  2. Plan bug: `interviews.py` snippet carried a stray `_q_payload`/`q_difficulty` leftover -> implemented the clean version (inline question dicts).
  3. Plan bug: Task 3.1 test had an unused `pool = [...]` line -> would trip ruff F841, removed (kept the `q.skill_id == "python"` assertion).
  4. Voice UI: `TemplateResponse(name, ctx)` (old signature) raised `TypeError: unhashable type: 'dict'` under this Starlette -> new signature `TemplateResponse(request, name, ctx)`.
  5. Engine: redundant `int(round(...))` -> `round(...)` for RUF046 (behavior-identical in Python 3).
- **Verified end-to-end on live Postgres** (throwaway smoke, deleted after): upload -> matches -> start -> 8 answers -> `finished=True`, `score` set, `per_skill` populated, `len(transcript)==8`, re-answer on a finished interview -> 409.
- **Edge-case regression tests added (quality review):** 409 on answering a finished interview, 400 on start with no matches computed, 404 for unknown student/interview.
- **Handoff to Phase 4:** dashboard links to `/interviews/{id}` (HTML, Web Speech voice in/out + always-on text fallback) and can start an interview via `POST /api/students/{sid}/interviews`; per-skill summary available at `GET /api/interviews/{id}` -> `feedback.per_skill`.

### Task 3.1: Question bank
**Files:** Create `data/questions.json`, `app/interview/bank.py`, Test in `tests/test_interview.py`
- [ ] Shape: `{"role": "backend", "skill_id": "python", "difficulty": 1, "question": "...", "keywords": ["list", "dict", "generator"]}` — ~90 questions: 3 roles × (3 difficulty levels × ~8–10 generic + per-skill for the 15 most common required skills). Difficulty: 0 easy, 1 medium, 2 hard.
- [ ] `bank.py`: `select(role, difficulty, exclude_ids, prefer_skills) -> Question` (deterministic: prefer gap skills, then role questions, never repeats within session).
- [ ] Test: selection respects difficulty, excludes used ids, prefers a given skill when present.

### Task 3.2: Adaptive engine
**Files:** Create `app/interview/engine.py`, `app/models.py` (Interview table exists), Test in `tests/test_interview.py`
- [ ] State machine:
  ```python
  class InterviewSession:
      def __init__(self, student_id, job_id, bank, llm, start_difficulty=1): ...
      def next_question(self) -> Question            # None when length reached (8)
      def answer(self, text: str) -> TurnResult      # {score 0..100, feedback_line, next_difficulty}
      def finish(self) -> InterviewSummary           # {score, per_skill: {...}, feedback, transcript}
  ```
  - Evaluation (LLM task `evaluate_answer`): MockLLM scores by keyword overlap (0–70) + length heuristic (+30 max) → 0..100; Bedrock/OpenAI get a rubric prompt returning `{"score": int, "feedback": str}` JSON.
  - Adaptation: `score >= 70 → difficulty = min(2, d+1)`; `score < 40 → max(0, d-1)`; else keep.
  - Summary: score = mean of turn scores; per-skill breakdown by question.skill_id; final feedback via `interview_feedback` task.
- [ ] API (`app/api/routes/interviews.py`): `POST /api/students/{sid}/interviews {job_id?}` → start session, persist `Interview` row, return `{interview_id, first_question}`; `POST /api/interviews/{id}/answers {text}` → turn result or final summary when finished; `GET /api/interviews/{id}`.
- [ ] Tests (MockLLM): easy session of 8 turns; answering with all keywords → difficulty increases (assert on returned `next_difficulty`); finish → score in 0..100, transcript length 8, per-skill keys non-empty.
- [ ] Commit per task.

### Task 3.3: Browser voice UI
**Files:** Create `app/templates/student/interview.html`, `app/static/app.js`, route `GET /interviews/{id}` (HTML)
- [ ] `app.js`: `webkitSpeechRecognition` (continuous=false, interim off) captures user speech → POST answer; `speechSynthesis.speak(new SpeechSynthesisUtterance(q.question))` reads the question; render turn feedback; on finish fetch summary and render score + per-skill bars.
- [ ] Guard: if `SpeechRecognition` unavailable (Firefox) → fall back to text input (input element always present).
- [ ] Manual check (Chrome): run a full 8-question interview voice-in/voice-out; transcript persisted in DB.
- [ ] Commit `feat: adaptive interview engine + voice UI`.
- [ ] **Gate:** suite green; commit `chore: phase 3 complete`.

## 9. Phase 4 — Student career dashboard

> Expand into its own detailed plan before coding.


### Phase 4 STATUS: COMPLETE (2026-11-01)
- Detailed plan: `2026-11-01-ps06-phase4-dashboard.md`. Tasks 4.1/4.2/4.3 executed via subagent-driven development with spec + quality review.
- Final gate: full 78 passed (76 unit + 2 live pgvector integration), 76 passed/2 deselected for `not integration`, ruff clean. 8 tests in `tests/test_dashboard.py`.
- **New HTML router** `app/api/routes/pages.py` (Jinja2 + Tailwind CDN + HTMX CDN + Chart.js CDN; no Node/build): `GET /` (student picker), `POST /upload` (303 -> profile), `GET /students/{sid}` (profile + skill chips + HTMX resume re-upload fragment), `GET /students/{sid}/matches` (explainable cards with quoted resume evidence), `GET /students/{sid}/plan` (checklist + progress bar + narrative), `GET /students/{sid}/progress` (Chart.js readiness line + event table), `GET /students/{sid}/interviews` (hub linking to the Phase 3 `/interviews/{id}` voice UI).
- **Refactor:** extracted `create_student_core` + `create_student_core_resume` from `students.py` so `POST /upload` and the re-upload fragment reuse the exact parse/persist/embed path (API JSON response shape unchanged; all prior API/planner/progress/interview tests pass).
- **Review-driven fixes (caught by review + TDD + live-PG smoke):**
  1. matches band badge: template checked `medium`, but real bands are `strong/moderate/developing/early` -> re-keyed to the real values.
  2. matches route: `JobMatch.breakdown` (`asdict(JobScore)`) stores `matched_required`/`missing_required`/preferred as dicts `{skill,name,category,evidence}`, not id strings -> route maps them to display names + evidence.
  3. **CRITICAL (live-PG smoke only):** `build_plan` added `PlanItem`s before the parent `UpskillPlan` was flushed -> `ForeignKeyViolation` on Postgres (SQLite does not enforce FKs, so it was masked in the unit suite). Fixed with `db.flush()` after `db.add(plan)`.
- **Test-infra hardening:** the SQLite test engine now runs `PRAGMA foreign_keys=ON` (tests/conftest.py), so the fast unit suite enforces referential integrity like production Postgres. All 76 unit tests pass under enforced FKs.
- **Verified end-to-end on live Postgres** (throwaway smoke, deleted after): upload -> profile -> matches -> generate plan -> complete item -> progress chart -> interview hub, all 200.
- **Handoff to Phase 5:** admin analytics reads JobMatch / PlanItem / StudentSkill / Job across the whole cohort; `band` values are `strong/moderate/developing/early`.

### Task 4.1: Template base + profile page
**Files:** Create `app/templates/base.html`, `app/templates/student/profile.html`, routes `GET /` (student picker/login-by-name, no auth in v1) and `GET /students/{sid}`
- [ ] `base.html`: Tailwind via CDN, HTMX via CDN, top nav (Profile / Matches / Plan / Progress / Interview), flash area. No build step, no Node.
- [ ] Profile: name, branch/year/GPA, skill chips grouped by category with level badge, resume re-upload button (HTMX `hx-post` → replace fragment).
- [ ] Tests: `GET /students/{sid}` returns 200 containing the student's name and ≥1 skill chip (TestClient with `Jinja2Templates`).
- [ ] Commit.

### Task 4.2: Matches page (explainable)
- [ ] `GET /students/{sid}/matches` HTML: ranked cards — score dial, band badge, title/company/salary, expandable breakdown: ✅ matched required (with evidence line quoted from resume), ❌ missing required, ➕ preferred. "Generate upskilling plan for this job" button (HTMX → Phase 2 endpoint, redirects to plan page).
- [ ] Tests: 200; contains "missing" section when gaps exist; contains evidence text from the fixture resume.
- [ ] Commit.

### Task 4.3: Plan + Progress pages
- [ ] Plan page: checklist (HTMX toggle → complete endpoint), progress bar (done hours / total), narrative block, per-item course link + hours + provider.
- [ ] Progress page: Chart.js line chart of readiness over time (`GET /api/students/{sid}/progress`), event table.
- [ ] Tests: 200 + key fragments present; HTMX toggle posts succeed (TestClient with `HX-Request: true`).
- [ ] **Gate:** full manual walkthrough with a real resume → matches → plan → complete items → progress chart updates. Commit `chore: phase 4 complete`.

## 10. Phase 5 — Placement-cell analytics dashboard


### Phase 5 STATUS: COMPLETE (2026-11-01)
- Detailed plan: `2026-11-01-ps06-phase5-analytics.md`. Tasks 5.1 (analytics + admin) and 5.2 (`--demo-cohort` CLI) executed via subagent-driven development with spec + quality review.
- Final gate: full 83 passed (81 unit + 2 live pgvector integration), 81 passed/2 deselected for `not integration`, ruff clean. 4 tests in `tests/test_analytics.py` + 1 in `tests/test_cli.py`.
- **Deliverables:** `app/analytics/cohort.py` (pure, Session-in/dicts-out): `skill_gap_report` (demand x under-coverage, sorted desc), `readiness_distribution` (band of mean top-3 match score), `top_requested_courses` (most common plan-item courses), `programme_recommendations` (top gap skills -> best covering course). `app/api/routes/admin.py`: `GET /api/admin/analytics` (JSON) + `GET /admin` (Chart.js: bar = skill gaps, doughnut = readiness bands, tables for courses/recommendations). `seed_demo_cohort` in `seed.py` wired to `python -m app.cli seed --demo-cohort` (10 varied students, matches via `rank_jobs` (embeds on the fly when no vector store), a top-match plan each via MockLLM).
- **Review-driven notes:** gap report sorts by `(-gap_score, skill)` for a stable, deterministic #1; for the 6-student test fixture the deterministic #1 gap skill is `git` (asserted). `band` values are `strong/moderate/developing/early`.
- **Verified on live Postgres** (throwaway smoke, deleted): seeded a 10-student demo cohort -> `/api/admin/analytics` returned non-empty gap report (top3: git 4.2 / python 2.8 / linux 2.4), readiness `{early:7, developing:3}` (sums to 10), populated top courses; `GET /admin` 200 with both charts + tables.
- **Handoff to Phase 6:** all features wired; README + demo script + final QA gates + `v0.1.0` tag remain.

### Task 5.1: Cohort aggregations
**Files:** Create `app/analytics/cohort.py`, `app/api/routes/admin.py`, Test `tests/test_analytics.py`
- [ ] `cohort.py` functions (pure, take `Session` + data, return dicts — fully unit-testable on SQLite with a multi-student fixture):
  ```python
  def skill_gap_report(db, top_k_jobs_per_student=3) -> list[dict]
      # for each skill: demand = # distinct top-matched jobs requiring it;
      # coverage = % of students who have the skill;
      # gap_score = (1 - coverage) * demand  → sorted desc
  def readiness_distribution(db) -> dict[str, int]   # band -> student count (mean of top-3 match scores)
  def top_requested_courses(db, n=5) -> list[dict]   # from plan items across cohort
  def programme_recommendations(db, n=5) -> list[dict]  # top gap skills -> best courses
  ```
- [ ] API: `GET /api/admin/analytics` → JSON of all four; `GET /admin` → `analytics.html` (Chart.js: bar = skill gaps with demand labels, donut = readiness distribution, tables for courses/recommendations).
- [ ] Test fixture: 6 students with deliberately uneven skill sets + seeded matches → assert gap report ranks a known skill #1, distribution sums to 6.
- [ ] Commit per task.
- [ ] **Gate:** seed a realistic 10-student cohort via CLI (`python -m app.cli seed --demo-cohort`), verify `/admin` tells a coherent story. Commit `chore: phase 5 complete`.

## 11. Phase 6 — Polish and handoff


### Phase 6 STATUS: COMPLETE (2026-11-01)
- **README.md** rewritten: problem + feature list, ASCII architecture diagram, quickstart (uv sync → Postgres 17 + pgvector → `seed` → `uvicorn app.api.__main__:app`), `--demo-cohort`, no-DB `demo`, full API reference, provider-switching table + AWS Bedrock setup checklist (IAM user → enable Nova Lite in `ap-south-1` → `.env`), a 3-minute demo walkthrough, tests, and project layout.
- **Production LLM fix (CRITICAL, found in QA):** `app/api/__main__.py::_build_ctx` set `llm=None`, so in production `POST .../plans` returned 503 and interviews crashed (`None.complete`). Now `llm=get_llm(settings)` (MockLLM default; Bedrock/OpenAI via `.env`). Validated by a live smoke that imports the real `app.api.__main__:app` entry and exercises every route (all 200) with plans + an 8-turn interview using the wired LLM. (The run command is `app.api.__main__:app`, not `app.api.main:app` - the latter is the factory and has no module-level `app`.)
- **Final whole-project review** (Antigravity/Gemini, session `ps06`): dispatched against the live tree. Invariant checks passed (session-per-op, planner consumes `JobScore` from `rank_jobs`, `datetime.now(timezone.utc)` only, job vectors embed `job.description`, no secrets tracked, tests assert real behavior). **5 IMPORTANT runtime-crash risks found and fixed** (commit `0baf894`): (1) `start_interview` 404 when the top-match job is missing (parity with `plans.py`; precondition is FK-blocked on `JobMatch.job_id` so it is defensive + untestable via API); (2) `ctx.llm is None` → 503 guards in both interview routes; (3) engine `answer()` guards `get_question(...) is None`; (4) `answer_interview` catches the engine `ValueError` → clean 400 (not 500) and finishes gracefully when the question bank is exhausted; (5) `create_student_core_resume` now deletes prior skills itself (idempotent, self-contained) so re-upload can't hit a PK violation. +4 regression tests.
- **Final QA (post-review):** `uv run pytest -m "not integration" -q` = 85 passed; `uv run pytest -q` (incl. live PG/pgvector integration) = 87 passed; `uv run ruff check .` clean. Live production smoke (real `app.api.__main__:app` entry, Postgres, `get_llm` wired): `/`, `/admin`, `/api/admin/analytics`, student profile/matches/idempotent re-upload/plan/progress/interview hub, `/interviews/{id}` + 8 answers, plan generation, and item completion all returned 200.
- **Note (finding #6, by design not a bug):** the reviewer flagged request-scoped `Depends(get_session)` in routes. This is the sanctioned pattern (the plan explicitly allows request-scoped DI for HTTP handlers); the invariant "never store a Session on AppContext" is met — `AppContext` holds an `engine`, and the vector store is session-per-op. No refactor made.
- **Tag:** `v0.1.0` (points at the post-review commit).

### Tasks (done)
- [x] **README.md**: problem, architecture diagram (ASCII), quickstart (the Phase 0 commands), AWS Bedrock setup checklist (create account → IAM user → enable model access in ap-south-1 → `.env`), how to switch providers, demo script.
- [x] **Demo script (3 min):** upload sample resume → match explanation with evidence → generate plan → complete an item → voice mock interview → readiness line moves → `/admin` cohort story.
- [x] **Final QA:** `uv run pytest -m "not integration" -q` green; `uv run pytest -q` green; live route smoke (real `app.api.__main__:app` entry) all routes 200.
- [x] Tag `v0.1.0`.

## 12. Testing strategy (summary)

| Layer | Tool | Deps |
|---|---|---|
| Unit (taxonomy/extract/match/embed/plan/interview/analytics) | pytest + SQLite + HashingEmbedder + MemoryVectorStore + MockLLM | none |
| API | FastAPI TestClient, same in-memory stack | none |
| Integration (pgvector, seed embeddings) | `@pytest.mark.integration`, local Postgres `ps06_test` | Phase 0 |
| Bedrock/OpenAI | boto3/SDK mocked with monkeypatch (no network in tests) | none |
| Manual | CLI demo, browser walkthrough, voice interview | Chrome |

Gate for every phase: `uv run pytest -m "not integration" -q` fully green + commit.

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Bedrock free-tier limits/eligibility change by build time | LLM layer is provider-agnostic; `mock` default means zero dependency; re-check pricing page at Phase 2 start; OpenAI fallback documented |
| Model not available in ap-south-1 / model-access approval delay | `LLM_MODEL` is config-only; Titan models are widely available; document us-east-1 fallback |
| sentence-transformers model download (~90 MB) | Warm cache in Task 0.4; `HashingEmbedder` fallback keeps everything working offline |
| pgvector Windows binary version mismatch | Pin pgvector release matching PG major 17; integration test proves `CREATE EXTENSION` + `<=>` works before Phase 1.7b |
| Python 3.15 wheel gaps | Project pinned to 3.12 via uv managed CPython |
| Resume formats vary (scanned PDFs) | Out of scope for v1; parser returns best-effort text; document that text-layer PDFs are supported |
| Web Speech API missing in some browsers | Text input fallback is always rendered (Task 3.3) |

## 14. Self-review (spec coverage)

| Spec deliverable | Covered by |
|---|---|
| Student career dashboard | Phase 4 (Tasks 4.1–4.3) |
| Job-matching agent | Phase 1 (engine + API) + ReAct-style tool use in Phase 2 planner/Phase 3 engine (`think→act→observe` loop over tool functions: `score_job`, `select_question`, `evaluate_answer`) |
| Explainable skill-gap analyser | `score_job` breakdown + evidence lines (Task 1.5) + matches page (4.2) |
| Personalised upskilling planner | Phase 2 (2.2) |
| Adaptive mock interview module | Phase 3 (3.1–3.3) |
| Progress tracker | Phase 2 (2.3) + progress page (4.3) |
| Placement-cell analytics dashboard | Phase 5 |
| Expected outcome (targeted prep, measurable progress, cohort insights) | readiness series + `/admin` skill-gap report |

No unresolved placeholders in Phase 1; Phases 2–5 are interface-locked and will be expanded to the same TDD granularity at phase start.
