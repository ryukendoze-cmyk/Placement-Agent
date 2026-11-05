# PS-06 Phase 5 - Placement-Cell Analytics Dashboard - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Add cohort-level analytics (skill gaps, readiness distribution, most-requested courses, programme recommendations) as pure, unit-testable functions, expose them at `GET /api/admin/analytics` (JSON) and `GET /admin` (Chart.js page), and add a `--demo-cohort` CLI seed so `/admin` tells a coherent story out of the box.

**Architecture:** `app/analytics/cohort.py` holds four PURE functions that take a SQLModel `Session` (and knobs) and return JSON-safe `dict`/`list`/`dict[str,int]` - no FastAPI. They read `Student`, `StudentSkill`, `JobMatch` (breakdown + score), `Job.required_skills`, `PlanItem`, `Course`, `Skill`. `app/api/routes/admin.py` wraps them: JSON endpoint + an `analytics.html` page (Chart.js CDN: bar = skill gaps, donut = readiness bands, two tables). A `seed_demo_cohort` in `seed.py` (wired to `python -m app.cli seed --demo-cohort`) creates 10 varied students, runs their matches (`rank_jobs` embeds job descriptions on the fly when no vector store is present), and builds a top-match plan each (MockLLM narrative), so the admin page is populated.

**Conventions (inherited):** `datetime.now(timezone.utc)`; no Session on AppContext; ruff green (I001/F401/F841; FastAPI Depends/File/Form/Query/Path exempt); unit tests use SQLite (FKs enforced via conftest) + HashingEmbedder + MemoryVectorStore + MockLLM; `band` values are `strong/moderate/developing/early`. Gate per task: `uv run pytest -m "not integration" -q` + `uv run ruff check .` + commit. New Starlette signature `TemplateResponse(request, name, ctx)`.

---

## Task 5.1: Cohort analytics + admin API + admin page

**Files:**
- Create: `app/analytics/__init__.py` (empty), `app/analytics/cohort.py`
- Create: `app/api/routes/admin.py`
- Create: `app/templates/admin/analytics.html`
- Modify: `app/api/main.py` (include `admin.router`)
- Test: `tests/test_analytics.py`

- [x] **Step 1: Create `app/analytics/cohort.py`** with EXACTLY this content:
  ```python
  """Cohort-level placement analytics. Pure functions over a SQLModel Session.

  Each function takes a Session (and optional knobs) and returns JSON-safe
  dict / list / dict[str,int] results, so they are fully unit-testable on
  SQLite and directly serializable for the admin API.
  """
  from collections import Counter, defaultdict

  from sqlmodel import Session, select

  from ..matching.engine import band_for
  from ..models import Course, Job, JobMatch, PlanItem, Skill, Student, StudentSkill
  from ..plan.planner import _load_courses, pick_course_for_skill


  def _skill_names(db: Session) -> dict[str, str]:
      return {s.id: s.name for s in db.exec(select(Skill)).all()}


  def skill_gap_report(db: Session, top_k_jobs_per_student: int = 3) -> list[dict]:
      """For each skill demanded by top-matched jobs: demand (distinct such jobs),
      coverage (fraction of students who have the skill), and
      gap_score = (1 - coverage) * demand, sorted by gap_score desc (largest
      placement gaps first). Only skills demanded by at least one top job appear.
      """
      students = db.exec(select(Student)).all()
      total = len(students)
      if total == 0:
          return []
      top_job_ids: set[str] = set()
      have_skill: dict[str, set[str]] = defaultdict(set)
      for s in students:
          top = db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                        .order_by(JobMatch.score.desc()).limit(top_k_jobs_per_student)).all()
          for m in top:
              top_job_ids.add(m.job_id)
          for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == s.id)).all():
              have_skill[ss.skill_id].add(s.id)
      demand: dict[str, int] = defaultdict(int)
      for jid in top_job_ids:
          job = db.get(Job, jid)
          if job is None:
              continue
          for sk in set(job.required_skills):
              demand[sk] += 1
      names = _skill_names(db)
      report = []
      for sk, d in demand.items():
          coverage = len(have_skill.get(sk, ())) / total
          gap = round((1 - coverage) * d, 3)
          report.append({"skill": sk, "name": names.get(sk, sk), "demand": d,
                         "coverage": round(coverage, 3), "gap_score": gap})
      report.sort(key=lambda r: (-r["gap_score"], r["skill"]))
      return report


  def readiness_distribution(db: Session) -> dict[str, int]:
      """Band of each student's mean top-3 match score -> student count."""
      dist: dict[str, int] = {}
      for s in db.exec(select(Student)).all():
          scores = [m.score for m in db.exec(select(JobMatch).where(JobMatch.student_id == s.id)
                     .order_by(JobMatch.score.desc()).limit(3)).all()]
          if not scores:
              continue
          band = band_for(round(sum(scores) / len(scores)))
          dist[band] = dist.get(band, 0) + 1
      return dist


  def top_requested_courses(db: Session, n: int = 5) -> list[dict]:
      """Most common courses across all plan items in the cohort."""
      counts = Counter(it.course_id for it in db.exec(select(PlanItem)).all())
      courses = {c.id: c for c in db.exec(select(Course)).all()}
      out = []
      for cid, cnt in counts.most_common(n):
          c = courses.get(cid)
          if c is not None:
              out.append({"course_id": c.id, "title": c.title, "provider": c.provider, "count": cnt})
      return out


  def programme_recommendations(db: Session, n: int = 5) -> list[dict]:
      """Top gap skills -> a recommended (best covering) course for each."""
      out = []
      courses = _load_courses(db)
      for r in skill_gap_report(db):
          if len(out) >= n:
              break
          course = pick_course_for_skill(r["skill"], courses)
          if course is None:
              continue
          out.append({"skill": r["skill"], "name": r["name"], "gap_score": r["gap_score"],
                      "course_id": course["id"], "course_title": course["title"],
                      "provider": course["provider"]})
      return out
  ```
  Create `app/analytics/__init__.py` (empty).

- [x] **Step 2: Create `app/api/routes/admin.py`:**
  ```python
  from fastapi import APIRouter, Depends, Request
  from sqlmodel import Session

  from app.api.deps import get_session
  from app.analytics.cohort import (programme_recommendations, readiness_distribution,
                                    skill_gap_report, top_requested_courses)

  router = APIRouter()


  def _collect(session: Session) -> dict:
      return {
          "skill_gap_report": skill_gap_report(session),
          "readiness_distribution": readiness_distribution(session),
          "top_requested_courses": top_requested_courses(session),
          "programme_recommendations": programme_recommendations(session),
      }


  @router.get("/api/admin/analytics")
  def admin_analytics(session: Session = Depends(get_session)):
      return _collect(session)


  @router.get("/admin")
  def admin_page(request: Request, session: Session = Depends(get_session)):
      data = _collect(session)
      return request.app.state.templates.TemplateResponse(
          request, "admin/analytics.html", {**data, "analytics": data, "error": ""})
  ```

- [x] **Step 3: Include router in `app/api/main.py`** - add `admin` to the `from app.api.routes import ...` line and `app.include_router(admin.router)` (anywhere among the includes; it has unique paths `/admin` and `/api/admin/analytics`).

- [x] **Step 4: Create `app/templates/admin/analytics.html`** (extends base; context: `skill_gap_report`, `readiness_distribution`, `top_requested_courses`, `programme_recommendations`, `analytics`, `error`). Requirements:
  - Heading containing "analytics" (e.g. `<h1>...Placement-cell analytics</h1>`).
  - `<canvas id="gapChart" ...></canvas>` under a heading containing "Skill gaps".
  - `<canvas id="bandChart" ...></canvas>` under a heading containing "Readiness".
  - A "Most requested courses" `<table>` iterating `top_requested_courses` (columns Course/Provider/Requests), with an empty row when the list is empty.
  - A "Programme recommendations" `<table>` iterating `programme_recommendations` (columns Skill/Recommended course/Provider), with an empty row when empty.
  - Chart.js CDN `<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>` and a `<script>` that reads `const DATA = {{ analytics|tojson }};` and draws:
    - a **bar** chart into `#gapChart` from `DATA.skill_gap_report` (top 15: labels = `name`, values = `gap_score`), and
    - a **doughnut** chart into `#bandChart` from `DATA.readiness_distribution` (labels = band keys, values = counts).
  - Guard the chart drawing so an empty dataset doesn't throw (Chart.js handles empty arrays, but wrap in `if (DATA && DATA.skill_gap_report)`).

- [x] **Step 5: Create `tests/test_analytics.py`:**
  ```python
  from sqlmodel import Session

  from app.analytics.cohort import (programme_recommendations, readiness_distribution,
                                    skill_gap_report, top_requested_courses)

  COHORT = [
      ("Asha", "Skills: Python, FastAPI, Docker"),
      ("Rohan", "Skills: Python, Git, Linux"),
      ("Meera", "Skills: JavaScript, React, HTML"),
      ("Arjun", "Skills: pandas, statistics, SQL"),
      ("Karan", "Skills: Python, FastAPI, Docker"),
      ("Ananya", "Skills: HTML, CSS, JavaScript"),
  ]


  def _cohort(client):
      sids = []
      for name, body in COHORT:
          sid = client.post("/api/students", files={"file": ("r.txt", body.encode("utf-8"), "text/plain")},
                            data={"name": name}).json()["id"]
          client.get(f"/api/students/{sid}/matches")
          sids.append(sid)
      return sids


  def test_skill_gap_report_shape_and_ordering(client_engine):
      client, eng = client_engine
      _cohort(client)
      with Session(eng) as db:
          report = skill_gap_report(db)
      assert report, "expected a non-empty gap report"
      assert report[0]["demand"] > 0
      assert 0 <= report[0]["coverage"] < 1.0
      scores = [r["gap_score"] for r in report]
      assert scores == sorted(scores, reverse=True)
      # document the deterministic #1 skill for this fixture + seeded jobs
      top = report[0]
      assert top["skill"] in {r["skill"] for r in report}


  def test_readiness_distribution_sums_to_cohort(client_engine):
      client, eng = client_engine
      _cohort(client)
      with Session(eng) as db:
          dist = readiness_distribution(db)
      assert sum(dist.values()) == 6


  def test_courses_and_recommendations_after_plans(client_engine):
      client, eng = client_engine
      sids = _cohort(client)
      for sid in sids:
          assert client.post(f"/api/students/{sid}/plans", json={}).status_code in (200, 201)
      with Session(eng) as db:
          courses = top_requested_courses(db)
          recs = programme_recommendations(db)
      assert isinstance(courses, list)
      assert len(recs) >= 1
      assert all("course_title" in r for r in recs)


  def test_admin_analytics_json_and_page(client_engine):
      client, _ = client_engine
      _cohort(client)
      r = client.get("/api/admin/analytics")
      assert r.status_code == 200
      b = r.json()
      for key in ("skill_gap_report", "readiness_distribution",
                  "top_requested_courses", "programme_recommendations"):
          assert key in b
      h = client.get("/admin")
      assert h.status_code == 200
      assert "analytics" in h.text.lower()
      assert "readiness" in h.text.lower()
  ```
  NOTE: In `test_skill_gap_report_shape_and_ordering`, RUN the test once, print `report[0]["skill"]`, and strengthen the last assertion to the specific deterministic skill it is (e.g. `assert top["skill"] == "sql"` or whatever the run shows) with a comment. If the top skill is not stable/meaningful, keep the structural assertions and drop the specific-value assert.

- [x] **Step 6: Gate.** `uv run pytest tests/test_analytics.py -v` (4 pass) + `uv run pytest -m "not integration" -q` (expected **80 passed**: 76 + 4) + `uv run ruff check .` clean.

- [x] **Step 7: Commit** `feat: cohort analytics + admin dashboard (skill gaps, readiness, courses, recommendations)`

---

## Task 5.2: `--demo-cohort` CLI seeder

**Files:** Modify `app/seed.py` (add `seed_demo_cohort`), Modify `app/cli.py` (flag + wiring), Test in `tests/test_cli.py`.

- [x] **Step 1: Add to `app/seed.py`** (add imports at top: `from dataclasses import asdict`, `from uuid import uuid4`, and `from .matching.rank import rank_jobs`, `from .models import Job, JobMatch, Student, StudentSkill` (merge with existing model import), `from .plan.planner import build_plan`, `from .skills.extract import extract_skills, level_for`, `from .llm.mock import MockLLM` - add only what's not already imported):
  ```python
  _DEMO_COHORT = [
      ("Asha Patel", "CSE", 3, 8.2, "Skills: Python, FastAPI, PostgreSQL, Docker\nBackend internship building REST APIs."),
      ("Rohan Gupta", "CSE", 3, 7.8, "Skills: Python, SQL, Git, Linux\nData engineering internship."),
      ("Meera Iyer", "ECE", 3, 8.5, "Skills: JavaScript, React, HTML, CSS, TypeScript\nFrontend web project."),
      ("Arjun Singh", "IT", 4, 7.5, "Skills: Python, pandas, statistics, SQL\nAnalytics and dashboards."),
      ("Diya Nair", "CSE", 4, 8.0, "Skills: Docker, Kubernetes, AWS, CI/CD, Linux\nDevOps club."),
      ("Karan Mehta", "CSE", 3, 7.9, "Skills: Python, FastAPI, Docker, Git\nFull-stack app."),
      ("Ananya Rao", "ECE", 3, 8.1, "Skills: HTML, CSS, JavaScript, React\nResponsive web dev."),
      ("Vikram Shah", "IT", 4, 7.7, "Skills: Python, SQL, deep_learning, pandas\nML capstone."),
      ("Sneha Kulkarni", "CSE", 4, 8.3, "Skills: Python, Git, AWS, Docker\nCloud projects."),
      ("Aditya Verma", "CSE", 3, 7.6, "Skills: React, TypeScript, CSS, HTML, SQL\nMERN stack."),
  ]


  def seed_demo_cohort(db, *, taxonomy, embedder, vector_store=None, llm=None) -> int:
      """Create a varied 10-student cohort, run each student's job matches, and
      build a top-match upskilling plan per student. Returns the count created.
      `rank_jobs` embeds job descriptions on the fly when `vector_store` is None
      (sqlite / no pgvector), so this works without a vector store.
      """
      llm = llm or MockLLM()
      jobs = db.exec(select(Job)).all()
      created = 0
      for name, branch, year, gpa, text in _DEMO_COHORT:
          sid = f"stu-{uuid4().hex[:8]}"
          db.add(Student(id=sid, name=name, branch=branch, year=year, gpa=gpa))
          found = extract_skills(text, taxonomy)
          for skill_id, lines in found.items():
              db.add(StudentSkill(student_id=sid, skill_id=skill_id, mentions=len(lines),
                                  level=level_for(len(lines), lines), evidence=list(lines)))
          db.commit()
          student_vec = embedder.embed(text)
          if vector_store is not None:
              vector_store.upsert("student", sid, student_vec)
          student_skills = {ss.skill_id: list(ss.evidence) for ss in
                            db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
          ranked = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                             vector_store=vector_store, top=10)
          for job, js in ranked:
              db.add(JobMatch(id=f"m-{uuid4().hex[:8]}", student_id=sid, job_id=job.id,
                              score=js.score, band=js.band, breakdown=asdict(js)))
          db.commit()
          if ranked:
              try:
                  build_plan(db, sid, ranked[0][0], ranked[0][1], llm=llm)
              except Exception:
                  db.rollback()
          created += 1
      return created
  ```
  (Ensure `select` and the model names are imported in seed.py - it already imports `select` and several models; add `JobMatch`/`StudentSkill`/`Student` if not present.)

- [x] **Step 2: Wire the CLI flag in `app/cli.py`:**
  - Import: `from .seed import _json, seed, seed_demo_cohort`.
  - Change `def cmd_seed() -> int:` to `def cmd_seed(demo_cohort: bool = False) -> int:`.
  - Inside the `with Session(engine) as db:` block: capture `taxonomy = load_taxonomy()` once (use it for both `seed(...)` and the cohort), and after the seed + counts add:
    ```python
    n_cohort = 0
    if demo_cohort:
        n_cohort = seed_demo_cohort(db, taxonomy=taxonomy, embedder=embedder, vector_store=store)
    ```
    and add `if demo_cohort: print(f"demo cohort: {n_cohort} students")` near the other prints.
  - In `main`, for the `seed` subparser add `s = sub.add_parser("seed", ...)` then `s.add_argument("--demo-cohort", action="store_true", help="also seed a 10-student demo cohort with matches + plans")`, and change the dispatch to `return cmd_seed(demo_cohort=args.demo_cohort)`.

- [x] **Step 3: Add a CLI test** to `tests/test_cli.py`:
  ```python
  def test_seed_demo_cohort_creates_cohort(capsys, monkeypatch):
      from app.db import make_engine
      from app.embeddings.embedder import HashingEmbedder
      from sqlmodel import Session, select

      from app.models import JobMatch, Student, UpskillPlan

      eng = make_engine("sqlite://")
      monkeypatch.setattr(cli, "_make_engine", lambda url: eng)
      monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
      monkeypatch.setattr(cli.settings, "database_url", "sqlite://")
      rc = cli.main(["seed", "--demo-cohort"])
      assert rc == 0
      out = capsys.readouterr().out
      assert "demo cohort" in out
      with Session(eng) as db:
          assert len(db.exec(select(Student)).all()) >= 10
          assert len(db.exec(select(JobMatch)).all()) > 0
          assert len(db.exec(select(UpskillPlan)).all()) >= 1
  ```

- [x] **Step 4: Gate.** `uv run pytest tests/test_cli.py -v` (6 pass: 5 prior + 1 new) + `uv run pytest -m "not integration" -q` (expected **81 passed**) + `uv run ruff check .` clean.

- [x] **Step 5: Commit** `feat: --demo-cohort CLI seeder (10 students, matches, plans)`

- [x] **Step 6: Phase 5 gate.**
  - `uv run pytest -m "not integration" -q` green; `uv run pytest -q` (full, live PG) green; `uv run ruff check .` clean.
  - (Manual, optional) `uv run python -m app.cli seed --demo-cohort` against the dev Postgres, then open `/admin`.
  - Update master plan `2026-10-31-ps06-placement-agent.md`: add `### Phase 5 STATUS: COMPLETE` after `## 10. Phase 5`, and mark all `- [ ]` boxes `[x]` in this file.
  - Commit `chore: phase 5 complete`.

---

## Notes for the executing agent
- `band_for` lives in `app/matching/engine.py` (values `strong/moderate/developing/early`).
- `pick_course_for_skill` and `_load_courses` live in `app/plan/planner.py` (import them; do NOT reimplement).
- The admin page + analytics JSON read the SAME four functions; keep the JSON keys exactly `skill_gap_report`, `readiness_distribution`, `top_requested_courses`, `programme_recommendations`.
- Do NOT add a new table. All analytics derive from existing rows.
- The `_DEMO_COHORT` resumes must use taxonomy skill names that `extract_skills` resolves (Python, FastAPI, Docker, SQL, Git, Linux, JavaScript, React, HTML, CSS, TypeScript, pandas, statistics, AWS, CI/CD, Kubernetes, deep_learning are all valid).
- Keep the analytics functions free of FastAPI imports (pure data layer).
