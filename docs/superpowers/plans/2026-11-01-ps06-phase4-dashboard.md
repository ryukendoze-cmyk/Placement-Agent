# PS-06 Phase 4 - Student Career Dashboard - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Add a browser career dashboard (server-rendered Jinja2 + Tailwind CDN + HTMX CDN, Chart.js for the progress line) with a student picker, profile (skill chips + resume re-upload), explainable matches, an upskilling plan checklist, and a readiness progress chart. Links out to the Phase 3 voice interview.

**Architecture:** New HTML router `app/api/routes/pages.py` (prefix-less, e.g. `GET /`, `GET /students/{sid}`) renders templates from `app/templates/` via `request.app.state.templates` (already wired in Phase 3's `create_app`). It reads the SAME data the Phase 1-3 JSON APIs expose (Student, StudentSkill, JobMatch breakdown, UpskillPlan/PlanItem/Course, ProgressEvent, Interview) - no new tables. HTMX handles the plan-item toggle and resume re-upload (fragment swap). **No JSON API changes** except one low-risk extraction of `create_student_core` from `students.py` so the HTML `POST /upload` can reuse the exact parse/persist/embed path and then 303-redirect.

**Conventions (inherited):** `datetime.now(timezone.utc)`; never store a Session on AppContext; ruff green (I001/F401/F841/B008/RUF046/E711/E712; FastAPI Depends/File/Form exempt); unit tests use the `client`/`client_engine` fixtures (SQLite + HashingEmbedder + MemoryVectorStore + MockLLM); gate per task = `uv run pytest -m "not integration" -q` + `uv run ruff check .` + commit. Templates are full HTML documents extending `base.html` (Jinja `{% extends %}`).

---

## Task 4.1: base layout + student picker + profile (+ resume re-upload)

**Files:**
- Create: `app/api/routes/pages.py`
- Create: `app/templates/base.html`, `app/templates/index.html`
- Create: `app/templates/student/profile.html`, `app/templates/student/_skill_chips.html`
- Modify: `app/api/main.py` (include `pages.router` FIRST, so `GET /` is reachable before any `/api/...`)
- Modify: `app/api/routes/students.py` (extract `create_student_core`)
- Test: `tests/test_dashboard.py`

- [x] **Step 1: Extract `create_student_core` in `app/api/routes/students.py`.**
  Move the body of `create_student` (from `suffix = ...` through `skills = [_skill_payload(r, ctx) for r in rows]`) into a module function:
  ```python
  def create_student_core(session, ctx, file, name, branch, year, gpa):
      suffix = Path(file.filename or "").suffix.lower()
      if suffix not in SUPPORTED_SUFFIXES:
          raise HTTPException(400, f"unsupported resume format: {suffix or '(none)'}")
      try:
          text = parse_resume_bytes(_read_capped(file), suffix)
      except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:
          logger.exception("resume parse failed for %s", file.filename)
          raise HTTPException(400, f"could not parse resume: {e}") from e
      sid = f"stu-{uuid4().hex[:8]}"
      found = extract_skills(text, ctx.taxonomy)
      session.add(Student(id=sid, name=name, branch=branch, year=year, gpa=gpa))
      for skill_id, lines in found.items():
          session.add(StudentSkill(student_id=sid, skill_id=skill_id,
                                   mentions=len(lines), level=level_for(len(lines), lines),
                                   evidence=list(lines)))
      session.commit()
      ctx.vector_store.upsert("student", sid, ctx.embedder.embed(text))
      rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
      return sid, [_skill_payload(r, ctx) for r in rows]
  ```
  Rewrite `create_student` to call it and keep its EXACT JSON response shape:
  ```python
  @router.post("/api/students", status_code=201)
  def create_student(file: UploadFile = File(...), name: str = Form(...), branch: str = Form(""),
                     year: int = Form(3), gpa: float = Form(0.0),
                     session: Session = Depends(get_session), ctx=Depends(get_ctx)):
      sid, skills = create_student_core(session, ctx, file, name, branch, year, gpa)
      return {"id": sid, "name": name, "skills": skills, "skill_count": len(skills)}
  ```
  Do NOT change any other students.py behavior. The existing `test_api.py` upload tests MUST still pass unchanged.

- [x] **Step 2: Create `app/templates/base.html`** - a full HTML doc with `{% block content %}{% endblock %}`. Must include:
  - Tailwind CDN `<script src="https://cdn.tailwindcss.com"></script>`.
  - HTMX CDN `<script src="https://unpkg.com/htmx.org@1.9.12"></script>`.
  - A top nav with links, each carrying `?sid=` context or using the current sid. The nav needs the active student id, so pass `active_sid` (may be empty) into base via the template context. Nav links (use plain anchors): Profile `/students/{{ sid }}`, Matches `/students/{{ sid }}/matches`, Plan `/students/{{ sid }}/plan`, Progress `/students/{{ sid }}/progress`, Interview `/students/{{ sid }}/interviews`, and a "Home" link `/`. When `sid` is empty (index page), render the Home link only / disable the rest (simplest: hide the per-student nav when `not sid`).
  - A flash/error area: `{% block flash %}{% endblock %}` showing `{{ error or '' }}` if provided.

- [x] **Step 3: Create `app/templates/index.html`** (extends base, no sid):
  - A card listing existing students: for each `s` in `students` → `<a href="/students/{{ s.id }}">{{ s.name }}</a>` (show branch/year if present).
  - A "Add a student" `<form method="post" action="/upload" enctype="multipart/form-data">` with fields: `file` (type=file), `name` (required), `branch`, `year` (number, default 3), `gpa` (number, step 0.01, default 0).
  - The list must contain the student's `name` (tests assert this).

- [x] **Step 4: Create `app/templates/student/profile.html`** (extends base, `sid` set):
  - Heading with `{{ student.name }}`; meta line branch/year/GPA.
  - A re-upload form: `<form hx-post="/students/{{ sid }}/resume" hx-target="#skill-chips" hx-swap="innerHTML" enctype="multipart/form-data">` with a single `file` input (required) + a submit button. (Identity fields kept from the existing student; only skills are refreshed.)
  - A `{% include "student/_skill_chips.html" %}` inside `<div id="skill-chips">`.

- [x] **Step 5: Create `app/templates/student/_skill_chips.html`** (the swappable fragment, NOT extends base):
  - `{% for cat, items in skills_by_category.items() %}` a category label + chips. Each chip: `{{ it.name }}` plus a level badge (e.g. `<span class="badge">{{ it.level }}</span>`). Render `{{ skills_by_category|length }}` category groups. If empty, render a "No skills detected" line.
  - This fragment receives `skills_by_category` (dict: category -> list of {id,name,level,mentions,evidence}).

- [x] **Step 6: Create `app/api/routes/pages.py`:**
  ```python
  from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
  from fastapi.responses import RedirectResponse
  from sqlalchemy import delete
  from sqlmodel import Session, select

  from app.api.deps import get_ctx, get_session
  from app.api.routes.students import create_student_core, _skill_payload
  from app.models import Interview, Job, JobMatch, PlanItem, Student, StudentSkill, UpskillPlan, Course

  router = APIRouter()


  def _tpl(request: Request, name: str, **ctx):
      templates = request.app.state.templates
      return templates.TemplateResponse(request, name, ctx)


  @router.get("/")
  def index(request: Request, session: Session = Depends(get_session)):
      students = session.exec(select(Student).order_by(Student.created_at.desc())).all()
      return _tpl(request, "index.html", students=students, sid="", error="")


  @router.post("/upload")
  def upload(file: UploadFile = File(...), name: str = Form(...), branch: str = Form(""),
             year: int = Form(3), gpa: float = Form(0.0),
             session: Session = Depends(get_session), ctx=Depends(get_ctx)):
      sid, _skills = create_student_core(session, ctx, file, name, branch, year, gpa)
      return RedirectResponse(url=f"/students/{sid}", status_code=303)


  @router.get("/students/{sid}")
  def profile(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
      student = session.get(Student, sid)
      if student is None:
          raise HTTPException(404, "student not found")
      skills_by_category = _group_skills(session, ctx, sid)
      return _tpl(request, "student/profile.html", student=student, sid=sid,
                  skills_by_category=skills_by_category, error="")


  @router.post("/students/{sid}/resume")
  def reupload(sid: str, request: Request, file: UploadFile = File(...),
               session: Session = Depends(get_session), ctx=Depends(get_ctx)):
      student = session.get(Student, sid)
      if student is None:
          raise HTTPException(404, "student not found")
      session.execute(delete(StudentSkill).where(StudentSkill.student_id == sid))
      session.commit()
      _sid, skills = create_student_core_resume(session, ctx, sid, file)
      return _tpl(request, "student/_skill_chips.html", skills_by_category=_group_skills(session, ctx, sid))


  def _group_skills(session, ctx, sid):
      rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
      grouped: dict[str, list] = {}
      for ss in rows:
          p = _skill_payload(ss, ctx)
          grouped.setdefault(p["category"], []).append(p)
      for items in grouped.values():
          items.sort(key=lambda x: x["name"])
      return grouped
  ```
  NOTE: `create_student_core_resume(session, ctx, existing_sid, file)` is a second small helper in `students.py` that re-parses a file and (re)populates StudentSkill rows for an EXISTING sid + re-embeds the vector, returning `(sid, skills)`. Define it right after `create_student_core`:
  ```python
  def create_student_core_resume(session, ctx, sid, file):
      suffix = Path(file.filename or "").suffix.lower()
      if suffix not in SUPPORTED_SUFFIXES:
          raise HTTPException(400, f"unsupported resume format: {suffix or '(none)'}")
      try:
          text = parse_resume_bytes(_read_capped(file), suffix)
      except (ValueError, zipfile.BadZipFile, OSError, RuntimeError) as e:
          logger.exception("resume reparse failed for %s", file.filename)
          raise HTTPException(400, f"could not parse resume: {e}") from e
      found = extract_skills(text, ctx.taxonomy)
      for skill_id, lines in found.items():
          session.add(StudentSkill(student_id=sid, skill_id=skill_id,
                                   mentions=len(lines), level=level_for(len(lines), lines),
                                   evidence=list(lines)))
      session.commit()
      ctx.vector_store.upsert("student", sid, ctx.embedder.embed(text))
      rows = session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
      return sid, [_skill_payload(r, ctx) for r in rows]
  ```
  (Remove the unused `Interview/Job/JobMatch/PlanItem/UpskillPlan/Course` imports from pages.py in this task if not yet used - keep only what's imported to avoid F401; add them back in later tasks.)

- [x] **Step 7: Include router in `app/api/main.py`** - add `pages` to the `from app.api.routes import ...` line and `app.include_router(pages.router)` as the FIRST include (before students.router) so `GET /` resolves.

- [x] **Step 8: Write tests** in `tests/test_dashboard.py`:
  ```python
  def test_index_lists_students(client_engine):
      client, _ = client_engine
      sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                        data={"name": "Asha"}).json()["id"]
      h = client.get("/")
      assert h.status_code == 200
      assert "Asha" in h.text
      assert "upload" in h.text.lower()

  def test_profile_renders_name_and_skill_chip(client_engine):
      client, _ = client_engine
      sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, Docker\n", "text/plain")},
                        data={"name": "Bharat", "branch": "CSE", "year": 3, "gpa": 8.1}).json()["id"]
      h = client.get(f"/students/{sid}")
      assert h.status_code == 200
      assert "Bharat" in h.text
      assert "Python" in h.text          # >=1 skill chip
      assert "skill-chips" in h.text

  def test_resume_reupload_refreshes_skill_chips(client_engine):
      client, _ = client_engine
      sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                        data={"name": "Swap"}).json()["id"]
      # re-upload a resume that adds a new skill via the HTMX fragment endpoint
      r = client.post(f"/students/{sid}/resume", files={"file": ("r.txt", b"Skills: Python, Kubernetes\n", "text/plain")},
                      headers={"HX-Request": "true"})
      assert r.status_code == 200
      assert "Kubernetes" in r.text or "kubernetes" in r.text.lower()
  ```
  (If "Kubernetes" is not in the taxonomy, pick a skill that IS, e.g. "React" - verify against `data/skills.json` before finalizing; the assertion must use a real taxonomy skill id/name.)

- [x] **Step 9: Run gate.** `uv run pytest tests/test_dashboard.py -v` (3 pass) + `uv run pytest -m "not integration" -q` (expected 71: 68 + 3) + `uv run ruff check .` clean.

- [x] **Step 10: Commit** `feat: dashboard base layout, student picker, profile with resume re-upload`

---

## Task 4.2: Explainable matches page

**Files:** Create `app/templates/student/matches.html`; Modify `app/api/routes/pages.py` (add `GET /students/{sid}/matches`); Test in `tests/test_dashboard.py`.

- [x] **Step 1: Add route** to `pages.py`:
  ```python
  @router.get("/students/{sid}/matches")
  def matches_page(sid: str, request: Request, session: Session = Depends(get_session), ctx=Depends(get_ctx)):
      if session.get(Student, sid) is None:
          raise HTTPException(404, "student not found")
      evidence: dict[str, str] = {}
      for ss in session.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all():
          if ss.evidence:
              evidence[ss.skill_id] = ss.evidence[0]
      rows = session.exec(select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())).all()
      cards = []
      for m in rows:
          job = session.get(Job, m.job_id)
          if job is None:
              continue
          b = m.breakdown or {}
          cards.append({
              "job_id": job.id, "title": job.title, "company": job.company,
              "location": job.location, "salary_lpa": list(job.salary_lpa),
              "score": m.score, "band": m.band, "semantic_sim": b.get("semantic_sim", 0),
              "matched_required": [(s, evidence.get(s, "")) for s in b.get("matched_required", [])],
              "missing_required": list(b.get("missing_required", [])),
              "matched_preferred": list(b.get("matched_preferred", [])),
              "missing_preferred": list(b.get("missing_preferred", [])),
          })
      return _tpl(request, "student/matches.html", sid=sid, cards=cards,
                  has_matches=bool(cards), error="")
  ```
  (Import `Job` is already added; `JobMatch` too.)

- [x] **Step 2: Create `app/templates/student/matches.html`** (extends base, sid set):
  - If `not has_matches`: a "No matches yet" card with a button that runs matching then reloads: `<button hx-get="/api/students/{{ sid }}/matches" hx-indicator="#run" hx-on::after-request="location.reload()">Run matching</button>`.
  - Else, for each `c` in `cards`: a card with:
    - A score dial/badge: show `{{ c.score }}` and band `{{ c.band }}` (band badge colored: strong>=80, medium>=60, else low - use Tailwind classes).
    - `{{ c.title }}` @ `{{ c.company }}`, location, salary `{{ c.salary_lpa[0] }}-{{ c.salary_lpa[1] }} LPA`.
    - An expandable `<details>` **breakdown** with three sections, each ALWAYS rendered (so tests can find them):
      - `✅ Matched required:` list `{{ s }}` with an `<em>` evidence line `{{ ev }}` per `(s, ev)`.
      - `❌ Missing required:` list `{{ s }}` (this text must contain the word "Missing").
      - `➕ Preferred:` list matched then missing preferred.
    - A "Generate upskilling plan for this job" button: `<button onclick="genPlan('{{ sid }}','{{ c.job_id }}')">Generate plan</button>` with a top-level `<script>`:
      ```html
      <script>
      function genPlan(sid, job_id) {
        fetch("/api/students/" + sid + "/plans", {method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({job_id: job_id})})
          .then(function (r) { return r.json().then(function (j) { return {ok: r.ok, j: j}; }); })
          .then(function (x) { if (x.ok) { location.href = "/students/" + sid + "/plan"; }
                               else { alert("Could not generate plan: " + JSON.stringify(x.j)); } });
      }
      </script>
      ```
  - The matched-required evidence text must come from the fixture resume (tests assert a known evidence substring).

- [x] **Step 3: Add tests** to `tests/test_dashboard.py`:
  ```python
  def _seed_with_matches(client, name="Meera"):
      sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL\n", "text/plain")},
                        data={"name": name}).json()["id"]
      client.get(f"/api/students/{sid}/matches")
      return sid

  def test_matches_page_renders_explainable_cards(client_engine):
      client, _ = client_engine
      sid = _seed_with_matches(client)
      h = client.get(f"/students/{sid}/matches")
      assert h.status_code == 200
      assert "Missing" in h.text                 # missing-required section label
      assert "Python" in h.text                   # a matched skill with evidence
      assert "Generate plan" in h.text

  def test_matches_page_no_matches_state(client_engine):
      client, _ = client_engine
      sid = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                        data={"name": "NoMatch"}).json()["id"]
      h = client.get(f"/students/{sid}/matches")
      assert h.status_code == 200
      assert "Run matching" in h.text
  ```

- [x] **Step 4: Gate** `uv run pytest tests/test_dashboard.py -v` (5 pass) + full unit (expected 73) + ruff clean.

- [x] **Step 5: Commit** `feat: explainable matches dashboard page with evidence + plan CTA`

---

## Task 4.3: Plan checklist + Progress chart + Interview hub (+ gate)

**Files:** Create `app/templates/student/plan.html`, `app/templates/student/progress.html`, `app/templates/student/interviews.html`; Modify `pages.py` (add 3 routes); Test in `tests/test_dashboard.py`.

- [x] **Step 1: Add routes** to `pages.py`:
  ```python
  @router.get("/students/{sid}/plan")
  def plan_page(sid: str, request: Request, session: Session = Depends(get_session)):
      if session.get(Student, sid) is None:
          raise HTTPException(404, "student not found")
      plan = session.exec(select(UpskillPlan).where(UpskillPlan.student_id == sid)
                          .order_by(UpskillPlan.created_at.desc())).first()
      if plan is None:
          return _tpl(request, "student/plan.html", sid=sid, plan=None, error="")
      items = session.exec(select(PlanItem).where(PlanItem.plan_id == plan.id)
                           .order_by(PlanItem.order)).all()
      courses = {c.id: c for c in session.exec(select(Course)).all()}
      plan_items = [{
          "id": it.id, "order": it.order, "skill_id": it.skill_id, "hours": it.hours,
          "status": it.status,
          "course_title": courses[it.course_id].title if it.course_id in courses else "",
          "course_url": courses[it.course_id].url if it.course_id in courses else "",
          "course_provider": courses[it.course_id].provider if it.course_id in courses else "",
          "course_cost": courses[it.course_id].cost if it.course_id in courses else "",
      } for it in items]
      done_hours = sum(i["hours"] for i in plan_items if i["status"] == "done")
      total_hours = plan.total_hours or 0
      return _tpl(request, "student/plan.html", sid=sid, plan=plan, plan=plan,
                  plan_items=plan_items, done_hours=done_hours, total_hours=total_hours, error="")


  @router.get("/students/{sid}/progress")
  def progress_page(sid: str, request: Request, session: Session = Depends(get_session)):
      if session.get(Student, sid) is None:
          raise HTTPException(404, "student not found")
      events = session.exec(select(ProgressEvent).where(ProgressEvent.student_id == sid)
                            .order_by(ProgressEvent.at.asc())).all()
      best = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                          .order_by(JobMatch.score.desc())).first()
      return _tpl(request, "student/progress.html", sid=sid,
                  series=[{"at": e.at.isoformat(), "readiness": e.readiness} for e in events],
                  events=[{"id": e.id, "kind": e.kind, "readiness": e.readiness, "at": e.at.isoformat()} for e in events],
                  best_match=({"job_id": best.job_id, "score": best.score, "band": best.band} if best else None),
                  error="")


  @router.get("/students/{sid}/interviews")
  def interviews_page(sid: str, request: Request, session: Session = Depends(get_session)):
      if session.get(Student, sid) is None:
          raise HTTPException(404, "student not found")
      rows = session.exec(select(Interview).where(Interview.student_id == sid)
                          .order_by(Interview.started_at.desc())).all()
      items = []
      for iv in rows:
          job = session.get(Job, iv.job_id)
          items.append({"id": iv.id, "job_title": job.title if job else iv.job_id,
                        "finished": iv.finished, "score": iv.score,
                        "started_at": iv.started_at.isoformat()})
      top = session.exec(select(JobMatch).where(JobMatch.student_id == sid)
                         .order_by(JobMatch.score.desc())).first()
      return _tpl(request, "student/interviews.html", sid=sid, items=items,
                  top_job_id=(top.job_id if top else None), error="")
  ```
  Import additions for pages.py (cumulative across 4.2/4.3): `Course, Job, JobMatch, PlanItem, ProgressEvent, UpskillPlan, Interview` from `app.models`. Keep the import list free of unused names (ruff F401) - add each import only in the task that first uses it.

- [x] **Step 2: Create `app/templates/student/plan.html`** (extends base, sid set):
  - If `plan is none`: "No upskilling plan yet" + a "Generate a plan" button that calls the top-match: `<button onclick="genPlanTop('{{ sid }}')">Generate plan</button>` (reuse a `genPlanTop` that POSTs `/api/students/{{sid}}/plans` with no body, then redirects to `/students/{{sid}}/plan`).
  - Else: a progress bar `<div id="plan-bar" style="width:{{ pct }}%">` where `pct = (100 * done_hours / total_hours if total_hours else 0)`; show `{{ done_hours }} / {{ total_hours }} h`.
  - Narrative block: `{{ plan.narrative }}`.
  - Checklist: for each `it` in `plan_items` a row with a checkbox/button:
    `<button hx-post="/api/plans/{{ plan.id }}/items/{{ it.id }}/complete" hx-target="tr#row-{{ it.id }}" hx-swap="outerHTML" hx-include="closest form" class="complete-btn" data-item="{{ it.id }}">{{ 'Done' if it.status == 'done' else 'Mark done' }}</button>`
    - Because the complete endpoint returns JSON (not HTML), set `hx-swap="none"` and instead refresh: use `hx-on::after-request="location.reload()"` so the checkbox state updates. Use `hx-post` with `hx-swap="none" hx-on::after-request="location.reload()"`.
    - Show per item: skill id, course title (link to `course_url` target=_blank), provider, cost, hours, status badge.
  - The tests will assert the plan page contains a plan item course title and the progress bar element.

- [x] **Step 3: Create `app/templates/student/progress.html`** (extends base, sid set):
  - A `<canvas id="readinessChart"></canvas>` and a `<script>` that fetches `/api/students/{{ sid }}/progress` and draws a **Chart.js** line chart (CDN `<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>` in this template or base). X = `at` (short time), Y = `readiness`.
  - An **event table**: for each `e` in `events`: kind, readiness, `at`.
  - Show best match: `{{ best_match.score }} ({{ best_match.band }})` if present else "no match yet".
  - The test asserts the page contains `readinessChart` and `chart.js`/`Chart`.

- [x] **Step 4: Create `app/templates/student/interviews.html`** (extends base, sid set):
  - A "Start a new interview" card with a job selector: a `<select id="job">` defaulting to `{{ top_job_id }}` (top match) plus all jobs would be ideal but is optional; at minimum a "Start with top match" button:
    `<button onclick="startInterview('{{ sid }}', document.getElementById('job').value)">Start interview</button>`
    with:
    ```html
    <script>
    function startInterview(sid, job_id) {
      fetch("/api/students/" + sid + "/interviews", {method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(job_id ? {job_id: job_id} : {})})
        .then(function (r) { return r.json(); })
        .then(function (j) { location.href = "/interviews/" + j.interview_id; })
        .catch(function () { alert("Start the interview after running matches."); });
    }
    </script>
    ```
  - A list of past interviews: for each `iv` in `items` → `<a href="/interviews/{{ iv.id }}">{{ iv.job_title }}</a>` + finished/score.
  - The test asserts the page contains "interview" (lowercase) and, after starting, redirects are exercised at the API level (the test here just checks the page renders + lists).

- [x] **Step 5: Add tests** to `tests/test_dashboard.py`:
  ```python
  def test_plan_page_and_htmx_complete(client_engine):
      client, _ = client_engine
      sid = _seed_with_matches(client, name="Planner")
      plan = client.post(f"/api/students/{sid}/plans", json={}).json()   # top-match plan
      assert plan["items"], "expected a non-empty plan"
      h = client.get(f"/students/{sid}/plan")
      assert h.status_code == 200
      assert "plan-bar" in h.text
      assert plan["items"][0]["course_title"] in h.text or plan["items"][0]["skill_id"] in h.text
      # HTMX toggle: complete the first item, expect the readiness JSON and 200
      item = plan["items"][0]
      r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete",
                      headers={"HX-Request": "true"})
      assert r.status_code == 200
      assert "readiness" in r.json()


  def test_progress_page_renders_chart(client_engine):
      client, _ = client_engine
      sid = _seed_with_matches(client, name="Progress")
      plan = client.post(f"/api/students/{sid}/plans", json={}).json()
      client.post(f"/api/plans/{plan['id']}/items/{plan['items'][0]['id']}/complete")
      h = client.get(f"/students/{sid}/progress")
      assert h.status_code == 200
      assert "readinessChart" in h.text
      assert "chart.js" in h.text or "Chart" in h.text


  def test_interviews_page_renders(client_engine):
      client, _ = client_engine
      sid = _seed_with_matches(client, name="Intv")
      h = client.get(f"/students/{sid}/interviews")
      assert h.status_code == 200
      assert "interview" in h.text.lower()
      assert "Start" in h.text
  ```

- [x] **Step 6: Run gate.** `uv run pytest tests/test_dashboard.py -v` (8 pass) + `uv run pytest -m "not integration" -q` (expected 78: 68 + 10) + `uv run pytest -q` (full, live PG) + `uv run ruff check .` clean.

- [x] **Step 7: Manual smoke (optional but recommended):** `uv run uvicorn app.api.main:app --port 8000 --reload` is NOT needed for tests; instead verify the TestClient flow already covers upload→profile→matches→plan→complete→progress. If a quick live check is desired, run the existing `cmd_demo` CLI.

- [x] **Step 8: Commit** `feat: plan checklist, readiness progress chart, interview hub dashboard pages`

- [x] **Step 9: Phase 4 gate.**
  - `uv run pytest -m "not integration" -q` green; `uv run pytest -q` green; `uv run ruff check .` clean.
  - Update master plan `2026-10-31-ps06-placement-agent.md`: add `### Phase 4 STATUS: COMPLETE` after `## 9. Phase 4`, mark Task 4.1/4.2/4.3 boxes `[x]` (in BOTH the master plan and this file), note the `create_student_core`/`_resume` refactor, and handoff to Phase 5 (admin analytics reads JobMatch/PlanItem/StudentSkill across the cohort).
  - Commit `chore: phase 4 complete`.

---

## Notes for the executing agent
- **Do not change JSON API response shapes** except the internal extraction in students.py (behavior must be byte-identical; existing `test_api.py`, `test_planner.py`, `test_progress.py`, `test_interview.py` must pass unchanged).
- Templates extend `base.html`. The only non-extending template is `_skill_chips.html` (a fragment for HTMX swap).
- HTMX complete/toggle endpoints return JSON; use `hx-swap="none" hx-on::after-request="location.reload()"` to refresh the page (no partial HTML needed).
- Chart.js + Tailwind + HTMX all via CDN (no Node/build). The tests only assert on the rendered HTML strings (presence of element ids / keywords), they do NOT execute JS - so keep the required id/keyword strings exactly as asserted.
- `job-001` and the seeded 24 jobs/40 courses/124 skills are available in the test DB via `seed()`.
- Keep the `sid` variable in every per-student template's context (base nav uses it).
- If a test asserts a specific skill name (e.g. "Kubernetes"), FIRST confirm that skill exists in `data/skills.json` (taxonomy) and is extractable from a plain "Skills: X" resume line; otherwise swap in a verified skill (React, Docker, SQL, Git, etc.).
