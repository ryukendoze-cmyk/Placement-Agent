# JKLU CareerOS - PS-06 Autonomous Placement & Readiness Agent

**Version 3.0.0 "Production Edition"** - an agentic AI system for a college
**placement cell** that turns a student's resume into an explainable, actionable
career picture, with autonomous agents, human-in-the-loop controls, live job
intelligence, drive simulation, and a flagship SaaS-style interface.

Everything runs **offline by default** (deterministic `HashingEmbedder` +
`MockLLM`); swap in AWS Bedrock (free tier) or OpenAI via `.env` - no code
changes.

## What it does

| Area | Capability |
|---|---|
| **Explainable matching** | Ranks 24 real job descriptions against a resume with a transparent score (`0.6 required-coverage + 0.2 preferred + 0.2 semantic`) and shows exactly *which* required skills matched (with the resume line as evidence) and which are missing. |
| **Upskilling plans** | Skill gaps become an ordered, hour-costed course plan ordered by a **prerequisite DAG** (prerequisites before consumers), capped at 10 h/week, with an LLM narrative. |
| **Closed-loop readiness** | Mark a plan item done (optionally verified by a keyword-graded micro-quiz) -> skills are promoted, matches re-ranked, active `JobApplication` match scores updated, and a readiness event (`0.5 plan-completion + 0.5 best-match`) climbs over time. |
| **Adaptive mock interviews** | 8-turn voice-in/voice-out studio (Web Speech API, text fallback) over a **220+ question bank with reference outlines**; difficulty adapts to answer quality; each turn is scored on a **3-axis rubric** (technical accuracy, depth & examples, communication) with matched/missed keywords and an ideal-answer panel; completion writes a `ProgressEvent` + immutable `AuditLog`. |
| **Autonomous agent** | Iterative ReAct loop (`Goal -> Plan -> Execute -> Observe -> Replan/Repair -> Verify`) over **18 registered tools** (inspect, rank, plan, apply, simulate drive, search SOPs, audit resume, STAR rewrite), with a live interactive reasoning trace in the **Agent Brain** drawer. |
| **Grounded copilot** | RAG over institutional placement SOPs; every answer cites `[SOP-XX]` clauses with a grounding score; clickable citation pills. |
| **Human-in-the-loop** | Two-tier Approvals Center (`faculty_mentor` Tier-1 endorsement -> `tpo_admin` Tier-2 final sign-off) enforcing strict separation of duties for nominations, bootcamp launches, resume re-upload unlocks, and proficiency waivers; every decision is immutably audited. |
| **Placement Policy Studio** | Dedicated TPO Admin `/admin/policies` console to add (`SOP-19+`), edit, remove, or reset institutional placement SOPs and threshold parameters with live RAG re-indexing. |
| **Cohort & Student Analytics** | PS-06 TPO Cohort Readiness & Skill-Gap Intelligence Center (`/admin` with 6 KPIs, 4 charts, Student Readiness Matrix, Institutional Skill-Gap Priorities, Drive Eligibility Simulator) and Student Progress Center (`/students/{sid}/progress` with multi-day milestone progression, role-gap matrix, 4-week study load, and inline module completion). |
| **Campus Drives & Recruiter JNF** | TPO Smart JD parser (`POST /api/campus-drives/parse-jd`) and shareable Recruiter Job Notification Form (`/recruiter/jnf/{token}`) with branch normalization (`CSE`, `AI&DS`, `ECE`, `IT`, `ME`, `EE`, `CE`). |
| **Per-Student Live Jobs & Favorites** | Students can fetch live market jobs isolated strictly to their own workspace (`owner_student_id`), bookmark favorite opportunities/applications, and protect on-campus drives from dismissal. |
| **Flagship UI/UX** | App shell with sidebar + command bar (`Ctrl+K`), active-student switcher, persona/PIN authentication (`/login`), global toast & confirmation modal system (zero `alert()` / `confirm()`), interactive Agent Brain drawer, printable interview scorecard. |
| **RBAC + PII** | 3 roles (`student` / `faculty_mentor` / `tpo_admin`) with HMAC-signed session tokens; PII scrubbed + prompt-injection neutralized across resumes, interviews, agent goals, copilot queries, and opaque `id-<hash>` audit logs. |
| **Live job intelligence** | Pluggable scraper with `content_hash` idempotency, transparent sandbox fallback tagging (`is_sandbox_fallback`), per-source circuit breaker, and a two-stage ANN pipeline (pre-filter k=10, final top 10) that keeps local + scraped jobs in one ranked pool. |
| **Drive simulation + ROI** | TPO cohort simulator that evaluates every student against a *hypothetical* drive with the production SOP engine (eligible list, per-SOP bottleneck breakdown, top missing skills with recommended courses) plus a deterministic bootcamp-intervention ROI projection. |
| **Streaming agents** | ReAct agent (**18 tools**, real-time unbuffered SSE generator, observation-driven replanning, step repair, budget + cycle guards) and grounded copilot stream events over SSE (`POST /api/agent/run/stream`, `POST /api/copilot/ask/stream`) with a live event-timeline UI. |
| **Interview intelligence** | Per-turn speech-delivery telemetry (WPM banding, filler density, STAR structure, delivery score, coaching tip), per-question accuracy/latency tracking, and a session `speech_summary` on every finished transcript. |
| **Applications + endorsements** | `JobApplication` state machine (`submitted -> awaiting_mentor_endorsement -> awaiting_tpo_approval -> nominated -> shortlisted -> rejected`) with SOP-03 CGPA waivers, auto-nominate for eligible standard-tier matches, and a faculty mentor endorsement loop. |
| **Observability** | Public `/healthz` + `/readyz` probes, `/api/admin/system-telemetry` (p95 latency, token usage, circuit state, entity counts), and a `eval-benchmark` command for SOP RAG recall@k/MRR, planner DAG invariants, and agent grounding rate. |
| **Worker queue + LLM telemetry** | Optional background task queue (in-memory ThreadPool, Redis backend with graceful fallback) and instrumented LLM calls (token/cost accounting) with an automatic circuit breaker to the offline MockLLM. |

## Architecture

```
                         Resume (PDF / DOCX / TXT / PNG / JPEG, <=5 MB)
                                        |
                                        v
                              [ parse_resume_bytes ]  --> text  (PII scrubbed)
                                        |
                                        v
                          [ extract_skills ]        (124-skill taxonomy)
                                        |
                                        v
                    StudentSkill (skill_id, mentions, level, evidence lines)
                                        |
                    embed(text) --> student vector --> VectorStore  (pgvector | memory)
                                                    HNSW, ann_pre_filter_k=10 (two-stage)
                                        v
                    [ rank_jobs ]  0.6*required + 0.2*preferred + 0.2*semantic
                    local jobs (source=local)  +  scraped jobs (source=live, content_hash dedupe)
                                        |
                                        v
                    JobMatch (score, band, breakdown)  [persisted]
                                        |
     +--------+-----------+------------+-----------+------------+------------+
     |        |           |            |           |            |            |
     v        v           v            v           v            v            v
  [ plan ] [ interview ] [ complete ] [ agent    ] [ copilot  ] [ applications ] [ drive sim ]
  prereq   8-turn        readiness    ReAct      RAG          state machine    cohort sim (TPO)
  DAG +    adaptive      series       15 tools   [SOP-XX]     waiver + HITL    SOP engine +
  10h/week 3-axis rubric (re-rank)    plan->act  cites +      two-tier        bottleneck
  + quiz   + speech       matches      ->observe  grounding    endorsement    breakdown +
           telemetry      (WPM, fillers) ->verify score       loop            ROI projection
           (STAR, tip)              (SSE stream) approvals    (auto-nominate  (hypothetical
                                             center)           standard)       drive, no persist)
                                        |
                                        v
                    HITL approvals (nominate / bootcamp / waiver / application / drive)
                    pending_review -> approve|reject|endorse -> AuditLog (append-only, guarded)

    Production shell (Jinja2 + HTMX + Chart.js, no build step):
      /                          student picker + upload (app shell)
      /students/{sid}            profile + skill chips + verification badges + resume lock
      /students/{sid}/matches    explainable match cards + favorites + isolated live job fetch
      /students/{sid}/plan       prereq-ordered plan + progress bar + verify quiz
      /students/{sid}/progress   PS-06 competency center, milestone curve, role-gap matrix & module tracker
      /students/{sid}/interviews interview hub
      /interviews/{id}           voice studio (rubric bars + speech telemetry + scorecard)
      /copilot                   grounded copilot (cited answers, SSE streaming)
      /recruiter/jnf/{token}     public recruiter Job Notification Form (JNF)
      /admin                     PS-06 cohort readiness, skill-gap priorities, drive simulator & eval telemetry
      /admin/policies            TPO Policy Studio (add, edit, remove, reset SOPs + RAG sync)
      /admin/approvals           HITL approvals center
      /admin/audit               audit-trail inspector (PII-scrubbed)
      /healthz  /readyz          liveness + readiness probes
```

## Quickstart

Prereqs: [uv](https://docs.astral.sh/uv/), Python 3.12, and a local
PostgreSQL 17 with the `vector` (pgvector) extension.

```powershell
# 1. Install dependencies (creates .venv)
uv sync

# 2. Postgres: create the dev + test databases and enable pgvector
#    (Windows, trust auth; adapt the service/login for your install)
psql -U postgres -c "CREATE DATABASE ps06;"
psql -U postgres -c "CREATE DATABASE ps06_test;"
psql -U postgres -d ps06      -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -U postgres -d ps06_test -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Configure (optional - the .env defaults work with trust auth on 127.0.0.1)
Copy-Item .env.example .env

# 4. Seed taxonomy (124 skills) + 24 jobs + 40 courses (+ job vectors on Postgres)
uv run python -m app.cli seed

# 5. Run the app  -->  http://127.0.0.1:8010
uv run uvicorn app.api.__main__:app --port 8010
```

Then open <http://127.0.0.1:8010/>.

### Docker (one command, no Postgres install)

With [Docker](https://docker.com) (Docker Desktop / Engine + Compose v2):

```powershell
docker compose up --build
```

This builds the app image (multi-stage, non-root, `uv sync --frozen`), starts
`pgvector/pgvector:pg17` (with the `vector` extension auto-created) and
`redis:7-alpine`, seeds the taxonomy + **demo cohort**, and serves the app on
<http://localhost:8010>. Stop with `Ctrl+C` (or `docker compose down -v` to
wipe the data volumes).

### Seed the demo cohort (recommended for demos)

Populate the full **12-student demo cohort** so every screen is alive at
startup - matches, plans, multi-point progress, finished interview transcripts
with **speech-delivery telemetry**, agent traces, 4 pending approvals, and
three flagship **JobApplications** (one `nominated`, one `awaiting_tpo_approval`
bound to the nomination approval, one `awaiting_mentor_endorsement` on a
CGPA-waiver path):

```powershell
uv run python -m app.cli seed --demo-cohort
```

### No-database demo

Rank a resume against `data/jobs.json` with no Postgres at all:

```powershell
uv run python -m app.cli demo --file tests/fixtures/resume.txt --name Bharat --top 5
```

## LLM provider (default: offline `mock`)

Narratives, answer scoring, interview feedback and copilot answers all go
through one provider factory (`app/llm/base.py:get_llm`). Set `LLM_PROVIDER`
in `.env`:

| Provider | What it does | Setup |
|---|---|---|
| `mock` (default) | Deterministic, offline keyword/length scoring + template narratives. Zero deps. | none |
| `bedrock` | Amazon Nova via `boto3.converse` (free tier) | see checklist below |
| `openai` | `openai` SDK chat completion | `uv add openai`, set `OPENAI_API_KEY` |

### AWS Bedrock setup checklist

1. **IAM user:** in the IAM console create a user with *programmatic access* and
   copy the access key id + secret access key.
2. **Enable the model (region matters):** in the Bedrock console, set the region
   to `ap-south-1`, go to **Model access**, and request access to
   *Amazon Nova Lite* (`amazon.nova-lite-v1:0`). Access is granted
   automatically for the free tier.
3. **IAM policy:** attach a policy allowing `bedrock:InvokeModel` (the managed
   `AmazonBedrockFullAccess` works for a local dev setup).
4. **Configure `.env`:**
   ```dotenv
   LLM_PROVIDER=bedrock
   LLM_MODEL=amazon.nova-lite-v1:0
   AWS_REGION=ap-south-1
   AWS_ACCESS_KEY_ID=AKIA...
   AWS_SECRET_ACCESS_KEY=...
   ```
   `boto3` is already a dependency. (If the model is not yet available in
   `ap-south-1`, set `AWS_REGION=us-east-1` as a fallback.)

## Live demo script (3 minutes)

1. **Shell** - the app opens on the flagship shell: sidebar, brand
   (*JKLU CareerOS - Autonomous Placement & Readiness Agent*), student switcher,
   role pill (Student/Faculty/TPO) and `Ctrl+K` command bar. Switch roles and
   watch the badge + toast confirm.
2. **Matches** - open a student's **Matches**: score ring, band, SOP eligibility
   badge, and a *Why this score?* breakdown (matched required skills with the
   quoted resume line, missing required skills, preferred skills).
3. **Plan** - **Generate plan for this job**: the gap skills become a
   prerequisite-ordered, hour-costed plan (10 h/week cap); complete an item and
   watch the progress bar; a micro-quiz can verify the skill first.
4. **Progress** - readiness chart climbs; events show resume parse, match,
   plan, completion and interview milestones; export the dossier.
5. **Interview** - start a mock interview: 8-turn stepper, difficulty badge,
   per-turn rubric bars, matched/missed keywords, collapsible ideal answer,
   voice pulse + timer, and a **printable scorecard** with the overall rubric.
6. **Copilot** - ask *"What score do I need for Dream drives?"*: the answer
   cites `[SOP-XX]` clauses as clickable pills with a grounding score, and the
   step trace shows the retrieval.
7. **Agent Brain** - open the drawer: the latest autonomous run's plan, tool
   calls, verification and recovery, step by step.
8. **Approvals** - `/admin/approvals`: approve a nomination or bootcamp
   proposal (**Approve & Execute**), and see the decision land in
   `/admin/audit` (immutable, with the executed mutation).
9. **Analytics** - `/admin`: KPI strip, skill-gap bars, readiness doughnut,
   top courses, recommendations; **Launch Bootcamp (HITL)** queues an approval;
   export CSV/JSON.

## API (JSON)

- `POST /api/students` - upload resume -> `{id, name, skills[]}`
- `GET  /api/students/{sid}` / `GET /api/students/{sid}/skills` - profile + skills
- `GET  /api/students/{sid}/matches` - ranked, explainable matches (persists `JobMatch`)
- `POST /api/students/{sid}/matches/{job_id}/simulate` - what-if skill acquisition
- `GET  /api/students/{sid}/resume/xray` / `POST .../resume/improve-bullets` - ATS X-Ray + AI bullets
- `POST /api/students/{sid}/plans` `{job_id?}` / `GET /api/students/{sid}/plans` / `GET /api/plans/{id}`
- `GET  /api/plans/{id}/items/{item}/quiz` / `POST .../verify` - keyword-graded micro-quiz
- `POST /api/plans/{id}/items/{item}/complete` - record readiness (re-ranks matches)
- `GET  /api/students/{sid}/progress` - readiness series + events
- `POST /api/students/{sid}/interviews` / `POST /api/interviews/{id}/answers` / `GET /api/interviews/{id}`
- `POST /api/copilot/ask` - grounded SOP answer with citations
- `POST /api/agent/run` / `GET /api/agent/runs` - autonomous agent runs + traces
- `POST /api/approvals` / `GET /api/approvals` / `POST .../approve|reject` - HITL queue
- `GET  /api/admin/analytics` / `GET /api/admin/audit` - cohort aggregates + scrubbed audit trail
- `GET  /api/admin/policies` / `POST /api/admin/policies` / `PUT /api/admin/policies/{sop_id}` / `DELETE /api/admin/policies/{sop_id}` / `POST /api/admin/policies/reset` - live SOP policy management
- `POST /api/students/{sid}/matches/{job_id}/favorite` - toggle favorite job/application
- `POST /api/campus-drives/parse-jd` / `POST /api/campus-drives` / `POST /api/campus-drives/jnf-links` - Smart JD & Recruiter JNF workflow
- `POST /api/auth/switch-role` / `GET /api/auth/role` - demo role cookie
- `POST /auth/login` / `POST /auth/logout` - HMAC-signed session tokens (v3)
- `POST /api/agent/run/stream` / `POST /api/copilot/ask/stream` - SSE event streams (v3)
- `POST /api/students/{sid}/applications` / `GET /api/students/{sid}/applications` / `GET /api/admin/applications` - application pipeline (v3)
- `POST /api/approvals/{appr_id}/endorse` (faculty mentor) / `POST /api/approvals/{appr_id}/approve` (TPO) - two-tier application approval (v3)
- `POST /api/speech/analyze` - speech-delivery analytics on a transcript (v3)
- `POST /api/jobs/scrape` / `GET /api/jobs` (source split) - live job intelligence (v3)
- `POST /api/admin/simulate-drive` - cohort drive eligibility simulation + bootcamp ROI (v3)
- `GET  /api/admin/system-telemetry` / `GET /api/admin/eval-benchmark` - ops observability (v3)
- `GET  /healthz` / `GET /readyz` - liveness + readiness probes (v3, public)
- `GET  /api/tasks` / `GET /api/tasks/{task_id}` - background task queue status (v3)

## Tests

```powershell
uv run pytest -m "not integration" -q   # unit: SQLite (FKs enforced) + HashingEmbedder + MockLLM
uv run pytest -m integration -q         # live Postgres/pgvector (needs local PG)
uv run pytest -q                        # everything
uv run ruff check .                     # linter
```

The unit suite runs on an in-memory SQLite engine with `PRAGMA foreign_keys=ON`,
so referential-integrity bugs (the kind Postgres would catch) fail fast without
a database.

## Project layout

```
app/
  api/            FastAPI app: main.py (create_app/AppContext), routes/, deps.py
  agent/          orchestrator.py - ReAct loop (18 tools, budget/cycle guards, SSE)
  analytics/      cohort.py (pure cohort aggregations), drive_simulator.py (TPO drive sim + ROI)
  embeddings/     embedder.py (Hashing/Local), store.py (Memory/PgVector, upsert_many, two-stage ANN)
  eval/           benchmark.py - SOP RAG recall@k/MRR + planner DAG + grounding audit
  interview/      bank.py (220+ Q bank + outlines + select), engine.py (adaptive + rubric),
                  speech_analytics.py (WPM/filler/STAR telemetry + aggregation)
  llm/            base.py (factory), mock.py (rubric-aware), bedrock.py, openai.py,
                  telemetry.py (token/cost accounting + circuit breaker)
  matching/       engine.py (score_job/JobScore/band_for), rank.py (rank_jobs, ann_pre_filter_k)
  plan/           planner.py (build_plan, 10h/week), prereq.py (DAG), verification.py (quiz)
  rag/            policy_rag.py - 18 SOP retrieval + citation verification + eligibility
  resume/         parser.py (PDF/DOCX/TXT/PNG/JPEG -> text)
  scraper/        service.py, providers.py, pipeline.py, normalizer.py, ai_query.py
  security/       rbac.py (3 roles + HMAC sessions), pii.py, guardrails.py, ratelimit.py, audit.py
  skills/         extract.py (text -> taxonomy skills + evidence)
  taxonomy/       taxonomy.py (124-skill taxonomy)
  templates/      Jinja2: base.html (shell), index, copilot, student/*, admin/*
  static/         app.js (interview studio + speech telemetry + global toast)
  workers/        task_queue.py (in-memory ThreadPool | Redis, graceful fallback)
  cli.py          seed / seed --demo-cohort / demo / eval-benchmark
  models.py       SQLModel tables (incl. AuditLog, AgentRun, AgentStep, ApprovalRequest, JobApplication)
  db.py, config.py, seed.py
data/             skills.json (124), jobs.json (24), courses.json (40), questions.json (220+), placement_sops.json (18)
tests/            unit + integration (marker) suites
docs/             COMPLETION-REPORT.md, superpowers/plans/
Dockerfile, docker-compose.yml, docker/initdb/, .github/workflows/ci.yml
```

## Status

**v3.0.0 "Production Edition"** - all 25 milestones complete (M1-M13 baseline +
M14-M25 production build), each passed an adversarial Gate C review (see
`docs/COMPLETION-REPORT.md` for the 100-point rubric mapping and demo script):

- **v0.1.0 (Phases 0-6)** - matching, planning, interviews, dashboard, analytics, docs
- **Phase 1** - shell foundation, RBAC (3 roles), PII scrubbing, audit skeleton
- **Phase 2** - autonomous agent orchestrator (plan/execute/verify/recover) + reasoning trace
- **Phase 3** - RAG policy copilot with citation guard + eligibility engine
- **Phase 4** - HITL approvals (nominate / bootcamp / waiver) + immutable audit trail
- **Phase 5** - prereq DAG plans, 10 h/week schedule, micro-quiz verification, closed-loop completion
- **Phase 6** - multi-rubric interview studio (89-Q bank, rubric bars, ideal answers, scorecard)
- **Phase 7** - flagship SaaS UI/UX (shell, switcher, role pill, toasts, copilot/approvals/audit pages)
- **Phase 8** - 12-student demo cohort, smoke, docs, final review
- **M14** - defect remediation, proficiency-aware planner (DAG-validated), expanded verifier, route-wide RBAC/audit
- **M15** - security hardening (HMAC-signed sessions, prompt-injection sanitizer, rate limits)
- **M16** - true iterative ReAct agent (15 tools, budget/cycle guards, `awaiting_input` clarification)
- **M17** - SSE streaming (agent run + copilot ask) with live progressive event timeline
- **M18** - interview intelligence: adaptive follow-ups, project deep-dives, per-turn speech-delivery telemetry
- **M19** - JobApplication pipeline + SOP-03 waivers + two-tier HITL (mentor endorsement -> TPO approval)
- **M20** - TPO cohort drive-eligibility simulator + deterministic bootcamp ROI projection + simulate-drive endpoint
- **M21** - DB & vector scaling (12 indexed columns, `upsert_many`, two-stage ANN `ann_pre_filter_k`)
- **M22** - worker task queue (memory | Redis) + LLM telemetry with circuit breaker
- **M23** - observability (`/healthz`, `/readyz`, system-telemetry) + eval benchmark
- **M24** - containerization (multi-stage Dockerfile, compose with pgvector + Redis) + GitHub Actions CI
- **M25** - flagship demo cohort v3 (application states + speech telemetry) + docs sync + final audit


## Desktop & Mobile Demo (Electron, PWA & Android)

All desktop, mobile, and PWA shells connect to the FastAPI backend (`http://127.0.0.1:8010` or LAN IP) so every fetch stays same-origin. Use demo persona PIN **`2026`** on the login screen to switch between `student`, `faculty_mentor`, and `tpo_admin` personas.

### 1. Progressive Web App (PWA)

- Served directly by FastAPI at `/manifest.webmanifest` and `/sw.js` with icons in `app/static/icons/`.
- Open `http://127.0.0.1:8010` in Chrome/Edge and click **Install JKLU CareerOS** from the address bar to run in standalone desktop/mobile mode.

### 2. Electron Desktop (`desktop/`)

```powershell
cd desktop
npm install
npm run desktop:dev
```

- Automatically probes `http://127.0.0.1:8010/healthz`; if unreachable, spawns `uv run uvicorn app.api.__main__:app --port 8010` with a live startup splash screen and polls `/readyz`.
- Configure a remote server URL anytime via **Settings -> Server Connection**.
- Package a Windows portable `.exe` and NSIS installer with `npm run desktop:build` (outputs to `desktop/dist/`). Note: on locked-down npm setups, ensure `npm config set ignore-scripts false` (or `npm rebuild electron`) so Electron's binary downloads during install.

### 3. Capacitor Android & LAN Demo (`mobile/`)

**Option A — USB Cable (ADB Reverse, works offline without Wi-Fi):**

```powershell
adb reverse tcp:8010 tcp:8010
```

In the Android app's **Server Connection** screen, tap **ADB Reverse (127.0.0.1:8010)**, tap **Test Connection**, and tap **Launch CareerOS**.

**Option B — Campus Wi-Fi / Mobile Hotspot (LAN):**

```powershell
.\scripts\serve_lan.ps1
```

The script prints your host LAN IPv4 address, the Windows Firewall rule command (`New-NetFirewallRule ...`), and the mobile/JNF URLs. Enter `http://<LAN-IP>:8010` in the Android app's **Server Connection** screen (or `http://10.0.2.2:8010` on the Android Emulator).

**Building the Android APK (requires JDK 17 + Android SDK):**

```powershell
cd mobile
npm install
npx cap sync android
npx cap open android
```
