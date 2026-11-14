# JKLU CareerOS: revised swarm prompt (v2, based on the current repo)

## Part A: How the old prompt differs from the repo (and what I changed)

| # | The old prompt assumed | What the repo actually has (commit `7efd360`) | Fix in v2 |
|---|---|---|---|
| 1 | A **React** frontend in `frontend/` | Server-rendered **Jinja2 + Tailwind + htmx + vanilla JS** in `app/templates/` and `app/static/`. There is no `frontend/` folder and no build step. | Keep the templates. A React rewrite two days before the buildathon (16 Nov) is not worth the risk. |
| 2 | Policy tiers: Super dream ≥ 10, Dream 5–10, "Regular" | **SOP-01** in `data/placement_sops.json`: Standard < 8.0, Dream 8.0–14.99, Super-Dream ≥ 15.0 LPA, using the **top of the CTC band**. This is already coded in `app/matching/engine.py::job_tier()` with ids `standard / dream / super_dream`. | The old thresholds broke your own Rule 2 (zero hallucination). v2 reads the thresholds from SOP-01 and uses "Standard", not "Regular". |
| 3 | "Internship" is a policy tier | No SOP defines an internship tier | `experience_level = Internship` gives `policy_tier = null` and a flag. No tier is invented. |
| 4 | Defaults for CGPA, backlogs and branches come from "JKLU standard policy" | No SOP defines a default CGPA. SOP-03 sets backlog limits per tier (Super-Dream 0, Dream ≤ 1). Empty `allowed_branches` already means "open to all". | Defaults come from SOP-03 or an explicit config file. Every default is listed in `defaults_applied` with its source. |
| 5 | Build a new skill dictionary (`c++`, `next.js`, `machine learning`) | 124-skill taxonomy with aliases in `data/skills.json`. Ids include `cplusplus`, `nextjs`, `machine_learning`. `app/skills/extract.py` already does grounded extraction with an evidence check (`_grounded`). | Reuse the taxonomy and the extractor. Text that doesn't map to a skill goes into `unmapped_skill_mentions` and never becomes a required skill. |
| 6 | Build PDF/DOCX extraction | `app/resume/parser.py::parse_resume_bytes` already handles pdf, docx (including tables), txt and OCR images | Generalise and reuse it. |
| 7 | Build a separate approvals queue | HITL already exists: `ApprovalRequest` + `execute_approved()` dispatch in `app/api/routes/approvals.py` + the `/admin/approvals` page | Add one new kind, `campus_drive_publish`. Approval goes through the existing queue, audit and page. |
| 8 | Build an audit trail | `log_audit()` (`app/security/audit.py`) and an immutable `AuditLog` table (UPDATE/DELETE blocked) are already shown at `/admin/audit` | Use them, add a `trace_id` to group related rows, and add new action names. |
| 9 | Agent 3 builds JWT auth from scratch | HMAC-SHA256 signed session tokens with `exp` (JWT-equivalent), `/login`, `/auth/login`, `/auth/logout`, `/api/auth/switch-role`, a role switcher in the top bar and sidebar, and persona cards on the login page | Agent 3 becomes a hardening job. **Role id is `faculty_mentor`, not `faculty`.** |
| 10 | — | **Real RBAC holes:** (a) the `X-Demo-Role` header overrides identity in every mode, so `curl -H "X-Demo-Role: tpo_admin"` gets TPO powers; (b) with no cookie at all, the default role is `tpo_admin` | Close both when `AUTH_MODE=strict`. Tests keep demo mode. |
| 11 | "100% offline" | `base.html` loads **Tailwind from cdn.tailwindcss.com, htmx from unpkg, a library from jsdelivr and Google Fonts**. Offline, the UI breaks. `LocalEmbedder` downloads a HuggingFace model on first run and has no fallback. | Vendor all assets into `app/static/vendor/`, fall back to `HashingEmbedder`, and add a test that blocks external URLs. |
| 12 | `/api/v1/...` paths | Every route uses `/api/...` | Use `/api/campus-drives/...` to match. |
| 13 | Capacitor `webDir` points at "the frontend build directory" | There is no build directory. FastAPI serves every page, and every `fetch` is a relative, same-origin call. | Electron and Capacitor are thin shells that **load the server URL**. No API-base-URL refactor and no CORS work is needed (same origin). The "Server Connection" screen belongs in the shell. |
| 14 | `allow_origins=["*"]` with cookies | Browsers reject `*` together with credentials | Use an explicit allowlist, and only if a shell actually needs it. |
| 15 | Status `PENDING_APPROVAL` vs `PENDING_TPO_APPROVAL` (inconsistent) | — | One enum: `DRAFT / PENDING_TPO_APPROVAL / PUBLISHED / REJECTED`. |
| 16 | Four agents with overlapping files (`base.html`, `models.py`, `matches.html` at 125 KB) | — | Five workers with **exclusive file ownership** plus a dedicated **verifier** (726 existing tests must stay green). |

**Priority order (the deadline is 16 Nov):** P0 campus-drive pipeline, offline vendoring and RBAC holes → P1 Smart JD UI, JNF and responsive layout → P2 Electron → P3 Capacitor APK.

---

## Part B: Paste-ready swarm prompt

````markdown
# MASTER CONTEXT (all workers read first)

Repo: ps06-placement-agent (local working copy). Project: JKLU CareerOS, PS-06 Intelligent
Placement Readiness & Career Matching Agent. Buildathon demo: 16 Nov 2026. Ship small and
correct, not big and broken.

## Actual stack (do NOT assume React)
- Backend: FastAPI + SQLModel. PostgreSQL+pgvector in prod; SQLite in-memory in tests
  (tests/conftest.py uses MockLLM + HashingEmbedder + MemoryVectorStore).
  Run: `uv run uvicorn app.api.__main__:app --port 8010`. Tests: `uv run pytest -q`
  (~726 tests, all must stay green). Lint: `uv run ruff check .`
- Frontend: Jinja2 templates in app/templates (base.html, student/matches.html,
  admin/approvals.html, admin/audit.html, auth/login.html) + Tailwind + htmx + vanilla JS.
  All fetch() calls are relative/same-origin.
- No Alembic: new tables come from SQLModel.metadata.create_all. New columns on existing
  tables MUST be added to app/db.py::_ensure_columns (idempotent ALTER).

## Reuse these, don't rebuild them
| Need | Existing code |
|---|---|
| Doc text extraction | app/resume/parser.py::parse_resume_bytes (pdf/docx/txt/OCR) |
| Skill grounding | data/skills.json (124 canonical ids + aliases), app/taxonomy/taxonomy.py, app/skills/extract.py (extract_skills, _grounded, _safe_json) |
| Branch normalisation | app/taxonomy/branches.py::normalize_branch (CSE, IT, ECE, EE, ME, CE, AI&DS, CHE, BT) |
| Policy tiers | SOP-01 params in data/placement_sops.json; app/matching/engine.py::job_tier |
| Eligibility | app/matching/engine.py::job_eligibility (GPA + branch + seniority) |
| LLM | app/llm/base.py LLMClient.complete(system, user, json_mode); providers mock/openai/bedrock; circuit breaker in app/llm/telemetry.py |
| Matching | app/matching/rank.py::rank_jobs; app/api/routes/students.py::_rank_and_persist; ctx.vector_store.upsert("job", id, vec); ctx.job_vectors cache |
| Background work | ctx.task_queue.submit(kind, fn, payload) (app/workers/task_queue.py) |
| HITL | ApprovalRequest model + approvals.py::execute_approved(kind dispatch) + /admin/approvals page |
| Audit | app/security/audit.py::log_audit (AuditLog is immutable) + /admin/audit page |
| RBAC | app/security/rbac.py: require_role(...), roles = student / faculty_mentor / tpo_admin (hierarchical) |
| Campus source | Job.source == "local" is already rendered as "On-Campus (Local)" / "On-Campus · JKLU" in matches.html |

## Non-negotiable rules
1. Agentic + recovery: every intelligent workflow runs plan → act → validate → recover.
   An LLM failure (timeout > 6 s, invalid JSON, schema failure, grounding failure, circuit
   open, provider = mock/offline) falls back automatically to a deterministic static algorithm.
2. Zero hallucination: tiers come ONLY from SOP-01 (Standard < 8.0, Dream 8.0–14.99,
   Super-Dream ≥ 15.0 LPA, using the TOP of the CTC band). Backlog defaults come ONLY from
   SOP-03 (Super-Dream 0, Dream ≤ 1, Standard → null + flag "departmental clearance").
   No SOP defines a default CGPA, so the default is null (no cutoff), flagged. Skills must be
   canonical taxonomy ids AND appear in the JD text. Branches go through normalize_branch.
   Internships: policy_tier = null + flag (SOP-01 classifies CTC offers only). Monthly
   stipends are NOT converted into a tier.
3. HITL: drive creation or JNF submission never changes Job/JobMatch rows. It creates a
   CampusDrive(PENDING_TPO_APPROVAL) + ApprovalRequest(kind="campus_drive_publish").
   Only TPO approval publishes and triggers cohort re-matching. faculty_mentor cannot approve.
4. Offline-first + RBAC: no CDN, no Google Auth, no runtime downloads. Every route is guarded
   except an explicit public allowlist.
5. Traceability: every step writes log_audit(...) with detail.trace_id so one JD's whole
   life can be filtered in /admin/audit.

## Shared contract (all workers code against this)

### Status and tier enums
approval_status: DRAFT | PENDING_TPO_APPROVAL | PUBLISHED | REJECTED
policy_tier (id): super_dream | dream | standard | null   (labels: "Super Dream", "Dream", "Standard")
experience_level: Fresher | Internship
work_mode: On-site | Remote | Hybrid

### CampusDriveSchema (JSON)
{
  "id": "cd-<hex8>", "trace_id": "tr-<hex8>",
  "company": "str", "role_title": "str", "location": "str", "work_mode": "On-site|Remote|Hybrid",
  "ctc_lpa_min": "float|null", "ctc_lpa_max": "float|null", "salary_disclosed": "bool",
  "stipend_inr_per_month": "int|null",
  "policy_tier": "super_dream|dream|standard|null", "policy_tier_source": "SOP-01|null",
  "experience_level": "Fresher|Internship",
  "source_portal": "On-Campus (Local)", "source_badge": "On-Campus · JKLU",
  "origin": "tpo_smart_jd|recruiter_jnf",
  "eligibility": { "min_cgpa": "float|null", "allowed_branches": ["CSE", "..."],
                   "max_active_backlogs": "int|null", "graduation_year": "int|null" },
  "required_skills": ["python", "fastapi", "postgresql"],   // canonical taxonomy ids only
  "preferred_skills": ["docker"],
  "unmapped_skill_mentions": ["kotlin multiplatform"],        // shown to TPO, never matched
  "raw_jd_text": "str",
  "parser_metadata": {
    "engine_used": "llm_structured|static_fallback|hybrid_merge",
    "fallback_reason": "null|offline_mode|forced_static|llm_timeout|llm_invalid_json|llm_schema_validation|llm_grounding_failed|circuit_open|llm_error",
    "attempts": [{ "engine": "llm|static", "status": "ok|failed", "latency_ms": 0, "error": "str|null" }],
    "confidence_score": 0.0,
    "field_confidence": { "company": 0.9 },
    "evidence": { "ctc_lpa_max": "CTC: 12 LPA", "min_cgpa": "CGPA >= 7.0" },
    "missing_fields_flagged": ["graduation_year"],
    "defaults_applied": [{ "field": "max_active_backlogs", "value": 1, "source": "SOP-03" }]
  },
  "approval_status": "...", "approval_id": "apr-<hex8>|null", "job_id": "job-campus-<hex8>|null",
  "created_by": "user_id|external_recruiter", "reviewed_by": "str|null"
}

### Endpoints (prefix /api/campus-drives)
| Method & path | Role | Behaviour |
|---|---|---|
| POST /parse-jd (multipart file .pdf/.docx/.txt OR form `text`; `force_parser=auto\|llm\|static`; `simulate_failure=none\|timeout\|invalid_json`, demo mode + tpo_admin only) | tpo_admin | Returns a pre-filled CampusDriveSchema (status DRAFT, not persisted) + trace_id. |
| POST / (body: CampusDriveSchema + `publish_now: bool` + `confirm: bool`) | tpo_admin | Persists CampusDrive + ApprovalRequest. If publish_now && confirm, auto-approves in the same transaction (reviewed_by = TPO, note "direct TPO sign-off") and goes through the SAME publish handler. |
| GET / ?status= , GET /{id} | faculty_mentor+ | List / detail. |
| PATCH /{id} | tpo_admin | Edit while PENDING; re-validate; audit field-level diff. |
| POST /{id}/approve, POST /{id}/reject (note) | tpo_admin | Delegate to the existing approvals approve/reject so /api/approvals/{apr}/approve and this route are equivalent and idempotent. |
| POST /jnf-links (expires_hours) | tpo_admin | Returns { token, url: "/jnf/<token>" }. Token = HMAC-signed with exp (reuse encode_session style). |
| GET /jnf/{token} (HTML page, outside /api) | PUBLIC (valid token) | Recruiter intake form. |
| POST /public-jnf (token + file or fields) | PUBLIC (valid token, rate-limited) | Parses with auto mode, saves PENDING_TPO_APPROVAL, origin=recruiter_jnf. Never publishes. |

### Publish handler (execute_approved branch for kind "campus_drive_publish")
1. Validate again (grounding + SOP checks). 2. Upsert Job(id="job-campus-<hex8>", source="local",
salary_lpa=[min,max] or [0,0], salary_disclosed, min_gpa (null→0.0), allowed_branches,
max_backlogs, required/preferred skills, description=raw_jd_text). 3. Embed → vector_store.upsert
("job", ...) + ctx.job_vectors. 4. Submit cohort re-match to ctx.task_queue (every student:
_rank_and_persist-equivalent; compute eligibility + missing required skills). 5. Store the
cohort summary {students_total, eligible, ineligible_by_reason, top_gaps[skill→count],
top_matches[5]} on CampusDrive and in audit. Do NOT auto-create UpskillPlans.

### Audit action names (detail always includes trace_id)
campus_drive.jd_extracted · campus_drive.parse_attempt · campus_drive.fallback_triggered ·
campus_drive.validated · campus_drive.eligibility_inferred · campus_drive.submitted ·
campus_drive.edited · approval.approved / approval.rejected (existing) · campus_drive.published ·
campus_drive.cohort_rematched · jnf.link_created · jnf.submitted · auth.login · auth.logout ·
auth.role_switched · auth.denied

### Public route allowlist (the only unauthenticated paths)
/login, /auth/*, /static/*, /jnf/*, /api/campus-drives/public-jnf, /healthz, /readyz,
/manifest.webmanifest, /sw.js

## File ownership (edit ONLY your files; ask the owner on the board otherwise)
| Worker | Owns |
|---|---|
| drive-backend | app/campus/** (new), app/api/routes/campus_drives.py (new), app/api/main.py (router include only), app/models.py (append only), app/db.py (_ensure_columns), app/api/routes/approvals.py (new kind branch only), app/matching/engine.py (backlog check in job_eligibility), app/llm/mock.py (only if needed), tests/test_campus_drive_*.py, tests/fixtures/jd/** |
| drive-ui | app/templates/student/matches.html, app/templates/campus/** (new), app/templates/admin/approvals.html, app/templates/admin/audit.html (badge colours), app/static/campus_drive.js (new) |
| platform | app/security/rbac.py, app/api/routes/auth.py, app/templates/auth/login.html, app/templates/base.html, app/config.py, app/seed.py (persona only), app/embeddings/embedder.py, app/api/__main__.py, app/static/vendor/** (new), app/static/responsive.css (new), data/personas.json (new), tests/test_platform_*.py |
| packaging | desktop/** (new), mobile/** (new), app/static/manifest.webmanifest, app/static/sw.js, app/static/icons/**, .dockerignore, README "Desktop & Mobile" section, scripts/serve_lan.ps1 |
| verifier | read-only on app code; owns tests/e2e/** and docs/DEMO_SCRIPT.md |

Each author runs `uv run pytest -q` + `uv run ruff check .` before saying "done". Post on the
board: files changed, tests added, test count, and any contract deviation (which needs coordinator OK).
````

---

### Worker brief: `drive-backend` (P0)

````markdown
You own the campus-drive backend pipeline. Read MASTER CONTEXT first.

Build app/campus/:
1. ingest.py: extract_jd_text(bytes, suffix) wraps parse_resume_bytes (pdf/docx/txt) +
   raw text. Normalise whitespace, keep bullets (•, -, *, 1.) as "- " lines, strip email
   headers/signatures heuristically. Max 2 MB; reject other types with 415.
2. static_parser.py: pure-Python and deterministic (same input → same output; unit-test it):
   - CTC: "₹12 LPA", "12.5 Lakhs per annum", "CTC: 8-10 LPA", "INR 12,00,000", "12L";
     "Stipend: 40,000/month" → stipend_inr_per_month (NOT a CTC); "not disclosed / as per
     industry standards / competitive" → salary_disclosed=false. Record an evidence snippet.
   - Eligibility: "CGPA >= 7.0", "7 CGPA and above", "minimum 7.5 GPA", "70% throughout"
     (percentage: keep as evidence + flag; do NOT convert to CGPA unless a conversion SOP
     exists). Backlogs: "no active backlogs" → 0, "max 1 backlog" → 1. Branches: split
     "B.Tech CSE/AI/ECE" etc. → normalize_branch; "all branches" → [].
     Graduation year: "2027 batch", "graduating in 2027".
   - Skills: reuse app/skills/extract.py + taxonomy aliases (word-boundary). Anything else
     that looks like tech goes into unmapped_skill_mentions. Split required vs preferred by
     section headers ("Must have", "Good to have", "Preferred", "Bonus").
   - Company / role / location / work_mode: "Company:", "Role:", "Position:", "Location:",
     first-heading heuristics, Remote / Hybrid / WFH / On-site keywords.
   - Tier: call policy.tier_for(ctc_lpa_max) that reads SOP-01 parameters from
     data/placement_sops.json (keep engine.job_tier consistent; add a test asserting both agree).
   - Defaults (policy.py): only from SOP-03 or data/campus_drive_defaults.json (new,
     clearly labelled institutional config, e.g. graduation_year). Every default goes into
     defaults_applied + missing_fields_flagged.
   - Confidence: per-field (regex hit with evidence = 0.9, heuristic = 0.6, default = 0.3),
     overall = weighted mean.
3. llm_parser.py: one strict-JSON prompt (taxonomy ids + SOP tier rules in the prompt),
   ctx.llm.complete(..., json_mode=True) in a worker thread with a hard 6 s timeout,
   Pydantic validation, then a GROUNDING VALIDATOR: drop skills not in the taxonomy or not
   in the text, drop a CTC/CGPA whose number doesn't appear in the text, recompute the tier
   from SOP-01 (never trust the LLM tier).
4. agent.py::parse_campus_jd(text, mode, ctx): plan = [extract → llm (if allowed) →
   validate → static → merge]. auto: provider "mock" or circuit open → static with
   fallback_reason offline_mode/circuit_open. If the LLM succeeds, run static as well and
   do a field-level hybrid merge (LLM value wins only if grounded; otherwise the static
   value; conflicts go into missing_fields_flagged as "conflict:<field>").
   simulate_failure injects timeout/invalid_json for the live demo. Log every attempt.
5. Models: CampusDrive table (JSON columns for eligibility, skills, parser_metadata,
   cohort_summary). Job gets max_backlogs: int | None (+ _ensure_columns);
   job_eligibility checks student.backlogs against it.
6. Routes + publish handler exactly as in the contract (kind "campus_drive_publish" in
   execute_approved; faculty_mentor blocked). Re-match runs on the task queue and the task
   id is returned so the UI can poll /api/tasks/{id}.
7. tests/fixtures/jd/: ≥ 8 JDs: super-dream PDF, dream DOCX, standard email text, internship
   with stipend, salary undisclosed, messy forwarded email, branch-restricted, one JD with a
   fake skill ("Blockchain-Quantum-Rust++") that must NOT be extracted.
   Tests: tier boundaries 7.99 / 8.0 / 14.99 / 15.0; determinism; LLM timeout / invalid JSON
   / ungrounded skill → fallback with the correct reason; HITL (no Job row before approval);
   student and faculty get 403 on approve/publish; public-jnf with a bad/expired token → 401;
   audit rows share the trace_id.
Done when: the endpoints work against Postgres and SQLite, the full suite is green, and you've
posted a curl walkthrough on the board.
````

### Worker brief: `drive-ui` (P1, can start right away against the contract)

````markdown
You own the TPO Smart JD + JNF + approvals UI in the existing Jinja/Tailwind/htmx style
(look at matches.html action bar + modal patterns; reuse colours #1D5C42 etc. and the
_skill_chips.html look). Read MASTER CONTEXT first. Until drive-backend lands, use a JS
mock that returns a fixture CampusDriveSchema.

1. matches.html action bar (next to Fetch Live Jobs / Remove All Live Jobs / Re-run
   matching): "+ Add Campus Drive (Smart JD)" and "Share Recruiter JNF Link". Rendered
   enabled for role == "tpo_admin"; for other roles shown disabled with a tooltip saying
   "TPO Admin role required". Put the modal markup in templates/campus/_smart_jd_modal.html
   and include it; put the JS in static/campus_drive.js (keep matches.html edits small;
   DO NOT rename existing ids/classes used for the layout).
2. Smart JD modal: tabs Upload (drag-drop .pdf/.docx/.txt) | Paste email. Parser mode
   segmented control: Auto (LLM + Static Fallback) / Force Static / Force LLM; in demo
   mode a small "Simulate LLM failure" select (none/timeout/invalid JSON).
   "Parse & Auto-Fill" → POST /api/campus-drives/parse-jd.
   Agent Explainability Banner: engine chip (LLM / Static Deterministic / Hybrid merge),
   fallback reason in plain English, confidence bar, attempts timeline (engine, ms, status),
   "defaults applied (SOP-03 / config)" list, conflicts. Hover on a field shows its evidence
   snippet from the JD.
   Editable form: company, role, location, work mode, CTC min/max + "Undisclosed" checkbox,
   stipend (internship), policy badge recomputed live in JS with the SOP-01 thresholds
   (8.0 / 15.0, using ctc max; null for internships/undisclosed, labelled "Not classified"),
   min CGPA, allowed branch multi-chips (CSE IT ECE EE ME CE AI&DS CHE BT), max backlogs,
   graduation year, required/preferred skill chips (add via autocomplete over GET taxonomy;
   if no endpoint exists, ask drive-backend to expose GET /api/campus-drives/taxonomy),
   unmapped mentions shown greyed out.
   Buttons: "Submit to Approval Queue" and "Approve & Publish Drive Immediately" (a confirm
   dialog that sends confirm: true). After publishing, show a cohort re-match progress toast
   (poll /api/tasks/{id}) and then the summary (eligible N, top gaps).
3. JNF: a "Share Recruiter JNF Link" drawer calls POST /jnf-links and shows a copyable full
   URL (use location.origin; also show a hint for the LAN IP for phone demos) + a QR code
   only if it can be generated offline (vendor a tiny QR lib in static/vendor or skip it).
   templates/campus/jnf_public.html: a standalone recruiter page (does NOT extend base.html
   nav; no role switcher) with upload/paste + fields, submit → POST /public-jnf, success
   screen "Submitted for TPO review".
4. admin/approvals.html: render kind "campus_drive_publish" as a rich card (company, role,
   tier badge, CTC, eligibility, skill chips, origin badge "Recruiter JNF" / "TPO Smart JD",
   parser engine chip) with Edit (opens the same modal prefilled → PATCH), Approve
   (Tier-2, TPO only), Reject (note). audit.html: badge colour for campus_drive.* and jnf.*.
Done when: the full flow works by mouse in the browser for all 3 roles, there are no console
errors, and it is usable at 390 px width.
````

### Worker brief: `platform` (P0: offline + RBAC holes; P1: responsive)

````markdown
You own auth hardening, offline-readiness and responsive layout. Read MASTER CONTEXT first.
Auth ALREADY exists (signed HMAC-SHA256 session tokens with exp, /login, /auth/login,
/auth/logout, /api/auth/switch-role, role switchers in base.html). Do not rewrite it; harden it.

P0 offline:
1. Download and vendor into app/static/vendor/: the Tailwind Play CDN script, htmx 1.9.12, the
   jsdelivr dependency used in templates, and the Google Fonts (woff2 + local @font-face css).
   Replace every external <script>/<link> in templates. Add tests/test_platform_offline.py
   that fails if any template references http(s):// for script/link/font assets.
2. Embedder: if LocalEmbedder fails to load (no cached model / offline), fall back to
   HashingEmbedder with a logged warning; support HF_HUB_OFFLINE=1. Add a
   `uv run python -m app.cli warm-cache` (or a README step) that pre-downloads the model.
P0 RBAC:
3. AUTH_MODE=strict (set in .env for the demo; tests stay on demo): ignore X-Demo-Role /
   X-Demo-Student-ID headers; no session → HTML pages 303 to /login and /api/* 401, except
   the public allowlist in MASTER CONTEXT; DEFAULT_ROLE open mode is disabled. 403s write
   auth.denied audit rows (rate-limit duplicate audit spam).
4. Personas (data/personas.json; PINs stored as pbkdf2 hashes, demo PIN documented in the README):
   rohit.shetty@jklu.edu.in → student (bind to a seeded "Rohit Shetty" student: add him to
   the demo cohort in seed.py, idempotent), faculty.mentor@jklu.edu.in → faculty_mentor,
   tpo.admin@jklu.edu.in → tpo_admin. Keep the existing roll-number student login working.
   POST /auth/login validates email domain + PIN. Login/logout/switch-role write audit rows.
   switch-role stays available for judges in strict mode only when DEMO_ROLE_SWITCH=true,
   and is audited (from → to, actor).
5. login.html: "JKLU CareerOS Offline SSO" card with email + PIN and "1-Click Judge Demo
   Personas" (3 cards, one per role) that sign in directly; Sign Out in base.html posts to
   /auth/logout; the top-bar role switcher shows the signed-in identity.
P1 responsive:
6. app/static/responsive.css (included by base.html): at ≤ 768 px the left sidebar becomes a
   slide-out drawer behind a hamburger in the top bar (focus trap, Esc closes, overlay);
   the matches 2-column list/detail split stacks (detail below the list or a full-screen
   sheet with a back button), using the existing ids/classes in matches.html (coordinate with
   drive-ui; do not edit matches.html yourself). Touch targets ≥ 44 px. Desktop unchanged.
7. CORS: only if packaging reports a real need; then use an explicit allowlist
   (http://127.0.0.1:8010, http://localhost:8010, capacitor://localhost, http://localhost)
   with allow_credentials=True. Never "*" with credentials.
Done when: the app renders fully with Wi-Fi off, the header-spoof curl gets 401/403 in
strict mode, and the suite is green in demo mode.
````

### Worker brief: `packaging` (P2/P3, start after platform's vendoring lands)

````markdown
You own the desktop/mobile shells. Read MASTER CONTEXT first. The UI is server-rendered by
FastAPI, so the shells LOAD THE SERVER URL; there is no frontend build to bundle and no
API-base-URL refactor. Keep all Node tooling inside desktop/ and mobile/ (no root
package.json; add both to .dockerignore).

1. PWA (cheap win): app/static/manifest.webmanifest (name "JKLU CareerOS", icons 192/512,
   theme #1D5C42) + sw.js that caches the static shell/vendor assets; register it in base.html
   via a 3-line snippet you hand to platform. Desktop Chrome on localhost can then "Install app".
2. desktop/ (Electron): main.js opens a 1440x900 window. On start it probes
   http://127.0.0.1:8010/healthz; if that is down it spawns
   `uv run uvicorn app.api.__main__:app --port 8010` from the repo root, polls /readyz (≤ 30 s,
   splash screen with logs) and loads the URL. The child process is killed on quit. A
   Settings menu → "Server Connection" changes the URL (persisted). Scripts: desktop:dev,
   desktop:build (electron-builder, Windows portable/nsis).
3. mobile/ (Capacitor Android): appId in.edu.jklu.careeros, appName "JKLU CareerOS",
   webDir "www". www/index.html is a "Server Connection" screen: enter
   http://192.168.x.x:8010, a Test button hits /healthz, save it in localStorage, then
   navigate to it. Configure server.cleartext, android.allowMixedContent and
   server.allowNavigation so the LAN URL opens INSIDE the webview (check against the
   Capacitor docs; if wildcard IP navigation doesn't work, fall back to build-time server.url
   from an env var and document it). Commit the generated android/ project.
   If JDK 17 / the Android SDK are missing on this machine, deliver the config +
   README steps and say so on the board. Don't block.
4. scripts/serve_lan.ps1: runs uvicorn with --host 0.0.0.0 --port 8010, prints the LAN IPs
   and the Windows Firewall rule command. README "Desktop & Mobile demo" section.
Done when: the Electron window opens the app on a clean start, and either the APK installs and
connects over a hotspot or the documented blocker is posted.
````

### Worker brief: `verifier` (starts immediately)

````markdown
You are the independent verifier. You do not write app code. Read MASTER CONTEXT first.
1. Baseline now: `uv run pytest -q` + `uv run ruff check .`; post the pass count.
2. After each author says "done": rerun the suite, review the diff against the rules
   (grounding: no invented tiers/cutoffs/skills; HITL: no Job/JobMatch writes before
   approval; RBAC on every new route; an audit row per step with trace_id; file ownership
   respected). Post concrete defects with file:line to the owner.
3. tests/e2e/: a scripted end-to-end check against a running server: login per persona →
   parse each tests/fixtures/jd file in auto/static/llm+simulate_failure → assert SOP-01
   tiers, no non-taxonomy skills, the fake skill is absent → submit → as student and faculty,
   approve gets 403 → as TPO approve → Job(source="local") exists, the task completes,
   matches include the drive for eligible students only → the audit trail shows the full
   trace. Also: X-Demo-Role spoof is rejected in strict mode; an expired JNF token gets 401;
   with LLM_PROVIDER=openai and a bogus OPENAI_BASE_URL the parser still returns
   static_fallback in < 8 s.
4. Offline check: no external asset URLs in rendered pages; the app boots with
   HF_HUB_OFFLINE=1 and no network.
5. Write docs/DEMO_SCRIPT.md: a 4-minute judge walkthrough mapped to the rubric (agentic +
   recovery via simulated LLM failure, SOP grounding, HITL approve, RBAC role switch,
   audit trace, mobile/desktop).
Report PASS/FAIL per area to the coordinator; nothing is "done" without your PASS.
````

---

## Part C: Suggested swarm setup

| Worker | Workspace | Starts | Depends on |
|---|---|---|---|
| `drive-backend` | `share` | now | — |
| `platform` | `share` | now | — |
| `drive-ui` | `share` | now (with a mock) | drive-backend contract |
| `verifier` | `inherit` | now (baseline) | each author's "done" |
| `packaging` | `share` | after platform's vendoring | platform |

The coordinator merges branches in this order: platform (vendoring + RBAC) → drive-backend → drive-ui → packaging. File ownership is disjoint, so the merges should be clean.
