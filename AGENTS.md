# JKLU CareerOS Project Rules

## Keep the Know More Page (`/know-more`) Synchronized

Whenever you add, modify, or remove any feature, UI element, or backend behavior in this repository, you MUST update the corresponding documentation in the `/know-more` page (`tpo_admin`-only) before finishing the task:

1. **What triggers an update**:
   - Any added, renamed, or removed FastAPI route (`app/api/routes/*.py`).
   - Any added, modified, or removed UI section, page, modal, drawer, button, toggle, filter, dropdown, or keyboard shortcut (`app/templates/**/*.html`, `app/static/*.js`).
   - Any change to RBAC permissions, auth modes, process workflows, AI pipelines, scoring formulas, scraper providers, or CLI commands (`app/**/*.py`).

2. **Where to update**:
   - Update the matching domain partial under `app/templates/know_more/`:
     - `_shell_auth_agent.html` (Shell, Nav, Theme, Command Bar, Auth/RBAC, Agent Brain, Security, PWA/Desktop/Mobile)
     - `_cohort_profile.html` (Home Cohort Directory, AI Resume Onboarding, Student Profile, Skill Chips, X-Ray, Bullet Improver)
     - `_matches_scraper_drives.html` (Matches, Filters/Toggles, Applications, Live Scraper, Smart JD, Campus Drives, Public JNF)
     - `_plan_progress_interview.html` (Upskill Planner, Quiz/Swap/Waiver/Simulate Modals, Progress Analytics, Mock Interviews, STT/TTS, Anti-Bluff Rubric)
     - `_copilot_admin_ops.html` (Copilot RAG, Admin Analytics, Drive Simulator, Approvals, Policies, Audit, CLI/LLM/Embeddings/Workers)
   - Update `app/templates/know_more.html` if top-level sections, role filters, or architecture pipeline summaries change.
   - Update `tests/test_know_more.py` to cover newly added controls, routes, or behaviors (and preserve `tpo_admin`-only RBAC enforcement on `/know-more`).

3. **Writing & UI Style (User-First, No Code Dumps)**:
   - Write in **plain, everyday layman language** for TPO Admins and end users — NEVER dump raw URL paths (`/api/...`, `/login`), CSS IDs (`#btn-id`), or Python/JS function names in the visible guide text.
   - Show each control using a **visual UI button/toggle/pill preview** matching its **exact visible label** in the app, paired with **Where it is located**, **Who can use it**, and **What it does** (100% accurate to live code behavior).
   - Keep each app screen cleanly separated into its own visual section.

4. **Verification**:
   - Ensure every documented button, toggle, screen, and workflow accurately reflects the live code (never leave stale descriptions or unverified claims).
   - Run `uv run pytest tests/test_know_more.py -q` alongside the relevant feature tests.
