# JKLU CareerOS — 4-Minute Buildathon Judge Walkthrough

> **Demo Goal:** Demonstrate an offline-first, zero-hallucination, HITL-governed Campus Drive & Placement Agent (PS-06) running end-to-end across Web, Desktop (Electron / PWA), and Mobile (Capacitor).

---

## Pre-Flight Checklist (30 seconds before judges arrive)

1. Start the server in strict RBAC + offline-ready mode:
   ```powershell
   $env:AUTH_MODE = "strict"
   $env:HF_HUB_OFFLINE = "1"
   uv run uvicorn app.api.__main__:app --host 0.0.0.0 --port 8010
   ```
   *(Or run `./scripts/serve_lan.ps1` to display LAN / JNF / Mobile URLs.)*
2. Open `http://127.0.0.1:8010/login` in Chrome / Electron.
3. Have a terminal ready for the RBAC spoof check and a sample JD (`.pdf` or email text containing a fake skill `"Blockchain-Quantum-Rust++"`).

---

## Step 1 — Offline-First & Strict RBAC (0:00 – 0:35)

**Rubric Focus:** Security, Offline Resilience, Zero External Dependencies.

1. **Block Header Spoofing:** In the terminal, attempt to bypass auth using the legacy demo header:
   ```bash
   curl -i -H "X-Demo-Role: tpo_admin" http://127.0.0.1:8010/api/campus-drives
   ```
   - **Observe:** `401 Unauthorized` (or `403`). In `AUTH_MODE=strict`, unsigned headers and missing session cookies are rejected.
2. **1-Click Judge Personas:** On `/login`, click **Sign in as TPO Admin** (`tpo.admin@jklu.edu.in`) with Demo PIN `2026`.
3. **Offline Proof:** Open DevTools → Network tab (or disable Wi-Fi). Every CSS, JS, and font asset loads from `/static/vendor/*` with zero external CDN calls (`cdn.tailwindcss.com`, `unpkg.com`, Google Fonts removed).

---

## Step 2 — TPO Smart JD Ingestion & Agentic Recovery (0:35 – 1:25)

**Rubric Focus:** Agentic `plan → act → validate → recover` Workflow & Explainability.

1. Navigate to the Matches / Campus Drives workspace (`/`) and click **`+ Add Campus Drive (Smart JD)`**.
2. Upload a campus JD (or paste a messy forwarded recruiter email).
3. In the parser controls, select **`Simulate LLM Failure: Timeout`** and click **`Parse & Auto-Fill`**.
4. **Observe (< 6 seconds):**
   - The LLM attempt times out cleanly; the agent automatically falls back to `static_fallback` (`fallback_reason: llm_timeout`).
   - The **Agent Explainability Banner** displays:
     - Engine chip (`Static Deterministic`) + fallback reason in plain English.
     - Confidence score bar and attempts timeline (`llm: failed` → `static: ok` with latency in ms).
     - **Defaults Applied (SOP-03 / config):** e.g., `max_active_backlogs` inferred from SOP-03 tier rules, `min_cgpa: null` flagged as missing rather than invented.
   - Hover over extracted fields (CTC, CGPA, skills) to inspect **exact evidence snippets** from the raw JD text.

---

## Step 3 — Zero-Hallucination Grounding (1:25 – 2:00)

**Rubric Focus:** Policy Grounding (SOP-01 / SOP-03) & Taxonomy Enforcement.

1. **SOP-01 Tier Boundaries:**
   - Show that CTC `12.0 LPA` (or `8–14.99 LPA`) maps to **`Dream`** (`SOP-01`), `< 8.0 LPA` maps to **`Standard`**, and `>= 15.0 LPA` maps to **`Super Dream`** (always using the **top** of the CTC band).
   - Toggle **Internship** with a monthly stipend (e.g., `₹40,000/month`): `policy_tier` becomes `null` (**Not classified**) — stipends are never hallucinated into annual LPA tiers.
2. **Skill Grounding:**
   - Point out a non-taxonomy / fake skill in the JD (e.g., `"Blockchain-Quantum-Rust++"`).
   - **Observe:** It is isolated in **`unmapped_skill_mentions`** for TPO awareness and **excluded** from `required_skills` / `preferred_skills` (which only accept canonical IDs from the 124-skill `data/skills.json` taxonomy).
3. Click **`Submit for TPO Approval`** (or **`Save & Publish`** if demonstrating direct TPO sign-off; for the HITL demo, submit to `PENDING_TPO_APPROVAL`).

---

## Step 4 — Public Recruiter JNF Flow (2:00 – 2:35)

**Rubric Focus:** External Intake Without Compromising Governance.

1. Click **`Share Recruiter JNF Link`** in the action bar to generate an HMAC-signed, time-limited link (`/jnf/<token>`).
2. Open `/jnf/<token>` in an incognito window (no login required; invalid/expired tokens return `401`).
3. Submit a recruiter JD via the public JNF form.
4. **Observe:** The submission lands in `PENDING_TPO_APPROVAL` (`origin: recruiter_jnf`). No `Job` or `JobMatch` row is created yet — recruiters cannot mutate the student matching pool directly.

---

## Step 5 — 2-Tier HITL Governance & Cohort Re-Matching (2:35 – 3:15)

**Rubric Focus:** Role Separation (`faculty_mentor` vs `tpo_admin`) & Automated Cohort Impact.

1. Switch role / sign in as **Faculty Mentor** (`faculty.mentor@jklu.edu.in`, PIN `2026`) and open `/admin/approvals`.
2. Attempt to approve the pending `campus_drive_publish` request:
   - **Observe:** Blocked with **`403 Forbidden`** — `faculty_mentor` can view drives but cannot publish campus drives.
3. Switch back to **TPO Admin** (`tpo.admin@jklu.edu.in`) and click **Approve & Publish**.
4. **Observe:**
   - A `Job(id="job-campus-...", source="local")` is created with badge **`On-Campus · JKLU`**.
   - Background cohort re-matching runs via `task_queue`, computing student eligibility (GPA + branch + SOP-03 `max_backlogs`) and missing required skills.
   - The Campus Drive detail / approval card summarizes **Eligible Students**, **Ineligible Reasons**, and **Top Skill Gaps**.

---

## Step 6 — End-to-End Traceability (3:15 – 3:40)

**Rubric Focus:** Immutable Audit Trail.

1. Open `/admin/audit` and filter by the drive's `trace_id` (`tr-...`).
2. **Observe:** Every lifecycle step is linked under one `trace_id` in the immutable `AuditLog` table:
   - `campus_drive.jd_extracted` → `campus_drive.parse_attempt` → `campus_drive.fallback_triggered` → `campus_drive.validated` → `campus_drive.submitted` → `approval.approved` → `campus_drive.published` → `campus_drive.cohort_rematched`.

---

## Step 7 — Desktop (Electron / PWA) & Mobile (Capacitor) Showcase (3:40 – 4:00)

**Rubric Focus:** Multi-Platform Readiness.

1. **PWA / Desktop:** Show `/manifest.webmanifest` & `/sw.js` installability, or open the Electron wrapper (`desktop/`) pointing to the local server.
2. **Mobile (Capacitor / 390px):** Resize browser to `390px` (or show the Android Capacitor app connected via `adb reverse tcp:8010 tcp:8010`) demonstrating the responsive slide-over drawer and touch-friendly Smart JD & Approvals cards.
