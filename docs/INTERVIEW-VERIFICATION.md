# Mock interview skill verification

New interviews snapshot every non-verified `StudentSkill` in
`Interview.feedback._verification`. Unverified and needs-reverification claims
come before other claimed skills. The snapshot includes canonical skill IDs,
claimed levels, sanitized evidence and job/resume context; later profile edits
cannot remove an interview target. There is no database migration.

Each target receives a primary assessment question before general practice.
The base question goal is `max(8, number of targets)`. A contextual follow-up
may add one turn per primary question, without consuming the base goal. Sessions
therefore have a finite goal, but no eight-turn cutoff that drops claims.
Follow-ups are enabled by default and can be disabled explicitly. A non-answer
still receives a low assessment score and moves to the next skill.

With a live provider, `TASK: interview_question` authors a new question and
factual grading outline from the exact target, evidence, claimed depth, job
context and recent answers. No bank question or reference is supplied in this
prompt. Both primary questions and follow-ups use this path. The optional project
deep-dive setting emphasizes architecture, failure modes and measurements.

The provider telemetry adapter exposes `complete_live`, which refuses mock
circuit-breaker substitutions. Failed or malformed question generation enters
explicit offline practice for that question and records `generation_error`.
Later turns try the live provider again (subject to its circuit breaker).
Question sources are `live_ai`, `offline_bank`, `offline_resume_template`, or
`offline_template`. Mock providers and missing providers use offline material.
The UI, API and persisted transcript expose these sources.

Offline selection restricts the bank to the exact target skill, then chooses
role/difficulty and avoids used IDs. Unsupported skills or exhausted references
receive an explicitly limited practice template; they are never silently skipped
or assessed under an unrelated skill. The supplemental bank contains mechanism,
debugging and design scenarios at three difficulty levels for all 136 canonical
skills. Run `python scripts/expand_interview_bank.py` to rebuild these scenarios
from the original factual references without altering the original questions.

Question generation and grading have separate provenance. Live grading bypasses
mock substitution too. If grading fails, a deterministic practice score and
`offline_fallback` source are recorded. Such scores, and unsupported-skill
templates without factual references, cannot promote or demote a skill. Mock
practice retains the existing offline scoring and update behavior; it is not
evidence of a live model's assessment quality.

Coverage reports distinguish `pending`, `asked`, and `assessed`. Assessed means an
answer was evaluated, not that the skill passed. Skill updates still require the
existing score thresholds: normally 80 to verify/promote; 70 for the existing
reverification/basic-level cases; below 45 flags unverified; remaining assessed
scores flag needs-reverification. Only skills actually scored with an eligible
reference/evaluator are updated on finalization. This is an interview assessment,
not an independently executed coding test or proof of project authorship.

`POST /api/interviews/{id}/finish` explicitly ends early, preserving queued and
unanswered claims with incomplete coverage and `completion_reason=ended_early`.
Unanswered skills keep their profile verification state. Early exits receive a
distinct progress event and audit action. Merely leaving the page keeps the
session resumable and does not finalize it.
Early or limited attempts are excluded from completed-interview readiness gates,
cohort aggregates and completed-assessment progress statistics.

Unfinished historical interviews receive a verification snapshot when resumed.
Their already displayed question is graded against its original reference and
labeled `legacy_question`; subsequent questions follow complete coverage and live
authoring rules. Completed historical transcripts remain unchanged. Direct
`InterviewSession` callers without a plan retain legacy behavior for seed data.

Regression coverage is in `tests/test_interview_complete_coverage.py`, alongside
the existing interview, rubric, strict-scoring and intelligence suites. Tests use
an injected live-provider double and mock/offline paths; live smoke checks are
separate and do not write student records.

## Implementation files

- Session flow and authoring: `app/interview/engine.py`,
  `app/interview/generation.py`, `app/interview/status.py`,
  `app/api/routes/interviews.py`, `app/llm/telemetry.py`.
- UI and history: `app/static/app.js`, `app/templates/student/interview.html`,
  `app/templates/student/interviews.html`, `app/api/routes/pages.py`.
- Completed-assessment consumers: `app/rag/policy_rag.py`, `app/agent/tools.py`,
  `app/analytics/cohort.py`, `app/analytics/student_progress.py`,
  `app/plan/solver.py`.
- Offline content: `data/questions.json`, `scripts/expand_interview_bank.py`.
- Regression tests: `tests/test_interview_complete_coverage.py`,
  `tests/test_interview.py`, `tests/test_interview_rubric.py`,
  `tests/test_interview_skill_verification.py`,
  `tests/test_interview_strict_scoring_and_print.py`,
  `tests/test_m18_interview_intelligence.py`.
- Documentation: `README.md`, `docs/INTERVIEW-VERIFICATION.md`.

## Validation

The final targeted run passed 193 tests across the interview suites, LLM and
observability suites, student progress, policies, cohort analytics, planning and
agent grounding. Ruff, JavaScript syntax and whitespace checks passed.

A headless Chromium smoke check exercised launch defaults, coverage, offline
provenance, answer submission, early exit and persisted summary reload with no
page errors. A separate synthetic-candidate smoke check against the configured
OpenAI-compatible API successfully authored a question, graded an answer and
generated its contextual follow-up, all with `live_ai` provenance. No real
student records were used in either smoke check. This validates integration,
not the model's accuracy across every skill or the entire application test suite.
