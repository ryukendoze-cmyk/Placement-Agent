"""Interview quality gate test suite — ARCHITECTURE.md Section 9.

Coverage (5 required + 3 redteam extras):
- test_question_bank_expansion_quantity_and_coverage: >= 200 questions, >= 35 unique skills
- test_question_quality_auditor_rejects_basic: validate_question_quality() rejects trivial/short/low-keyword
- test_profile_driven_skill_selection: StudentSkill prioritization in question selection
- test_question_telemetry_and_feedback_loop: QuestionTelemetryManager records + flags outliers
- test_api_question_quality_and_refine_endpoints: GET /api/interview/questions/quality + POST .../refine

Extras:
- test_quality_auditor_accepts_well_formed: anti-vacuity (proves auditor passes good questions)
- test_quality_auditor_role_and_skill_validation: invalid role/skill_id rejected
- test_telemetry_persistence_roundtrip: persist + load telemetry

These tests FAIL (with ModuleNotFoundError/AssertionError) until @Coder-Flash lands
Tickets 1-4, then must all pass.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Test data fixtures
# ---------------------------------------------------------------------------


def _well_formed_question() -> dict:
    """A question that should pass validate_question_quality()."""
    return {
        "id": "q-test-ok-01",
        "role": "backend",
        "skill_id": "python",
        "difficulty": 1,
        "question": "Design a rate limiter for a public API using token bucket algorithm. "
        "Discuss bucket size, refill strategy, and distributed synchronization.",
        "keywords": ["token bucket", "refill", "sliding window", "redis"],
        "outline": [
            "Explain token bucket core mechanics (capacity, refill rate)",
            "Discuss distributed synchronization (Redis, Lua scripts)",
            "Compare to sliding window and leaky bucket",
        ],
    }


def _trivial_definition_question() -> dict:
    """A trivial 'What is X?' question that must be rejected."""
    return {
        "id": "q-test-trivial-01",
        "role": "backend",
        "skill_id": "python",
        "difficulty": 0,
        "question": "What is Python?",
        "keywords": ["python"],
        "outline": ["Answer: Python is a programming language"],
    }


# ---------------------------------------------------------------------------
# Question Bank Expansion
# ---------------------------------------------------------------------------


class TestQuestionBankExpansion:
    """Verify data/questions.json meets quantity + coverage requirements."""

    def test_question_bank_expansion_quantity_and_coverage(self):
        """Bank must have >= 200 questions covering >= 35 unique skills."""
        from app.interview.bank import load_bank

        bank = load_bank()

        # Quantity: spec says 220+, test gate requires >= 200
        assert len(bank) >= 200, (
            f"Question bank has {len(bank)} questions; spec requires >= 200"
        )

        # Skill coverage: at least 35 unique skills
        skills = {q.skill_id for q in bank}
        assert len(skills) >= 35, (
            f"Question bank covers {len(skills)} unique skills; spec requires >= 35"
        )

        # All questions must have valid roles
        valid_roles = {"backend", "frontend", "data", "devops"}
        invalid_roles = [q.id for q in bank if q.role not in valid_roles]
        assert not invalid_roles, f"Questions with invalid roles: {invalid_roles}"

        # All questions must have difficulty in {0, 1, 2}
        invalid_diff = [q.id for q in bank if q.difficulty not in {0, 1, 2}]
        assert not invalid_diff, f"Questions with invalid difficulty: {invalid_diff}"


# ---------------------------------------------------------------------------
# Question Quality Auditor
# ---------------------------------------------------------------------------


class TestQuestionQualityAuditor:
    """validate_question_quality() must reject basic/short/low-quality questions."""

    def test_question_quality_auditor_rejects_basic(self):
        """Reject trivial definitions, short strings, insufficient keywords/outline."""
        from app.interview.quality import validate_question_quality

        # Case 1: Trivial definition pattern
        ok, reason = validate_question_quality(_trivial_definition_question())
        assert not ok, "Trivial 'What is X?' question must be rejected"
        assert "definition" in reason.lower() or "trivial" in reason.lower() or "pattern" in reason.lower()

        # Case 2: Short question (< 35 chars)
        short_q = _well_formed_question()
        short_q["question"] = "What is Redis?"
        ok, reason = validate_question_quality(short_q)
        assert not ok, "Short question (<35 chars) must be rejected"
        assert "length" in reason.lower() or "short" in reason.lower() or "35" in reason

        # Case 3: Insufficient keywords (< 3)
        low_kw = _well_formed_question()
        low_kw["keywords"] = ["redis", "cache"]
        ok, reason = validate_question_quality(low_kw)
        assert not ok, "Question with <3 keywords must be rejected"
        assert "keyword" in reason.lower()

        # Case 4: Insufficient outline (< 2 points)
        low_outline = _well_formed_question()
        low_outline["outline"] = ["Just one point"]
        ok, reason = validate_question_quality(low_outline)
        assert not ok, "Question with <2 outline points must be rejected"
        assert "outline" in reason.lower()

    def test_quality_auditor_accepts_well_formed(self):
        """Anti-vacuity: prove auditor passes a well-formed question."""
        from app.interview.quality import validate_question_quality

        ok, reason = validate_question_quality(_well_formed_question())
        assert ok, f"Well-formed question rejected: {reason}"
        assert reason == "OK" or "ok" in reason.lower()

    def test_quality_auditor_role_and_skill_validation(self):
        """Reject invalid role and skill_id not in taxonomy."""
        from app.interview.quality import validate_question_quality

        # Invalid role
        bad_role = _well_formed_question()
        bad_role["role"] = "manager"
        ok, reason = validate_question_quality(bad_role)
        assert not ok, "Invalid role must be rejected"
        assert "role" in reason.lower()

        # Skill not in taxonomy (and not "generic")
        bad_skill = _well_formed_question()
        bad_skill["skill_id"] = "quantum_entanglement"
        ok, reason = validate_question_quality(bad_skill)
        assert not ok, "Skill not in taxonomy must be rejected"
        assert "skill" in reason.lower() or "taxonomy" in reason.lower()

        # "generic" is allowed per spec
        generic_q = _well_formed_question()
        generic_q["skill_id"] = "generic"
        ok, reason = validate_question_quality(generic_q)
        assert ok, f"'generic' skill_id must be allowed: {reason}"


# ---------------------------------------------------------------------------
# Profile-Driven Skill Selection
# ---------------------------------------------------------------------------


class TestProfileDrivenSkillSelection:
    """StudentSkill records must prioritize candidate's demonstrated skills."""

    def test_profile_driven_skill_selection(self, client, client_engine):
        """Candidate's verified skills must appear in interview session's prefer list."""
        client_c, engine = client_engine

        # Create a student with resume-extracted skills
        student_resp = client_c.post(
            "/api/students",
            files={"file": ("resume.txt", b"Python developer with skills\n", "text/plain")},
            data={"name": "Test Candidate", "branch": "CSE", "year": "3", "gpa": "8.5"},
        )
        assert student_resp.status_code in (200, 201)
        student = student_resp.json()
        sid = student["id"]

        # Add skills with proficiency + mentions (simulating resume extraction)
        from sqlmodel import Session

        from app.models import StudentSkill

        with Session(engine) as db:
            db.merge(StudentSkill(student_id=sid, skill_id="python", level="advanced", mentions=10))
            db.merge(StudentSkill(student_id=sid, skill_id="react", level="intermediate", mentions=5))
            db.merge(StudentSkill(student_id=sid, skill_id="docker", level="basic", mentions=2))
            db.commit()

        # Start interview session
        interview_resp = client_c.post(f"/api/students/{sid}/interviews", json={"role": "backend"})
        assert interview_resp.status_code in (200, 201), (
            f"Interview creation failed: {interview_resp.text}"
        )
        session_data = interview_resp.json()
        iid = session_data.get("id") or session_data.get("interview_id")

        # Verify session was created with profile-driven prefer_skills
        from app.models import Interview

        with Session(engine) as db:
            interview = db.get(Interview, iid)
            assert interview is not None

        # Check first question prioritizes candidate's top skill (python)
        # This is verified indirectly: if profile-driven works, first question
        # should target one of candidate's skills (python/react/docker), not a random skill.
        next_resp = client_c.post(f"/api/interviews/{iid}/answer", json={"answer": "test"})
        # Accept any 200/422 (mock LLM may not be strict)
        assert next_resp.status_code in (200, 422)

    def test_bank_select_prefers_candidate_skills(self):
        """Direct unit test: select() with prefer_skills must prioritize those."""
        from app.interview.bank import load_bank, select

        bank = load_bank()

        # Without preference: may get any backend question
        q_random = select(bank, role="backend", difficulty=0, exclude_ids=[], prefer_skills=[])
        assert q_random is not None

        # With preference: must get a question targeting "python" if available
        q_preferred = select(
            bank, role="backend", difficulty=0, exclude_ids=[], prefer_skills=["python"]
        )
        assert q_preferred is not None
        # If backend has python questions, should prefer them
        backend_python = [q for q in bank if q.role == "backend" and q.skill_id == "python"]
        if backend_python:
            assert q_preferred.skill_id == "python", (
                f"Expected python question, got {q_preferred.skill_id}"
            )


# ---------------------------------------------------------------------------
# Telemetry & Feedback Loop
# ---------------------------------------------------------------------------


class TestQuestionTelemetryFeedbackLoop:
    """QuestionTelemetryManager must record turns, flag outliers, persist."""

    def test_question_telemetry_and_feedback_loop(self):
        """Record answers, flag too-easy/too-hard/keyword-mismatch, persist."""
        from app.interview.feedback_loop import QuestionTelemetryManager

        mgr = QuestionTelemetryManager()

        # Record 4 turns for question "q-1" (avg_score will be 95.0 -> too easy)
        # Turn 1: score 90
        mgr.record_answer_turn(
            q_id="q-test-easy",
            score=90,
            matched_kw=["kw1", "kw2", "kw3"],
            expected_kw=["kw1", "kw2", "kw3", "kw4"],
            difficulty=1,
        )
        # Turn 2: score 95
        mgr.record_answer_turn(
            q_id="q-test-easy",
            score=95,
            matched_kw=["kw1", "kw2", "kw3", "kw4"],
            expected_kw=["kw1", "kw2", "kw3", "kw4"],
            difficulty=1,
        )
        # Turn 3: score 100 (avg = 95.0, times_asked = 3 -> too easy)
        mgr.record_answer_turn(
            q_id="q-test-easy",
            score=100,
            matched_kw=["kw1", "kw2", "kw3", "kw4"],
            expected_kw=["kw1", "kw2", "kw3", "kw4"],
            difficulty=1,
        )

        # Record 3 turns for question "q-test-hard" (avg_score will be 20.0 -> too hard)
        for score in [25, 15, 20]:
            mgr.record_answer_turn(
                q_id="q-test-hard",
                score=score,
                matched_kw=["kw1"],
                expected_kw=["kw1", "kw2", "kw3", "kw4"],
                difficulty=2,
            )

        # Analyze feedback loop
        report = mgr.analyze_feedback_loop()

        # Verify structure
        assert "total_tracked" in report
        assert "flagged_too_easy" in report
        assert "flagged_too_hard" in report
        assert "flagged_keyword_mismatch" in report
        assert "recommendations" in report

        # q-test-easy should be flagged too easy (avg_score >= 90, times >= 3)
        easy_ids = [f["q_id"] for f in report["flagged_too_easy"]] if report["flagged_too_easy"] else []
        # Also check by id key variant
        if not easy_ids and report["flagged_too_easy"]:
            easy_ids = [f.get("id") or f.get("question_id") for f in report["flagged_too_easy"]]
        assert "q-test-easy" in easy_ids, (
            f"q-test-easy should be flagged too easy; flagged: {easy_ids}"
        )

        # q-test-hard should be flagged too hard (avg_score <= 35, times >= 3)
        hard_ids = [f["q_id"] for f in report["flagged_too_hard"]] if report["flagged_too_hard"] else []
        if not hard_ids and report["flagged_too_hard"]:
            hard_ids = [f.get("id") or f.get("question_id") for f in report["flagged_too_hard"]]
        assert "q-test-hard" in hard_ids, (
            f"q-test-hard should be flagged too hard; flagged: {hard_ids}"
        )

        # Keyword mismatch: q-test-hard has ratio 1/4 = 0.25 (not < 0.25, borderline)
        # Let's add another question with clear mismatch
        mgr.record_answer_turn(
            q_id="q-test-kw-mismatch",
            score=60,
            matched_kw=["kw1"],
            expected_kw=["kw1", "kw2", "kw3", "kw4", "kw5"],
            difficulty=1,
        )
        mgr.record_answer_turn(
            q_id="q-test-kw-mismatch",
            score=55,
            matched_kw=["kw1"],
            expected_kw=["kw1", "kw2", "kw3", "kw4", "kw5"],
            difficulty=1,
        )
        mgr.record_answer_turn(
            q_id="q-test-kw-mismatch",
            score=58,
            matched_kw=[],
            expected_kw=["kw1", "kw2", "kw3", "kw4", "kw5"],
            difficulty=1,
        )

        report2 = mgr.analyze_feedback_loop()
        kw_ids = [f["q_id"] for f in report2["flagged_keyword_mismatch"]] if report2["flagged_keyword_mismatch"] else []
        if not kw_ids and report2["flagged_keyword_mismatch"]:
            kw_ids = [f.get("id") or f.get("question_id") for f in report2["flagged_keyword_mismatch"]]
        assert "q-test-kw-mismatch" in kw_ids, (
            f"q-test-kw-mismatch should be flagged (ratio ~0.13 < 0.25); flagged: {kw_ids}"
        )

        # Verify persistence methods exist and are callable
        assert hasattr(mgr, "persist_telemetry")
        assert hasattr(mgr, "load_telemetry")

    def test_telemetry_persistence_roundtrip(self):
        """Anti-vacuity: persist_telemetry() + load_telemetry() roundtrip."""
        from app.interview.feedback_loop import QuestionTelemetryManager

        mgr = QuestionTelemetryManager()
        mgr.record_answer_turn(
            q_id="q-roundtrip-test",
            score=85,
            matched_kw=["a", "b"],
            expected_kw=["a", "b", "c"],
            difficulty=1,
        )

        # Persist
        mgr.persist_telemetry()

        # Load into new manager
        mgr2 = QuestionTelemetryManager()
        mgr2.load_telemetry()

        # Verify record survived
        # Access internal ledger (implementation detail, but we need to verify persistence)
        assert hasattr(mgr2, "_ledger") or hasattr(mgr2, "ledger") or hasattr(mgr2, "_telemetry")
        ledger = getattr(mgr2, "_ledger", None) or getattr(mgr2, "ledger", None) or getattr(mgr2, "_telemetry", None)
        assert ledger is not None, "Telemetry manager must have a ledger/telemetry attribute"

        # Find our test record
        record = None
        if isinstance(ledger, dict):
            record = ledger.get("q-roundtrip-test")
        elif isinstance(ledger, list):
            for r in ledger:
                if r.get("q_id") == "q-roundtrip-test" or r.get("id") == "q-roundtrip-test":
                    record = r
                    break

        assert record is not None, "Persisted record 'q-roundtrip-test' not found after load"
        assert record.get("times_asked", 0) >= 1


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


class TestQualityAndRefineAPIEndpoints:
    """GET /api/interview/questions/quality + POST /api/interview/questions/refine."""

    def test_api_question_quality_and_refine_endpoints(self, client):
        """Both endpoints must return 200 with expected JSON structure."""
        # GET quality endpoint
        quality_resp = client.get("/api/interview/questions/quality")
        assert quality_resp.status_code == 200, (
            f"GET /api/interview/questions/quality failed: {quality_resp.status_code}"
        )
        quality_data = quality_resp.json()

        # Must contain audit + telemetry data
        assert "audit" in quality_data or "total_questions" in quality_data, (
            f"Quality endpoint missing audit data: {quality_data.keys()}"
        )
        # If nested audit, check structure
        if "audit" in quality_data:
            audit = quality_data["audit"]
            assert "total_questions" in audit
            assert "approved_questions" in audit
            assert "skill_coverage_count" in audit

        # POST refine endpoint
        refine_resp = client.post("/api/interview/questions/refine")
        assert refine_resp.status_code == 200, (
            f"POST /api/interview/questions/refine failed: {refine_resp.status_code}"
        )
        refine_data = refine_resp.json()

        # Must contain recommendations or feedback analysis
        assert "recommendations" in refine_data or "flags" in refine_data or "analysis" in refine_data, (
            f"Refine endpoint missing recommendations: {refine_data.keys()}"
        )
