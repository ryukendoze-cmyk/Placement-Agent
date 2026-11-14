from __future__ import annotations

import io
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session, select

from app.api.main import AppContext, create_app
from app.api.routes import campus_drives
from app.campus.agent import parse_campus_jd
from app.campus.ingest import MAX_JD_BYTES, extract_jd_text
from app.campus.policy import tier_for
from app.campus.static_parser import parse_jd_static
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.llm.telemetry import LLMTelemetryTracker
from app.matching.engine import job_eligibility, job_tier
from app.models import (
    AuditLog,
    CampusDrive,
    Job,
    JobMatch,
    Student,
    StudentSkill,
    UpskillPlan,
)
from app.security.rbac import encode_session
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "jd"
TPO_HDR = {"X-Demo-Role": "tpo_admin"}
FACULTY_HDR = {"X-Demo-Role": "faculty_mentor"}
STUDENT_HDR = {"X-Demo-Role": "student"}


def _fk_on(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _build_campus_client():
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
    init_db(eng)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    with Session(eng) as db:
        seed(db, ctx.taxonomy)
    app = create_app(ctx)
    existing_paths = {getattr(r, "path", "") for r in app.routes}
    if "/api/campus-drives/taxonomy" not in existing_paths:
        app.include_router(campus_drives.router)
    return TestClient(app), eng, ctx


def test_policy_tier_boundaries_and_engine_parity():
    assert tier_for(7.99) == ("standard", "SOP-01")
    assert tier_for(8.0) == ("dream", "SOP-01")
    assert tier_for(14.99) == ("dream", "SOP-01")
    assert tier_for(15.0) == ("super_dream", "SOP-01")
    assert tier_for(12.0, experience_level="Internship") == (None, None)
    assert tier_for(12.0, salary_disclosed=False) == (None, None)
    assert tier_for(None) == (None, None)
    assert tier_for(0) == (None, None)

    for upper in (0.0, 4.5, 7.99, 8.0, 12.0, 14.99, 15.0, 22.0):
        j = Job(
            id=f"j-{upper}",
            title="Engineer",
            company="Test",
            description="Test",
            salary_lpa=[upper, upper],
        )
        t, _ = tier_for(upper)
        if upper > 0:
            assert t == job_tier(j)


def test_ingest_formats_limits_and_cleaning():
    pdf_bytes = (FIXTURE_DIR / "01_super_dream_pdf.pdf").read_bytes()
    pdf_text = extract_jd_text(pdf_bytes, ".pdf")
    assert "Razorpay Technologies" in pdf_text
    assert "- Python" in pdf_text

    docx_bytes = (FIXTURE_DIR / "02_dream_docx.docx").read_bytes()
    docx_text = extract_jd_text(docx_bytes, ".docx")
    assert "FinEdge Analytics" in docx_text
    assert "- Java" in docx_text

    messy_bytes = (FIXTURE_DIR / "06_messy_forwarded_email.txt").read_bytes()
    messy_text = extract_jd_text(messy_bytes, ".txt")
    assert "From: hr-notifications" not in messy_text
    assert "Sent from my iPhone" not in messy_text
    assert "- Python" in messy_text
    assert "- Terraform" in messy_text

    # Oversize > 2 MB -> HTTP 413
    try:
        extract_jd_text(b"x" * (MAX_JD_BYTES + 1), ".txt")
        raise AssertionError("expected 413")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 413

    # Unsupported extension -> HTTP 415
    try:
        extract_jd_text(b"hello", ".exe")
        raise AssertionError("expected 415")
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 415


def test_static_parser_determinism_and_all_fixtures():
    tax = load_taxonomy()
    assert len(list(FIXTURE_DIR.iterdir())) >= 8

    # 01 Super Dream PDF
    t1 = extract_jd_text((FIXTURE_DIR / "01_super_dream_pdf.pdf").read_bytes(), ".pdf")
    p1 = parse_jd_static(t1, tax)
    p1_again = parse_jd_static(t1, tax)
    assert p1 == p1_again  # deterministic
    assert p1["company"] == "Razorpay Technologies"
    assert p1["policy_tier"] == "super_dream"
    assert p1["policy_tier_source"] == "SOP-01"
    assert p1["ctc_lpa_min"] == 18.0 and p1["ctc_lpa_max"] == 22.0
    assert p1["eligibility"]["min_cgpa"] == 7.5
    assert p1["eligibility"]["max_active_backlogs"] == 0
    assert p1["eligibility"]["allowed_branches"] == ["CSE", "AI&DS", "IT"]
    assert set(["python", "fastapi", "postgresql", "redis"]).issubset(p1["required_skills"])
    assert set(["docker", "kubernetes"]).issubset(p1["preferred_skills"])

    # 02 Dream DOCX
    t2 = extract_jd_text((FIXTURE_DIR / "02_dream_docx.docx").read_bytes(), ".docx")
    p2 = parse_jd_static(t2, tax)
    assert p2["policy_tier"] == "dream"
    assert p2["ctc_lpa_max"] == 12.5
    assert p2["eligibility"]["min_cgpa"] == 7.0
    assert p2["eligibility"]["max_active_backlogs"] == 1
    assert p2["eligibility"]["allowed_branches"] == ["CSE", "ECE"]
    assert "java" in p2["required_skills"] and "spring" in p2["required_skills"]
    assert "docker" in p2["preferred_skills"] and "kafka" in p2["preferred_skills"]

    # 03 Standard Email
    t3 = extract_jd_text((FIXTURE_DIR / "03_standard_email.txt").read_bytes(), ".txt")
    p3 = parse_jd_static(t3, tax)
    assert p3["policy_tier"] == "standard"
    assert p3["ctc_lpa_max"] == 6.5
    assert p3["eligibility"]["max_active_backlogs"] is None
    assert "departmental clearance" in p3["parser_metadata"]["missing_fields_flagged"]

    # 04 Internship with stipend
    t4 = extract_jd_text((FIXTURE_DIR / "04_internship_stipend.txt").read_bytes(), ".txt")
    p4 = parse_jd_static(t4, tax)
    assert p4["experience_level"] == "Internship"
    assert p4["stipend_inr_per_month"] == 40000
    assert p4["policy_tier"] is None
    assert p4["ctc_lpa_max"] is None
    assert p4["eligibility"]["allowed_branches"] == []

    # 05 Salary undisclosed
    t5 = extract_jd_text((FIXTURE_DIR / "05_salary_undisclosed.txt").read_bytes(), ".txt")
    p5 = parse_jd_static(t5, tax)
    assert p5["salary_disclosed"] is False
    assert p5["policy_tier"] is None
    assert p5["eligibility"]["allowed_branches"] == []
    assert p5["eligibility"]["min_cgpa"] is None
    assert "min_cgpa" in p5["parser_metadata"]["missing_fields_flagged"]

    # 06 Messy forwarded email with INR 12,00,000 + 70% throughout
    t6 = extract_jd_text((FIXTURE_DIR / "06_messy_forwarded_email.txt").read_bytes(), ".txt")
    p6 = parse_jd_static(t6, tax)
    assert p6["ctc_lpa_max"] == 12.0
    assert p6["policy_tier"] == "dream"
    assert p6["eligibility"]["min_cgpa"] is None  # % never converted to CGPA
    assert "percentage_cutoff_unconverted" in p6["parser_metadata"]["missing_fields_flagged"]
    assert "percentage_criteria" in p6["parser_metadata"]["evidence"]
    assert p6["eligibility"]["graduation_year"] == 2027

    # 07 Branch restricted + 12L
    t7 = extract_jd_text((FIXTURE_DIR / "07_branch_restricted.txt").read_bytes(), ".txt")
    p7 = parse_jd_static(t7, tax)
    assert p7["ctc_lpa_max"] == 12.0
    assert p7["policy_tier"] == "dream"
    assert p7["eligibility"]["allowed_branches"] == ["ECE", "EE"]
    assert p7["eligibility"]["max_active_backlogs"] == 0
    assert "cplusplus" in p7["required_skills"]

    # 08 Fake & unmapped skills must NEVER enter required_skills or preferred_skills
    t8 = extract_jd_text((FIXTURE_DIR / "08_fake_unmapped_skills.txt").read_bytes(), ".txt")
    p8 = parse_jd_static(t8, tax)
    all_extracted = set(p8["required_skills"]) | set(p8["preferred_skills"])
    assert "rust" not in all_extracted
    assert "kotlin" not in all_extracted
    assert "Blockchain-Quantum-Rust++" not in all_extracted
    unmapped_low = [u.lower() for u in p8["unmapped_skill_mentions"]]
    assert "blockchain-quantum-rust++" in unmapped_low
    assert "kotlin multiplatform" in unmapped_low


def test_agent_modes_timeouts_grounding_and_circuit_breaker(monkeypatch):
    client, eng, ctx = _build_campus_client()
    jd_text = (FIXTURE_DIR / "01_super_dream_pdf.pdf").read_bytes()
    text = extract_jd_text(jd_text, ".pdf")

    with Session(eng) as s:
        # 1. Auto with MockLLM -> offline_mode
        res_auto = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_auto["parser_metadata"]["engine_used"] == "static_fallback"
        assert res_auto["parser_metadata"]["fallback_reason"] == "offline_mode"

        # 2. Forced static -> forced_static
        res_static = parse_campus_jd(text, mode="static", ctx=ctx, session=s)
        assert res_static["parser_metadata"]["engine_used"] == "static_fallback"
        assert res_static["parser_metadata"]["fallback_reason"] == "forced_static"

        # 3. Forced LLM with MockLLM -> hybrid_merge
        res_llm = parse_campus_jd(text, mode="llm", ctx=ctx, session=s)
        assert res_llm["parser_metadata"]["engine_used"] == "hybrid_merge"
        assert res_llm["parser_metadata"]["fallback_reason"] is None

        # 4. Simulated failures
        res_sim_to = parse_campus_jd(
            text, mode="auto", simulate_failure="timeout", ctx=ctx, session=s, actor_role="tpo_admin"
        )
        assert res_sim_to["parser_metadata"]["fallback_reason"] == "llm_timeout"

        res_sim_bad = parse_campus_jd(
            text, mode="auto", simulate_failure="invalid_json", ctx=ctx, session=s, actor_role="tpo_admin"
        )
        assert res_sim_bad["parser_metadata"]["fallback_reason"] == "llm_invalid_json"

        # 5. Real ThreadPoolExecutor timeout fallback
        import app.campus.llm_parser as lp
        monkeypatch.setattr(lp, "LLM_TIMEOUT_SECONDS", 0.1)

        class SlowLLM:
            provider = "openai"

            def complete(self, system, user, json_mode=False):
                time.sleep(0.5)
                return "{}"

        ctx.llm = SlowLLM()
        res_to = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_to["parser_metadata"]["engine_used"] == "static_fallback"
        assert res_to["parser_metadata"]["fallback_reason"] == "llm_timeout"

        # 6. Invalid JSON -> llm_invalid_json
        class BadJsonLLM:
            provider = "openai"

            def complete(self, system, user, json_mode=False):
                return "not valid json {{"

        ctx.llm = BadJsonLLM()
        res_ij = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_ij["parser_metadata"]["fallback_reason"] == "llm_invalid_json"

        # 7. Schema error -> llm_schema_validation
        class BadSchemaLLM:
            provider = "openai"

            def complete(self, system, user, json_mode=False):
                return json.dumps({"eligibility": "should_be_an_object"})

        ctx.llm = BadSchemaLLM()
        res_sv = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_sv["parser_metadata"]["fallback_reason"] == "llm_schema_validation"

        # 8. Ungrounded hallucinated skill / CTC -> llm_grounding_failed
        class HallucinatingLLM:
            provider = "openai"

            def complete(self, system, user, json_mode=False):
                return json.dumps({
                    "company": "Razorpay Technologies",
                    "role_title": "SDE",
                    "ctc_lpa_min": 99.0,
                    "ctc_lpa_max": 99.0,
                    "salary_disclosed": True,
                    "required_skills": ["blockchain_quantum_fake", "rust"],
                    "preferred_skills": [],
                })

        ctx.llm = HallucinatingLLM()
        res_hg = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_hg["parser_metadata"]["fallback_reason"] == "llm_grounding_failed"
        assert res_hg["ctc_lpa_max"] == 22.0
        assert "rust" not in res_hg["required_skills"]

        # 9. Grounded LLM hybrid merge + conflict detection
        class GroundedConflictLLM:
            provider = "openai"

            def complete(self, system, user, json_mode=False):
                return json.dumps({
                    "company": "Razorpay Technologies",
                    "role_title": "Software Development Engineer - Backend",
                    "location": "Bengaluru",
                    "work_mode": "Hybrid",
                    "ctc_lpa_min": 18.0,
                    "ctc_lpa_max": 22.0,
                    "salary_disclosed": True,
                    "experience_level": "Fresher",
                    "eligibility": {
                        "min_cgpa": 7.5,
                        "allowed_branches": ["CSE"],
                        "max_active_backlogs": 0,
                        "graduation_year": 2027,
                    },
                    "required_skills": ["python", "fastapi"],
                    "preferred_skills": ["docker"],
                })

        ctx.llm = GroundedConflictLLM()
        res_hm = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_hm["parser_metadata"]["engine_used"] == "hybrid_merge"
        assert "conflict:allowed_branches" in res_hm["parser_metadata"]["missing_fields_flagged"]

        # 10. Circuit breaker open -> circuit_open
        tracker = LLMTelemetryTracker()
        for _ in range(3):
            tracker.record("openai", "sys", error=True)
        ctx.llm_telemetry = tracker
        res_cb = parse_campus_jd(text, mode="auto", ctx=ctx, session=s)
        assert res_cb["parser_metadata"]["fallback_reason"] == "circuit_open"


def test_hitl_workflow_rbac_eligibility_and_audit():
    client, eng, _ctx = _build_campus_client()

    # Taxonomy accessible
    r_tax = client.get("/api/campus-drives/taxonomy", headers=FACULTY_HDR)
    from app.taxonomy.taxonomy import load_taxonomy

    assert r_tax.status_code == 200
    assert len(r_tax.json()) == len(load_taxonomy().all())
    assert len(r_tax.json()) >= 136

    # Student / faculty blocked on parse-jd
    assert client.post("/api/campus-drives/parse-jd", json={"text": "Company: Acme"}, headers=STUDENT_HDR).status_code == 403
    assert client.post("/api/campus-drives/parse-jd", json={"text": "Company: Acme"}, headers=FACULTY_HDR).status_code == 403

    # Parse multipart PDF as TPO
    pdf_bytes = (FIXTURE_DIR / "01_super_dream_pdf.pdf").read_bytes()
    r_parse = client.post(
        "/api/campus-drives/parse-jd",
        files={"file": ("jd.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        data={"force_parser": "auto", "simulate_failure": "timeout"},
        headers=TPO_HDR,
    )
    assert r_parse.status_code == 200
    draft = r_parse.json()
    assert draft["approval_status"] == "DRAFT"
    assert draft["policy_tier"] == "super_dream"
    assert draft["parser_metadata"]["fallback_reason"] == "llm_timeout"
    trace_id = draft["trace_id"]

    # Parse MUST NOT persist CampusDrive or Job rows; seed 3 test students for rematch check
    with Session(eng) as s:
        assert len(s.exec(select(CampusDrive)).all()) == 0
        initial_jobs = len(s.exec(select(Job)).all())
        initial_plans = len(s.exec(select(UpskillPlan)).all())
        s.add(Student(id="stu-1", name="Eligible Ace", branch="CSE", year=4, gpa=8.5, backlogs=0, resume_text="Python FastAPI PostgreSQL"))
        s.add(Student(id="stu-2", name="Backlog Bob", branch="CSE", year=4, gpa=8.0, backlogs=1, resume_text="Python"))
        s.add(Student(id="stu-3", name="Mech Maya", branch="ME", year=4, gpa=8.2, backlogs=0, resume_text="CAD"))
        s.add(StudentSkill(student_id="stu-1", skill_id="python", mentions=2, level="intermediate", evidence=["Python"]))
        s.add(StudentSkill(student_id="stu-1", skill_id="fastapi", mentions=1, level="intermediate", evidence=["FastAPI"]))
        s.commit()

    # Submit to Approval Queue (no Job created yet!)
    r_sub = client.post("/api/campus-drives", json=draft, headers=TPO_HDR)
    assert r_sub.status_code == 201
    created = r_sub.json()
    drive_id = created["id"]
    apr_id = created["approval_id"]
    assert created["approval_status"] == "PENDING_TPO_APPROVAL"
    assert created["job_id"] is None

    with Session(eng) as s:
        assert len(s.exec(select(Job)).all()) == initial_jobs

    # Faculty mentor can view list/detail, CANNOT edit or approve
    assert client.get("/api/campus-drives", headers=FACULTY_HDR).status_code == 200
    assert client.get(f"/api/campus-drives/{drive_id}", headers=FACULTY_HDR).status_code == 200
    assert client.patch(f"/api/campus-drives/{drive_id}", json={"location": "Remote"}, headers=FACULTY_HDR).status_code == 403
    assert client.post(f"/api/campus-drives/{drive_id}/approve", headers=FACULTY_HDR).status_code == 403
    assert client.post(f"/api/approvals/{apr_id}/approve", headers=FACULTY_HDR).status_code == 403
    assert client.post(f"/api/campus-drives/{drive_id}/approve", headers=STUDENT_HDR).status_code == 403

    # Edit while pending
    r_patch = client.patch(
        f"/api/campus-drives/{drive_id}",
        json={"eligibility": {"min_cgpa": 7.0}},
        headers=TPO_HDR,
    )
    assert r_patch.status_code == 200
    assert r_patch.json()["eligibility"]["min_cgpa"] == 7.0

    # Approve via /api/campus-drives/{id}/approve -> creates Job, rematches cohort, idempotent
    r_app = client.post(f"/api/campus-drives/{drive_id}/approve", json={"note": "Verified JD"}, headers=TPO_HDR)
    assert r_app.status_code == 200
    app_data = r_app.json()
    assert app_data["campus_drive"]["approval_status"] == "PUBLISHED"
    job_id = app_data["campus_drive"]["job_id"]
    assert job_id and job_id.startswith("job-campus-")
    summary = app_data["campus_drive"]["cohort_summary"]
    assert summary["students_total"] == 3
    assert summary["eligible"] == 1
    assert len(summary["top_matches"]) <= 5

    # Idempotent repeat approve via /api/approvals/{apr_id}/approve
    r_app2 = client.post(f"/api/approvals/{apr_id}/approve", headers=TPO_HDR)
    assert r_app2.status_code == 200
    assert r_app2.json()["status"] == "approved"

    with Session(eng) as s:
        job = s.get(Job, job_id)
        assert job is not None
        assert job.source == "local"
        assert job.scraper_engine == "campus_drive"
        assert job.max_backlogs == 0
        # No UpskillPlans auto-created
        assert len(s.exec(select(UpskillPlan)).all()) == initial_plans
        # Backlog eligibility check on new job vs student with backlogs=1
        stu_with_backlog = Student(id="stu-backlog-test", name="Backlog Student", branch="CSE", gpa=8.5, backlogs=1)
        elig_ok, reasons = job_eligibility(stu_with_backlog, job)
        assert not elig_ok
        assert any("SOP-03" in r for r in reasons)
        # JobMatch persisted for students
        assert len(s.exec(select(JobMatch).where(JobMatch.job_id == job_id)).all()) > 0
        # Full audit trace shares trace_id
        audits = s.exec(select(AuditLog)).all()
        traced = [a.action for a in audits if (a.detail or {}).get("trace_id") == trace_id]
        for expected_action in (
            "campus_drive.jd_extracted",
            "campus_drive.parse_attempt",
            "campus_drive.fallback_triggered",
            "campus_drive.validated",
            "campus_drive.eligibility_inferred",
            "campus_drive.submitted",
            "campus_drive.edited",
            "campus_drive.published",
            "campus_drive.cohort_rematched",
        ):
            assert expected_action in traced


def test_direct_publish_now_and_reject_and_public_jnf():
    client, eng, _ctx = _build_campus_client()

    # 1. Direct publish_now + confirm -> immediately PUBLISHED with task_id
    r_pub = client.post(
        "/api/campus-drives",
        json={
            "company": "DirectHire Corp",
            "role_title": "Backend Engineer",
            "location": "Bengaluru",
            "work_mode": "Remote",
            "ctc_lpa_min": 10.0,
            "ctc_lpa_max": 12.0,
            "salary_disclosed": True,
            "experience_level": "Fresher",
            "required_skills": ["python", "fastapi"],
            "publish_now": True,
            "confirm": True,
        },
        headers=TPO_HDR,
    )
    assert r_pub.status_code == 201
    pub_data = r_pub.json()
    assert pub_data["approval_status"] == "PUBLISHED"
    assert pub_data["policy_tier"] == "dream"
    assert pub_data["task_id"]
    r_task = client.get(f"/api/tasks/{pub_data['task_id']}", headers=TPO_HDR)
    assert r_task.status_code == 200
    assert r_task.json()["status"] == "completed"

    # 2. Reject flow flips CampusDrive.approval_status == REJECTED
    r_pend = client.post(
        "/api/campus-drives",
        json={"company": "RejectCorp", "role_title": "QA Engineer", "ctc_lpa_max": 6.0, "salary_disclosed": True},
        headers=TPO_HDR,
    )
    assert r_pend.status_code == 201
    rej_id = r_pend.json()["id"]
    rej_apr = r_pend.json()["approval_id"]
    r_rej = client.post(f"/api/approvals/{rej_apr}/reject", json={"note": "Incomplete"}, headers=TPO_HDR)
    assert r_rej.status_code == 200
    assert client.get(f"/api/campus-drives/{rej_id}", headers=TPO_HDR).json()["approval_status"] == "REJECTED"

    # 3. Public JNF flow
    r_link = client.post("/api/campus-drives/jnf-links", json={"expires_hours": 48}, headers=TPO_HDR)
    assert r_link.status_code == 200
    tok = r_link.json()["token"]
    assert client.get(f"/jnf/{tok}").status_code == 200
    assert client.get("/jnf/tampered.token").status_code == 401

    expired_tok = encode_session({"kind": "jnf_intake", "jnf_id": "jnf-old", "exp": int(time.time()) - 10})
    assert client.get(f"/jnf/{expired_tok}").status_code == 401
    assert client.post("/api/campus-drives/public-jnf", json={"token": expired_tok, "company": "X", "role_title": "Y"}).status_code == 401

    r_jnf = client.post(
        "/api/campus-drives/public-jnf",
        data={
            "token": tok,
            "company": "External Recruiter Co",
            "role_title": "Data Analyst",
            "ctc_lpa_max": "9.0",
            "salary_disclosed": "true",
            "allowed_branches": "CSE, IT",
            "text": "Required Skills: Python, SQL, Pandas",
        },
    )
    assert r_jnf.status_code == 201
    jnf_drive = r_jnf.json()
    assert jnf_drive["approval_status"] == "PENDING_TPO_APPROVAL"
    assert jnf_drive["origin"] == "recruiter_jnf"
    assert jnf_drive["created_by"] == "external_recruiter"
    assert jnf_drive["policy_tier"] == "dream"
    assert jnf_drive["job_id"] is None
