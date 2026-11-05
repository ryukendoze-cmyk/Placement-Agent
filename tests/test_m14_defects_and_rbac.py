"""Milestone 14: core defect remediation, proficiency-aware planner, verifier
expansion, and route-wide RBAC/audit hardening."""
import re

from sqlmodel import Session, select

from app.agent.verifier import build_known_entities, verify_text
from app.interview.engine import InterviewSummary
from app.models import AuditLog, StudentSkill
from app.plan.planner import build_plan, pick_course_for_skill


def _seed_student(client, resume=b"Skills: Python, FastAPI, PostgreSQL, Docker\n"):
    r = client.post("/api/students",
                    files={"file": ("r.txt", resume, "text/plain")},
                    data={"name": "M14 Probe"})
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


# ---------------------------------------------------------------- Fix #1
def test_promote_skills_multiple_and_generic_last(client_engine):
    """Loop-scope bug: every modified row must be staged, and a trailing
    'generic' (row=None) must not drop earlier promotions."""
    from app.api.routes.interviews import _promote_skills_from_summary

    client, eng = client_engine
    sid = _seed_student(client)
    summary = InterviewSummary(
        score=85,
        per_skill={"python": 90, "fastapi": 88, "generic": 95},
        feedback="strong senior answers",
    )
    with Session(eng) as db:
        _promote_skills_from_summary(db, sid, summary)
    with Session(eng) as db:
        rows = {r.skill_id: r for r in db.exec(
            select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
        assert rows["python"].level in ("intermediate", "advanced")
        assert rows["fastapi"].level in ("intermediate", "advanced")
        assert any("interview verified" in e for e in rows["python"].evidence)
        assert "generic" not in rows


def test_promote_creates_new_skill_only_for_real_taxonomy_skill(client_engine):
    from app.api.routes.interviews import _promote_skills_from_summary

    client, eng = client_engine
    r = client.post("/api/students",
                    files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                    data={"name": "M14 New"})
    sid = r.json()["id"]
    summary = InterviewSummary(score=90, per_skill={"redis": 95, "generic": 99},
                               feedback="verified")
    with Session(eng) as db:
        _promote_skills_from_summary(db, sid, summary)
    with Session(eng) as db:
        rows = {r.skill_id: r for r in db.exec(
            select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
        assert "redis" in rows
        assert rows["redis"].level == "intermediate"
        assert "generic" not in rows


def test_promote_remedial_records_gap_without_promotion(client_engine):
    from app.api.routes.interviews import _promote_skills_from_summary

    client, eng = client_engine
    sid = _seed_student(client)
    with Session(eng) as db:
        before = {r.skill_id: r.level for r in db.exec(
            select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
    with Session(eng) as db:
        _promote_skills_from_summary(
            db, sid, InterviewSummary(score=20, per_skill={"python": 15},
                                      feedback="weak"))
    with Session(eng) as db:
        row = db.get(StudentSkill, (sid, "python"))
        assert row.level == before["python"]
        assert any("remedial gap" in e for e in row.evidence)


# ---------------------------------------------------------------- Fix #2
def test_create_plan_passes_levels_and_populates_proficiency_gaps(client_engine):
    client, eng = client_engine
    r = client.post("/api/students",
                    files={"file": ("r.txt",
                                    b"Skills: Python, FastAPI, PostgreSQL, Docker, Git\n",
                                    "text/plain")},
                    data={"name": "M14 Levels"})
    sid = r.json()["id"]
    # force python to basic so a proficiency gap can arise against job-001
    with Session(eng) as db:
        ss = db.get(StudentSkill, (sid, "python"))
        if ss is not None:
            ss.level = "basic"
            db.add(ss)
            db.commit()
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    assert r.status_code in (200, 201), r.text


# ---------------------------------------------------------------- Fix #3
def test_pick_course_target_level_sorting():
    from app.seed import _json
    courses = _json("courses.json")
    # sql has course-011 (beginner), course-013 (intermediate), course-030 (advanced)
    assert pick_course_for_skill("sql", courses)["id"] == "course-011"
    assert pick_course_for_skill("sql", courses, "beginner")["id"] == "course-011"
    assert pick_course_for_skill("sql", courses, "basic")["id"] == "course-011"
    assert pick_course_for_skill("sql", courses, "intermediate")["id"] == "course-013"
    assert pick_course_for_skill("sql", courses, "advanced")["id"] == "course-030"


def test_build_plan_upgrades_proficiency_instead_of_empty_plan(client_engine):
    """Student owns every required job-001 skill at basic; role needs
    intermediate -> a non-empty upgrade plan must be produced."""
    client, eng = client_engine
    r = client.post("/api/students",
                    files={"file": ("r.txt",
                                    b"Skills: Python, FastAPI, PostgreSQL, Docker, Git\n",
                                    "text/plain")},
                    data={"name": "M14 Upgrade"})
    sid = r.json()["id"]
    with Session(eng) as db:
        from app.llm.mock import MockLLM
        from app.matching.engine import score_job
        from app.models import Job
        from app.taxonomy.taxonomy import load_taxonomy
        job = db.get(Job, "job-001")
        tax = load_taxonomy()
        req = [x for x in job.required_skills if tax.get_opt(x)]
        # student owns all required skills, all at basic
        student_skills = {x: [] for x in req}
        levels = {x: "basic" for x in req}
        js = score_job(student_skills, job, 0.8, tax, levels=levels)
        assert js.missing_required == []
        plan, items = build_plan(db, sid, job, js, llm=MockLLM(), levels=levels)
        assert len(items) >= 1
        assert plan.total_hours == sum(i.hours for i in items)


def test_build_plan_missing_required_path_unchanged(client_engine):
    """Existing behaviour: a real missing required skill is still emitted."""
    client, eng = client_engine
    r = client.post("/api/students",
                    files={"file": ("r.txt",
                                    b"Skills: Python, FastAPI, PostgreSQL, Docker\n",
                                    "text/plain")},
                    data={"name": "M14 Missing"})
    sid = r.json()["id"]
    with Session(eng) as db:
        from app.embeddings.embedder import HashingEmbedder
        from app.llm.mock import MockLLM
        from app.matching.rank import rank_jobs
        from app.models import Job
        from app.taxonomy.taxonomy import load_taxonomy
        job = db.get(Job, "job-001")
        ss = db.exec(select(StudentSkill)
                     .where(StudentSkill.student_id == sid)).all()
        skills = {s.skill_id: list(s.evidence) for s in ss}
        vec = HashingEmbedder().embed("Python FastAPI PostgreSQL Docker")
        js = rank_jobs([job], vec, skills, HashingEmbedder(), load_taxonomy())[0][1]
        plan, items = build_plan(db, sid, job, js, llm=MockLLM())
        assert items[0].skill_id == "git"


# ---------------------------------------------------------------- Fix #4
def test_apply_completion_promotes_intermediate_to_advanced_on_senior_job(client_engine):
    from app.plan.planner import apply_plan_item_completion

    client, eng = client_engine
    sid = _seed_student(client)
    client.get(f"/api/students/{sid}/matches")
    with Session(eng) as db:
        from app.models import Job, PlanItem, UpskillPlan
        job = Job(id="job-senior-x", title="Senior Backend Engineer",
                  company="X", description="senior role", required_skills=["python"],
                  preferred_skills=[], experience_years=5)
        db.add(job)
        plan = UpskillPlan(id=f"plan-{sid}-job-senior-x", student_id=sid,
                           job_id=job.id, total_hours=10, narrative="")
        db.add(plan)
        db.flush()
        db.add(PlanItem(id=f"{plan.id}-i1", plan_id=plan.id, order=1,
                        skill_id="python", course_id="course-001", hours=10,
                        status="pending", week=1))
        ss = db.get(StudentSkill, (sid, "python"))
        ss.level = "intermediate"
        db.add(ss)
        db.commit()
        ctx = client.app.state.ctx
        apply_plan_item_completion(db, ctx, plan, db.get(PlanItem, f"{plan.id}-i1"))
        assert db.get(StudentSkill, (sid, "python")).level == "advanced"


# ---------------------------------------------------------------- Fix #5
def test_verifier_matches_external_jobs_interviews_plans(client_engine):
    client, _eng = client_engine
    client  # fixture keeps the in-memory DB + app alive
    known = {"job-ext-deadbeef00", "int-1234abcd", "plan-stu-1234abcd-job-001"}
    res = verify_text(
        "refs job-ext-deadbeef00 int-1234abcd plan-stu-1234abcd-job-001 job-999",
        known)
    assert "job-ext-deadbeef00" in res["verified"]
    assert "int-1234abcd" in res["verified"]
    assert "plan-stu-1234abcd-job-001" in res["verified"]
    assert "job-999" in res["unverified"]


def test_build_known_entities_includes_interview_and_plans(client_engine):
    client, eng = client_engine
    sid = _seed_student(client)
    with Session(eng) as db:
        known = build_known_entities(db)
    assert sid in known


def test_verifier_patterns_compiled():
    from app.agent.verifier import PATTERNS
    names = {n for n, _ in PATTERNS}
    assert {"job", "interview", "plan"} <= names
    job_pat = dict(PATTERNS)["job"]
    assert job_pat.search("job-001") and job_pat.search("job-ext-deadbeef00")


# ---------------------------------------------------------------- Fix #6 RBAC
def test_student_cannot_read_other_student_plan(client_engine):
    client, eng = client_engine
    sid_a = _seed_student(client)
    sid_b = _seed_student(client)
    client.get(f"/api/students/{sid_a}/matches")
    r = client.post(f"/api/students/{sid_a}/plans", json={"job_id": "job-001"})
    plan_id = r.json()["id"]
    hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_b}
    assert client.get(f"/api/students/{sid_a}/plans", headers=hdr).status_code == 403
    assert client.get(f"/api/plans/{plan_id}", headers=hdr).status_code == 403
    ok = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_a}
    assert client.get(f"/api/plans/{plan_id}", headers=ok).status_code == 200


def test_student_cannot_start_interview_for_other_student(client_engine):
    client, eng = client_engine
    sid_a = _seed_student(client)
    sid_b = _seed_student(client)
    hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_b}
    r = client.post(f"/api/students/{sid_a}/interviews", json={"job_id": "job-001"},
                    headers=hdr)
    assert r.status_code == 403


def test_student_cannot_read_or_answer_other_student_interview(client_engine):
    client, eng = client_engine
    sid_a = _seed_student(client)
    sid_b = _seed_student(client)
    ok_a = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_a}
    bad_b = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_b}
    r = client.post(f"/api/students/{sid_a}/interviews", json={"job_id": "job-001"},
                    headers=ok_a)
    assert r.status_code == 201, r.text
    iid = r.json()["id"]
    assert client.get(f"/api/interviews/{iid}", headers=bad_b).status_code == 403
    assert client.post(f"/api/interviews/{iid}/answers", json={"text": "hi"},
                       headers=bad_b).status_code == 403
    assert client.get(f"/interviews/{iid}", headers=bad_b).status_code == 403
    assert client.get(f"/api/interviews/{iid}", headers=ok_a).status_code == 200
    assert client.get(f"/interviews/{iid}", headers=ok_a).status_code == 200


def test_agent_run_scoped_to_student(client_engine):
    client, eng = client_engine
    sid_a = _seed_student(client)
    sid_b = _seed_student(client)
    hdr_a = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_a}
    r = client.post("/api/agent/run", json={"goal": "rank jobs for me",
                                            "student_id": sid_b}, headers=hdr_a)
    assert r.status_code == 403
    r = client.post("/api/agent/run", json={"goal": "rank jobs for me"}, headers=hdr_a)
    assert r.status_code == 201, r.text
    run_id = r.json()["id"]
    assert r.json()["student_id"] == sid_a
    hdr_b = {"X-Demo-Role": "student", "X-Demo-Student-ID": sid_b}
    assert client.get(f"/api/agent/runs/{run_id}", headers=hdr_b).status_code == 403
    listed = client.get("/api/agent/runs", headers=hdr_b).json()
    assert all(x["student_id"] == sid_b for x in listed)


def test_agent_runs_demo_mode_unchanged(client):
    r = client.post("/api/agent/run", json={"goal": "inspect my profile"})
    assert r.status_code == 201
    runs = client.get("/api/agent/runs").json()
    assert len(runs) >= 1


def test_refine_questions_requires_privileged_role(client):
    assert client.post("/api/interview/questions/refine").status_code == 200
    assert client.post("/api/interview/questions/refine",
                       headers={"X-Demo-Role": "student"}).status_code == 403
    assert client.post("/api/interview/questions/refine",
                       headers={"X-Demo-Role": "tpo_admin"}).status_code == 200


def test_get_quiz_and_verify_scoped_404_before_scope(client_engine):
    client, eng = client_engine
    hdr = {"X-Demo-Role": "student", "X-Demo-Student-ID": "stu-deadbeef"}
    assert client.get("/api/plans/plan-nope/items/i1/quiz",
                      headers=hdr).status_code == 404
    assert client.post("/api/plans/plan-nope/items/i1/verify", json={},
                       headers=hdr).status_code == 404


# ---------------------------------------------------------------- Fix #7 Audit
def test_audit_rows_written_on_key_events(client_engine):
    client, eng = client_engine
    sid = _seed_student(client)
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    plan = r.json()
    item = plan["items"][0]
    client.get(f"/api/plans/{plan['id']}/items/{item['id']}/quiz")
    with Session(eng) as db:
        actions = {a.action for a in db.exec(select(AuditLog)).all()}
    assert "student.resume_uploaded" in actions
    assert "plan.generated" in actions


def test_audit_rows_written_on_quiz_verify_and_agent_run(client_engine, monkeypatch):
    import app.plan.verification as verification
    client, eng = client_engine
    sid = _seed_student(client)
    client.get(f"/api/students/{sid}/matches")
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    plan = r.json()
    item = plan["items"][0]
    monkeypatch.setattr(
        verification, "check_answers",
        lambda skill_id, answers: (True, 100, [
            {"question_id": "q1", "correct": True, "hit_ratio": 1.0, "explanation": "x"},
            {"question_id": "q2", "correct": True, "hit_ratio": 1.0, "explanation": "x"},
        ]))
    v = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/verify",
                    json={"answers": ["a", "b"]})
    assert v.json()["passed"] is True
    client.post("/api/agent/run", json={"goal": "rank my top matches"})
    with Session(eng) as db:
        actions = {a.action for a in db.exec(select(AuditLog)).all()}
    assert "plan.quiz_verified" in actions
    assert "agent.run_completed" in actions


def test_audit_jobs_scraped_written():
    import types

    from app.db import init_db, make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    from app.models import AuditLog as AL
    from app.scraper.pipeline import ingest_scraped_jobs
    from app.taxonomy.taxonomy import load_taxonomy

    tax = load_taxonomy()
    eng = make_engine("sqlite://")
    init_db(eng)
    ctx = types.SimpleNamespace(taxonomy=tax, llm=None,
                                embedder=HashingEmbedder(),
                                vector_store=MemoryVectorStore(), job_vectors={})
    raw = [{"title": "QA Engineer", "company": "Acme Corp",
            "description": "QA role requiring manual testing and automation.",
            "job_url": "https://example.com/qa-acme-1"}]
    with Session(eng) as db:
        result = ingest_scraped_jobs(raw, db, ctx, tax)
        if result.get("ingested_count"):
            assert db.exec(select(AL).where(AL.action == "jobs.scraped")).first() is not None


def test_log_audit_helper_stages_row():
    from app.db import init_db, make_engine
    from app.models import AuditLog as AL
    from app.security.audit import log_audit
    eng = make_engine("sqlite://")
    init_db(eng)
    with Session(eng) as db:
        row = log_audit(db, action="x.y", entity_type="t", entity_id="e1",
                        detail={"k": 1})
        assert row.id.startswith("aud-")
        db.commit()
        assert db.exec(select(AL).where(AL.action == "x.y")).first() is not None


def test_plan_curve_advanced_target_senior(client_engine):
    """pick_course advanced ordering falls back to intermediate/beginner."""
    from app.seed import _json
    courses = [c for c in _json("courses.json") if "golang_none" in c["skill_ids"]] or [
        {"id": "c-b", "level": "beginner", "cost": "free", "hours": 5,
         "skill_ids": ["zzz"]},
        {"id": "c-i", "level": "intermediate", "cost": "paid", "hours": 6,
         "skill_ids": ["zzz"]},
    ]
    assert pick_course_for_skill("zzz", courses, "advanced")["id"] == "c-i"
    alone = [{"id": "c-b", "level": "beginner", "cost": "free", "hours": 5,
              "skill_ids": ["zzz"]}]
    assert pick_course_for_skill("zzz", alone, "advanced")["id"] == "c-b"


def test_verifier_plan_token_boundary():
    """Plan id capture must not swallow trailing punctuation."""
    from app.agent.verifier import _referenced_ids
    found = {tok for _, tok in _referenced_ids("see plan-stu-aa-job-001, ok")}
    assert "plan-stu-aa-job-001" in found
    assert not re.search(r"[.,]", " ".join(found))


def test_verifier_plan_ignores_english_hyphenations():
    """'plan-based' / 'plan-driven' are English, not hallucinated ids."""
    from app.agent.verifier import _referenced_ids
    res = verify_text("We recommend a plan-based approach and a plan-driven schedule.", set())
    assert res["unverified"] == []
    assert res["grounding_score"] == 100
    found = {tok for _, tok in _referenced_ids("plan-based plan-driven plan-stu-aa-job-001")}
    assert "plan-based" not in found
    assert "plan-driven" not in found
    assert "plan-stu-aa-job-001" in found
