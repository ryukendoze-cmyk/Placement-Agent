from sqlmodel import Session, select

from app.models import StudentSkill
from app.plan.planner import build_plan, pick_course_for_skill


class _StubsLLM:
    """Returns whatever JSON payload it was configured with."""

    def __init__(self, payload: str):
        self._payload = payload

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        return self._payload


def _seed_student_with_gaps(client_engine):
    """Upload a resume that matches job-001 (Backend Python) but is missing git + redis."""
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Planner Probe"},
    )
    sid = r.json()["id"]
    # trigger matches so JobMatch rows exist
    client.get(f"/api/students/{sid}/matches")
    return client, eng, sid


def test_pick_course_prefers_covers_skill_then_beginner_then_free():
    from app.seed import _json
    courses = _json("courses.json")
    # git is covered by course-027 (beginner, free, 3h) only
    c = pick_course_for_skill("git", courses)
    assert c is not None
    assert "git" in c["skill_ids"]
    # sql has 3 courses: course-011 (beginner, paid, 18h), course-013 (intermediate, free, 12h), course-030 (advanced, paid, 16h)
    c_sql = pick_course_for_skill("sql", courses)
    assert c_sql["id"] == "course-011"  # beginner beats free/intermediate


def test_build_plan_orders_required_before_preferred_and_sums_hours(client_engine):
    client, eng, sid = _seed_student_with_gaps(client_engine)
    # build_plan for job-001 against this student
    with Session(eng) as db:
        from app.embeddings.embedder import HashingEmbedder
        from app.llm.mock import MockLLM
        from app.matching.rank import rank_jobs
        from app.models import Job
        from app.taxonomy.taxonomy import load_taxonomy
        job = db.get(Job, "job-001")
        student_skills = {
            ss.skill_id: list(ss.evidence)
            for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
        }
        student_vec = HashingEmbedder().embed("Python FastAPI PostgreSQL Docker")
        ranked = rank_jobs([job], student_vec, student_skills, HashingEmbedder(), load_taxonomy())
        plan, items = build_plan(db, sid, job, ranked[0][1], llm=MockLLM())

    assert plan.total_hours == sum(i.hours for i in items)
    # job-001 missing: required [git], preferred [redis] -> 2 items, required first
    assert len(items) >= 2
    assert items[0].skill_id == "git"
    assert plan.narrative  # MockLLM produced non-empty narrative


def test_plan_api_roundtrip(client_engine):
    client, eng, sid = _seed_student_with_gaps(client_engine)
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    assert r.status_code in (200, 201)
    body = r.json()
    plan_id = body["id"]
    assert body["total_hours"] > 0
    assert body["narrative"]
    # list plans
    r2 = client.get(f"/api/students/{sid}/plans")
    assert any(p["id"] == plan_id for p in r2.json())
    # fetch single
    r3 = client.get(f"/api/plans/{plan_id}")
    assert r3.json()["id"] == plan_id


def _build_client_with_llm(llm):
    """Mirror of tests/conftest._build_client but with an explicit llm (incl. None)."""
    from fastapi.testclient import TestClient

    from app.api.main import AppContext, create_app
    from app.db import init_db, make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.embeddings.store import MemoryVectorStore
    from app.seed import seed
    from app.taxonomy.taxonomy import load_taxonomy

    eng = make_engine("sqlite://")
    init_db(eng)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=llm,
    )
    with Session(eng) as db:
        seed(db, ctx.taxonomy)
    return TestClient(create_app(ctx)), eng


def _seed_student(client, name: str = "Planner Probe") -> str:
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": name},
    )
    assert r.status_code in (200, 201)
    return r.json()["id"]


def test_create_plan_returns_503_when_llm_missing():
    client, _eng = _build_client_with_llm(None)
    with client:
        sid = _seed_student(client)
        r = client.post(f"/api/students/{sid}/plans", json={"job_id": "job-001"})
    assert r.status_code == 503


def test_create_plan_without_matches_returns_400():
    from app.llm.mock import MockLLM

    client, _eng = _build_client_with_llm(MockLLM())
    with client:
        sid = _seed_student(client)
        # resume uploaded but /matches never called -> no JobMatch rows
        r = client.post(f"/api/students/{sid}/plans", json={})
    assert r.status_code == 400


def test_create_plan_defaults_to_top_match(client_engine):
    client, eng, sid = _seed_student_with_gaps(client_engine)
    r = client.post(f"/api/students/{sid}/plans", json={})
    assert r.status_code == 201
    with Session(eng) as db:
        from app.models import JobMatch
        top = db.exec(
            select(JobMatch).where(JobMatch.student_id == sid).order_by(JobMatch.score.desc())
        ).first()
    assert r.json()["job_id"] == top.job_id


def test_build_plan_tolerates_non_dict_narrative_json(client_engine):
    """List/number/string/dict-less JSON narrative falls back to raw text - never AttributeError->500."""
    from sqlalchemy import delete

    from app.matching.engine import JobScore
    from app.models import Job, PlanItem, UpskillPlan

    _client, eng, sid = _seed_student_with_gaps(client_engine)
    with Session(eng) as db:
        job = db.get(Job, "job-001")
        job_score = JobScore(
            job_id=job.id, score=50, band="moderate",
            matched_required=[], missing_required=[], matched_preferred=[],
            missing_preferred=[], semantic_sim=0.5,
        )
        for payload in ('["a", "list"]', "42", '"just a string"', "{}"):
            # deterministic plan id - replace prior plan like routes/plans.py does
            prior = db.get(UpskillPlan, f"plan-{sid}-{job.id}")
            if prior is not None:
                db.execute(delete(PlanItem).where(PlanItem.plan_id == prior.id))
                db.delete(prior)
                db.commit()
            plan, _items = build_plan(db, sid, job, job_score, llm=_StubsLLM(payload))
            # non-dict JSON and empty dict fall back cleanly to a grounded deterministic narrative
            assert isinstance(plan.narrative, str) and len(plan.narrative.strip()) > 10
            assert plan.narrative != payload
            assert db.get(UpskillPlan, plan.id).narrative == plan.narrative  # persisted, not rolled back


def test_build_plan_skips_gaps_with_no_covering_course(client_engine):
    """rust exists in taxonomy but no course covers it -> item skipped, not fabricated."""
    from app.llm.mock import MockLLM
    from app.matching.engine import JobScore
    from app.models import Course, Job

    _client, eng, sid = _seed_student_with_gaps(client_engine)
    with Session(eng) as db:
        covered = {sk for c in db.exec(select(Course)).all() for sk in c.skill_ids}
        from app.taxonomy.taxonomy import load_taxonomy
        uncovered = [s.id for s in load_taxonomy().all() if s.id not in covered]
        assert "rust" in uncovered  # sanity: chosen gap really has no course

        job = db.get(Job, "job-016")
        job_score = JobScore(
            job_id=job.id, score=50, band="moderate",
            matched_required=[],
            missing_required=[{"skill": "kubernetes", "name": "Kubernetes", "category": "cloud"}],
            matched_preferred=[],
            missing_preferred=[
                {"skill": "rust", "name": "Rust", "category": "languages"},        # no course
                {"skill": "monitoring", "name": "Monitoring", "category": "ops"},  # covered
            ],
            semantic_sim=0.5,
        )
        plan, items = build_plan(db, sid, job, job_score, llm=MockLLM())

        skill_ids = [i.skill_id for i in items]
        assert "rust" not in skill_ids                        # skipped, not fabricated
        assert "monitoring" in skill_ids                      # covered gap appears
        assert "kubernetes" in skill_ids                      # required gap appears
        assert plan.total_hours == sum(i.hours for i in items)
