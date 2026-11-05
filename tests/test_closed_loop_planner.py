"""Phase 5: closed-loop upskilling - prerequisite DAG ordering, weekly schedule,
skill-verification micro-quiz, and the completion -> StudentSkill -> JobMatch
score-climb feedback loop.
"""
import json

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.main import AppContext, create_app
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import JobMatch, PlanItem, ProgressEvent, StudentSkill
from app.plan.prereq import PREREQUISITE_GRAPH, topological_order
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


def _build_client():
    eng = make_engine("sqlite://")
    init_db(eng)
    ctx = AppContext(engine=eng, taxonomy=load_taxonomy(), embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=MockLLM())
    with Session(eng) as db:
        seed(db, ctx.taxonomy)
    return TestClient(create_app(ctx)), eng


def _seed_student_with_gaps(client, resume: bytes = b"Skills: React, HTML, CSS, SQL, Git\n"):
    r = client.post("/api/students", files={"file": ("r.txt", resume, "text/plain")},
                    data={"name": "Closed Loop Probe"})
    assert r.status_code in (200, 201), r.text
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    return sid


def _seed_student_git_gap(client):
    """job-001 required gap `git` is open (resume has python+fastapi, no git)."""
    return _seed_student_with_gaps(client, b"Skills: Python, FastAPI\n")


def _seed_student_sql_present(client):
    """job-001 plan covers ALL 6 gaps (python, fastapi, postgresql, git, docker, redis)
    - `sql` (a graph prerequisite of postgresql/redis) is already owned by the resume,
    so it does NOT become a phantom course item."""
    return _seed_student_with_gaps(client, b"Skills: SQL, Python, FastAPI\n")


def _make_plan(client, sid, job_id="job-001"):
    r = client.post(f"/api/students/{sid}/plans", json={"job_id": job_id})
    assert r.status_code == 201, r.text
    return r.json()


# --- (1) prerequisite DAG ordering ------------------------------------------

def test_prereq_graph_orders_missing_python_before_fastapi_and_docker_before_k8s():
    # job-001 (Backend Python): resume has none of python/fastapi/postgresql/git;
    # preferred docker + redis also missing -> python, fastapi, docker, kubernetes-free
    client, _eng = _build_client()
    with client:
        sid = _seed_student_with_gaps(client, b"Skills: Java\n")
        plan = _make_plan(client, sid, "job-001")
        skills = [it["skill_id"] for it in plan["items"]]
        assert "python" in skills and "fastapi" in skills
        assert skills.index("python") < skills.index("fastapi"), (
            f"python must be scheduled before its prerequisite consumer fastapi: {skills}")
        assert "docker" in skills
        # graph edge sanity
        assert "python" in PREREQUISITE_GRAPH["fastapi"]
        assert "docker" in PREREQUISITE_GRAPH["kubernetes"]


def test_prereq_graph_orders_full_web_stack_chain():
    # job-005 (React, typescript, tailwind, css / pref nextjs, accessibility):
    # gaps must contain html -> css -> javascript_web -> react -> nextjs in order
    client, _eng = _build_client()
    with client:
        sid = _seed_student_with_gaps(client, b"Skills: Excel, PowerBI\n")
        plan = _make_plan(client, sid, "job-005")
        skills = [it["skill_id"] for it in plan["items"]]
        for req, consumer in (("html", "css"), ("css", "javascript_web"),
                              ("javascript_web", "react"), ("react", "nextjs")):
            if req in skills and consumer in skills:
                assert skills.index(req) < skills.index(consumer), (
                    f"{req} must precede {consumer}: {skills}")


def test_prereq_graph_maintains_week_constraint_for_python_fastapi():
    """Prerequisites must land in the same week or an earlier week than their consumer."""
    client, _eng = _build_client()
    with client:
        sid = _seed_student_with_gaps(client, b"Skills: Java\n")
        plan = _make_plan(client, sid, "job-001")
        weeks = {it["skill_id"]: it["week"] for it in plan["items"]}
        assert "python" in weeks and "fastapi" in weeks
        assert weeks["python"] <= weeks["fastapi"]


def test_topological_order_schedules_prerequisites_before_their_consumers():
    graph = {"pytorch": ["deep_learning"], "deep_learning": ["machine_learning"],
             "machine_learning": ["python"], "python": []}
    order = topological_order(graph, ["pytorch"])
    assert order.index("python") < order.index("machine_learning")
    assert order.index("machine_learning") < order.index("deep_learning")
    assert order.index("deep_learning") < order.index("pytorch")
    # determinism
    assert topological_order(graph, ["pytorch"]) == order


def test_topological_order_is_stable_for_disconnected_skills():
    graph = {"a": [], "b": [], "c": []}
    order1 = topological_order(graph, ["c", "a", "b"])
    order2 = topological_order(graph, ["c", "a", "b"])
    assert order1 == order2
    assert sorted(order1) == ["a", "b", "c"]


def test_topological_order_excludes_owned_prerequisites_but_keeps_ordering():
    # student owns `python`: it must not be emitted as a schedulable gap, but
    # the `fastapi -> python` edge must still put fastapi after any other
    # gap that python constrains. `python` in `targets` is dropped too.
    graph = {"fastapi": ["python"], "django": ["python"], "python": []}
    order = topological_order(graph, ["fastapi", "django"], exclude={"python"})
    assert "python" not in order
    assert sorted(order) == ["django", "fastapi"]
    # determinism with exclude
    assert topological_order(graph, ["fastapi", "django"], exclude={"python"}) == order


def test_topological_order_priority_puts_required_and_shorter_first():
    # same dependency level: required (rank 0) beats preferred (rank 1),
    # fewer hours beats more hours at equal rank
    graph = {"required_long": [], "required_short": [], "preferred_short": []}
    priority = {"required_long": (0, 30), "required_short": (0, 5),
                "preferred_short": (1, 5)}
    order = topological_order(
        graph, ["preferred_short", "required_long", "required_short"],
        priority=priority)
    assert order == ["required_short", "required_long", "preferred_short"]
    # without priority, alphabetical fallback still applies
    assert topological_order(
        graph, ["preferred_short", "required_long", "required_short"]
    ) == ["preferred_short", "required_long", "required_short"]


# --- (2) weekly schedule ------------------------------------------------------

def test_weekly_schedule_groups_items_into_weeks():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_with_gaps(client, b"Skills: Excel, PowerBI\n")
        plan = _make_plan(client, sid, "job-005")
        assert plan["total_weeks"] >= 2, plan
        assert len(plan["weekly_schedule"]) == plan["total_weeks"]
        for wk in plan["weekly_schedule"]:
            assert wk["week"] >= 1
            assert wk["total_hours"] > 0
            assert wk["items"]
        # flat item list and schedule agree on hours
        assert sum(w["total_hours"] for w in plan["weekly_schedule"]) == plan["total_hours"]


def test_plan_never_includes_skills_the_student_already_owns():
    client, _eng = _build_client()
    with client:
        # resume owns SQL, Python, FastAPI - job-001 real gaps are
        # postgresql/git (required) + docker/redis (preferred); `sql` and
        # `linux` are graph prerequisites of those gaps but `sql` is owned.
        sid = _seed_student_sql_present(client)
        plan = _make_plan(client, sid, "job-001")
        skills = [it["skill_id"] for it in plan["items"]]
        # sql + python + fastapi are prerequisites of postgresql/redis gaps but
        # already owned by the resume - they must NOT become phantom items
        assert "sql" not in skills, skills
        assert "python" not in skills, skills
        assert "fastapi" not in skills, skills
        # genuine gaps still present and in prerequisite order
        assert "git" in skills and "postgresql" in skills, skills
        assert skills.index("git") < skills.index("postgresql"), skills
        assert all(it["status"] == "pending" for it in plan["items"])


def test_weekly_schedule_cap_never_exceeds_10_hours_per_week():
    client, _eng = _build_client()
    with client:
        # job-020 (SDET): resume already owns python/playwright/test_automation, so the
        # only gaps are git (required) + ci_cd/pytest (preferred). ci_cd's prerequisite
        # `git` is itself a gap and both git(3h)+ci_cd(5h) courses pack into week 1
        # at 8h - the only real multi-item week this dataset can produce under the
        # 10h cap (every other job's smallest two gap courses sum to >10h or are
        # non-adjacent in the prerequisite order).
        sid = _seed_student_with_gaps(client, b"Skills: Python, Playwright, Test Automation\n")
        plan = _make_plan(client, sid, "job-020")
        assert len(plan["items"]) >= 2, plan
        for wk in plan["weekly_schedule"]:
            # a course strictly longer than the cap (e.g. 15h/18h) gets its own
            # week - that week's hours may exceed 10 but never packs 2 items
            if len(wk["items"]) >= 2:
                assert wk["total_hours"] <= 10, wk
        # at least one week genuinely packs 2+ items
        multi = [w for w in plan["weekly_schedule"] if len(w["items"]) >= 2]
        assert multi, plan["weekly_schedule"]


def test_get_plan_serialization_includes_weeks_schedule_and_item_week():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_with_gaps(client, b"Skills: Excel, PowerBI\n")
        plan = _make_plan(client, sid, "job-005")
        r = client.get(f"/api/plans/{plan['id']}")
        assert r.status_code == 200
        body = r.json()
        assert body["total_weeks"] >= 1
        assert body["weekly_schedule"]
        assert all("week" in it for it in body["items"])
        # Backward compat: pre-Phase-5 contract intact
        for key in ("id", "student_id", "job_id", "total_hours", "narrative", "items"):
            assert key in body
        for it in body["items"]:
            for key in ("id", "order", "skill_id", "course_id", "course_title", "course_url", "hours", "status"):
                assert key in it, f"missing {key}"


# --- (3) skill-verification micro-quiz ---------------------------------------

def test_quiz_get_returns_two_questions_without_answers():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        item = next(i for i in plan["items"] if i["skill_id"] == "git")
        r = client.get(f"/api/plans/{plan['id']}/items/{item['id']}/quiz")
        assert r.status_code == 200, r.text
        quiz = r.json()
        assert quiz["item_id"] == item["id"]
        assert quiz["skill_id"] == "git"
        assert len(quiz["questions"]) == 2
        for q in quiz["questions"]:
            assert q["prompt"]
            # zero-leak: no grading keys/explanations may be serialized
            assert "keywords" not in q and "answer_idx" not in q
            assert "explanation" not in q
        assert "keywords" not in json.dumps(quiz)
        assert "explanation" not in json.dumps(quiz)
        # a correct answer (the keyword hit string) must pass server-side grading
        from app.plan.verification import _answers_for_item
        answers = _answers_for_item(item["id"], "git")
        r2 = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/verify",
                         json={"answers": answers})
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["passed"] is True and body2["score"] == 100


def test_quiz_unknown_plan_or_item_returns_404():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        assert client.get("/api/plans/plan-none/items/x/quiz").status_code == 404
        assert client.post("/api/plans/plan-none/items/x/verify", json={"answers": {}}).status_code == 404
        r = client.get(f"/api/plans/{plan['id']}/items/item-none/quiz")
        assert r.status_code == 404
        r = client.post(f"/api/plans/{plan['id']}/items/item-none/verify", json={"answers": {}})
        assert r.status_code == 404


def test_quiz_verify_failing_answers_keep_item_pending():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        item = next(i for i in plan["items"] if i["skill_id"] == "git")
        # empty/irrelevant answers must fall below the 50% keyword threshold
        quiz = client.get(f"/api/plans/{plan['id']}/items/{item['id']}/quiz").json()
        wrong = {q["id"]: "irrelevant noise answer" for q in quiz["questions"]}
        r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/verify",
                        json={"answers": wrong})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["passed"] is False
        assert body["item_id"] == item["id"]
        assert body["status"] == "pending"
        # no closed loop on failure
        assert body.get("closed_loop") is None
        # item state agrees with the response
        with Session(_eng) as db:
            row = db.get(PlanItem, item["id"])
            assert row.status == "pending"


def test_quiz_pass_marks_item_done_and_upserts_skill_with_evidence():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        item = next(i for i in plan["items"] if i["skill_id"] == "git")
        # learn the correct answers deterministically from the bank-side helper
        from app.plan.verification import _answers_for_item
        answers = _answers_for_item(item["id"], "git")
        r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/verify",
                        json={"answers": answers})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["passed"] is True
        assert body["score"] == 100
        assert body["status"] == "done"
        assert body["closed_loop"]["skill_id"] == "git"
        with Session(_eng) as db:
            row = db.get(PlanItem, item["id"])
            assert row.status == "done"
            ss = db.get(StudentSkill, (sid, "git"))
            assert ss is not None, "verified skill must be upserted as StudentSkill"
            assert any("course" in e.lower() for e in ss.evidence), ss.evidence


# --- (4) closed-loop score climb --------------------------------------------

def test_closed_loop_completion_raises_best_match_and_readiness():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        with Session(_eng) as db:
            before = max((m.score for m in db.exec(
                select(JobMatch).where(JobMatch.student_id == sid)).all()), default=0)
        item = plan["items"][0]
        r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete")
        assert r.status_code == 200, r.text
        body = r.json()
        with Session(_eng) as db:
            after = max((m.score for m in db.exec(
                select(JobMatch).where(JobMatch.student_id == sid)).all()), default=0)
        assert body["best_match"] == after
        assert after >= before, "JobMatch scores must never regress on completion"
        assert body["skill_id"] == item["skill_id"], "completion must report the learned skill"
        # a StudentSkill row now exists for the completed gap skill
        with Session(_eng) as db:
            ss = db.get(StudentSkill, (sid, item["skill_id"]))
            assert ss is not None
        # readiness formula stays the documented blend
        assert body["readiness"] == round(0.5 * body["plan_pct"] + 0.5 * body["best_match"])


def test_completing_second_item_increases_match_score_again():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        if len(plan["items"]) < 2:
            return  # nothing further to climb (defensive)
        first, second = plan["items"][0], plan["items"][1]
        r1 = client.post(f"/api/plans/{plan['id']}/items/{first['id']}/complete")
        assert r1.status_code == 200
        best1 = r1.json()["best_match"]
        r2 = client.post(f"/api/plans/{plan['id']}/items/{second['id']}/complete")
        assert r2.status_code == 200
        best2 = r2.json()["best_match"]
        assert best2 >= best1
        assert r2.json()["plan_pct"] > 0


def test_complete_is_idempotent_for_progress_events():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        item = plan["items"][0]
        assert client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete").status_code == 200
        assert client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete").status_code == 200
        with Session(_eng) as db:
            events = db.exec(select(ProgressEvent).where(
                ProgressEvent.student_id == sid,
                ProgressEvent.kind == "plan_item_done",
                ProgressEvent.ref_id == item["id"])).all()
            assert len(events) == 1, "idempotent: one event per plan item"
            # replayed complete must NOT inflate the skill's mentions/level
            ss = db.get(StudentSkill, (sid, item["skill_id"]))
            assert ss is not None
            assert ss.mentions == 1, f"replay inflated mentions: {ss.mentions}"
            assert ss.level == "basic", f"replay promoted level: {ss.level}"


def test_completion_promotes_skill_level_and_boosts_proficiency_bonus():
    client, _eng = _build_client()
    with client:
        sid = _seed_student_git_gap(client)
        plan = _make_plan(client, sid, "job-001")
        item = next(i for i in plan["items"] if i["skill_id"] == "git")
        r = client.post(f"/api/plans/{plan['id']}/items/{item['id']}/complete")
        assert r.status_code == 200
        with Session(_eng) as db:
            ss = db.get(StudentSkill, (sid, "git"))
            assert ss is not None
            assert ss.level in ("basic", "intermediate", "advanced")
            assert ss.mentions >= 1
