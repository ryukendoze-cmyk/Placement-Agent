"""Comprehensive regression tests for Courses & Upskilling Plan fixes."""
import json
import re
from pathlib import Path

from sqlmodel import Session, select

from app.matching.engine import JobScore
from app.models import AuditLog, Course, Job, JobMatch, Student, StudentSkill
from app.plan.planner import (
    apply_plan_item_completion,
    build_plan,
    ensure_remedial_plan_items,
    pick_course_for_skill,
)
from app.plan.prereq import PREREQUISITE_GRAPH, topological_order
from app.plan.verification import (
    PASSING_SCORE,
    _answers_for_item,
    _grade_question,
    check_answers,
    generate_quiz,
)


def test_courses_catalog_urls_and_full_job_coverage():
    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    jobs = json.loads(Path("data/jobs.json").read_text(encoding="utf-8"))
    skills = {s["id"] for s in json.loads(Path("data/skills.json").read_text(encoding="utf-8"))}

    assert len(courses) >= 75
    covered = set()
    bad_patterns = (
        "udemy.com",
        "edx.org/learn/",
        "youtube.com/results",
        "flockjay",
        "coursera.org/specializations/google-ux-design",
    )
    for c in courses:
        Course(**c)
        assert c["url"].startswith("https://"), c["id"]
        assert not any(bad in c["url"] for bad in bad_patterns), f"Banned URL in {c['id']}: {c['url']}"
        assert c.get("url_verified") is True
        assert c.get("cost_tier") in ("free", "freemium", "paid")
        assert isinstance(c.get("key_topics"), list) and len(c["key_topics"]) >= 2
        assert isinstance(c.get("syllabus"), list) and len(c["syllabus"]) >= 2
        for sk in c["skill_ids"]:
            assert sk in skills
            covered.add(sk)

    needed = {
        sk for j in jobs for sk in (j.get("required_skills", []) + j.get("preferred_skills", []))
    }
    assert needed.issubset(covered), f"Missing job skills: {sorted(needed - covered)}"

    for req_pref in (
        "ansible",
        "bootstrap",
        "chrome_devtools",
        "figma",
        "matplotlib",
        "mlops",
        "postman",
        "pytest",
        "redux",
    ):
        assert req_pref in covered

    assert pick_course_for_skill("sql", courses)["id"] == "course-011"
    assert pick_course_for_skill("sql", courses, "intermediate")["id"] == "course-013"
    assert pick_course_for_skill("sql", courses, "advanced")["id"] == "course-030"


def test_prereq_dag_fixes_and_acyclic():
    for root_skill in ("redis", "rest_api", "kafka", "hadoop", "test_automation"):
        assert PREREQUISITE_GRAPH[root_skill] == []
    assert PREREQUISITE_GRAPH["javascript"] == ["html"]
    assert PREREQUISITE_GRAPH["rails"] == ["ruby"]

    ordered = topological_order(PREREQUISITE_GRAPH, ["rest_api", "redis"])
    assert "html" not in ordered
    assert "css" not in ordered
    assert "sql" not in ordered


def test_verification_hardened_word_boundary_and_passing_score():
    assert PASSING_SCORE == 70

    # Substring concatenation exploit & negation attack both fail
    assert _grade_question("nonsemantic antiaccessibility", ["semantic", "accessibility"]) == (False, 0.0)
    assert _grade_question("not semantic, no accessibility", ["semantic", "accessibility"]) == (False, 0.0)
    # Genuine plural answer passes
    assert _grade_question("Python decorators wrap functions", ["decorator", "function"]) == (True, 1.0)

    # All catalog + job skills have 2 specific questions; 1/2 fails (< 70%)
    courses = json.loads(Path("data/courses.json").read_text(encoding="utf-8"))
    jobs = json.loads(Path("data/jobs.json").read_text(encoding="utf-8"))
    all_skills = sorted(
        {sk for c in courses for sk in c["skill_ids"]}
        | {sk for j in jobs for sk in j.get("required_skills", []) + j.get("preferred_skills", [])}
    )
    for sk in all_skills:
        qs = generate_quiz(sk)
        assert len(qs) == 2, sk
        assert all(q["skill_id"] == sk for q in qs), sk
        ans = _answers_for_item("i1", sk)
        ok, score, _ = check_answers(sk, ans)
        assert ok and score == 100
        half_ok, half_score, _ = check_answers(sk, {qs[0]["id"]: ans[qs[0]["id"]]})
        assert not half_ok and half_score == 50


def test_build_plan_dedup_course_ids_and_grounded_narrative_fallback(client_engine):
    _client, eng = client_engine
    with Session(eng) as db:
        shared = Course(
            id="course-900",
            title="Shared Infra Pack",
            provider="Official",
            url="https://kubernetes.io/docs/tutorials/kubernetes-basics/",
            skill_ids=["kubernetes", "monitoring"],
            level="beginner",
            hours=2,
            cost="free",
        )
        db.merge(shared)
        db.merge(Student(id="stu-dedup", name="Test Student", branch="CSE", year=4, cgpa=8.5, backlogs=0, resume_text="Linux Docker"))
        db.merge(StudentSkill(student_id="stu-dedup", skill_id="linux", mentions=1, level="basic", evidence=["linux"]))
        db.merge(StudentSkill(student_id="stu-dedup", skill_id="docker", mentions=1, level="basic", evidence=["docker"]))
        db.commit()

        job = db.get(Job, "job-016")
        js = JobScore(
            job_id=job.id,
            score=50,
            band="moderate",
            matched_required=[],
            missing_required=[{"skill": "kubernetes", "name": "Kubernetes", "category": "cloud"}],
            matched_preferred=[],
            missing_preferred=[{"skill": "monitoring", "name": "Monitoring", "category": "ops"}],
            semantic_sim=0.5,
        )

        class _HallucinatingLLM:
            def complete(self, system, user, json_mode=False):
                payload = json.loads(user)
                assert "schedule" in payload and "weeks" in payload
                return json.dumps({
                    "narrative": "Take https://evil-hallucination.invalid/fake and course-999 per SOP-99."
                })

        plan, items = build_plan(db, "stu-dedup", job, js, llm=_HallucinatingLLM())
        course_ids = [it.course_id for it in items]
        assert course_ids.count("course-900") == 1
        assert plan.total_hours == sum(it.hours for it in items)
        assert "evil-hallucination.invalid" not in plan.narrative
        assert "course-999" not in plan.narrative
        assert "Shared Infra Pack" in plan.narrative

        class _BrokenLLM:
            def complete(self, system, user, json_mode=False):
                raise RuntimeError("API timeout")

        for it in items:
            db.delete(it)
        db.delete(plan)
        db.commit()
        plan2, _items2 = build_plan(db, "stu-dedup", job, js, llm=_BrokenLLM())
        assert plan2.total_hours > 0 and "Week 1" in plan2.narrative


def test_verify_quiz_promotes_status_and_remedial_items_and_ui(client_engine):
    client, eng = client_engine
    up = client.post(
        "/api/students",
        data={"name": "Plan Tester", "branch": "CSE", "year": "4", "cgpa": "8.4", "backlogs": "0"},
        files={"file": ("resume.txt", b"Skills: Python, SQL, FastAPI\n", "text/plain")},
    )
    assert up.status_code in (200, 201), up.text
    sid = up.json()["id"]
    client.get(f"/api/students/{sid}/matches")

    with Session(eng) as db:
        ss_sql = db.get(StudentSkill, (sid, "sql"))
        if ss_sql is None:
            db.add(StudentSkill(student_id=sid, skill_id="sql", mentions=1, level="basic", evidence=["sql"], verification_status="unverified"))
        else:
            ss_sql.verification_status = "unverified"
            db.add(ss_sql)
        db.commit()

        touched = ensure_remedial_plan_items(db, sid, ["sql", "pytest"])
        assert len(touched) >= 1

    page = client.get(f"/students/{sid}/plan")
    assert page.status_code == 200
    html = page.text
    assert "30-day curriculum" not in html
    assert "plan-bar" in html
    assert "plan-quiz-modal" in html
    assert "Verify &amp; Complete" in html or "Verify & Complete" in html
    assert not re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", html)
    assert "alert(" not in html

    plans_resp = client.get(f"/api/students/{sid}/plans")
    assert plans_resp.status_code == 200
    plans = plans_resp.json()
    assert plans
    item = next(it for it in plans[0]["items"] if it["skill_id"] == "sql")
    assert isinstance(item["key_topics"], list) and item["key_topics"]
    assert isinstance(item["syllabus"], list) and item["syllabus"]
    assert item["url_verified"] is True

    q_resp = client.get(f"/api/plans/{plans[0]['id']}/items/{item['id']}/quiz")
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert q_data["passing_score"] == 70
    assert "keywords" not in json.dumps(q_data)

    ans = _answers_for_item(item["id"], item["skill_id"])
    v_resp = client.post(
        f"/api/plans/{plans[0]['id']}/items/{item['id']}/verify",
        json={"answers": ans},
    )
    assert v_resp.status_code == 200
    v_body = v_resp.json()
    assert v_body["passed"] is True
    assert v_body["closed_loop"]["verification_status"] == "verified"

    with Session(eng) as db:
        refreshed = db.get(StudentSkill, (sid, "sql"))
        assert refreshed.verification_status == "verified"
        assert refreshed.last_verified_score == 100
        aud = db.exec(
            select(AuditLog).where(
                AuditLog.entity_id == item["id"],
                AuditLog.action == "plan.quiz_verified",
            )
        ).first()
        assert aud is not None


def test_apply_completion_preserves_eligibility_and_senior_upgrades(client_engine):
    client, eng = client_engine
    up = client.post(
        "/api/students",
        data={"name": "Senior Gap Tester", "branch": "CSE", "year": "4", "cgpa": "8.4", "backlogs": "10"},
        files={"file": ("resume.txt", b"Skills: PostgreSQL, Python\n", "text/plain")},
    )
    assert up.status_code in (200, 201), up.text
    sid = up.json()["id"]

    with Session(eng) as db:
        db.merge(StudentSkill(student_id=sid, skill_id="postgresql", mentions=1, level="basic", evidence=["pg"], verification_status="claimed"))
        db.commit()

        senior_job = db.exec(select(Job).where(Job.experience_years >= 3)).first()
        if senior_job is not None:
            js = JobScore(
                job_id=senior_job.id,
                score=65,
                band="moderate",
                matched_required=[{"skill": "postgresql"}],
                missing_required=[{"skill": "kubernetes", "name": "Kubernetes", "category": "cloud"}],
                matched_preferred=[],
                missing_preferred=[],
                proficiency_gaps=[{"skill": "postgresql", "current_level": "basic", "target_level": "advanced"}],
                semantic_sim=0.6,
            )
            plan, items = build_plan(db, sid, senior_job, js, llm=None)
            skills = {it.skill_id for it in items}
            assert "postgresql" in skills
            out = apply_plan_item_completion(db, client.app.state.ctx, plan, items[0])
            assert out["status"] == "done"
            matches = db.exec(select(JobMatch).where(JobMatch.student_id == sid)).all()
            assert any(m.breakdown.get("seniority_penalty", 0) < 0 for m in matches)
            aud = db.exec(
                select(AuditLog).where(
                    AuditLog.entity_id == items[0].id,
                    AuditLog.action == "plan.item_completed",
                )
            ).first()
            assert aud is not None
