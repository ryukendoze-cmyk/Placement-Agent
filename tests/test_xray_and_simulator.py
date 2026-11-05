import io
from types import SimpleNamespace

from sqlmodel import Session, select

from app.matching.engine import higher_level, job_eligibility, job_tier, score_job
from app.matching.simulator import simulate_counterfactual
from app.resume.xray import improve_weak_bullets, pick_weak_bullets, xray_resume
from app.taxonomy.taxonomy import Taxonomy, load_taxonomy

T = Taxonomy([
    {"id": "python", "name": "Python", "category": "languages", "aliases": ["python"]},
    {"id": "fastapi", "name": "FastAPI", "category": "web_backend", "aliases": ["fastapi"]},
    {"id": "postgresql", "name": "PostgreSQL", "category": "databases", "aliases": ["postgresql"]},
    {"id": "docker", "name": "Docker", "category": "devops_cloud", "aliases": ["docker"]},
    {"id": "aws", "name": "AWS", "category": "devops_cloud", "aliases": ["aws"]},
])
JOB = SimpleNamespace(id="job-1", required_skills=["python", "fastapi", "postgresql"],
                      preferred_skills=["docker", "aws"])


# ---- tech-debt: unknown job skill must not crash --------------------------
def test_unknown_job_skill_is_skipped_not_crash():
    job = SimpleNamespace(id="j-x", required_skills=["python", "no-such-skill-xyz"], preferred_skills=[])
    s = score_job({"python": ["l"]}, job, sim=0.5, taxonomy=T)
    assert {m["skill"] for m in s.matched_required} == {"python"}
    assert s.score > 0


# ---- proficiency-aware scoring (backward compatible) ----------------------
def test_score_job_levels_none_matches_base():
    s = score_job({"python": ["l"], "fastapi": ["l"], "postgresql": ["l"], "docker": ["l"]}, JOB, sim=0.9, taxonomy=T)
    assert s.proficiency_bonus == 0
    assert s.score == 88  # 0.6*1.0 + 0.2*0.5 + 0.2*0.9


def test_proficiency_raises_score_and_caps_at_100():
    base = score_job({"python": ["l"], "fastapi": ["l"], "postgresql": ["l"]}, JOB, sim=0.9, taxonomy=T)
    adv = score_job({"python": ["l"], "fastapi": ["l"], "postgresql": ["l"]}, JOB, sim=0.9, taxonomy=T,
                    levels={"python": "advanced", "fastapi": "advanced", "postgresql": "advanced"})
    assert adv.score > base.score
    assert adv.proficiency_bonus == 10
    capped = score_job({"python": ["l"], "fastapi": ["l"], "postgresql": ["l"], "docker": ["l"]}, JOB, sim=1.0,
                       taxonomy=T, levels={"python": "advanced", "fastapi": "advanced", "postgresql": "advanced"})
    assert capped.score <= 100


def test_matched_required_carries_level():
    s = score_job({"python": ["l"]}, JOB, sim=0.5, taxonomy=T, levels={"python": "advanced"})
    assert s.matched_required[0]["level"] == "advanced"


def test_higher_level_never_downgrades():
    assert higher_level("advanced", "intermediate") == "advanced"
    assert higher_level("basic", "intermediate") == "intermediate"
    assert higher_level(None, "intermediate") == "intermediate"


def test_job_tier_boundaries():
    assert job_tier(SimpleNamespace(salary_lpa=[0, 7])) == "standard"
    assert job_tier(SimpleNamespace(salary_lpa=[8, 14])) == "dream"
    assert job_tier(SimpleNamespace(salary_lpa=[15, 25])) == "super_dream"
    assert job_tier(SimpleNamespace(salary_lpa=None)) == "standard"


def test_job_eligibility_reasons():
    ok, reasons = job_eligibility(SimpleNamespace(gpa=7.0, branch="CSE"),
                                  SimpleNamespace(min_gpa=8.0, allowed_branches=[]))
    assert ok is False and any("GPA" in r for r in reasons)
    ok, reasons = job_eligibility(SimpleNamespace(gpa=9.0, branch="CSE"),
                                  SimpleNamespace(min_gpa=0.0, allowed_branches=["ECE"]))
    assert ok is False and any("Branch" in r for r in reasons)
    ok, reasons = job_eligibility(SimpleNamespace(gpa=9.0, branch="cse"),
                                  SimpleNamespace(min_gpa=0.0, allowed_branches=["CSE"]))
    assert ok is True and reasons == []


# ---- counterfactual simulator ---------------------------------------------
def test_simulator_adding_missing_skill_raises_score():
    res = simulate_counterfactual({"python": ["l"]}, JOB, ["fastapi", "postgresql"], 0.9, T,
                                  courses_for_skill={"fastapi": {"id": "c1", "hours": 4},
                                                     "postgresql": {"id": "c2", "hours": 6}})
    assert res["projected_score"] > res["current_score"]
    assert res["delta"] > 0
    assert res["impacted_skills"] == ["fastapi", "postgresql"]
    assert res["total_hours"] == 10
    assert res["missing_required_after"] == []


def test_simulator_dedupes_course_hours():
    res = simulate_counterfactual({"python": ["l"]}, JOB, ["fastapi", "postgresql"], 0.9, T,
                                  courses_for_skill={"fastapi": {"id": "c1", "hours": 8},
                                                     "postgresql": {"id": "c1", "hours": 8}})
    assert res["total_hours"] == 8  # same course counted once


def test_simulator_neutral_skill_zero_delta():
    res = simulate_counterfactual({"python": ["l"], "fastapi": ["l"], "postgresql": ["l"]}, JOB, ["rust"], 0.9, T,
                                  courses_for_skill={"rust": {"id": "c9", "hours": 20}})
    assert res["delta"] == 0
    assert "rust" in res["neutral_skills"]
    assert res["impacted_skills"] == []


def test_simulate_endpoint(client_engine):
    client, _ = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
                    data={"name": "Sim"})
    sid = r.json()["id"]
    res = client.post(f"/api/students/{sid}/matches/job-001/simulate", json={"skills": ["postgresql", "git"]})
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == "job-001"
    assert {"current_score", "projected_score", "delta", "total_hours"} <= set(body)
    assert body["projected_score"] >= body["current_score"]


# ---- Resume X-Ray ----------------------------------------------------------
RICH = """
Bharat Kumar
bharat@example.com | +91 98765 43210 | Bengaluru
github.com/bharat

EDUCATION
B.Tech CSE, 2023, JKLU (CGPA 8.5)

PROJECTS
- Built a FastAPI REST API serving 50k requests/day, reducing p95 latency by 40%
- Developed a PostgreSQL data pipeline processing 2M rows in 3ms

EXPERIENCE
- Engineered a Docker + Kubernetes deployment that cut infra cost by 25%
- Led a team of 3 to ship an ML dashboard used by 120 users
"""


def test_xray_rich_resume_is_deterministic_and_scored():
    r1 = xray_resume(RICH, load_taxonomy())
    r2 = xray_resume(RICH, load_taxonomy())
    assert r1 == r2
    assert 0 <= r1["ats_score"] <= 100
    assert r1["ats_score"] >= 55
    assert r1["sections"]["education"] and r1["sections"]["projects"] and r1["sections"]["experience"]
    assert any("github.com" in link for link in r1["links"])
    assert r1["metrics"]["quantified_bullets"] >= 2
    assert r1["metrics"]["action_verb_bullets"] >= 2
    assert 0 <= r1["sub_scores"]["impact_quantification"] <= 100


def test_xray_poor_resume_scores_low():
    r = xray_resume("I like computers. I did some stuff. Web related things and coding in general across various projects.",
                    load_taxonomy())
    assert r["ats_score"] < 50
    assert r["metrics"]["quantified_bullets"] == 0


def test_pick_weak_bullets_excludes_headers_and_skill_lists():
    text = ("EDUCATION\n"
            "B.Tech CSE at Some University in 2023 with overall good grades\n"
            "Tech: Python, Git, Docker\n"
            "This was a long enough line about general web development work in 2022")
    got = pick_weak_bullets(text, load_taxonomy())
    assert len(got) <= 3
    assert all(g.strip() != "EDUCATION" for g in got)
    assert all("Python, Git, Docker" not in g for g in got)


def test_improve_bullets_uses_only_present_skills():
    text = ("PROJECTS\n"
            "Worked on a web app that handles user requests and shows data on a page\n"
            "Made some database changes to store information better for the team")
    got = improve_weak_bullets(text, ["Python", "Docker"], __import__("app.llm.mock", fromlist=["MockLLM"]).MockLLM(),
                               taxonomy=load_taxonomy())
    assert 1 <= len(got) <= 3
    assert all(item["improved"] for item in got)
    assert any("Python" in item["improved"] for item in got)
    for item in got:
        assert "Rust" not in item["improved"] and "Kubernetes" not in item["improved"]


# ---- tech-debt: parser + analytics + serialization ------------------------
def test_docx_table_text_is_extracted():
    from docx import Document

    from app.resume.parser import parse_resume_bytes
    doc = Document()
    doc.add_paragraph("Skills and experience summary")
    table = doc.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Python"
    table.cell(0, 1).text = "PostgreSQL"
    buf = io.BytesIO()
    doc.save(buf)
    text = parse_resume_bytes(buf.getvalue(), ".docx")
    assert "Python" in text and "PostgreSQL" in text


def test_readiness_distribution_always_has_all_bands(client_engine):
    from app.analytics.cohort import readiness_distribution
    _client, eng = client_engine
    with Session(eng) as db:
        dist = readiness_distribution(db)
    assert set(dist.keys()) == {"strong", "moderate", "developing", "early"}


def test_matches_salary_lpa_null_guard(client_engine):
    from app.models import Job
    client, eng = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")}, data={"name": "S"})
    sid = r.json()["id"]
    with Session(eng) as db:
        job = db.get(Job, "job-001")
        job.salary_lpa = None
        db.add(job)
        db.commit()
    m = client.get(f"/api/students/{sid}/matches").json()
    row = next(x for x in m if x["job_id"] == "job-001")
    assert row["salary_lpa"] == [0, 0]


def test_matches_payload_has_tier_and_eligibility(client_engine):
    client, _ = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python\n", "text/plain")},
                    data={"name": "S", "branch": "CSE", "year": 3, "gpa": 8.0})
    sid = r.json()["id"]
    m = client.get(f"/api/students/{sid}/matches").json()
    assert m
    top = m[0]
    assert top["tier"] in {"standard", "dream", "super_dream"}
    assert "eligible" in top and "proficiency_bonus" in top and "eligibility_reasons" in top


def test_reupload_refreshes_stale_matches(client_engine):
    from app.models import JobMatch, StudentSkill
    client, eng = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")}, data={"name": "R"})
    sid = r.json()["id"]
    client.get(f"/api/students/{sid}/matches")
    client.post(f"/students/{sid}/resume", files={"file": ("r.txt", b"Skills: React, TypeScript\n", "text/plain")})
    with Session(eng) as db:
        skills = {ss.skill_id for ss in db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()}
        assert "react" in skills and "python" not in skills
        assert db.exec(select(JobMatch).where(JobMatch.student_id == sid)).first() is not None


def test_resume_xray_endpoint(client_engine):
    client, _ = client_engine
    r = client.post("/api/students", files={"file": ("r.txt", b"Skills: Python, Docker\nBuilt an API handling 5k requests.\n", "text/plain")},
                    data={"name": "X"})
    sid = r.json()["id"]
    res = client.get(f"/api/students/{sid}/resume/xray")
    assert res.status_code == 200
    body = res.json()
    assert "ats_score" in body and "sub_scores" in body and "sections" in body


def test_demo_cohort_students_have_resume_text(client_engine):
    from app.embeddings.embedder import HashingEmbedder
    from app.models import Student
    from app.seed import seed_demo_cohort
    _client, eng = client_engine
    with Session(eng) as db:
        seed_demo_cohort(db, taxonomy=load_taxonomy(), embedder=HashingEmbedder())
        students = db.exec(select(Student)).all()
        assert students and all((s.resume_text or "").strip() for s in students)


def test_simulator_marks_proficiency_upgrade_as_impacted():
    # student already has python at basic; simulating an upgrade must register impact
    res = simulate_counterfactual({"python": ["l"]}, JOB, ["python"], 0.9, T,
                                  levels={"python": "basic"},
                                  courses_for_skill={"python": {"id": "c1", "hours": 5}})
    assert res["impacted_skills"] == ["python"]
    assert res["neutral_skills"] == []
