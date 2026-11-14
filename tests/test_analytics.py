from sqlmodel import Session

from app.analytics.cohort import (
    branch_readiness_breakdown,
    cohort_executive_kpis,
    enriched_skill_gap_matrix,
    programme_recommendations,
    readiness_distribution,
    role_track_readiness,
    skill_gap_report,
    student_readiness_matrix,
    top_requested_courses,
)

COHORT = [
    ("Asha", "Skills: Python, FastAPI, Docker"),
    ("Rohan", "Skills: Python, Git, Linux"),
    ("Meera", "Skills: JavaScript, React, HTML"),
    ("Arjun", "Skills: pandas, statistics, SQL"),
    ("Karan", "Skills: Python, FastAPI, Docker"),
    ("Ananya", "Skills: HTML, CSS, JavaScript"),
]


def _cohort(client):
    sids = []
    for name, body in COHORT:
        sid = client.post("/api/students", files={"file": ("r.txt", body.encode("utf-8"), "text/plain")},
                          data={"name": name}).json()["id"]
        client.get(f"/api/students/{sid}/matches")
        sids.append(sid)
    return sids


def test_skill_gap_report_shape_and_ordering(client_engine):
    client, eng = client_engine
    _cohort(client)
    with Session(eng) as db:
        report = skill_gap_report(db)
    assert report, "expected a non-empty gap report"
    assert report[0]["demand"] > 0
    assert 0 <= report[0]["coverage"] < 1.0
    scores = [r["gap_score"] for r in report]
    assert scores == sorted(scores, reverse=True)
    top = report[0]
    assert top["skill"] == "git"  # Git: demanded by 5 top jobs, held by 1/6 students (gap 4.167)


def test_readiness_distribution_sums_to_cohort(client_engine):
    client, eng = client_engine
    _cohort(client)
    with Session(eng) as db:
        dist = readiness_distribution(db)
    assert sum(dist.values()) == 6


def test_courses_and_recommendations_after_plans(client_engine):
    client, eng = client_engine
    sids = _cohort(client)
    for sid in sids:
        assert client.post(f"/api/students/{sid}/plans", json={}).status_code in (200, 201)
    with Session(eng) as db:
        courses = top_requested_courses(db)
        recs = programme_recommendations(db)
    assert isinstance(courses, list)
    assert len(recs) >= 1
    assert all("course_title" in r for r in recs)


def test_admin_analytics_json_and_page(client_engine):
    client, _ = client_engine
    _cohort(client)
    r = client.get("/api/admin/analytics")
    assert r.status_code == 200
    b = r.json()
    for key in (
        "skill_gap_report",
        "readiness_distribution",
        "top_requested_courses",
        "programme_recommendations",
        "executive_kpis",
        "branch_readiness",
        "role_track_readiness",
        "student_matrix",
        "enriched_skill_gaps",
    ):
        assert key in b
    h = client.get("/admin")
    assert h.status_code == 200
    assert "analytics" in h.text.lower()
    assert "readiness" in h.text.lower()
    for elem_id in (
        "gapChart",
        "bandChart",
        "branchReadinessChart",
        "roleReadinessChart",
        "cohort-student-matrix-card",
        "cohort-skill-intervention-card",
    ):
        assert elem_id in h.text
    assert 'id="launch-bootcamp"' not in h.text
    assert "Most requested courses" not in h.text
    assert "youtube" not in h.text.lower()


def test_v4_cohort_intelligence_functions(client_engine):
    client, eng = client_engine
    sids = _cohort(client)
    for sid in sids[:2]:
        assert client.post(f"/api/students/{sid}/plans", json={}).status_code in (200, 201)
    with Session(eng) as db:
        kpis = cohort_executive_kpis(db)
        assert kpis["total_students"] == 6
        assert 0.0 <= kpis["mean_readiness_index"] <= 100.0
        assert 0.0 <= kpis["verified_authenticity_pct"] <= 100.0

        branches = branch_readiness_breakdown(db)
        assert any(b["branch"] == "CSE" for b in branches)

        tracks = role_track_readiness(db)
        assert len(tracks) == 5
        assert {t["track"] for t in tracks} == {
            "Backend & Cloud",
            "Frontend & Full-Stack",
            "Data & AI/ML",
            "DevOps & SRE",
            "Core ECE / Embedded",
        }

        matrix = student_readiness_matrix(db)
        assert len(matrix) == 6
        assert [r["name"] for r in matrix] == sorted(r["name"] for r in matrix)

        gaps = enriched_skill_gap_matrix(db)
        assert gaps
        assert all("youtube" not in g["intervention_type"].lower() for g in gaps)
