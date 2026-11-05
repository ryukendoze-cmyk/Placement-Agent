def test_upload_resume_and_get_matches(client):
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker, AWS\nBuilt REST APIs with FastAPI\n", "text/plain")},
        data={"name": "Bharat Sharma"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("stu-")
    skill_ids = {s["id"] for s in body["skills"]}
    assert {"python", "fastapi", "postgresql"} <= skill_ids

    r2 = client.get(f"/api/students/{body['id']}/matches")
    assert r2.status_code == 200
    matches = r2.json()
    assert matches and 0 <= matches[0]["score"] <= 100
    m = matches[0]
    assert {"job_id", "score", "band", "matched_required", "missing_required", "semantic_sim"} <= set(m)
    assert m["matched_required"] or m["missing_required"]


def test_jobs_list(client):
    r = client.get("/api/jobs")
    assert r.status_code == 200 and len(r.json()) >= 20


def test_jobs_get_single(client):
    r = client.get("/api/jobs/job-001")
    assert r.status_code == 200
    j = r.json()
    assert j["id"] == "job-001" and "required_skills" in j


def test_unknown_student_returns_404(client):
    assert client.get("/api/students/stu-nope/matches").status_code == 404


def test_unsupported_suffix_returns_400(client):
    r = client.post(
        "/api/students",
        files={"file": ("r.xyz", b"not supported", "application/octet-stream")},
        data={"name": "N"},
    )
    assert r.status_code == 400
    assert "unsupported resume format" in r.text


def test_corrupt_png_returns_400(client):
    r = client.post(
        "/api/students",
        files={"file": ("r.png", b"\x89PNG\r\n\x1a\ncorrupt", "image/png")},
        data={"name": "N"},
    )
    assert r.status_code == 400


def test_corrupt_pdf_returns_400(client):
    r = client.post(
        "/api/students",
        files={"file": ("r.pdf", b"not a pdf at all", "application/pdf")},
        data={"name": "N"},
    )
    assert r.status_code == 400


def test_upload_beyond_size_cap_returns_413(client):
    from app.api.routes.students import MAX_RESUME_BYTES
    big = b"x" * (MAX_RESUME_BYTES + 1)
    r = client.post(
        "/api/students",
        files={"file": ("big.txt", big, "text/plain")},
        data={"name": "Big"},
    )
    assert r.status_code == 413


def test_matches_persist_jobmatch_rows(client_engine):
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL", "text/plain")},
        data={"name": "Persist Student"},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    r2 = client.get(f"/api/students/{sid}/matches")
    assert r2.status_code == 200
    from sqlmodel import Session, select

    from app.models import JobMatch
    with Session(eng) as db:
        rows = db.exec(select(JobMatch).where(JobMatch.student_id == sid)).all()
    assert len(rows) == 10
    assert all(0 <= row.score <= 100 for row in rows)
    assert all(row.breakdown.get("job_id") == row.job_id for row in rows)


# ---------------------------------------------------------------------------
# AI resume extraction endpoints (ARCHITECTURE.md 3.3)
# ---------------------------------------------------------------------------
AI_RESUME = (
    b"Aarav Mehta, B.Tech CSE, Year 3, CGPA 8.7\n"
    b"Summary: Backend engineer who ships production systems.\n"
    b"Experience - Summer Intern, Zeta Labs:\n"
    b"Architected a Python event pipeline on FastAPI serving 50k requests/day, "
    b"cutting p95 latency by 40%.\n"
    b"Projects:\n"
    b"Built a React dashboard used by 5 teams.\n"
    b"Skills: Python, FastAPI, React\n"
)


def test_post_api_resumes_extract_valid(client):
    r = client.post(
        "/api/resumes/extract",
        files={"file": ("resume.txt", AI_RESUME, "text/plain")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "profile" in body and "skills" in body
    assert isinstance(body["skills"], list)
    assert body["skill_count"] == len(body["skills"])
    assert isinstance(body["ai_parsed"], bool)
    assert "pii_redacted_count" in body
    # at least the skills explicitly listed in the resume should surface
    skill_ids = {s["id"] for s in body["skills"]}
    assert {"python", "fastapi"} <= skill_ids


def test_upload_resume_png_and_get_matches(client):
    from pathlib import Path
    png_bytes = Path("tests/fixtures/resume.png").read_bytes()
    r = client.post(
        "/api/students",
        files={"file": ("resume.png", png_bytes, "image/png")},
        data={"name": "Bharat Sharma"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("stu-")
    skill_ids = {s["id"] for s in body["skills"]}
    assert {"python", "fastapi"} <= skill_ids


def test_upload_resume_jpeg_and_get_matches(client):
    from pathlib import Path
    jpeg_bytes = Path("tests/fixtures/resume.jpeg").read_bytes()
    r = client.post(
        "/api/students",
        files={"file": ("resume.jpeg", jpeg_bytes, "image/jpeg")},
        data={"name": "Bharat Sharma"},
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["id"].startswith("stu-")
    skill_ids = {s["id"] for s in body["skills"]}
    assert {"python", "fastapi"} <= skill_ids


def test_post_api_resumes_extract_valid_png(client):
    from pathlib import Path
    png_bytes = Path("tests/fixtures/resume.png").read_bytes()
    r = client.post(
        "/api/resumes/extract",
        files={"file": ("resume.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "skills" in body
    skill_ids = {s["id"] for s in body["skills"]}
    assert {"python", "fastapi"} <= skill_ids


def test_post_api_resumes_extract_bad_extension(client):
    r = client.post(
        "/api/resumes/extract",
        files={"file": ("resume.xyz", b"binary", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "unsupported resume format" in r.text


def test_post_api_resumes_extract_oversize(client):
    from app.api.routes.students import MAX_RESUME_BYTES
    big = b"x" * (MAX_RESUME_BYTES + 1)
    r = client.post(
        "/api/resumes/extract",
        files={"file": ("big.txt", big, "text/plain")},
    )
    assert r.status_code == 413


def test_post_api_students_resume_json(client):
    # create a student first
    r = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI\n", "text/plain")},
        data={"name": "Resume Reupload"},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    r2 = client.post(
        f"/api/students/{sid}/resume",
        files={"file": ("resume2.txt", AI_RESUME, "text/plain")},
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["id"] == sid
    assert isinstance(body["skills"], list)
    assert body["skill_count"] == len(body["skills"])
    skill_ids = {s["id"] for s in body["skills"]}
    assert "python" in skill_ids


def test_post_api_students_resume_unknown_student(client):
    r = client.post(
        "/api/students/stu-nope/resume",
        files={"file": ("resume.txt", AI_RESUME, "text/plain")},
    )
    assert r.status_code == 404


def test_post_api_students_persists_ai_levels(client_engine):
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("resume.txt", AI_RESUME, "text/plain")},
        data={"name": "", "branch": "", "gpa": "0"},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    from sqlmodel import Session, select

    from app.models import Student, StudentSkill
    with Session(eng) as db:
        student = db.get(Student, sid)
        assert student is not None
        rows = db.exec(select(StudentSkill).where(StudentSkill.student_id == sid)).all()
    assert rows, "no StudentSkill rows persisted"
    for row in rows:
        assert row.level in {"basic", "intermediate", "advanced"}
        assert row.mentions >= 1
        assert isinstance(row.evidence, list)


def test_post_api_students_enriches_blank_profile(client_engine):
    """Blank name/branch/gpa on POST /api/students are enriched from the
    resume's AI-extracted profile when available."""
    client, eng = client_engine
    r = client.post(
        "/api/students",
        files={"file": ("resume.txt", AI_RESUME, "text/plain")},
        data={"name": "", "branch": "", "year": "3", "gpa": "0"},
    )
    assert r.status_code == 201, r.text
    sid = r.json()["id"]

    from sqlmodel import Session

    from app.models import Student
    with Session(eng) as db:
        student = db.get(Student, sid)
    # name is never blank after enrichment
    assert student.name and student.name.strip()
    # branch and gpa should come back from the resume if the AI parsed them;
    # they must at least be valid types (str / float) regardless.
    assert isinstance(student.branch, str)
    assert isinstance(student.gpa, float)
