from app import cli
from app.db import make_engine
from app.embeddings.embedder import HashingEmbedder


def test_seed_command_summary_output(capsys, monkeypatch):
    """seed must init a real engine, seed real data and print counts.

    Three monkeypatches are the only seams: _make_engine (sqlite instead of
    Postgres, so no DB needed in unit tests), _make_embedder (hashing, so no
    sentence-transformers model load) and settings.database_url (sqlite, so
    _make_store returns None — no pgvector store over a sqlite session).
    """
    monkeypatch.setattr(cli, "_make_engine", lambda url: make_engine("sqlite://"))
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(cli.settings, "database_url", "sqlite://")
    rc = cli.main(["seed"])
    out = capsys.readouterr().out
    from app.seed import _json
    from app.taxonomy.taxonomy import load_taxonomy

    expected_skills = str(len(load_taxonomy().all()))
    expected_jobs = str(len(_json("jobs.json")))
    expected_courses = str(len(_json("courses.json")))

    assert rc == 0
    assert expected_skills in out
    assert expected_jobs in out
    assert expected_courses in out
    for word in ("skills", "jobs", "courses"):
        assert word in out


def test_seed_command_refuses_unsupported_vector_dim(capsys, monkeypatch):
    """PgVectorStore only supports dim=384; a non-384 dim must fail cleanly, not crash."""
    monkeypatch.setattr(cli.settings, "database_url", "postgresql+psycopg://u@localhost/db")
    monkeypatch.setattr(cli.settings, "embedding_dim", 512)
    rc = cli.main(["seed"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "384" in err


def test_seed_command_handles_db_errors_friendly(capsys, monkeypatch):
    """SQLAlchemy errors (dead Postgres, etc.) must print the friendly error, not a traceback."""
    import sqlalchemy.exc

    def _boom(url):
        raise sqlalchemy.exc.OperationalError(
            "psycopg.OperationalError", None, Exception("connection refused")
        )

    monkeypatch.setattr(cli, "_make_engine", _boom)
    monkeypatch.setattr(
        cli.settings, "database_url", "postgresql+psycopg://postgres@127.0.0.1:5432/ps06"
    )
    rc = cli.main(["seed"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "seed failed" in err


def test_demo_ranks_jobs_with_explainable_table(tmp_path, capsys, monkeypatch):
    p = tmp_path / "resume.txt"
    p.write_text("Skills: Python, FastAPI, PostgreSQL", encoding="utf-8")
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    rc = cli.main(["demo", "--file", str(p), "--name", "T", "--top", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "rank" in out
    assert "missing required" in out
    assert out.strip().count("\n") >= 3
    assert any(band in out for band in ("developing", "moderate", "strong", "early"))


def test_demo_bad_file_returns_1(tmp_path, capsys):
    p = tmp_path / "r.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\nnot a resume")
    rc = cli.main(["demo", "--file", str(p), "--name", "T"])
    captured = capsys.readouterr()
    assert rc == 1
    assert "could not" in (captured.out + captured.err)


def test_demo_png_resume(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    rc = cli.main(["demo", "--file", "tests/fixtures/resume.png", "--name", "Bharat", "--top", "3"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Bharat" in out


def test_unknown_command_returns_nonzero():
    assert cli.main(["bogus"]) == 2


def test_seed_demo_cohort_creates_cohort(capsys, monkeypatch):
    from sqlmodel import Session, select

    from app.db import make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.models import JobMatch, Student, UpskillPlan

    eng = make_engine("sqlite://")
    monkeypatch.setattr(cli, "_make_engine", lambda url: eng)
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(cli.settings, "database_url", "sqlite://")
    rc = cli.main(["seed", "--demo-cohort"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "demo cohort" in out
    with Session(eng) as db:
        assert len(db.exec(select(Student)).all()) >= 10
        assert len(db.exec(select(JobMatch)).all()) > 0
        assert len(db.exec(select(UpskillPlan)).all()) >= 1

def test_cli_purge_live_jobs_command(tmp_path, capsys, monkeypatch):
    import json

    from sqlmodel import Session, select

    from app.db import make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.models import Job

    eng = make_engine("sqlite://")
    Job.metadata.create_all(eng)
    with Session(eng) as db:
        db.add(Job(id="job-live-cli", title="Live Job", company="Corp", description="d",
                   job_url="https://www.naukri.com/job/1", source="naukri"))
        db.add(Job(id="job-local-cli", title="Local Job", company="JKLU", description="d",
                   job_url="https://jklu.edu.in/drive/1", source="local"))
        db.commit()

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = out_dir / "raw_scraped_jobs.json"
    raw_file.write_text(json.dumps([{"title": "temp"}]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(cli, "_make_engine", lambda url: eng)
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(cli.settings, "database_url", "sqlite://")

    rc = cli.main(["purge-live-jobs"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Purged 1 live job(s)" in out

    with Session(eng) as db:
        remaining = db.exec(select(Job)).all()
        assert len(remaining) == 1
        assert remaining[0].id == "job-local-cli"

    assert json.loads(raw_file.read_text(encoding="utf-8")) == []


def test_cli_purge_live_jobs_keep_output(tmp_path, capsys, monkeypatch):
    import json

    from app.db import make_engine
    from app.embeddings.embedder import HashingEmbedder
    from app.models import Job

    eng = make_engine("sqlite://")
    Job.metadata.create_all(eng)

    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_file = out_dir / "raw_scraped_jobs.json"
    raw_file.write_text(json.dumps([{"title": "preserve me"}]), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(cli, "_make_engine", lambda url: eng)
    monkeypatch.setattr(cli, "_make_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(cli.settings, "database_url", "sqlite://")

    rc = cli.main(["purge-live-jobs", "--keep-output"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Purged 0 live job(s)" in out
    assert json.loads(raw_file.read_text(encoding="utf-8")) == [{"title": "preserve me"}]

