"""M24: structural validation of deployment artifacts (no docker required)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    p = ROOT / name
    assert p.exists(), f"missing deployment artifact: {name}"
    return p.read_text(encoding="utf-8")


def test_dockerfile_multi_stage_nonroot_healthcheck():
    df = _read("Dockerfile")
    lines = [l.strip() for l in df.splitlines()]
    froms = [l for l in lines if l.startswith("FROM")]
    assert len(froms) >= 2, "must be multi-stage"
    assert any("python:3.12-slim" in l for l in froms)
    assert "uv sync --frozen --no-dev" in df
    for asset in ("COPY app/", "job_scraper/", "data/", "cli.py", "main.py"):
        assert asset in df, f"Dockerfile must copy {asset}"
    assert "COPY pyproject.toml uv.lock" in df or "COPY pyproject.toml" in df
    users = [l for l in lines if l.startswith("USER")]
    assert users and all(u.split()[1] not in ("root", "0") for u in users)
    assert any(l.startswith("EXPOSE 8010") for l in lines)
    assert "HEALTHCHECK" in df
    assert "http://127.0.0.1:8010/healthz" in df


def test_dockerignore_excludes_vcs_and_cache():
    di = _read(".dockerignore")
    patterns = {l.strip() for l in di.splitlines() if l.strip() and not l.startswith("#")}
    for needed in (".venv", ".git", ".pytest_temp", ".pytest_cache",
                   ".ruff_cache", "__pycache__", ".env"):
        assert needed in patterns, f".dockerignore must exclude {needed}"


def test_docker_compose_services_and_wiring():
    cfg = yaml.safe_load(_read("docker-compose.yml"))
    svcs = cfg["services"]
    assert {"db", "redis", "web"} <= set(svcs)
    db = svcs["db"]
    assert db["image"] == "pgvector/pgvector:pg17"
    assert "healthcheck" in db
    assert db["environment"].get("POSTGRES_HOST_AUTH_METHOD") == "trust"
    redis = svcs["redis"]
    assert redis["image"] == "redis:7-alpine"
    assert "healthcheck" in redis
    web = svcs["web"]
    assert web["build"] == "."
    dep = web["depends_on"]
    assert "db" in dep and "redis" in dep
    for name in ("db", "redis"):
        assert dep[name].get("condition") == "service_healthy"
    assert any("8010" in str(p) for p in web["ports"])
    cmd = web["command"]
    assert "seed --demo-cohort" in cmd
    assert "uvicorn app.api.__main__:app" in cmd
    assert "--port 8010" in cmd
    env = web["environment"]
    assert env["DATABASE_URL"] == "postgresql+psycopg://postgres@db:5432/ps06"
    assert env["REDIS_URL"] == "redis://redis:6379/0"
    init_sql = ROOT / "docker" / "initdb" / "01-create-vector.sql"
    assert init_sql.exists()
    assert "CREATE EXTENSION" in init_sql.read_text(encoding="utf-8")
    assert "vector" in init_sql.read_text(encoding="utf-8")
    mounts = " ".join(str(m) for m in db.get("volumes", []))
    assert "initdb" in mounts and "/docker-entrypoint-initdb.d" in mounts


def test_ci_workflow_ruff_unit_and_pgvector_integration():
    wf = yaml.safe_load(_read(".github/workflows/ci.yml"))
    jobs = wf["jobs"]
    assert len(jobs) >= 2
    all_steps = " ".join(
        str(s.get("run", "")) for j in jobs.values() for s in j.get("steps", []))
    assert "ruff check ." in all_steps
    assert 'pytest -m "not integration"' in all_steps
    assert "pytest -m integration" in all_steps
    svc_found = False
    for j in jobs.values():
        for svc in (j.get("services") or {}).values():
            if "pgvector/pgvector:pg17" in str(svc.get("image", "")):
                svc_found = True
    assert svc_found, "integration job must spin up a pgvector/pgvector:pg17 service"


def test_entrypoints_and_healthz_route_are_real():
    main_py = (ROOT / "app" / "api" / "__main__.py").read_text(encoding="utf-8")
    assert "app = create_app" in main_py, "uvicorn target app.api.__main__:app must define app"
    admin = (ROOT / "app" / "api" / "routes" / "admin.py").read_text(encoding="utf-8")
    assert '"/healthz"' in admin, "healthcheck URL must map to a real route"
    cli = (ROOT / "app" / "cli.py").read_text(encoding="utf-8")
    assert "eval-benchmark" in cli or "seed" in cli
    assert "demo_cohort" in cli


def test_dev_dependencies_declared_for_artifact_tests():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "pyyaml" in pyproject, "yaml parsing in artifact tests needs a declared dev dep"
