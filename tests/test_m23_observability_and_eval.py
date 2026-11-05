"""M23: /healthz, /readyz, system telemetry, and the automated eval benchmark."""
import math

from sqlmodel import Session, select

from app.models import AgentRun, AgentStep, Job, Student

# ---------------------------------------------------------------------------
# 23.1 health / readiness / telemetry endpoints
# ---------------------------------------------------------------------------

def test_healthz_public_and_shape(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "ps06-careeros", "version": "3.0.0"}
    # public: even an explicit student role must not be blocked
    assert client.get("/healthz", headers={"x-demo-role": "student"}).status_code == 200


def test_readyz_ready_on_seeded_client(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert all(v is True for v in body["checks"].values())
    for key in ("database", "taxonomy_skills", "sops", "question_bank", "embedder"):
        assert body["checks"][key] is True


def test_system_telemetry_shape_and_rbac(client):
    r = client.get("/api/admin/system-telemetry")
    assert r.status_code == 200
    body = r.json()
    am = body["agent_metrics"]
    for key in ("total_runs", "avg_latency_ms", "p95_latency_ms",
                "avg_grounding_score", "fallback_rate", "tool_invocation_counts"):
        assert key in am
    assert set(body["llm"].keys()) >= {"providers", "totals"}
    ent = body["entities"]
    for key in ("students", "jobs", "local_jobs", "scraped_jobs", "matches",
                "plans", "interviews", "applications", "approvals", "audit_logs"):
        assert key in ent
    assert ent["local_jobs"] + ent["scraped_jobs"] == ent["jobs"]
    assert client.get("/api/admin/system-telemetry",
                      headers={"x-demo-role": "student"}).status_code == 403


def test_system_telemetry_agent_metrics_computed(client):
    eng = client.app.state.ctx.engine
    run1 = AgentRun(id="run-aaaaaaaa", student_id=None, goal="g1",
                    status="completed", total_latency_ms=100,
                    citations={"verified": ["job-001"], "grounding_score": 80},
                    fallback_triggered=False)
    run2 = AgentRun(id="run-bbbbbbbb", student_id=None, goal="g2",
                    status="completed", total_latency_ms=300,
                    citations={"verified": []}, fallback_triggered=True)
    with Session(eng) as db:
        db.add(run1)
        db.add(run2)
        db.add(AgentStep(id="step-aaaaaaaa", run_id="run-aaaaaaaa", step_order=0,
                         tool_name="search_jobs"))
        db.add(AgentStep(id="step-bbbbbbbb", run_id="run-aaaaaaaa", step_order=1,
                         tool_name="rank_jobs"))
        db.add(AgentStep(id="step-cccccccc", run_id="run-bbbbbbbb", step_order=0,
                         tool_name="verify_policy"))
        db.commit()
    body = client.get("/api/admin/system-telemetry").json()
    am = body["agent_metrics"]
    assert am["total_runs"] == 2
    assert am["avg_latency_ms"] == 200.0
    assert am["p95_latency_ms"] == 300
    assert am["avg_grounding_score"] == 80.0
    assert am["fallback_rate"] == 0.5
    assert am["tool_invocation_counts"] == {"search_jobs": 1, "rank_jobs": 1,
                                            "verify_policy": 1}


def test_eval_benchmark_endpoint(client):
    r = client.get("/api/admin/eval-benchmark")
    assert r.status_code == 200
    body = r.json()
    for key in ("sop_rag", "matching", "planner_dag", "agent_grounding", "summary"):
        assert key in body
    assert body["summary"]["rag_recall_at_3"] >= 0.9
    assert body["planner_dag"]["violations"] == 0
    assert body["planner_dag"]["cycles"] == 0
    assert body["agent_grounding"]["interception_rate"] == 1.0
    assert client.get("/api/admin/eval-benchmark",
                      headers={"x-demo-role": "student"}).status_code == 403


# ---------------------------------------------------------------------------
# 23.2 run_evaluation_benchmark deep checks
# ---------------------------------------------------------------------------

def _bench(client):
    from app.eval.benchmark import run_evaluation_benchmark
    ctx = client.app.state.ctx
    with Session(client.app.state.ctx.engine) as session:
        return run_evaluation_benchmark(session, ctx)


def test_benchmark_sop_rag_metrics(client):
    rag = _bench(client)["sop_rag"]
    assert rag["queries"] == 10
    assert rag["recall_at_3"] >= 0.9
    assert rag["mean_reciprocal_rank"] > 0.5
    assert rag["citation_grounding_score"] == 100


def test_benchmark_matching(client):
    m = _bench(client)["matching"]
    assert m["fresher_senior"]["senior_eligible"] is False
    assert "experienced" in m["fresher_senior"]["reasons"][0].lower()
    assert m["fresher_senior"]["fresher_eligible"] is True
    d = m["depth_monotonicity"]
    assert d["basic"] < d["intermediate"] < d["advanced"]
    s = m["sop_eligibility"]
    assert s["low_gpa_eligible"] is False
    assert s["low_gpa_gpa_cutoff_passed"] is False
    assert s["high_gpa_eligible"] is True
    assert s["high_gpa_gpa_cutoff_passed"] is True


def test_benchmark_planner_dag(client):
    dag = _bench(client)["planner_dag"]
    assert dag["profiles"] == 10
    assert dag["violations"] == 0
    assert dag["cycles"] == 0
    assert dag["valid"] is True


def test_benchmark_grounding_interception(client):
    g = _bench(client)["agent_grounding"]
    assert g["interception_rate"] == 1.0
    assert set(g["hallucinated"]) <= set(g["unverified"])
    assert {"job-999", "job-ext-deadbeef00", "SOP-99"} <= set(g["unverified"])


def test_benchmark_leaves_no_db_rows(client):
    _bench(client)
    with Session(client.app.state.ctx.engine) as db:
        assert db.exec(select(Student)).all().__len__() >= 0
        leftovers = [s.id for s in db.exec(select(Student)).all() if s.id.startswith("eval-stu-")]
        job_left = [j.id for j in db.exec(select(Job)).all() if j.id.startswith("eval-job-")]
        assert leftovers == []
        assert job_left == []


def test_benchmark_savepoint_safe_with_committing_vector_store(client):
    """Regression: PgVectorStore.upsert commits the session, which would break
    the nested SAVEPOINT inside _sop_eligibility_benchmark if rank_jobs tried
    to upsert eval-job-gpa."""
    from types import SimpleNamespace

    from app.eval.benchmark import run_evaluation_benchmark

    ctx = client.app.state.ctx

    class _CommittingStore:
        def __init__(self, session):
            self._session = session

        def get(self, kind, ref_id):
            return None

        def upsert(self, kind, ref_id, vector):
            self._session.commit()
            raise AssertionError("benchmark must not upsert ephemeral job into vector_store")

    with Session(ctx.engine) as session:
        pg_like_ctx = SimpleNamespace(
            engine=ctx.engine,
            taxonomy=ctx.taxonomy,
            vector_store=_CommittingStore(session),
            embedder=ctx.embedder,
            job_vectors={},
        )
        res = run_evaluation_benchmark(session, pg_like_ctx)
        assert res["summary"]["all_passed"] is True


def test_p95_math_edge_cases():
    from app.api.routes.admin import _p95
    assert _p95([]) == 0.0
    assert _p95([42]) == 42
    assert _p95([100, 300]) == 300
    vals = list(range(1, 21))
    assert _p95(vals) == math.ceil(0.95 * 20)  # index 18 (0-based) of sorted 1..20


def test_cli_eval_benchmark(monkeypatch):
    import app.cli as app_cli
    from app.db import make_engine
    from app.embeddings.embedder import HashingEmbedder
    eng = make_engine("sqlite://")
    monkeypatch.setattr(app_cli, "_make_engine", lambda url: eng)
    monkeypatch.setattr(app_cli, "_make_embedder", lambda: HashingEmbedder())
    monkeypatch.setattr(app_cli, "_make_store", lambda db, url: None)
    rc = app_cli.main(["eval-benchmark"])
    assert rc == 0
