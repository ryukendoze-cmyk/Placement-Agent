from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import socket
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.api.main import AppContext, create_app
from app.campus.agent import run_cohort_rematch, validate_drive_payload
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.models import (
    AgentRun,
    AuditLog,
    CampusDrive,
    Job,
    JobMatch,
    PlanItem,
    UpskillPlan,
)
from app.scraper.normalizer import _is_safe_public_http_url, fetch_external_job_description
from app.scraper.providers import StealthHarvesterProvider
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy
from app.workers.task_queue import TaskQueueManager, TaskState
from job_scraper.core.client import BrowserStealthContext
from job_scraper.extractors.base import BaseExtractor
from job_scraper.extractors.indeed import parse_indeed_html

TPO_HDR = {"X-Demo-Role": "tpo_admin"}
STUDENT_HDR = {"X-Demo-Role": "student"}


def _make_file_client(tmp_path: Path):
    eng = make_engine(f"sqlite:///{tmp_path / 'audit_test.db'}")
    init_db(eng)
    ctx = AppContext(
        engine=eng,
        taxonomy=load_taxonomy(),
        embedder=HashingEmbedder(),
        vector_store=MemoryVectorStore(),
        llm=MockLLM(),
    )
    with Session(eng) as db:
        seed(db, ctx.taxonomy)
    return TestClient(create_app(ctx)), eng, ctx


# ---------------------------------------------------------------------------
# SCR-01: Path confinement & .json check on /api/jobs/scrape/ingest-output
# ---------------------------------------------------------------------------


def test_scr01_ingest_output_rejects_outside_dir_and_non_json(
    client_engine, monkeypatch, tmp_path: Path
):
    client, _eng = client_engine
    allowed_dir = (tmp_path / "output").resolve()
    allowed_dir.mkdir()
    outside = tmp_path / "secret.json"
    outside.write_text("[]", encoding="utf-8")
    non_json = allowed_dir / "notes.txt"
    non_json.write_text("[]", encoding="utf-8")

    import app.api.routes.jobs as jobs_route

    monkeypatch.setattr(jobs_route, "INGEST_OUTPUT_DIR", allowed_dir)

    r_traversal = client.post(
        "/api/jobs/scrape/ingest-output",
        json={"file_path": str(outside)},
        headers=TPO_HDR,
    )
    assert r_traversal.status_code == 403

    r_rel_escape = client.post(
        "/api/jobs/scrape/ingest-output",
        json={"file_path": "../secret.json"},
        headers=TPO_HDR,
    )
    assert r_rel_escape.status_code == 403

    r_non_json = client.post(
        "/api/jobs/scrape/ingest-output",
        json={"file_path": str(non_json)},
        headers=TPO_HDR,
    )
    assert r_non_json.status_code == 400


# ---------------------------------------------------------------------------
# SCR-02: SSRF DNS resolution, redirect validation, and response byte cap
# ---------------------------------------------------------------------------


def test_scr02_ssrf_dns_redirect_and_byte_cap(monkeypatch):
    monkeypatch.delenv("PS06_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("PS06_SANDBOX_SCRAPER", raising=False)
    assert _is_safe_public_http_url("http://127.0.0.1/job") is False

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host == "evil.nip.io":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]
        if host == "local.internal":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 0))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert _is_safe_public_http_url("https://evil.nip.io/job") is False
    assert _is_safe_public_http_url("https://example.com/job") is True

    requested_urls: list[str] = []

    class _FakeStreamResp:
        def __init__(self, status_code: int, headers: dict[str, str], chunks: list[bytes]):
            self.status_code = status_code
            self.headers = headers
            self._chunks = chunks
            self.encoding = "utf-8"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def iter_bytes(self, chunk_size: int = 8192):
            yield from self._chunks

    class _FakeClient:
        def __init__(self, **kwargs):
            self.follow_redirects = kwargs.get("follow_redirects")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def stream(self, method: str, url: str, headers=None):
            assert self.follow_redirects is False
            requested_urls.append(url)
            if url == "https://example.com/redirect":
                return _FakeStreamResp(302, {"location": "http://local.internal/secret"}, [])
            # Return 1.2 MB body in chunks to verify 1 MB cap
            big_chunk = b"<p>" + (b"a" * 100_000) + b"</p>"
            return _FakeStreamResp(200, {}, [big_chunk] * 12)

    monkeypatch.setattr(httpx, "Client", _FakeClient)

    assert fetch_external_job_description("https://example.com/redirect") == ""
    assert requested_urls == ["https://example.com/redirect"]

    text = fetch_external_job_description("https://example.com/large")
    assert len(text.encode("utf-8")) <= 1_000_000


# ---------------------------------------------------------------------------
# SCR-03: RBAC on agent reply_run & actor_role validation
# ---------------------------------------------------------------------------


def test_scr03_agent_reply_rbac_and_actor_role_validation(client_engine, monkeypatch):
    client, eng = client_engine
    with Session(eng) as db:
        db.add(
            AgentRun(
                id="run-global-admin",
                student_id=None,
                actor_role="tpo_admin",
                goal="Audit cohort drives",
                status="awaiting_input",
            )
        )
        db.commit()

    r_student = client.post(
        "/api/agent/runs/run-global-admin/reply",
        json={"reply": "continue"},
        headers=STUDENT_HDR,
    )
    assert r_student.status_code == 403

    import app.agent.orchestrator as orch

    monkeypatch.setattr(
        orch,
        "resume_agent_run",
        lambda ctx, session, run_id, user_reply: session.get(AgentRun, run_id),
    )
    r_tpo = client.post(
        "/api/agent/runs/run-global-admin/reply",
        json={"reply": "continue"},
        headers=TPO_HDR,
    )
    assert r_tpo.status_code == 200

    r_bad_role = client.post(
        "/api/agent/run",
        json={"goal": "hello", "actor_role": "superuser"},
        headers=TPO_HDR,
    )
    assert r_bad_role.status_code == 400


# ---------------------------------------------------------------------------
# SCR-04 & SCR-11 & QA-01: job_scraper fixes
# ---------------------------------------------------------------------------


def test_scr04_scr11_job_scraper_fixes(monkeypatch):
    assert BaseExtractor.location_match("Bengaluru, India", "remote") is False
    assert BaseExtractor.location_match("Remote - India", "remote") is True
    assert parse_indeed_html("<html><body></body></html>") == []

    stopped = {"called": False}

    class _FakeChromium:
        async def launch(self, **kwargs):
            raise RuntimeError("browser launch boom")

        async def new_context(self, **kwargs):
            raise RuntimeError("unused")

    class _FakePW:
        chromium = _FakeChromium()
        firefox = _FakeChromium()
        webkit = _FakeChromium()

        async def stop(self):
            stopped["called"] = True

    class _FakePlaywrightCM:
        async def start(self):
            return _FakePW()

        async def __aenter__(self):
            return _FakePW()

        async def __aexit__(self, *args):
            stopped["called"] = True
            return False

    monkeypatch.setattr(
        "playwright_stealth.Stealth.hook_playwright_context", lambda self, ctx: None
    )
    monkeypatch.setattr(
        "playwright.async_api.async_playwright", lambda: _FakePlaywrightCM()
    )

    async def _run_ctx():
        ctx = BrowserStealthContext()
        with pytest.raises(RuntimeError, match="browser launch boom"):
            await ctx.__aenter__()

    asyncio.run(_run_ctx())
    assert stopped["called"] is True

    seen_timeouts: list[float] = []

    async def _fake_wait_for(coro, timeout=None):
        seen_timeouts.append(float(timeout))
        return await coro

    class _FakeDummyExtractor:
        async def search(self, *args, **kwargs):
            return []

    import job_scraper.pipeline as js_pipeline

    monkeypatch.setattr(js_pipeline.asyncio, "wait_for", _fake_wait_for)
    monkeypatch.setitem(
        js_pipeline.EXTRACTOR_REGISTRY, "indeed", lambda session: _FakeDummyExtractor()
    )

    class _FakeClientCM:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(js_pipeline, "StealthSession", _FakeClientCM)
    asyncio.run(js_pipeline.run_pipeline(keywords="python", portals=["indeed"], limit=10))
    assert seen_timeouts and seen_timeouts[0] >= 30.0


# ---------------------------------------------------------------------------
# SCR-05: TaskQueueManager handler registry, FIFO blpop, and state helpers
# ---------------------------------------------------------------------------


class _FakeFIFORedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.blpop_calls = 0

    def ping(self):
        return True

    def close(self):
        return None

    def rpush(self, key: str, value: str):
        self.lists.setdefault(key, []).append(value)

    def blpop(self, key: str, timeout: int = 1):
        self.blpop_calls += 1
        items = self.lists.get(key) or []
        if not items:
            return None
        return (key, items.pop(0))

    def hset(self, key: str, field: str | None = None, value: str | None = None, mapping: dict[str, str] | None = None):
        bucket = self.hashes.setdefault(key, {})
        if mapping:
            bucket.update(mapping)
        if field is not None and value is not None:
            bucket[field] = value

    def hget(self, key: str, field: str):
        return self.hashes.get(key, {}).get(field)

    def sadd(self, key: str, value: str):
        self.sets.setdefault(key, set()).add(value)

    def smembers(self, key: str):
        return set(self.sets.get(key, set()))

    def expire(self, key: str, ttl: int):
        return True

    def scard(self, key: str) -> int:
        return len(self.sets.get(key, set()))

    def spop(self, key: str, count: int = 1):
        s = self.sets.get(key, set())
        out = []
        for _ in range(count):
            if not s:
                break
            out.append(s.pop())
        return out


def test_scr05_task_queue_register_fifo_and_record_state():
    fake_redis = _FakeFIFORedis()
    tq_submitter = TaskQueueManager()
    tq_worker = TaskQueueManager()
    tq_submitter._redis = fake_redis
    tq_worker._redis = fake_redis

    order: list[str] = []

    def handler(payload: Any) -> dict[str, Any]:
        order.append(str(payload))
        return {"handled": payload}

    tq_worker.register("demo.work", handler)
    t1 = tq_submitter.submit("demo.work", handler, "first")
    t2 = tq_submitter.submit("demo.work", handler, "second")
    state1 = tq_submitter.get(t1)
    assert state1 is not None and state1.status == "queued"

    # Worker pops FIFO via blpop and runs registered handler without local submit
    pop1 = fake_redis.blpop(tq_worker._queue_name, timeout=1)
    pop2 = fake_redis.blpop(tq_worker._queue_name, timeout=1)
    assert pop1 is not None and pop2 is not None
    job1 = json.loads(pop1[1])
    job2 = json.loads(pop2[1])
    assert job1["task_id"] == t1 and job2["task_id"] == t2
    tq_worker._run(job1["task_id"], job1["kind"], job1["payload"])
    tq_worker._run(job2["task_id"], job2["kind"], job2["payload"])
    assert order == ["first", "second"]

    # record_state persists externally without directly mutating _states
    tq_worker.record_state(
        TaskState(task_id="ext-1", kind="demo.work", status="completed", result={"ok": True})
    )
    ext_state = tq_submitter.get("ext-1")
    assert ext_state is not None and ext_state.status == "completed"


# ---------------------------------------------------------------------------
# SCR-06: StealthHarvesterProvider live vs offline isolation & non-blocking pool
# ---------------------------------------------------------------------------


def test_scr06_stealth_harvester_live_skips_stale_output_and_shutdown(
    monkeypatch, tmp_path: Path
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "normalized_jobs.json").write_text(
        json.dumps([{"title": "Stale Job", "company": "OldCorp", "job_url": "https://old.example/1"}]),
        encoding="utf-8",
    )

    provider = StealthHarvesterProvider(output_dir=out_dir)
    assert provider._live_enabled({}) is False

    # Offline run reads normalized_jobs.json
    offline_jobs = provider.fetch_jobs("python")
    assert len(offline_jobs) == 1 and offline_jobs[0]["title"] == "Stale Job"

    # Live run returning empty does NOT fall back to stale output file
    monkeypatch.setattr(provider, "_run_live", lambda *a, **kw: [])
    live_jobs = provider.fetch_jobs("python", live=True)
    assert live_jobs == []

    # Executor shutdown(wait=False, cancel_futures=True) used on timeout in running loop
    shutdown_calls: list[tuple[bool, bool]] = []
    fresh_provider = StealthHarvesterProvider(output_dir=out_dir)

    class _TrackingPool:
        def __init__(self, *args, **kwargs):
            pass

        def submit(self, fn, coro, *args, **kwargs):
            if asyncio.iscoroutine(coro):
                coro.close()

            class _TimeoutFuture:
                def result(self, timeout=None):
                    raise TimeoutError("timed out")

            return _TimeoutFuture()

        def shutdown(self, wait=True, cancel_futures=False):
            shutdown_calls.append((wait, cancel_futures))

    import job_scraper.pipeline as js_pipeline

    async def _noop_pipeline(**kwargs):
        return {}

    monkeypatch.setattr(js_pipeline, "run_pipeline", _noop_pipeline)
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", _TrackingPool)

    async def _inside_loop():
        return fresh_provider._run_live(["python"], "India", 5, {})

    res = asyncio.run(_inside_loop())
    assert res == []
    assert shutdown_calls == [(False, True)]


# ---------------------------------------------------------------------------
# SCR-07: Plans validation, transactional rollback, and verify_quiz args
# ---------------------------------------------------------------------------


def test_scr07_plans_validation_atomic_replace_and_verify_quiz(
    client_engine, monkeypatch
):
    client, eng = client_engine
    r_stud = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Plan Audit Student"},
    )
    assert r_stud.status_code == 201
    sid = r_stud.json()["id"]
    client.get(f"/api/students/{sid}/matches")

    r_bad = client.post(
        f"/api/students/{sid}/plans",
        json={"job_id": "job-001", "weekly_cap": "not-an-int"},
    )
    assert r_bad.status_code == 400

    r_ok = client.post(
        f"/api/students/{sid}/plans",
        json={"job_id": "job-001", "weekly_cap": 8, "horizon_weeks": 4},
    )
    assert r_ok.status_code == 201
    plan_id = r_ok.json()["id"]
    first_item = r_ok.json()["items"][0]

    # Failed rebuild must preserve prior UpskillPlan + PlanItems
    import app.api.routes.plans as plans_route

    monkeypatch.setattr(
        plans_route,
        "build_plan",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("solver failed")),
    )
    with pytest.raises(RuntimeError, match="solver failed"):
        client.post(
            f"/api/students/{sid}/plans",
            json={"job_id": "job-001", "weekly_cap": 10},
        )
    with Session(eng) as db:
        assert db.get(UpskillPlan, plan_id) is not None
        assert len(db.exec(select(PlanItem).where(PlanItem.plan_id == plan_id)).all()) >= 1

    # verify_quiz invokes QuestionTelemetryManager.record_answer_turn with q_id + int difficulty
    recorded_calls: list[dict[str, Any]] = []
    from app.interview.feedback_loop import QuestionTelemetryManager

    monkeypatch.setattr(
        QuestionTelemetryManager,
        "record_answer_turn",
        lambda self, **kw: recorded_calls.append(kw),
    )
    r_quiz = client.post(
        f"/api/plans/{plan_id}/items/{first_item['id']}/verify",
        json={"answers": ["answer one", "answer two"]},
    )
    assert r_quiz.status_code == 200
    assert recorded_calls
    assert recorded_calls[0]["q_id"] == f"plan_quiz:{first_item['skill_id']}"
    assert isinstance(recorded_calls[0]["difficulty"], int)


# ---------------------------------------------------------------------------
# SCR-08: Pre-validation before parse_campus_jd & numeric validation
# ---------------------------------------------------------------------------


def test_scr08_public_jnf_validates_before_audit_commit(client_engine):
    client, eng = client_engine
    r_tok = client.post(
        "/api/campus-drives/jnf-links",
        json={"expires_hours": 24},
        headers=TPO_HDR,
    )
    assert r_tok.status_code == 200
    token = r_tok.json()["token"]

    with Session(eng) as db:
        before_audits = len(db.exec(select(AuditLog)).all())
        before_drives = len(db.exec(select(CampusDrive)).all())

    r_missing = client.post(
        "/api/campus-drives/public-jnf",
        data={"token": token, "location": "Jaipur"},
    )
    assert r_missing.status_code == 400

    with Session(eng) as db:
        after_audits = len(db.exec(select(AuditLog)).all())
        after_drives = len(db.exec(select(CampusDrive)).all())
    assert after_audits == before_audits
    assert after_drives == before_drives

    tax = load_taxonomy()
    with pytest.raises(ValueError):
        validate_drive_payload(
            {"company": "Acme", "role_title": "SDE", "eligibility": {"min_cgpa": "abc"}},
            tax,
        )
    with pytest.raises(ValueError):
        validate_drive_payload(
            {"company": "Acme", "role_title": "SDE", "eligibility": {"min_cgpa": True}},
            tax,
        )
    with pytest.raises(ValueError):
        validate_drive_payload({"company": "Acme", "role_title": "SDE", "ctc_lpa_min": "nan"}, tax)


# ---------------------------------------------------------------------------
# SCR-09 & SCR-10 & QA-05: speech_transcribe, pagination/RL, interview races
# ---------------------------------------------------------------------------


def test_scr09_speech_transcribe_windows_safe_tempfile(client_engine, monkeypatch):
    client, _eng = client_engine
    seen_paths: list[str] = []

    class _Seg:
        def __init__(self, text: str) -> None:
            self.text = text

    class _FakeWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, path: str):
            seen_paths.append(path)
            with open(path, "rb") as fh:
                assert fh.read() == b"audio-payload-123"
            return ([_Seg("hello"), _Seg("world")], None)

    fake_fw = types.ModuleType("faster_whisper")
    fake_fw.WhisperModel = _FakeWhisperModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

    resp = client.post(
        "/api/speech/transcribe",
        files={"file": ("voice.webm", b"audio-payload-123", "audio/webm")},
        headers=STUDENT_HDR,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "faster_whisper"
    assert body["transcript"] == "hello world"
    assert seen_paths and not os.path.exists(seen_paths[0])


def test_scr10_jobs_pagination_and_stream_rate_limit(client_engine, monkeypatch):
    client, _eng = client_engine
    r_all = client.get("/api/jobs")
    assert r_all.status_code == 200
    all_jobs = r_all.json()
    assert isinstance(all_jobs, list)
    if len(all_jobs) >= 2:
        r_page = client.get("/api/jobs?limit=1&offset=1")
        assert r_page.status_code == 200
        assert r_page.json() == [all_jobs[1]]

    from app.config import settings
    from app.scraper.service import CompositeJobScraper
    from app.security.ratelimit import limiter

    limiter.reset()
    monkeypatch.setattr(settings, "rate_limit_enabled", True)
    monkeypatch.setattr(CompositeJobScraper, "fetch_jobs_streaming", lambda *a, **kw: [])
    try:
        for _ in range(10):
            assert client.post("/api/jobs/scrape/stream", json={"live": False}, headers=TPO_HDR).status_code == 200
        r_limited = client.post("/api/jobs/scrape/stream", json={"live": False}, headers=TPO_HDR)
        assert r_limited.status_code == 429
    finally:
        limiter.reset()


def test_qa05_concurrent_interview_answers_and_finish_are_serialized(tmp_path: Path):
    client, _eng, _ctx = _make_file_client(tmp_path)
    r_stud = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL, Docker\n", "text/plain")},
        data={"name": "Race Student"},
    )
    assert r_stud.status_code == 201
    sid = r_stud.json()["id"]

    r_ans_iv = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    assert r_ans_iv.status_code == 201
    iid_ans = r_ans_iv.json()["interview_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(
            client.post,
            f"/api/interviews/{iid_ans}/answer",
            json={"text": "First detailed answer mentioning python api async fastapi."},
        )
        f2 = pool.submit(
            client.post,
            f"/api/interviews/{iid_ans}/answer",
            json={"text": "Second detailed answer covering postgresql indexing sql."},
        )
        res1, res2 = f1.result(), f2.result()

    assert res1.status_code == 200
    assert res2.status_code == 200
    r_get = client.get(f"/api/interviews/{iid_ans}")
    assert r_get.status_code == 200
    answered = [t for t in r_get.json()["transcript"] if t.get("score") is not None]
    assert len(answered) == 2

    r_fin_iv = client.post(f"/api/students/{sid}/interviews", json={"job_id": "job-001"})
    assert r_fin_iv.status_code == 201
    iid_fin = r_fin_iv.json()["interview_id"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        ff1 = pool.submit(client.post, f"/api/interviews/{iid_fin}/finish")
        ff2 = pool.submit(client.post, f"/api/interviews/{iid_fin}/finish")
        codes = sorted([ff1.result().status_code, ff2.result().status_code])

    assert codes == [200, 409]


# ---------------------------------------------------------------------------
# SCR-12: run_cohort_rematch preserves flags & enforces job visibility
# ---------------------------------------------------------------------------


def test_scr12_cohort_rematch_visibility_and_preserves_flags(tmp_path: Path):
    client, eng, ctx = _make_file_client(tmp_path)
    r_stud = client.post(
        "/api/students",
        files={"file": ("r.txt", b"Skills: Python, FastAPI, PostgreSQL\n", "text/plain")},
        data={"name": "Cohort Rematch Student"},
    )
    assert r_stud.status_code == 201
    sid = r_stud.json()["id"]

    with Session(eng) as db:
        job_open = Job(
            id="job-open-vis",
            title="Campus Backend Intern",
            company="OpenCorp",
            description="Python FastAPI SQL",
            required_skills=["python"],
            preferred_skills=[],
            status="active",
            is_approved=True,
        )
        job_hidden = Job(
            id="job-hidden-vis",
            title="Private Other Student Live Job",
            company="HiddenCorp",
            description="Python FastAPI PostgreSQL",
            required_skills=["python"],
            preferred_skills=[],
            is_live=True,
            owner_student_id="stu-other-owner",
        )
        db.add(job_open)
        db.add(job_hidden)
        drive = CampusDrive(
            id="cd-rematch-test",
            company="OpenCorp",
            role_title="Campus Backend Intern",
            status="active",
            job_id="job-open-vis",
        )
        db.add(drive)
        existing_match = db.exec(
            select(JobMatch).where(JobMatch.student_id == sid, JobMatch.job_id == "job-001")
        ).first()
        if existing_match is None:
            existing_match = JobMatch(
                id=f"jm-fav-{sid}",
                student_id=sid,
                job_id="job-001",
                score=80,
                band="strong",
                is_favorite=True,
            )
        else:
            existing_match.is_favorite = True
        db.add(existing_match)
        dismissed_match = JobMatch(
            id=f"jm-dis-{sid}",
            student_id=sid,
            job_id="job-002",
            score=40,
            band="dismissed",
            breakdown={"dismissed": True},
        )
        db.add(dismissed_match)
        db.commit()

    summary = run_cohort_rematch(
        ctx, drive_id="cd-rematch-test", job_id="job-open-vis", trace_id="tr-test-1"
    )
    assert summary["students_total"] >= 1

    with Session(eng) as db:
        matches = db.exec(select(JobMatch).where(JobMatch.student_id == sid)).all()
        matched_ids = {m.job_id for m in matches}
        assert "job-hidden-vis" not in matched_ids
        preserved_fav = next((m for m in matches if m.job_id == "job-001"), None)
        assert preserved_fav is not None
        assert preserved_fav.is_favorite is True
        preserved_dis = next((m for m in matches if m.job_id == "job-002"), None)
        assert preserved_dis is not None
        assert preserved_dis.band == "dismissed"
        assert preserved_dis.breakdown.get("dismissed") is True
