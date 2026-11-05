"""M22: distributed background worker queue + LLM token/cost telemetry + circuit breaker."""
import sys
import time
import types

import pytest

from app.config import Settings


def _settings(**overrides):
    base = dict(llm_provider="mock", llm_model="m", aws_region="us-east-1",
                aws_access_key_id=None, aws_secret_access_key=None,
                openai_api_key=None, openai_model="gpt-4o-mini", openai_base_url=None)
    base.update(overrides)
    return Settings(**base)


def _wait_status(queue, task_id, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = queue.get(task_id)
        if st is not None and st.status in ("completed", "failed"):
            return st
        time.sleep(0.01)
    return queue.get(task_id)


# ---------------------------------------------------------------------------
# 22.1 TaskQueueManager - memory backend
# ---------------------------------------------------------------------------

def test_memory_task_submit_executes_and_tracks_state():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    try:
        assert q.backend == "memory"
        tid = q.submit("double", lambda p: p * 2, payload=21)
        st = _wait_status(q, tid)
        assert st is not None
        assert st.task_id == tid
        assert st.kind == "double"
        assert st.status == "completed"
        assert st.progress_pct == 100
        assert st.result == 42
        assert st.error is None
        assert st.created_at and st.completed_at
        d = st.to_dict()
        for key in ("task_id", "kind", "status", "progress_pct", "result",
                    "error", "created_at", "completed_at"):
            assert key in d
    finally:
        q.shutdown()


def test_memory_task_zero_arg_fn_supported():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    try:
        calls = []
        def job():
            calls.append("ran")
            return "ok"
        tid = q.submit("noop", job)
        st = _wait_status(q, tid)
        assert st.status == "completed"
        assert st.result == "ok"
        assert calls == ["ran"]
    finally:
        q.shutdown()


def test_memory_task_exception_marks_failed():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    try:
        def boom(p):
            raise RuntimeError("kaboom")
        tid = q.submit("boom", boom, payload=1)
        st = _wait_status(q, tid)
        assert st.status == "failed"
        assert "kaboom" in (st.error or "")
        assert st.result is None
        assert st.completed_at
    finally:
        q.shutdown()


def test_memory_task_list_newest_first_and_kind_filter():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    try:
        t1 = q.submit("a", lambda p: p, payload=1)
        time.sleep(0.02)
        t2 = q.submit("b", lambda p: p, payload=2)
        time.sleep(0.02)
        t3 = q.submit("a", lambda p: p, payload=3)
        time.sleep(0.2)  # let all settle
        all_tasks = q.list()
        assert [t.task_id for t in all_tasks] == [t3, t2, t1]
        a_tasks = q.list(kind="a")
        assert {t.task_id for t in a_tasks} == {t1, t3}
        assert len(q.list(limit=1)) == 1
    finally:
        q.shutdown()


def test_memory_task_concurrent_execution():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    try:
        t0 = time.time()
        ids = [q.submit("sleep", lambda p: time.sleep(0.25), payload=None) for _ in range(4)]
        for tid in ids:
            st = _wait_status(q, tid, timeout=10.0)
            assert st.status == "completed"
        elapsed = time.time() - t0
        assert elapsed < 0.9, f"4x0.25s tasks should overlap, took {elapsed:.2f}s"
    finally:
        q.shutdown()


def test_task_queue_shutdown_clean():
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager()
    tid = q.submit("x", lambda p: "done", payload=0)
    _wait_status(q, tid)
    q.shutdown(wait=True)
    q.shutdown(wait=True)  # idempotent


# ---------------------------------------------------------------------------
# 22.1 TaskQueueManager - redis backend (lazy import, resilience, fake client)
# ---------------------------------------------------------------------------

class _FakeRedis:
    def __init__(self):
        self.q: list[str] = []
        self.hashes: dict[str, dict] = {}
        self.index: set[str] = set()
        self.closed = False

    @classmethod
    def from_url(cls, url, **kw):
        return cls()

    def ping(self):
        return True

    def rpush(self, key, value):
        self.q.append(value)
        return len(self.q)

    def brpop(self, key, timeout=1):
        if self.q:
            return [key.encode(), self.q.pop(0).encode()]
        return None

    def hset(self, key, mapping=None, **kw):
        self.hashes.setdefault(key, {}).update(mapping or {})
        return 1

    def hget(self, key, field):
        return self.hashes.get(key, {}).get(field)

    def sadd(self, key, *members):
        self.index.update(members)
        return len(members)

    def smembers(self, key):
        return set(self.index)

    def close(self):
        self.closed = True


def test_redis_backend_falls_back_when_package_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "redis", None)  # forces ImportError on `import redis`
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager(backend="redis", redis_url="redis://localhost:6399/0")
    try:
        assert q.backend == "memory"
        tid = q.submit("demo", lambda p: p + 1, payload=1)
        st = _wait_status(q, tid)
        assert st.status == "completed" and st.result == 2
    finally:
        q.shutdown()


def test_redis_backend_falls_back_when_ping_fails(monkeypatch):
    boom = types.ModuleType("redis")
    class _Dead:
        @classmethod
        def from_url(cls, url, **kw):
            c = cls()
            return c
        def ping(self):
            raise ConnectionError("no redis here")
    boom.Redis = _Dead
    monkeypatch.setitem(sys.modules, "redis", boom)
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager(backend="redis", redis_url="redis://localhost:6399/0")
    try:
        assert q.backend == "memory"
        tid = q.submit("demo", lambda p: p + 1, payload=1)
        assert _wait_status(q, tid).status == "completed"
    finally:
        q.shutdown()


def test_redis_backend_consumer_executes_task(monkeypatch):
    fake = _FakeRedis()
    mod = types.ModuleType("redis")
    mod.Redis = type("Redis", (), {"from_url": staticmethod(lambda url, **kw: fake),
                                   "ping": lambda self: True})
    monkeypatch.setitem(sys.modules, "redis", mod)
    from app.workers.task_queue import TaskQueueManager
    q = TaskQueueManager(backend="redis", redis_url="redis://fake/0")
    try:
        assert q.backend == "redis"
        tid = q.submit("square", lambda p: p * p, payload=7)
        st = _wait_status(q, tid, timeout=6.0)
        assert st is not None
        assert st.status == "completed"
        assert st.result == 49
        assert st.progress_pct == 100
        listed = q.list()
        assert any(t.task_id == tid for t in listed)
        q.shutdown(wait=True)
        assert fake.closed
    except BaseException:
        q.shutdown(wait=True)
        raise


# ---------------------------------------------------------------------------
# 22.2 LLMTelemetryTracker - tokens, cost, circuit breaker
# ---------------------------------------------------------------------------

def _clock(start=1_000_000.0):
    state = {"t": start}
    def now():
        return state["t"]
    def advance(s):
        state["t"] += s
    return now, advance


def test_tracker_token_estimation_and_cost():
    from app.llm.telemetry import LLMTelemetryTracker, estimate_tokens
    assert estimate_tokens("") == 0
    assert estimate_tokens("abc") == 1  # max(1, 3//4)
    assert estimate_tokens("a" * 400) == 100
    tr = LLMTelemetryTracker(now_fn=_clock()[0])
    tr.record("openai", "a" * 400, "b" * 800, latency_ms=120.0)
    stats = tr.stats()
    p = stats["providers"]["openai"]
    assert p["calls"] == 1
    assert p["prompt_tokens"] == 100
    assert p["completion_tokens"] == 200
    assert p["total_tokens"] == 300
    assert p["estimated_cost_usd"] == pytest.approx((100 * 0.15 + 200 * 0.60) / 1_000_000.0)
    assert p["total_latency_ms"] == pytest.approx(120.0)
    assert stats["totals"]["total_calls"] == 1
    assert stats["totals"]["total_tokens"] == 300


def test_tracker_usage_metadata_overrides_estimate():
    from app.llm.telemetry import LLMTelemetryTracker
    tr = LLMTelemetryTracker(now_fn=_clock()[0])
    tr.record("bedrock", "a" * 400, "b" * 800, usage={"prompt_tokens": 111, "completion_tokens": 222})
    p = tr.stats()["providers"]["bedrock"]
    assert p["prompt_tokens"] == 111
    assert p["completion_tokens"] == 222
    assert p["estimated_cost_usd"] == pytest.approx((111 * 0.05 + 222 * 0.20) / 1_000_000.0)


class _FailLLM:
    def __init__(self, fail_times=999):
        self.fail_times = fail_times
        self.calls = 0
        self.stream_calls = 0

    def complete(self, system, user, json_mode=False):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("network down")
        return "live-answer"

    def stream(self, system, user, json_mode=False):
        self.stream_calls += 1
        raise ConnectionError("stream down")
        yield  # pragma: no cover


def _telemetry_llm(primary, tracker):
    from app.llm.telemetry import TelemetryLLM
    llm = TelemetryLLM(_settings(llm_provider="openai", openai_api_key="sk-test"),
                       tracker=tracker)
    llm.primary = primary
    return llm


def test_circuit_trips_after_three_consecutive_failures():
    from app.llm.telemetry import LLMTelemetryTracker
    now, advance = _clock()
    tr = LLMTelemetryTracker(now_fn=now)
    llm = _telemetry_llm(_FailLLM(fail_times=999), tr)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            llm.complete("s", "u")
    assert tr.should_use_fallback("openai") is True
    primary = llm.primary
    out = llm.complete("s", "u")  # 4th call: routed to MockLLM, no network
    assert primary.calls == 3, "open-circuit call must not hit the live provider"
    assert out and out != "live-answer"
    assert tr.stats()["providers"]["openai"]["fallbacks"] == 1


def test_circuit_success_resets_streak():
    from app.llm.telemetry import LLMTelemetryTracker
    now, advance = _clock()
    tr = LLMTelemetryTracker(now_fn=now)
    llm = _telemetry_llm(_FailLLM(fail_times=2), tr)
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    llm.primary = _FailLLM(fail_times=0)  # success resets the streak
    assert llm.complete("s", "u") == "live-answer"
    llm.primary = _FailLLM(fail_times=999)
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    assert tr.should_use_fallback("openai") is False, "only 2 consecutive failures after reset"


def test_circuit_ignores_failures_spaced_beyond_window():
    from app.llm.telemetry import LLMTelemetryTracker
    now, advance = _clock()
    tr = LLMTelemetryTracker(now_fn=now)
    llm = _telemetry_llm(_FailLLM(fail_times=999), tr)
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    advance(55.0)
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    advance(15.0)  # t=70: first failure (t=0) is outside the 60s window
    with pytest.raises(ConnectionError):
        llm.complete("s", "u")
    assert tr.should_use_fallback("openai") is False


def test_circuit_cooldown_expiry_restores_live():
    from app.llm.telemetry import LLMTelemetryTracker
    now, advance = _clock()
    tr = LLMTelemetryTracker(now_fn=now)
    llm = _telemetry_llm(_FailLLM(fail_times=999), tr)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            llm.complete("s", "u")
    assert tr.should_use_fallback("openai") is True
    advance(31.0)
    assert tr.should_use_fallback("openai") is False
    llm.primary = _FailLLM(fail_times=0)
    assert llm.complete("s", "u") == "live-answer"


def test_telemetry_llm_stream_records_success_and_fallback():
    from app.llm.telemetry import LLMTelemetryTracker
    now, advance = _clock()
    tr = LLMTelemetryTracker(now_fn=now)

    class _StreamOK:
        def stream(self, system, user, json_mode=False):
            yield "hel"
            yield "lo"
    llm = _telemetry_llm(_StreamOK(), tr)
    assert "".join(llm.stream("s", "u")) == "hello"
    p = tr.stats()["providers"]["openai"]
    assert p["calls"] == 1
    assert p["completion_tokens"] == 1  # "hello" -> max(1, 5//4)

    # circuit open -> stream routes to mock without touching the failing primary
    fail = _FailLLM(fail_times=999)
    llm2 = _telemetry_llm(fail, tr)
    for _ in range(3):
        with pytest.raises(ConnectionError):
            llm2.complete("s", "u")
    assert tr.should_use_fallback("openai") is True
    chunks = list(llm2.stream("s", "u"))
    assert chunks, "open circuit must stream from the mock fallback"
    assert fail.stream_calls == 0
    assert tr.stats()["providers"]["openai"]["fallbacks"] == 1


def test_get_instrumented_llm_wraps_provider():
    from app.llm.mock import MockLLM
    from app.llm.telemetry import TelemetryLLM, get_instrumented_llm
    llm = get_instrumented_llm(_settings())
    assert isinstance(llm, TelemetryLLM)
    assert isinstance(llm.primary, MockLLM)
    out = llm.complete("greet", "world")
    assert isinstance(out, str) and out
    assert tr_stats_calls(llm) == 1


def tr_stats_calls(llm):
    return llm.tracker.stats()["providers"][llm.provider]["calls"]


# ---------------------------------------------------------------------------
# 22.1 API endpoints
# ---------------------------------------------------------------------------

def test_api_tasks_endpoints(client):
    r = client.get("/api/tasks")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0
    assert body["tasks"] == []
    assert body["backend"] == "memory"
    queue = client.app.state.ctx.task_queue
    assert queue is not None

    tid = queue.submit("demo", lambda p: p * 2, payload=21)
    st = _wait_status(queue, tid)
    assert st.status == "completed" and st.result == 42

    r = client.get(f"/api/tasks/{tid}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["result"] == 42

    assert client.get("/api/tasks/does-not-exist").status_code == 404
    assert client.get("/api/tasks", headers={"x-demo-role": "student"}).status_code == 403
    assert client.get(f"/api/tasks/{tid}", headers={"x-demo-role": "student"}).status_code == 403
    r = client.get("/api/tasks", headers={"x-demo-role": "tpo_admin"})
    assert r.status_code == 200
    assert r.json()["count"] == 1
