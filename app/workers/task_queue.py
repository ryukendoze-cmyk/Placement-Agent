"""Pluggable background task runner (Milestone 22).

``TaskQueueManager`` executes long-running jobs (scrape/ranking/OCR) outside
the request path and tracks their lifecycle:

- ``TASK_BACKEND=memory`` (default): in-process ``ThreadPoolExecutor``; state
  lives in a lock-guarded dict. Deterministic, offline, test-friendly.
- ``TASK_BACKEND=redis`` (when ``REDIS_URL`` is set): task state persisted in
  Redis hashes, queue in a Redis list, executed by an in-process daemon
  consumer thread (``BRPOP`` with a 1s timeout - never an infinite block).
  The ``redis`` package is an optional dependency: if it is missing or the
  server is unreachable the manager falls back to the memory backend so boot
  never crashes (offline-first invariant).
"""
from __future__ import annotations

import inspect
import json
import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class TaskState:
    task_id: str
    kind: str
    status: str = "queued"            # queued | running | completed | failed
    progress_pct: int = 0
    result: object = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class TaskQueueManager:
    def __init__(self, backend: str = "memory", redis_url: str | None = None,
                 max_workers: int = 4, queue_name: str = "ps06:tasks:queue",
                 state_prefix: str = "ps06:tasks:", index_key: str = "ps06:tasks:index"):
        self.backend = "memory"
        self._queue_name = queue_name
        self._state_prefix = state_prefix
        self._index_key = index_key
        self._registry: dict[str, Callable] = {}
        self._states: dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._consumer: threading.Thread | None = None
        self._redis = None
        self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers or 4)),
            thread_name_prefix="ps06-task",
        )

        if (backend or "memory").lower() == "redis" and redis_url:
            self._init_redis(redis_url)

    @classmethod
    def from_settings(cls, settings) -> "TaskQueueManager":
        return cls(backend=getattr(settings, "task_backend", "memory") or "memory",
                   redis_url=getattr(settings, "redis_url", None))

    # -- redis wiring (lazy, optional dependency) -----------------------------

    def _init_redis(self, redis_url: str) -> None:
        try:
            import redis  # noqa: F401  (optional dependency, imported lazily)
            client = redis.Redis.from_url(
                redis_url, socket_connect_timeout=2, socket_timeout=2)
            client.ping()
        except Exception as e:
            logger.warning("Redis task backend unavailable (%s); "
                           "falling back to in-memory task queue.", e)
            return
        self._redis = client
        self.backend = "redis"
        self._consumer = threading.Thread(target=self._consumer_loop,
                                          name="ps06-task-consumer", daemon=True)
        self._consumer.start()

    def _consumer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._redis.brpop(self._queue_name, timeout=1)
            except Exception as e:
                logger.warning("task queue brpop failed (%s); retrying.", e)
                self._stop_event.wait(0.2)
                continue
            if item is None:
                continue
            _queue, raw = item
            try:
                job = json.loads(raw)
                if self._executor is not None:
                    self._executor.submit(self._run, job["task_id"], job.get("kind"), job.get("payload"))
                else:
                    self._run(job["task_id"], job.get("kind"), job.get("payload"))
            except Exception as e:
                logger.warning("malformed task job dropped: %s", e)

    # -- public API ------------------------------------------------------------

    def submit(self, kind: str, fn: Callable, payload: object = None) -> str:
        self._registry[kind] = fn
        task_id = uuid.uuid4().hex
        state = TaskState(task_id=task_id, kind=kind)
        if self._redis is not None:
            self._save_state(state)
            self._redis.sadd(self._index_key, task_id)
            self._redis.rpush(self._queue_name, json.dumps(
                {"task_id": task_id, "kind": kind, "payload": payload}))
            return task_id
        with self._lock:
            self._states[task_id] = state
            if len(self._states) > 2000:
                terminal = [
                    k for k, v in self._states.items()
                    if v.status in ("completed", "failed")
                ]
                for k in terminal[: len(self._states) - 1500]:
                    self._states.pop(k, None)
        self._executor.submit(self._run, task_id, kind, payload)
        return task_id

    def get(self, task_id: str) -> TaskState | None:
        if self._redis is not None:
            return self._load_state(task_id)
        with self._lock:
            return self._states.get(task_id)

    def list(self, kind: str | None = None, limit: int = 50) -> list[TaskState]:
        if self._redis is not None:
            ids = [i.decode() if isinstance(i, bytes) else str(i)
                   for i in self._redis.smembers(self._index_key)]
            states = [s for i in ids if (s := self._load_state(i)) is not None]
        else:
            with self._lock:
                states = list(self._states.values())
        if kind:
            states = [s for s in states if s.kind == kind]
        states.sort(key=lambda s: s.created_at, reverse=True)
        return states[:max(1, int(limit))]

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        if self._consumer is not None:
            self._consumer.join(timeout=2.0 if wait else 0)
            self._consumer = None
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
            self._redis = None
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None

    # -- internals -------------------------------------------------------------

    def _dispatch(self, kind: str, payload: object) -> object:
        fn = self._registry.get(kind)
        if fn is None:
            raise KeyError(f"no handler registered for task kind {kind!r}")
        try:
            takes_arg = len(inspect.signature(fn).parameters) > 0
        except (TypeError, ValueError):
            takes_arg = True
        return fn(payload) if takes_arg else fn()

    def _run(self, task_id: str, kind: str, payload: object) -> None:
        self._update(task_id, status="running")
        try:
            result = self._dispatch(kind, payload)
        except Exception as e:
            self._update(task_id, status="failed", error=str(e))
            return
        self._update(task_id, status="completed", result=result, progress_pct=100)

    def _update(self, task_id: str, **changes) -> None:
        with self._lock:
            state = self._states.get(task_id)
            if state is None and self._redis is not None:
                state = self._load_state(task_id)
            if state is None:
                return
            for key, value in changes.items():
                setattr(state, key, value)
            if state.status in ("completed", "failed"):
                state.completed_at = datetime.now(timezone.utc).isoformat()
            if self._redis is not None:
                self._save_state(state)

    def _save_state(self, state: TaskState) -> None:
        self._redis.hset(f"{self._state_prefix}{state.task_id}",
                         mapping={"data": json.dumps(state.to_dict())})

    def _load_state(self, task_id: str) -> TaskState | None:
        raw = self._redis.hget(f"{self._state_prefix}{task_id}", "data")
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return TaskState(**json.loads(raw))
