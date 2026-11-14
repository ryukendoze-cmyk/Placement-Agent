"""Production ASGI entry point: uvicorn app.api.__main__:app --reload"""

from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel

from app.api.main import AppContext, create_app
from app.config import settings
from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder, LocalEmbedder
from app.embeddings.store import EmbeddingRow, MemoryVectorStore, PgVectorStore
from app.llm.telemetry import LLMTelemetryTracker, get_instrumented_llm
from app.taxonomy.taxonomy import load_taxonomy
from app.workers.task_queue import TaskQueueManager


class _SessionPerOpPgVectorStore:
    """PgVectorStore with a fresh short-lived session per operation.

    PgVectorStore commits on the session it holds; in production the ASGI app
    lives for hours while the request-scoped Session objects come and go, so
    each operation here opens its own session from the session factory.
    """

    def __init__(self, session_factory: sessionmaker, dim: int = 384):
        if dim != 384:
            raise ValueError("PgVectorStore only supports dim=384")
        self._factory = session_factory
        self.dim = dim
        self._engine = session_factory.kw["bind"]
        SQLModel.metadata.create_all(bind=self._engine, tables=[EmbeddingRow.__table__])

    def upsert(self, kind: str, ref_id: str, vector: list[float]) -> None:
        with self._factory() as db:
            PgVectorStore(db, dim=self.dim, ensure_table=False).upsert(kind, ref_id, vector)

    def search(self, kind: str, vector: list[float], k: int) -> list[tuple[str, float]]:
        with self._factory() as db:
            return PgVectorStore(db, dim=self.dim, ensure_table=False).search(kind, vector, k)

    def get(self, kind: str, ref_id: str) -> list[float] | None:
        with self._factory() as db:
            return PgVectorStore(db, dim=self.dim, ensure_table=False).get(kind, ref_id)


def backfill_memory_vectors(ctx: AppContext) -> int:
    """Warm an empty in-memory vector store from persisted Student and Job rows on startup."""
    store = getattr(ctx, "vector_store", None)
    embedder = getattr(ctx, "embedder", None)
    engine = getattr(ctx, "engine", None)
    if store is None or embedder is None or engine is None:
        return 0
    if not isinstance(store, MemoryVectorStore):
        return 0

    from sqlmodel import Session, select

    from app.models import Job, Student

    warmed = 0
    try:
        with Session(engine) as db:
            for st in db.exec(select(Student)).all():
                if store.get("student", st.id) is not None:
                    continue
                text = (st.resume_text or "").strip() or f"{st.name} {st.branch}".strip()
                if text:
                    store.upsert("student", st.id, embedder.embed(text))
                    warmed += 1
            for job in db.exec(select(Job)).all():
                vec = ctx.job_vectors.get(job.id) or store.get("job", job.id)
                if vec is None and (job.description or "").strip():
                    vec = embedder.embed(job.description)
                if vec is not None:
                    if store.get("job", job.id) is None:
                        store.upsert("job", job.id, vec)
                    ctx.job_vectors.setdefault(job.id, vec)
                    warmed += 1
    except Exception:
        pass
    return warmed


def _build_ctx() -> AppContext:
    engine = make_engine()
    init_db(engine)
    is_pg = (settings.database_url or "").startswith("postgres")
    embedder = LocalEmbedder() if settings.embedder == "local" else HashingEmbedder()

    if is_pg:
        store = _SessionPerOpPgVectorStore(
            sessionmaker(bind=engine, expire_on_commit=False), dim=embedder.dim
        )
    else:
        store = MemoryVectorStore()

    tracker = LLMTelemetryTracker()
    ctx = AppContext(
        engine=engine,
        taxonomy=load_taxonomy(),
        embedder=embedder,
        vector_store=store,
        llm=get_instrumented_llm(settings, tracker),
        llm_telemetry=tracker,
        task_queue=TaskQueueManager.from_settings(settings),
    )
    backfill_memory_vectors(ctx)
    return ctx


app = create_app(_build_ctx())
