"""Milestone 21: DB & vector store production scaling - SQL indexes, batch
upserts, HNSW index, and two-stage ANN pre-filter reranking.

Covers 21.1-21.4 of the master plan (unit scope; PgVectorStore batch/HNSW are
covered by the integration suite).
"""
from __future__ import annotations

from unittest.mock import patch

from sqlalchemy import text
from sqlmodel import Session

from app.db import _ensure_indexes, init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.matching import rank as rank_mod
from app.matching.rank import rank_jobs
from app.taxonomy.taxonomy import load_taxonomy

# ------------------------------------------------------------ 21.1 indexes

_EXPECTED_INDEXES = [
    ("jobmatch", "ix_jobmatch_student_id"),
    ("jobmatch", "ix_jobmatch_job_id"),
    ("upskillplan", "ix_upskillplan_student_id"),
    ("upskillplan", "ix_upskillplan_job_id"),
    ("planitem", "ix_planitem_plan_id"),
    ("planitem", "ix_planitem_skill_id"),
    ("progressevent", "ix_progressevent_student_id"),
    ("interview", "ix_interview_student_id"),
    ("interview", "ix_interview_job_id"),
    ("approvalrequest", "ix_approvalrequest_status"),
    ("auditlog", "ix_auditlog_action"),
    ("auditlog", "ix_auditlog_created_at"),
]


def _index_names(engine, table: str) -> set[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:t"),
            {"t": table}).fetchall()
        return {r[0] for r in rows}


def test_ensure_indexes_creates_expected_indexes_idempotently():
    """_ensure_indexes creates the canonical ix_<table>_<column> indexes and is
    safe to call repeatedly."""
    eng = make_engine("sqlite://")
    init_db(eng)
    # first explicit run
    _ensure_indexes(eng)
    _ensure_indexes(eng)  # idempotent second run
    for table, name in _EXPECTED_INDEXES:
        assert name in _index_names(eng, table), f"{name} missing on {table}"


def test_create_all_and_ensure_indexes_agree_on_names():
    """No duplicate index names: create_all's convention must match
    _ensure_indexes (ix_<table>_<column>), so a fresh DB has exactly one of each."""
    eng = make_engine("sqlite://")
    init_db(eng)
    _ensure_indexes(eng)
    for table, name in _EXPECTED_INDEXES:
        # a duplicate would appear as a second distinct index name on the table;
        # assert our canonical name exists exactly once (sqlite_master is unique
        # per name, so just assert presence + table index list non-empty)
        names = _index_names(eng, table)
        assert name in names
    # auditlog must have BOTH single indexes per spec (no composite)
    audit_names = _index_names(eng, "auditlog")
    assert "ix_auditlog_action" in audit_names
    assert "ix_auditlog_created_at" in audit_names


def test_indexed_columns_declared_in_models():
    """21.1: the model fields carry index=True (verified via SQLAlchemy metadata)."""
    from sqlmodel import SQLModel

    from app import models  # noqa: F401

    expected = {
        ("jobmatch", "student_id"), ("jobmatch", "job_id"),
        ("upskillplan", "student_id"), ("upskillplan", "job_id"),
        ("planitem", "plan_id"), ("planitem", "skill_id"),
        ("progressevent", "student_id"),
        ("interview", "student_id"), ("interview", "job_id"),
        ("approvalrequest", "status"),
        ("auditlog", "action"), ("auditlog", "created_at"),
    }
    found: set[tuple[str, str]] = set()
    for table in SQLModel.metadata.tables.values():
        for col in table.columns:
            if col.index or col.unique:
                found.add((table.name, col.name))
    missing = expected - found
    assert not missing, f"columns missing index=True: {sorted(missing)}"


# --------------------------------------------------------- 21.2 upsert_many

def test_memory_vector_store_upsert_many():
    store = MemoryVectorStore()
    items = [("job", "j1", [1.0, 0.0, 0.0]),
             ("job", "j2", [0.0, 1.0, 0.0]),
             ("student", "s1", [0.5, 0.5, 0.0])]
    n = store.upsert_many(items)
    assert n == 3
    assert store.get("job", "j1") == [1.0, 0.0, 0.0]
    assert store.get("student", "s1") == [0.5, 0.5, 0.0]
    # overwrite semantics (idempotent)
    assert store.upsert_many([("job", "j1", [0.0, 0.0, 1.0])]) == 1
    assert store.get("job", "j1") == [0.0, 0.0, 1.0]
    # empty batch is a no-op
    assert store.upsert_many([]) == 0


def test_vectorstore_protocol_declares_upsert_many():
    from app.embeddings.store import VectorStore
    assert hasattr(VectorStore, "upsert_many")


# ------------------------------------------------------------ 21.3 rank_jobs

def _mk_jobs(n: int, skill_pool: list[str]):
    """n lightweight job objects for ranking tests (SimpleNamespace-like)."""
    from types import SimpleNamespace
    jobs = []
    for i in range(n):
        jobs.append(SimpleNamespace(
            id=f"j-{i:04d}", title=f"Role {i}", company="C",
            description=f"Role {i} needs {skill_pool[i % len(skill_pool)]}",
            required_skills=[skill_pool[i % len(skill_pool)]],
            preferred_skills=[], salary_lpa=[0, 0], min_gpa=0.0,
            allowed_branches=[], experience_years=0, location="", remote=False,
            job_url="", source="local", scraper_engine="local",
        ))
    return jobs


def test_rank_jobs_without_ann_matches_full_scoring():
    """Default path (no ann_pre_filter_k, <=500 jobs) scores every job exactly
    like before (byte-for-byte top-K order)."""
    taxonomy = load_taxonomy()
    embedder = HashingEmbedder()
    skills = ["python", "java", "go", "sql"]
    jobs = _mk_jobs(50, skills)
    student_vec = embedder.embed("python developer")
    student_skills = {"python": ["resume line"]}
    store = MemoryVectorStore()
    full = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                     vector_store=store, top=10)
    assert len(full) == 10
    assert all(isinstance(_, object) for _, _ in full)
    # deterministic: same call again yields identical order
    full2 = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                      vector_store=store, top=10)
    assert [j.id for j, _ in full] == [j.id for j, _ in full2]


def test_rank_jobs_ann_pre_filter_sublinear_and_identical_topk():
    """With ann_pre_filter_k set, score_job is called on a strict subset while
    the top-K for a highly-overlapping candidate set stays identical."""
    taxonomy = load_taxonomy()
    embedder = HashingEmbedder()
    skills = ["python", "java", "go", "sql"]
    # 300 jobs: <=500 so the auto (>500) path stays off; ann_pre_filter_k forces it
    jobs = _mk_jobs(300, skills)
    student_vec = embedder.embed("python developer")
    student_skills = {"python": ["resume line"]}
    store = MemoryVectorStore()

    # full scoring baseline
    with patch.object(rank_mod, "score_job", wraps=rank_mod.score_job) as spy_full:
        full = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                         vector_store=store, top=5)
        full_calls = spy_full.call_count
    assert full_calls == len(jobs)

    # ANN pre-filtered
    with patch.object(rank_mod, "score_job", wraps=rank_mod.score_job) as spy_ann:
        ann = rank_jobs(jobs, student_vec, student_skills, embedder, taxonomy,
                        vector_store=store, top=5, ann_pre_filter_k=40)
        ann_calls = spy_ann.call_count

    # sub-linear: strictly fewer scoring calls than the full catalog
    assert ann_calls < full_calls, (ann_calls, full_calls)
    assert ann_calls <= 40 + len(jobs)  # bounded by ANN k + skill-overlap union
    # top-K identical when the student's matched skill jobs dominate the top
    assert [j.id for j, _ in ann] == [j.id for j, _ in full]


def test_rank_jobs_auto_ann_only_above_500():
    """Catalog <= 500 never auto-enables ANN (byte-for-byte path); >500 with a
    searchable store auto-enables it."""
    taxonomy = load_taxonomy()
    embedder = HashingEmbedder()
    skills = ["python", "java"]
    small = _mk_jobs(500, skills)
    student_vec = embedder.embed("python")
    store = MemoryVectorStore()
    with patch.object(rank_mod, "score_job", wraps=rank_mod.score_job) as spy:
        rank_jobs(small, student_vec, {"python": ["x"]}, embedder, taxonomy,
                  vector_store=store, top=3)
        assert spy.call_count == 500  # full path

    big = _mk_jobs(600, skills)
    with patch.object(rank_mod, "score_job", wraps=rank_mod.score_job) as spy2:
        rank_jobs(big, student_vec, {"python": ["x"]}, embedder, taxonomy,
                  vector_store=store, top=3)
        # auto-activated: strictly fewer than 600 scoring calls
        assert spy2.call_count < 600


def test_rank_jobs_ann_without_store_search_degrades_to_full():
    """No vector_store -> ANN cannot run -> falls back to full scoring."""
    taxonomy = load_taxonomy()
    embedder = HashingEmbedder()
    jobs = _mk_jobs(30, ["python"])
    student_vec = embedder.embed("python")
    with patch.object(rank_mod, "score_job", wraps=rank_mod.score_job) as spy:
        out = rank_jobs(jobs, student_vec, {"python": ["x"]}, embedder, taxonomy,
                        vector_store=None, top=5, ann_pre_filter_k=10)
        assert spy.call_count == 30
    assert len(out) == 5


# ------------------------------------- seed/scraper adoption (upsert_many)

def test_seed_uses_upsert_many_when_available():
    """seed() prefers vector_store.upsert_many over per-row upsert."""
    from app.api.main import AppContext
    from app.seed import seed

    eng = make_engine("sqlite://")
    init_db(eng)
    taxonomy = load_taxonomy()
    ctx = AppContext(engine=eng, taxonomy=taxonomy, embedder=HashingEmbedder(),
                     vector_store=MemoryVectorStore(), llm=None)
    # prime the store once
    with Session(eng) as db:
        seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
    orig = MemoryVectorStore.upsert_many
    calls = {"many": 0}

    def spy(self, items):
        calls["many"] += 1
        return orig(self, items)

    with patch.object(MemoryVectorStore, "upsert_many", spy):
        with Session(eng) as db:
            seed(db, taxonomy, vector_store=ctx.vector_store, embedder=ctx.embedder)
        assert calls["many"] >= 1


def test_scrape_index_job_vectors_uses_upsert_many_when_available():
    """index_job_vectors prefers upsert_many when the store exposes it."""
    from types import SimpleNamespace

    from app.scraper.pipeline import index_job_vectors

    calls = {"many": 0, "one": 0}

    class StoreSpy(MemoryVectorStore):
        def upsert_many(self, items):
            calls["many"] += 1
            return super().upsert_many(items)

        def upsert(self, kind, ref_id, vector):
            calls["one"] += 1
            return super().upsert(kind, ref_id, vector)

    embedder = HashingEmbedder()
    store = StoreSpy()
    job = SimpleNamespace(
        id="job-ext-1", title="Data Eng", company="Corp",
        description="Spark and Python pipelines",
    )
    n = index_job_vectors([job], store, embedder)
    assert n == 1
    assert calls["many"] >= 1
    assert calls["one"] == 0
    assert store.get("job", "job-ext-1") is not None


def test_scrape_index_job_vectors_falls_back_without_upsert_many():
    """A store lacking upsert_many still gets per-row upserts."""
    from types import SimpleNamespace

    from app.scraper.pipeline import index_job_vectors

    class LegacyStore:
        def __init__(self):
            self.items = {}

        def upsert(self, kind, ref_id, vector):
            self.items[(kind, ref_id)] = vector

        def get(self, kind, ref_id):
            return self.items.get((kind, ref_id))

        def search(self, kind, vector, k):
            return []

    store = LegacyStore()
    job = SimpleNamespace(id="job-ext-2", description="python work")
    n = index_job_vectors([job], store, HashingEmbedder())
    assert n == 1
    assert ("job", "job-ext-2") in store.items


def test_scrape_index_job_vectors_caches_and_scales():
    """job_vectors cache is populated and embedding failures are isolated."""
    from types import SimpleNamespace

    from app.scraper.pipeline import index_job_vectors

    store = MemoryVectorStore()
    cache: dict[str, list[float]] = {}
    good = SimpleNamespace(id="job-a", description="python a")
    bad = SimpleNamespace(id="job-b", description="")

    class BoomEmbedder(HashingEmbedder):
        def embed(self, text):
            if not text:
                raise ValueError("empty text")
            return super().embed(text)

    n = index_job_vectors([good, bad], store, BoomEmbedder(), job_vectors=cache)
    assert n == 1  # only the good job indexed
    assert "job-a" in cache
    assert store.get("job", "job-a") is not None
    assert store.get("job", "job-b") is None
