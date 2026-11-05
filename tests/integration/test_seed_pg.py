import pytest
from sqlalchemy import text
from sqlmodel import Session

from app.db import init_db, make_engine
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import PgVectorStore
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy

pytestmark = pytest.mark.integration


def test_seed_stores_job_vectors_in_pgvector():
    eng = make_engine("postgresql+psycopg://postgres@127.0.0.1:5432/ps06_test")
    init_db(eng)
    embedder = HashingEmbedder()   # deterministic + fast; LocalEmbedder is exercised manually via CLI
    with Session(eng) as db:
        store = PgVectorStore(db, dim=384)
        # clean slate: shared ps06_test retains rows from other integration tests (e.g. x/y in test_pgvector)
        db.execute(text("DELETE FROM embeddings WHERE kind='job'"))
        db.commit()
        seed(db, load_taxonomy(), vector_store=store, embedder=embedder)
        count = db.execute(text("SELECT count(*) FROM embeddings WHERE kind='job'")).scalar()
        assert count == 24
        # verify one vector is readable and 384-dim
        row = db.execute(text("SELECT embedding FROM embeddings WHERE kind='job' AND ref_id='job-001'")).first()
        assert row is not None
        vec = [float(x) for x in str(row[0]).strip().strip("[]").split(",")]
        assert len(vec) == 384
    # idempotency: seeding again must not fail or duplicate
    with Session(eng) as db:
        store = PgVectorStore(db, dim=384)
        seed(db, load_taxonomy(), vector_store=store, embedder=embedder)
        count = db.execute(text("SELECT count(*) FROM embeddings WHERE kind='job'")).scalar()
        assert count == 24
