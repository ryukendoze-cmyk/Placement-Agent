import pytest
from sqlalchemy import create_engine, text
from sqlmodel import Session

from app.embeddings.store import PgVectorStore

pytestmark = pytest.mark.integration


def test_pgvector_roundtrip():
    eng = create_engine("postgresql+psycopg://postgres@127.0.0.1:5432/ps06_test")
    with Session(eng) as db:
        store = PgVectorStore(db, dim=384)
        try:
            store.upsert("job", "x", [1.0] + [0.0] * 383)
            store.upsert("job", "y", [0.0, 1.0] + [0.0] * 382)
            res = store.search("job", [0.9, 0.1] + [0.0] * 382, k=1)
            assert res[0][0] == "x"
            assert res[0][1] > 0.99
            assert store.get("job", "x")[0] == pytest.approx(1.0)
            assert store.get("job", "missing") is None
        finally:
            db.execute(text("DELETE FROM embeddings WHERE kind='job' AND ref_id IN ('x','y')"))
            db.commit()
