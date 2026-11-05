import json
import math
from typing import Protocol

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, text
from sqlalchemy.orm import Session
from sqlmodel import Field, SQLModel


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


class VectorStore(Protocol):
    def upsert(self, kind: str, ref_id: str, vector: list[float]) -> None: ...
    def upsert_many(self, items: list[tuple[str, str, list[float]]]) -> int: ...
    def search(self, kind: str, vector: list[float], k: int) -> list[tuple[str, float]]: ...
    def get(self, kind: str, ref_id: str) -> list[float] | None: ...


class MemoryVectorStore:
    """In-memory cosine store — unit tests and SQLite mode."""

    def __init__(self):
        self._items: dict[tuple[str, str], list[float]] = {}

    def upsert(self, kind: str, ref_id: str, vector: list[float]) -> None:
        self._items[(kind, ref_id)] = list(vector)

    def upsert_many(self, items: list[tuple[str, str, list[float]]]) -> int:
        for kind, ref_id, vector in items:
            self._items[(kind, ref_id)] = list(vector)
        return len(items)

    def search(self, kind: str, vector: list[float], k: int) -> list[tuple[str, float]]:
        scored = [(rid, cosine(vector, v)) for (knd, rid), v in self._items.items() if knd == kind]
        scored.sort(key=lambda kv: -kv[1])
        return scored[:k]

    def get(self, kind: str, ref_id: str) -> list[float] | None:
        v = self._items.get((kind, ref_id))
        return list(v) if v is not None else None


class EmbeddingRow(SQLModel, table=True):
    __tablename__ = "embeddings"
    kind: str = Field(primary_key=True)
    ref_id: str = Field(primary_key=True)
    embedding: list[float] = Field(sa_column=Column(Vector(384)))


def _parse_vector(raw: str) -> list[float]:
    inner = raw.strip().strip("[]").strip()
    if not inner:
        return []
    return [float(x) for x in inner.split(",")]


class PgVectorStore:
    """pgvector-backed store over the `embeddings` table (384-dim only).

    The table schema (EmbeddingRow) is fixed at Vector(384); constructing with
    dim != 384 raises ValueError.
    """

    def __init__(self, session: Session, dim: int = 384, ensure_table: bool = True):
        if dim != 384:
            raise ValueError("PgVectorStore only supports dim=384 (EmbeddingRow is fixed at Vector(384))")
        self._session = session
        self.dim = dim
        self._hnsw_ensured = False
        if ensure_table:
            SQLModel.metadata.create_all(bind=session.get_bind(), tables=[EmbeddingRow.__table__])

    def _ensure_hnsw_index(self) -> None:
        """Create the HNSW cosine index once (idempotent) for ANN search.

        Older pgvector builds (< 0.5.0) lack HNSW; a failed CREATE INDEX aborts
        the transaction, so we roll back and fall back to sequential scans
        gracefully.
        """
        if self._hnsw_ensured:
            return
        try:
            self._session.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_embeddings_hnsw "
                "ON embeddings USING hnsw (embedding vector_cosine_ops)"
            ))
            self._session.commit()
        except Exception:
            # Leave the session clean (aborted transaction) before continuing.
            try:
                self._session.rollback()
            except Exception:
                pass
        self._hnsw_ensured = True

    def upsert(self, kind: str, ref_id: str, vector: list[float]) -> None:
        self._session.execute(
            text(
                "INSERT INTO embeddings (kind, ref_id, embedding) VALUES (:k, :r, :v) "
                "ON CONFLICT (kind, ref_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ),
            {"k": kind, "r": ref_id, "v": json.dumps(vector)},
        )
        self._session.commit()

    def upsert_many(self, items: list[tuple[str, str, list[float]]]) -> int:
        """Batch upsert: one parameterized statement for the whole list, one commit."""
        if not items:
            return 0
        self._ensure_hnsw_index()
        params = [{"k": k, "r": r, "v": json.dumps(v)} for k, r, v in items]
        self._session.execute(
            text(
                "INSERT INTO embeddings (kind, ref_id, embedding) "
                "VALUES (:k, :r, :v) "
                "ON CONFLICT (kind, ref_id) DO UPDATE SET embedding = EXCLUDED.embedding"
            ),
            params,
        )
        self._session.commit()
        return len(items)

    def search(self, kind: str, vector: list[float], k: int) -> list[tuple[str, float]]:
        rows = self._session.execute(
            text(
                "SELECT ref_id, 1 - (embedding <=> CAST(:v AS vector)) AS sim "
                "FROM embeddings WHERE kind = :k "
                "ORDER BY embedding <=> CAST(:v AS vector) LIMIT :n"
            ),
            {"k": kind, "v": json.dumps(vector), "n": k},
        ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def get(self, kind: str, ref_id: str) -> list[float] | None:
        row = self._session.execute(
            text("SELECT embedding FROM embeddings WHERE kind = :k AND ref_id = :r"),
            {"k": kind, "r": ref_id},
        ).first()
        if row is None:
            return None
        return _parse_vector(str(row[0]))
