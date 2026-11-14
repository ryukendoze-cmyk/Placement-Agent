import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event
from sqlmodel import Session

from app.api.main import AppContext, create_app
from app.config import settings
from app.db import init_db, make_engine

# Keep automated pytest runs deterministic in demo mode even when `.env`
# ships hardened strict production defaults (`AUTH_MODE=strict`).
settings.auth_mode = "demo"
settings.demo_role_switch = True
from app.embeddings.embedder import HashingEmbedder
from app.embeddings.store import MemoryVectorStore
from app.llm.mock import MockLLM
from app.seed import seed
from app.taxonomy.taxonomy import load_taxonomy


def _fk_on(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


def _build_client():
    eng = make_engine("sqlite://")
    event.listen(eng, "connect", _fk_on)
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
    app = create_app(ctx)
    return TestClient(app), eng


@pytest.fixture()
def client():
    c, _ = _build_client()
    with c:
        yield c


@pytest.fixture()
def client_engine():
    """(client, engine) for tests that must inspect persisted DB rows."""
    c, eng = _build_client()
    with c:
        yield c, eng
