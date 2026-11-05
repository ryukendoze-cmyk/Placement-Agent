from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import (
    admin,
    agent,
    applications,
    approvals,
    auth,
    copilot,
    interviews,
    jobs,
    pages,
    plans,
    students,
)
from app.embeddings.embedder import Embedder
from app.embeddings.store import VectorStore
from app.taxonomy.taxonomy import Taxonomy

BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass
class AppContext:
    engine: object            # sqlalchemy Engine
    taxonomy: Taxonomy
    embedder: Embedder
    vector_store: VectorStore
    llm: object | None = None
    job_vectors: dict[str, list[float]] = field(default_factory=dict)
    llm_telemetry: object | None = None
    task_queue: object | None = None


def create_app(ctx: AppContext) -> FastAPI:
    app = FastAPI(title="PS-06 Placement Agent", version="0.1.0")
    app.state.ctx = ctx
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.include_router(pages.router)
    app.include_router(students.router)
    app.include_router(jobs.router)
    app.include_router(plans.router)
    app.include_router(interviews.router)
    app.include_router(admin.router)
    app.include_router(agent.router)
    app.include_router(copilot.router)
    app.include_router(approvals.router)
    app.include_router(applications.router)
    app.include_router(auth.router)
    return app
