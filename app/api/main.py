import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import Session

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
from app.security.audit import log_audit
from app.security.rbac import current_role, role_explicitly_set
from app.taxonomy.taxonomy import Taxonomy

BASE_DIR = Path(__file__).resolve().parents[1]

_PUBLIC_EXACT = {
    "/login",
    "/healthz",
    "/readyz",
    "/manifest.webmanifest",
    "/sw.js",
    "/api/campus-drives/public-jnf",
}
_PUBLIC_PREFIXES = (
    "/auth/",
    "/static/",
    "/jnf/",
    "/api/campus-drives/public-jnf/",
)
_AUDIT_DENIED_LOCK = threading.Lock()
_AUDIT_DENIED_LAST: dict[tuple[int, str, str, int], float] = {}
_AUDIT_DENIED_WINDOW_S = 10.0


def _is_public_path(path: str, *, demo_role_switch: bool) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return True
    if demo_role_switch and path in ("/api/auth/switch-role", "/api/auth/role"):
        return True
    return False


def _audit_auth_denied_rate_limited(
    request: Request,
    *,
    actor: str,
    status_code: int,
) -> None:
    ctx = getattr(request.app.state, "ctx", None)
    engine = getattr(ctx, "engine", None) if ctx is not None else None
    if engine is None:
        return
    path = request.url.path
    now = time.monotonic()
    key = (id(engine), actor, path, status_code)
    with _AUDIT_DENIED_LOCK:
        last = _AUDIT_DENIED_LAST.get(key)
        if last is not None and (now - last) < _AUDIT_DENIED_WINDOW_S:
            return
        _AUDIT_DENIED_LAST[key] = now
        # Prune stale entries opportunistically so memory stays bounded.
        if len(_AUDIT_DENIED_LAST) > 2048:
            cutoff = now - _AUDIT_DENIED_WINDOW_S
            stale = [k for k, ts in _AUDIT_DENIED_LAST.items() if ts < cutoff]
            for k in stale:
                _AUDIT_DENIED_LAST.pop(k, None)
    try:
        with Session(engine) as db:
            log_audit(
                db,
                action="auth.denied",
                entity_type="route",
                entity_id=path,
                actor_role=actor,
                detail={
                    "trace_id": f"tr-{uuid4().hex[:8]}",
                    "path": path,
                    "method": request.method,
                    "status_code": status_code,
                },
                commit=True,
            )
    except Exception:
        pass


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
    # Warm in-memory vector store when booting over an existing non-Postgres database
    try:
        from app.api.__main__ import backfill_memory_vectors
        backfill_memory_vectors(ctx)
    except Exception:
        pass
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @app.middleware("http")
    async def rbac_strict_and_audit_middleware(request: Request, call_next):
        from app.config import settings

        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        is_strict = str(getattr(settings, "auth_mode", "demo")).lower() == "strict"
        allow_demo_switch = bool(getattr(settings, "demo_role_switch", True))

        if is_strict and not _is_public_path(path, demo_role_switch=allow_demo_switch):
            if not role_explicitly_set(request):
                if path.startswith("/api/"):
                    _audit_auth_denied_rate_limited(
                        request, actor="anonymous", status_code=401
                    )
                    return JSONResponse(
                        status_code=401,
                        content={"detail": "Authentication required"},
                    )
                return RedirectResponse(url="/login", status_code=303)

        response = await call_next(request)
        if response.status_code == 403 and path not in ("/auth/login", "/api/auth/switch-role"):
            _audit_auth_denied_rate_limited(
                request, actor=current_role(request), status_code=403
            )
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:8010",
            "http://localhost:8010",
            "capacitor://localhost",
            "http://localhost",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def webmanifest():
        return FileResponse(
            str(BASE_DIR / "static" / "manifest.webmanifest"),
            media_type="application/manifest+json",
        )

    @app.get("/sw.js", include_in_schema=False)
    def service_worker():
        return FileResponse(
            str(BASE_DIR / "static" / "sw.js"),
            media_type="application/javascript",
        )

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
    try:
        from app.api.routes import campus_drives
        app.include_router(campus_drives.router)
    except ImportError:
        pass
    return app
