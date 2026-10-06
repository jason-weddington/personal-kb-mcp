"""Personal KB web service — FastAPI application."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kb_core import Attribution, EmbeddingRetryConfig, create_postgres, create_sqlite

from kb_service.config import (
    build_agentic_config,
    build_embedding_config,
    build_embedding_retry_config,
    build_ingest_config,
    build_provider_config,
)
from kb_service.database import close_db, init_db
from kb_service.routes.admin_routes import router as admin_router
from kb_service.routes.auth_routes import router as auth_router
from kb_service.routes.chat_routes import router as chat_router
from kb_service.routes.cluster_ledger_routes import router as cluster_ledger_router
from kb_service.routes.embedding_queue_routes import router as embedding_queue_router
from kb_service.routes.ingest_routes import router as ingest_router
from kb_service.routes.kb_read_routes import router as kb_read_router
from kb_service.routes.kb_routes import router as kb_router
from kb_service.routes.kb_write_routes import router as kb_write_router
from kb_service.routes.listener_routes import router as listener_router
from kb_service.routes.map_delete_routes import router as map_delete_router
from kb_service.routes.map_eligibility_routes import router as map_eligibility_router
from kb_service.routes.map_lint_routes import router as map_lint_router
from kb_service.routes.map_loop_routes import router as map_loop_router
from kb_service.routes.map_op_routes import router as map_op_router
from kb_service.routes.maps_routes import router as maps_router
from kb_service.routes.nudge_routes import router as nudge_router
from kb_service.routes.query_routes import router as query_router
from kb_service.routes.settings_routes import router as settings_router
from kb_service.routes.telemetry_routes import router as telemetry_router

if TYPE_CHECKING:
    from kb_core import KnowledgeBase

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def mount_frontend(app: FastAPI, dist_dir: Path) -> bool:
    """Mount the built SPA from dist_dir onto app.

    Returns True if the SPA was mounted, False if dist_dir/index.html
    does not exist (no-op — all API routes continue to work normally).
    """
    if not (dist_dir / "index.html").is_file():
        return False

    assets_dir = dist_dir / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    resolved_dist = dist_dir.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (dist_dir / full_path).resolve()
        if candidate.is_file() and candidate.is_relative_to(resolved_dist):
            return FileResponse(candidate)
        return FileResponse(dist_dir / "index.html")

    return True


def _parse_int(env_var: str, default: int) -> int:
    """Parse an integer env var, falling back to a default on absence."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    return int(raw)


# Default SQLite data-DB path mirrors personal_kb.config.get_db_path (line 57).
# The raw string (including the leading ``~``) is passed through to kb-core's
# ``_open_sqlite``, which handles tilde expansion and parent-dir mkdir.
_DEFAULT_KB_DB_PATH = "~/.local/share/personal_kb/knowledge.db"


async def _open_kb() -> "KnowledgeBase":
    """Open the kb-core data DB, branching on ``KB_DATABASE_URL``.

    * ``KB_DATABASE_URL`` set AND non-empty -> ``create_postgres`` with the
      same kwargs as the original lifespan (hosted Postgres mode is
      byte-for-byte unchanged).
    * ``KB_DATABASE_URL`` unset OR empty -> ``create_sqlite`` opening the
      SQLite file at ``KB_DB_PATH`` (default ``~/.local/share/personal_kb/
      knowledge.db``). The path is passed RAW; kb-core expands ``~`` and
      mkdir's the parent dir downstream.

    Extracting this out of ``lifespan`` lets tests monkeypatch
    ``create_postgres`` and ``create_sqlite`` independently — driving
    lifespan via ``TestClient`` would otherwise reach an un-patched
    factory on whichever branch the env happens to take.
    """
    database_url = os.environ.get("KB_DATABASE_URL")
    if database_url:
        return await create_postgres(
            database_url,
            embedding_dim=_parse_int("KB_EMBEDDING_DIM", 1024),
            pool_min=_parse_int("KB_PG_POOL_MIN", 1),
            pool_max=_parse_int("KB_PG_POOL_MAX", 5),
            embedding=build_embedding_config(),
            providers=build_provider_config(),
            ingest=build_ingest_config(),
            agentic=build_agentic_config(),
            attribution=Attribution(),
            embedding_retry=build_embedding_retry_config(),
        )

    sqlite_path = os.environ.get("KB_DB_PATH", _DEFAULT_KB_DB_PATH)
    return await create_sqlite(
        sqlite_path,
        embedding_dim=_parse_int("KB_EMBEDDING_DIM", 1024),
        embedding=build_embedding_config(),
        providers=build_provider_config(),
        ingest=build_ingest_config(),
        agentic=build_agentic_config(),
        attribution=Attribution(),
        # The SQLite branch shares ONE aiosqlite connection with every request
        # handler — a worker commit could commit a half-written request
        # transaction (see KnowledgeBase.start_embedding_worker's docstring).
        # The worker is Postgres-only; always off here regardless of env.
        embedding_retry=EmbeddingRetryConfig(enabled=False),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle.

    Opens the service-auth tables (KB_SERVICE_DATABASE_URL, or a local SQLite
    ``service.db`` in no-auth mode — see ``kb_service.database``) and a single
    kb-core ``KnowledgeBase`` singleton (KB_DATABASE_URL or default SQLite),
    storing the latter on ``app.state.kb``. The Ollama embedder is opened
    ONCE here (it is not per-request safe).

    Shutdown (``stop_embedding_worker`` / ``kb.close`` / ``close_db``)
    always runs once ``app.state.kb`` has been assigned — wrapped in
    ``try/finally`` so an exception raised through the yielded phase (e.g. a
    request handler bug that propagates past ASGI error handling) still
    releases the worker task and both DB connections instead of leaking
    them. A failure constructing ``app.state.kb`` itself (before the
    ``try``) still fails app startup outright, same as before.
    """
    await init_db()

    app.state.kb = await _open_kb()
    try:
        await app.state.kb.start_embedding_worker()
        yield
    finally:
        await app.state.kb.stop_embedding_worker()
        await app.state.kb.close()
        await close_db()


app = FastAPI(title="Personal KB Web Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://localhost",
        f"https://{os.environ.get('HOSTNAME', 'localhost')}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(kb_router)
app.include_router(kb_read_router)
app.include_router(maps_router)
app.include_router(query_router)
app.include_router(settings_router)
app.include_router(kb_write_router)
app.include_router(ingest_router)
app.include_router(chat_router)
app.include_router(cluster_ledger_router)
app.include_router(listener_router)
app.include_router(telemetry_router)
app.include_router(embedding_queue_router)
app.include_router(map_eligibility_router)
app.include_router(map_lint_router)
app.include_router(nudge_router)
app.include_router(map_loop_router)
app.include_router(map_op_router)
app.include_router(map_delete_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


mount_frontend(app, FRONTEND_DIST)
