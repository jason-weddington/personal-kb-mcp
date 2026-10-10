"""Personal KB web service — FastAPI application."""

import asyncio
import importlib.metadata
import logging
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from kb_core import Attribution, EmbeddingRetryConfig, create_postgres, create_sqlite
from starlette.routing import Route

from kb_service import write_policy
from kb_service.config import (
    build_agentic_config,
    build_embedding_config,
    build_embedding_retry_config,
    build_ingest_config,
    build_provider_config,
)
from kb_service.database import check_database_config, close_db, init_db
from kb_service.mcp_server import McpEndpoint, create_mcp_server
from kb_service.mcp_server.observability import MCP_ENDPOINT_MARKER
from kb_service.mcp_server.server import _get_tool_prefix as _mcp_prefix
from kb_service.routes.admin_routes import router as admin_router
from kb_service.routes.auth_routes import router as auth_router
from kb_service.routes.chat_routes import router as chat_router
from kb_service.routes.cluster_ledger_routes import router as cluster_ledger_router
from kb_service.routes.embedding_queue_routes import router as embedding_queue_router
from kb_service.routes.event_routes import router as event_router
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
from kb_service.routes.metrics_routes import router as metrics_router
from kb_service.routes.nudge_routes import router as nudge_router
from kb_service.routes.prevention_routes import router as prevention_router
from kb_service.routes.query_routes import router as query_router
from kb_service.routes.settings_routes import router as settings_router
from kb_service.routes.surprise_routes import router as surprise_router
from kb_service.routes.telemetry_routes import router as telemetry_router
from kb_service.routes.turn_routes import record_validation_failure
from kb_service.routes.turn_routes import router as turn_router
from kb_service.supersession_log import log_reconcile_report
from kb_service.surprise_worker import (
    SurpriseCaptureWorker,
    should_start_surprise_worker,
)
from kb_service.turn_digest import surprise_capture_mode

if TYPE_CHECKING:
    from kb_core import KnowledgeBase

logger = logging.getLogger(__name__)

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"
PACKAGED_STATIC = Path(__file__).resolve().parent / "static"
STATIC_DIR_ENV = "KB_SERVICE_STATIC_DIR"


def resolve_static_dir(
    env_dir: str | None = None,
    dev_dist: Path | None = None,
    packaged: Path | None = None,
) -> tuple[Path, str] | None:
    """Pick the SPA directory to serve, or None if no candidate has index.html.

    Order: (a) KB_SERVICE_STATIC_DIR, (b) the checkout dev build at
    frontend/dist, (c) the UI packaged inside the wheel. Arguments default to
    the real environment / paths; tests pass explicit ones.
    """
    if env_dir is None:
        env_dir = os.environ.get(STATIC_DIR_ENV)
    candidates: list[tuple[str, Path]] = []
    if env_dir:
        candidates.append((STATIC_DIR_ENV, Path(env_dir)))
    candidates.append(("dev build", FRONTEND_DIST if dev_dist is None else dev_dist))
    candidates.append(("packaged", PACKAGED_STATIC if packaged is None else packaged))
    for source, path in candidates:
        if (path / "index.html").is_file():
            return path, source
    return None


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
        if full_path.startswith("api/") or full_path.startswith(".well-known/"):
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
    check_database_config()
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


async def _reconcile_supersession(kb: "KnowledgeBase") -> None:
    """Best-effort startup supersession reconcile; never blocks startup.

    Logs one INFO summary line and one WARNING per row whose
    ``superseded_by`` it had to change. After the first post-deploy startup
    any ``supersession-reconcile drift`` line is a defect signal: some
    writer left the invariant broken.
    """
    try:
        report = await kb.reconcile_supersession()
        log_reconcile_report(report, logger)
    except Exception as exc:
        logger.warning("supersession-reconcile failed: %s", exc)


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
    check_database_config()
    await init_db()

    app.state.kb = await _open_kb()
    await _reconcile_supersession(app.state.kb)
    app.state.surprise_drain_lock = asyncio.Lock()
    app.state.surprise_worker = None
    app.state.mcp_http_app = None
    try:
        await app.state.kb.start_embedding_worker()
        mode = surprise_capture_mode()
        if should_start_surprise_worker(mode, os.environ.get("KB_DATABASE_URL")):
            worker = SurpriseCaptureWorker(app.state.kb, app.state.surprise_drain_lock)
            await worker.start()
            app.state.surprise_worker = worker
        else:
            logger.info(
                "surprise_worker not started mode=%s (needs shadow/on and a"
                " Postgres KB_DATABASE_URL; use POST /api/kb/surprise/drain)",
                mode,
            )
        await write_policy.log_startup(
            mode, worker_started=app.state.surprise_worker is not None
        )
        # A fresh MCP app per lifespan entry: the mcp session manager's run()
        # can only be entered once per instance.
        mcp = create_mcp_server()
        names = sorted(t.name for t in await mcp.list_tools())
        logger.info(
            MCP_ENDPOINT_MARKER + " started path=/mcp prefix=%s tools=%d names=%s",
            _mcp_prefix(),
            len(names),
            ",".join(names),
        )
        app.state.mcp_http_app = mcp.http_app(
            path="/mcp", stateless_http=True, json_response=True
        )
        try:
            async with app.state.mcp_http_app.lifespan(app.state.mcp_http_app):
                yield
        except BaseExceptionGroup as group:
            # The MCP session manager's task group wraps an exception raised
            # through the yield; re-raise the original so callers see it.
            if len(group.exceptions) == 1:
                raise group.exceptions[0] from None
            raise
    finally:
        app.state.mcp_http_app = None
        if app.state.surprise_worker is not None:
            await app.state.surprise_worker.stop()
        await app.state.kb.stop_embedding_worker()
        await app.state.kb.close()
        await close_db()


app = FastAPI(title="Personal KB Web Service", version="0.1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Log /api/kb/turn 422s (loc and type only); body is FastAPI's default."""
    if request.url.path == "/api/kb/turn":
        raw = exc.body.get("harness") if isinstance(exc.body, dict) else None
        record_validation_failure(
            exc.errors(), raw[:64] if isinstance(raw, str) else None
        )
    return await request_validation_exception_handler(request, exc)


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
app.include_router(event_router)
app.include_router(turn_router)
app.include_router(surprise_router)
app.include_router(prevention_router)
app.include_router(metrics_router)
app.include_router(telemetry_router)
app.include_router(embedding_queue_router)
app.include_router(map_eligibility_router)
app.include_router(map_lint_router)
app.include_router(nudge_router)
app.include_router(map_loop_router)
app.include_router(map_op_router)
app.include_router(map_delete_router)
# MCP over streamable HTTP. Appended as a raw Route (not app.add_route) so the
# ASGI instance type-checks; it must precede the SPA catch-all mounted below.
app.router.routes.append(
    Route("/mcp", endpoint=McpEndpoint(), methods=["GET", "POST", "DELETE"])
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint.

    ``version`` and ``install_id`` (``sys.prefix`` of the serving process) let a
    local client detect a daemon left over from a different install.
    """
    try:
        version = importlib.metadata.version("personal-kb-web-service")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"status": "ok", "version": version, "install_id": sys.prefix}


_static = resolve_static_dir()
if _static is not None and mount_frontend(app, _static[0]):
    logger.info("Serving web UI from %s (%s)", _static[0], _static[1])
else:
    logger.info("No web UI found; serving the API only")
