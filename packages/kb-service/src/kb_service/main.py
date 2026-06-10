"""Personal KB web service — FastAPI application."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from kb_core import Attribution, create_postgres

from kb_service.config import (
    build_agentic_config,
    build_embedding_config,
    build_ingest_config,
    build_provider_config,
)
from kb_service.database import close_db, init_db
from kb_service.routes.admin_routes import router as admin_router
from kb_service.routes.auth_routes import router as auth_router
from kb_service.routes.kb_routes import router as kb_router
from kb_service.routes.maps_routes import router as maps_router
from kb_service.routes.query_routes import router as query_router
from kb_service.routes.settings_routes import router as settings_router

logger = logging.getLogger(__name__)


def _parse_int(env_var: str, default: int) -> int:
    """Parse an integer env var, falling back to a default on absence."""
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    return int(raw)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle.

    Opens the service-auth tables (KB_SERVICE_DATABASE_URL) and a single
    kb-core ``KnowledgeBase`` singleton (KB_DATABASE_URL), storing the latter
    on ``app.state.kb``. The Ollama embedder is opened ONCE here (it is not
    per-request safe).
    """
    await init_db()

    app.state.kb = await create_postgres(
        os.environ["KB_DATABASE_URL"],
        embedding_dim=_parse_int("KB_EMBEDDING_DIM", 1024),
        pool_min=_parse_int("KB_PG_POOL_MIN", 1),
        pool_max=_parse_int("KB_PG_POOL_MAX", 5),
        embedding=build_embedding_config(),
        providers=build_provider_config(),
        ingest=build_ingest_config(),
        agentic=build_agentic_config(),
        attribution=Attribution(),
    )

    yield

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
app.include_router(maps_router)
app.include_router(query_router)
app.include_router(settings_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}
