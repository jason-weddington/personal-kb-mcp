"""FastAPI application factory for the KB Explorer web server."""

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from personal_kb.db.backend import Database
from personal_kb.llm.provider import LLMProvider
from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

_STATIC_DIR = str(Path(__file__).resolve().parent.parent / "explorer" / "static")


def create_app_with_deps(
    db: Database,
    embedder: EmbeddingClient | None,
    query_llm: LLMProvider | None,
    synthesis_llm: LLMProvider | None = None,
    *,
    store: Any | None = None,
    graph_builder: Any | None = None,
    graph_enricher: Any | None = None,
    extraction_llm: LLMProvider | None = None,
    contributor: str | None = None,
    team: str | None = None,
) -> Any:
    """Create a FastAPI app using pre-existing deps from MCP lifespan.

    Returns a FastAPI application. The caller owns the lifecycle of db/embedder/llm.
    """
    from fastapi import FastAPI

    from personal_kb.web.routes import register_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from personal_kb.web.chat_history import ChatHistoryStore

        ch = await ChatHistoryStore.open()
        app.state.chat_history = ch
        try:
            yield
        finally:
            await ch.close()

    app = FastAPI(title="KB Explorer", lifespan=lifespan)

    from starlette.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    app.state.db = db
    app.state.embedder = embedder
    app.state.query_llm = query_llm
    app.state.synthesis_llm = synthesis_llm
    app.state.store = store
    app.state.graph_builder = graph_builder
    app.state.graph_enricher = graph_enricher
    app.state.extraction_llm = extraction_llm
    app.state.contributor = contributor
    app.state.team = team
    register_routes(app)
    return app


def create_app() -> Any:
    """Create a standalone FastAPI app that manages its own db/embedder/llm.

    Used by the CLI entry point (personal-kb-web).
    """
    from fastapi import FastAPI

    from personal_kb.config import (
        get_embedding_dim,
        get_log_level,
        get_query_provider,
    )
    from personal_kb.db.connection import create_connection
    from personal_kb.server import _create_llm, _create_synthesis_llm
    from personal_kb.web.routes import register_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from personal_kb.config import (
            get_contributor,
            get_extraction_provider,
            get_team,
        )
        from personal_kb.graph.builder import GraphBuilder
        from personal_kb.graph.enricher import GraphEnricher
        from personal_kb.store.knowledge_store import KnowledgeStore

        logging.basicConfig(
            level=getattr(logging, get_log_level()),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            stream=sys.stderr,
        )
        db = await create_connection(embedding_dim=get_embedding_dim())
        embedder = EmbeddingClient(db)
        store = KnowledgeStore(db)
        graph_builder = GraphBuilder(db)

        query_provider = get_query_provider()
        query_llm = _create_llm(query_provider)
        synthesis_llm = _create_synthesis_llm(query_provider)

        extraction_provider = get_extraction_provider()
        extraction_llm = _create_llm(extraction_provider)
        graph_enricher: GraphEnricher | None = None
        if extraction_llm is not None:
            graph_enricher = GraphEnricher(db, extraction_llm)

        app.state.db = db
        app.state.embedder = embedder
        app.state.query_llm = query_llm
        app.state.synthesis_llm = synthesis_llm
        app.state.store = store
        app.state.graph_builder = graph_builder
        app.state.graph_enricher = graph_enricher
        app.state.extraction_llm = extraction_llm
        app.state.contributor = get_contributor()
        app.state.team = get_team()

        try:
            yield
        finally:
            if synthesis_llm is not None:
                await synthesis_llm.close()
            if query_llm is not None:
                await query_llm.close()
            if extraction_llm is not None and extraction_llm is not query_llm:
                await extraction_llm.close()
            await embedder.close()
            await db.close()

    app = FastAPI(title="KB Explorer", lifespan=lifespan)

    from starlette.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    register_routes(app)
    return app
