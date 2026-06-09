"""FastAPI application factory for the KB Explorer web server.

The web channel hangs a :class:`~kb_core.knowledge_base.KnowledgeBase`
instance off ``app.state.kb`` and uses its facade methods
(:meth:`KnowledgeBase.search`, :meth:`KnowledgeBase.ask`,
:meth:`KnowledgeBase.summarize`, :meth:`KnowledgeBase.ingest_*`,
:meth:`KnowledgeBase.get`, …) for all reads and writes. The route
handlers do not reach into :mod:`personal_kb.tools` — the web → tools
backdoor is gone.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kb_core.config import Attribution, KbConfig, SqliteConfig
from kb_core.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from kb_core.db.backend import Database
    from kb_core.graph.builder import GraphBuilder
    from kb_core.graph.enricher import GraphEnricher
    from kb_core.llm.provider import LLMProvider
    from kb_core.search.embedder_protocol import Embedder
    from kb_core.store.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

_STATIC_DIR = str(Path(__file__).resolve().parent.parent / "explorer" / "static")


def _wrap_deps_as_kb(
    db: Database,
    embedder: Embedder | None,
    query_llm: LLMProvider | None,
    synthesis_llm: LLMProvider | None,
    *,
    store: KnowledgeStore | None,
    graph_builder: GraphBuilder | None,
    graph_enricher: GraphEnricher | None,
    extraction_llm: LLMProvider | None,
    contributor: str | None,
    team: str | None,
) -> KnowledgeBase:
    """Wrap pre-built dependencies in a :class:`KnowledgeBase` shell.

    Used by :func:`create_app_with_deps` when callers (the MCP-side
    explorer auto-start in :mod:`personal_kb.tools.kb_explore`, the
    web test suite) already own ``db`` / ``embedder`` / etc. The facade
    becomes the single source of truth on ``app.state.kb`` so the route
    handlers can call facade methods like any other consumer; ownership
    stays with the caller (all ``_owned_*`` flags are False, so
    :meth:`KnowledgeBase.close` is a no-op for these resources).
    """
    from kb_core.graph.builder import GraphBuilder as _GraphBuilder
    from kb_core.store.knowledge_store import KnowledgeStore as _KnowledgeStore

    # Minimal config sufficient for the routes' facade calls. The web
    # channel reads ``attribution`` (contributor/team plumbing) and the
    # ``agentic`` defaults via :meth:`KnowledgeBase.ask`/``summarize``;
    # the database/embedding/provider sub-configs are unused here
    # because the resources are already wired in.
    config = KbConfig(
        database=SqliteConfig(path=Path(":memory:")),
        attribution=Attribution(contributor=contributor, team=team),
    )
    return KnowledgeBase(
        config=config,
        db=db,
        store=store if store is not None else _KnowledgeStore(db),
        embedder=embedder,
        graph_builder=graph_builder if graph_builder is not None else _GraphBuilder(db),
        graph_enricher=graph_enricher,
        extraction_llm=extraction_llm,
        query_llm=query_llm,
        synthesis_llm=synthesis_llm,
        # Caller owns every resource — close() must not touch them.
        _owned_embedder=False,
        _owned_extraction_llm=False,
        _owned_query_llm=False,
        _owned_synthesis_llm=False,
    )


def _build_app(kb: KnowledgeBase) -> Any:
    """Build a FastAPI app wired to a :class:`KnowledgeBase` instance.

    The KB is stashed on ``app.state.kb`` for the route handlers. A
    lifespan opens the chat-history SQLite store and tears it down on
    shutdown. The KB itself is NOT closed here — callers own its
    lifecycle (see :func:`create_app_with_deps` and :func:`create_app`).
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

    app.state.kb = kb
    register_routes(app)
    return app


def create_app_with_deps(
    db: Database,
    embedder: Embedder | None,
    query_llm: LLMProvider | None,
    synthesis_llm: LLMProvider | None = None,
    *,
    store: KnowledgeStore | None = None,
    graph_builder: GraphBuilder | None = None,
    graph_enricher: GraphEnricher | None = None,
    extraction_llm: LLMProvider | None = None,
    contributor: str | None = None,
    team: str | None = None,
) -> Any:
    """Create a FastAPI app from pre-existing deps (MCP lifespan / tests).

    The MCP-side explorer auto-start in
    :mod:`personal_kb.tools.kb_explore` passes the lifespan-owned
    :class:`KnowledgeBase`'s components directly; the test suite uses
    the same entry point with fake doubles. Internally we re-wrap the
    deps in a :class:`KnowledgeBase` shell (no ownership transfer) so
    the route handlers all funnel through facade methods just like the
    standalone path.

    ``embedder`` is typed against the protocol
    :class:`~kb_core.search.embedder_protocol.Embedder` because the
    facade exposes its embedder accessor as the protocol — the previous
    concrete :class:`EmbeddingClient` annotation needed a
    ``type: ignore`` at the kb_explore → web boundary which is now
    unnecessary.
    """
    kb = _wrap_deps_as_kb(
        db,
        embedder,
        query_llm,
        synthesis_llm,
        store=store,
        graph_builder=graph_builder,
        graph_enricher=graph_enricher,
        extraction_llm=extraction_llm,
        contributor=contributor,
        team=team,
    )
    return _build_app(kb)


def create_app() -> Any:
    """Create a standalone FastAPI app that owns its own KB lifecycle.

    Used by the CLI entry point (``personal-kb-web``). Builds a full
    :class:`~kb_core.config.KbConfig` via
    :func:`personal_kb.config.build_kb_config` (snapshotting the same
    ``KB_*`` env vars as the MCP server), opens a
    :class:`~kb_core.knowledge_base.KnowledgeBase`, and tears it down
    on shutdown. The facade owns the database, embedder and LLM
    clients.
    """
    from fastapi import FastAPI

    from personal_kb.config import build_kb_config, get_log_level
    from personal_kb.web.routes import register_routes

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        from personal_kb.web.chat_history import ChatHistoryStore

        logging.basicConfig(
            level=getattr(logging, get_log_level()),
            format="%(asctime)s %(name)s %(levelname)s %(message)s",
            stream=sys.stderr,
        )

        kb_config = build_kb_config()
        kb = await KnowledgeBase.create(kb_config)
        app.state.kb = kb

        ch = await ChatHistoryStore.open()
        app.state.chat_history = ch

        try:
            yield
        finally:
            await ch.close()
            await kb.close()

    app = FastAPI(title="KB Explorer", lifespan=lifespan)

    from starlette.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    register_routes(app)
    return app
