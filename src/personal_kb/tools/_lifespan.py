"""Channel-side helper to extract a :class:`KnowledgeBase` from lifespan context.

The MCP server's lifespan stores a single ``"kb"`` key — the
:class:`~kb_core.knowledge_base.KnowledgeBase` facade. Every tool reads its
dependency through :func:`kb_from_lifespan`.

The helper also supports a backwards-compatible **loose-tuple** lifespan
(``{"db": ..., "store": ..., "embedder": ..., ...}``) so existing tool tests
that build a lifespan dict by hand keep working without churn. In that path the
helper builds an ephemeral :class:`KnowledgeBase` around the test-supplied
pieces; nothing is closed by the facade (the test owns the underlying handles).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb_core.knowledge_base import KnowledgeBase


def kb_from_lifespan(lifespan: dict[str, Any]) -> KnowledgeBase:
    """Return the :class:`KnowledgeBase` for this lifespan.

    Production lifespans store ``{"kb": kb}`` and the helper just returns it.
    Test lifespans that store a loose ``{"db": ..., "store": ..., ...}`` tuple
    get an ephemeral facade built around those pieces — same identity for ``db``
    / ``store`` / ``embedder`` so the test's assertions on counts and DB state
    keep observing the right thing.
    """
    from kb_core.knowledge_base import KnowledgeBase as _KnowledgeBase

    existing = lifespan.get("kb")
    if existing is not None:
        assert isinstance(existing, _KnowledgeBase)  # noqa: S101 — narrow Any → KnowledgeBase
        return existing

    # Backwards-compat: build an ephemeral KnowledgeBase wrapping the
    # test-supplied dependencies. None of these are owned by the facade — the
    # test fixture closes the DB itself in its teardown.
    from kb_core.config import Attribution, KbConfig
    from kb_core.graph.builder import GraphBuilder
    from kb_core.store.knowledge_store import KnowledgeStore

    db = lifespan["db"]
    store = lifespan.get("store") or KnowledgeStore(db)
    graph_builder = lifespan.get("graph_builder") or GraphBuilder(db)

    # Test fixtures historically supply only "query_llm"; the channel-side
    # ingest path used to read that same key. Map it into extraction_llm too
    # so kb.ingest_*() works (the facade keys ingest off extraction_llm).
    extraction_llm = (
        lifespan.get("llm_client") or lifespan.get("extraction_llm") or lifespan.get("query_llm")
    )

    config = KbConfig(
        attribution=Attribution(
            contributor=lifespan.get("contributor"),
            team=lifespan.get("team"),
        )
    )
    return _KnowledgeBase(
        config=config,
        db=db,
        store=store,
        embedder=lifespan.get("embedder"),
        graph_builder=graph_builder,
        graph_enricher=lifespan.get("graph_enricher"),
        extraction_llm=extraction_llm,
        query_llm=lifespan.get("query_llm"),
        synthesis_llm=lifespan.get("synthesis_llm"),
    )


__all__ = ["kb_from_lifespan"]
