"""Channel-side helpers to extract backend / KnowledgeBase from lifespan context.

Production lifespans:

* **HTTP mode** — ``{"backend": HttpBackend}``
* **Local mode** — ``{"kb": KnowledgeBase, "backend": LocalBackend}``

Test lifespans supply a loose ``{"db": ..., "store": ..., ...}`` tuple that
:func:`kb_from_lifespan` wraps in an ephemeral :class:`KnowledgeBase`.
:func:`backend_from_lifespan` wraps that KB in a :class:`LocalBackend`.

:func:`kb_from_lifespan` keeps working unchanged for existing tool tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from kb_core.knowledge_base import KnowledgeBase

    from personal_kb.backend.protocol import Backend


def kb_from_lifespan(lifespan: dict[str, Any]) -> KnowledgeBase:
    """Return the :class:`KnowledgeBase` for this lifespan.

    Production lifespans store ``{"kb": kb}`` and the helper just returns it.
    Test lifespans that store a loose ``{"db": ..., "store": ..., ...}`` tuple
    get an ephemeral facade built around those pieces — same identity for ``db``
    / ``store`` / ``embedder`` so the test's assertions on counts and DB state
    keep observing the right thing.

    Raises :class:`RuntimeError` if called in HTTP mode (no local KB).
    """
    from kb_core.knowledge_base import KnowledgeBase as _KnowledgeBase

    existing = lifespan.get("kb")
    if existing is not None:
        assert isinstance(existing, _KnowledgeBase)  # noqa: S101 — narrow Any → KnowledgeBase
        return existing

    # HTTP mode — no local KB available
    if "backend" in lifespan:
        from personal_kb.backend.http import HttpBackend

        if isinstance(lifespan["backend"], HttpBackend):
            raise RuntimeError(
                "kb_from_lifespan() called in HTTP mode — no local KnowledgeBase is available. "
                "Use backend_from_lifespan() instead."
            )

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


def backend_from_lifespan(lifespan: dict[str, Any]) -> Backend:
    """Return the :class:`~personal_kb.backend.protocol.Backend` for this lifespan.

    * Production HTTP mode: returns the :class:`HttpBackend` stored under
      ``"backend"``.
    * Production local mode: returns the :class:`LocalBackend` stored under
      ``"backend"``.
    * Test loose-dict lifespans: builds an ephemeral
      :class:`~kb_core.knowledge_base.KnowledgeBase` via
      :func:`kb_from_lifespan`, then wraps it in a :class:`LocalBackend`.
    """
    from personal_kb.backend.local import LocalBackend as _LocalBackend

    # Production path: backend is pre-built by server.py lifespan
    backend = lifespan.get("backend")
    if backend is not None:
        return cast("Backend", backend)

    # Test loose-dict path: build an ephemeral KB + LocalBackend
    kb = kb_from_lifespan(lifespan)
    return _LocalBackend(kb)


__all__ = ["backend_from_lifespan", "kb_from_lifespan"]
