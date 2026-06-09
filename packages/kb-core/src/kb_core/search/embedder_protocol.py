"""Structural typing for the embedding client used by search modules.

The concrete ``EmbeddingClient`` lives in ``personal_kb.search.embeddings``
(it reads env / Ollama config and is therefore not part of the kb_core
nucleus). The hybrid and vector search modules in kb_core depend only on
two methods — ``embed()`` and ``search_similar()`` — so they program
against this Protocol rather than importing the concrete class. Any
caller that implements both methods (the production ``EmbeddingClient``,
the test ``FakeEmbedder`` fixture, or a future provider) satisfies the
interface structurally.

This is purely a typing concession to satisfy the import-purity rule —
no behavior change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedder interface used by ``kb_core.search``.

    Implementations supply text-to-vector embedding and a vector KNN
    search. The KNN call's metadata filters mirror the SQL-level filters
    on the FTS leg so the hybrid RRF caller can trust both legs honor
    the same scoping.
    """

    async def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for ``text``, or ``None`` if unavailable."""
        ...

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 20,
        *,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return (entry_id, distance) pairs — lower distance is closer."""
        ...
