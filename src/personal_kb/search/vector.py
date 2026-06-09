"""KNN vector search via cosine distance."""

import logging

from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


async def vector_search(
    embedder: EmbeddingClient,
    query: str,
    limit: int = 20,
    *,
    project_ref: str | None = None,
    entry_type: str | None = None,
    tags: list[str] | None = None,
    contributor: str | None = None,
    team: str | None = None,
) -> list[tuple[str, float]]:
    """Search using cosine distance.

    Returns (entry_id, distance) pairs. Lower distance = better match.
    Returns empty list if embeddings are unavailable.

    Optional metadata filters are passed through to the embedder so the
    underlying KNN result set is restricted to matching entries at the
    SQL level — this prevents the vector leg of hybrid search from
    smuggling in wrong-type / wrong-project entries past the FTS leg's
    filters (the bug fixed in the hybrid search filter regression).
    """
    embedding = await embedder.embed(query)
    if embedding is None:
        return []

    try:
        return await embedder.search_similar(
            embedding,
            limit=limit,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
            contributor=contributor,
            team=team,
        )
    except Exception:
        logger.warning("Vector search failed", exc_info=True)
        return []
