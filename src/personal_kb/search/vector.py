"""KNN vector search via cosine distance."""

import logging

from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)


async def vector_search(
    embedder: EmbeddingClient,
    query: str,
    limit: int = 20,
) -> list[tuple[str, float]]:
    """Search using cosine distance.

    Returns (entry_id, distance) pairs. Lower distance = better match.
    Returns empty list if embeddings are unavailable.
    """
    embedding = await embedder.embed(query)
    if embedding is None:
        return []

    try:
        return await embedder.search_similar(embedding, limit=limit)
    except Exception:
        logger.warning("Vector search failed", exc_info=True)
        return []
