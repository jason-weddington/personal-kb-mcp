"""FTS5/BM25 full-text search."""

import logging

from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


async def fts_search(
    db: Database,
    query: str,
    limit: int = 20,
    project_ref: str | None = None,
    entry_type: str | None = None,
    tags: list[str] | None = None,
) -> list[tuple[str, float]]:
    """Search using FTS5 BM25 ranking.

    Returns (entry_id, bm25_score) pairs. Lower BM25 scores are better matches
    (FTS5 returns negative scores where more negative = better match).
    """
    if not query.strip():
        return []

    try:
        return await db.fts_search(
            query,
            limit=limit,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
        )
    except Exception:
        logger.warning("FTS search failed for query: %s", query, exc_info=True)
        return []
