"""FTS5/BM25 full-text search."""

import logging

from kb_core.db.backend import Database

logger = logging.getLogger(__name__)

# Log marker for an FTS-leg failure (the search then runs on the vector leg alone).
FTS_LEG_MARKER = "fts-leg"


async def fts_search(
    db: Database,
    query: str,
    limit: int = 20,
    project_ref: str | None = None,
    entry_type: str | None = None,
    tags: list[str] | None = None,
    contributor: str | None = None,
    team: str | None = None,
) -> list[tuple[str, float]]:
    """Search the full-text index of the backend.

    Results are the entries matching at least ``fts_min_coverage(n)`` of the
    n tokens from ``fts_query_tokens``, ordered by coverage descending, then
    score ascending (lower is better), then id. Returns (entry_id, score)
    pairs. On any backend exception, logs a ``fts-leg op=error`` warning and
    returns no hits.
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
            contributor=contributor,
            team=team,
        )
    except Exception:
        logger.warning("%s op=error query=%r", FTS_LEG_MARKER, query[:80], exc_info=True)
        return []
