"""FTS5/BM25 full-text search."""

import logging

import aiosqlite

logger = logging.getLogger(__name__)


async def fts_search(
    db: aiosqlite.Connection,
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
    fts_query = _escape_fts_query(query)
    if not fts_query:
        return []

    # Join FTS results back to knowledge_entries via rowid
    sql = """
        SELECT e.id, bm25(knowledge_fts) as score
        FROM knowledge_fts f
        JOIN knowledge_entries e ON e.rowid = f.rowid
        WHERE knowledge_fts MATCH ?
        AND e.is_active = 1
    """
    params: list[str | int] = [fts_query]

    if project_ref:
        sql += " AND e.project_ref = ?"
        params.append(project_ref)
    if entry_type:
        sql += " AND e.entry_type = ?"
        params.append(entry_type)
    if tags:
        # Tags are stored as space-separated text; check each tag is present
        for tag in tags:
            sql += " AND (' ' || e.tags || ' ') LIKE ?"
            params.append(f"% {tag} %")

    sql += " ORDER BY score LIMIT ?"
    params.append(limit)

    try:
        cursor = await db.execute(sql, params)
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]
    except Exception:
        logger.warning("FTS search failed for query: %s", query, exc_info=True)
        return []


def _escape_fts_query(query: str) -> str:
    """Convert a natural language query to a safe FTS5 query.

    Wraps each token in quotes to avoid FTS5 syntax errors from special chars.
    """
    tokens = query.split()
    if not tokens:
        return ""
    escaped = [f'"{token}"' for token in tokens]
    return " ".join(escaped)
