"""Reciprocal Rank Fusion (RRF) hybrid ranking."""

import logging
from datetime import UTC, datetime

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.db.backend import Database
from personal_kb.db.queries import get_entry
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.search.fts import fts_search
from personal_kb.search.vector import vector_search

logger = logging.getLogger(__name__)

# RRF constant — standard value from the literature
RRF_K = 60


async def hybrid_search(
    db: Database,
    embedder: EmbeddingClient | None,
    query: SearchQuery,
    *,
    contributor: str | None = None,
) -> tuple[list[SearchResult], int]:
    """Execute hybrid search combining FTS5 and vector similarity via RRF.

    Falls back to FTS-only when embeddings are unavailable.
    Returns (results, filtered_count) where filtered_count is the number
    of candidates removed by the relative score threshold.
    """
    # Filter-only path: no meaningful query text, just metadata filters
    query_text = query.query.strip()
    if not query_text or query_text == "*":
        if _has_filters(query):
            filter_results = await _filter_only_search(db, query)
            await _record_search_event(
                db, query.query, len(filter_results), None, "filter", contributor
            )
            return filter_results, 0
        # No query and no filters — nothing to search
        return [], 0

    fetch_limit = query.limit * 3  # Over-fetch for re-ranking

    # Always run FTS
    fts_results = await fts_search(
        db,
        query.query,
        limit=fetch_limit,
        project_ref=query.project_ref,
        entry_type=query.entry_type.value if query.entry_type else None,
        tags=query.tags,
        contributor=query.contributor,
        team=query.team,
    )

    # Try vector search
    vec_results: list[tuple[str, float]] = []
    match_source = "fts"
    if embedder is not None:
        vec_results = await vector_search(embedder, query.query, limit=fetch_limit)
        if vec_results:
            match_source = "hybrid"

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}

    for rank, (entry_id, _score) in enumerate(fts_results):
        rrf_scores[entry_id] = rrf_scores.get(entry_id, 0) + 1.0 / (RRF_K + rank + 1)

    for rank, (entry_id, _dist) in enumerate(vec_results):
        rrf_scores[entry_id] = rrf_scores.get(entry_id, 0) + 1.0 / (RRF_K + rank + 1)

    # Sort by combined RRF score (higher = better)
    sorted_ids = sorted(rrf_scores, key=lambda eid: rrf_scores[eid], reverse=True)

    # Apply relative relevance threshold
    pre_filter_count = len(sorted_ids)
    if sorted_ids and query.min_score_ratio > 0:
        top_score = rrf_scores[sorted_ids[0]]
        min_score = top_score * query.min_score_ratio
        sorted_ids = [eid for eid in sorted_ids if rrf_scores[eid] >= min_score]
    filtered_count = pre_filter_count - len(sorted_ids)

    # Build results
    now = datetime.now(UTC)
    results: list[SearchResult] = []
    for entry_id in sorted_ids[: query.limit]:
        entry = await get_entry(db, entry_id)
        if entry is None or not entry.is_active:
            continue

        # Hard filter by project_ref (vector search doesn't scope by project)
        if query.project_ref and entry.project_ref != query.project_ref:
            continue

        # Filter expired entries unless requested
        if not query.include_expired and entry.expires_at is not None:
            exp = entry.expires_at
            if not exp.tzinfo:
                exp = exp.replace(tzinfo=UTC)
            if now >= exp:
                continue

        # Apply confidence decay (reset clock on updates)
        anchor = entry.updated_at or entry.created_at or now
        eff_conf = compute_effective_confidence(
            entry.confidence_level,
            entry.entry_type,
            anchor,
            now,
            last_accessed=entry.last_accessed,
        )

        # Filter stale unless requested
        if not query.include_stale and eff_conf < 0.3:
            continue

        warning = staleness_warning(eff_conf, entry.entry_type)

        results.append(
            SearchResult(
                entry=entry,
                score=rrf_scores[entry_id],
                effective_confidence=eff_conf,
                staleness_warning=warning,
                match_source=match_source if vec_results else "fts",
            )
        )

    # Record search telemetry (fire-and-forget)
    event_top_score: float | None = rrf_scores[sorted_ids[0]] if sorted_ids else None
    await _record_search_event(
        db, query.query, len(results), event_top_score, match_source, contributor
    )

    return results, filtered_count


def _has_filters(query: SearchQuery) -> bool:
    """Check if the query has any metadata filters set."""
    return bool(
        query.project_ref or query.entry_type or query.tags or query.contributor or query.team
    )


async def _filter_only_search(
    db: Database,
    query: SearchQuery,
) -> list[SearchResult]:
    """List entries matching metadata filters without text search.

    Used when no query text is provided — returns entries ordered by
    creation date (newest first).
    """
    sql = "SELECT id FROM knowledge_entries WHERE is_active = 1"
    params: list[str | int] = []

    if query.project_ref:
        sql += " AND project_ref = ?"
        params.append(query.project_ref)
    if query.entry_type:
        sql += " AND entry_type = ?"
        params.append(query.entry_type.value)
    if query.tags:
        for tag in query.tags:
            sql += " AND (' ' || tags || ' ') LIKE ?"
            params.append(f"% {tag} %")
    if query.contributor:
        sql += " AND contributor = ?"
        params.append(query.contributor)
    if query.team:
        sql += " AND team = ?"
        params.append(query.team)

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(query.limit)

    cursor = await db.execute(sql, tuple(params))
    rows = await cursor.fetchall()

    now = datetime.now(UTC)
    results: list[SearchResult] = []
    for row in rows:
        entry = await get_entry(db, row[0])
        if entry is None:
            continue

        # Filter expired entries unless requested
        if not query.include_expired and entry.expires_at is not None:
            exp = entry.expires_at
            if not exp.tzinfo:
                exp = exp.replace(tzinfo=UTC)
            if now >= exp:
                continue

        anchor = entry.updated_at or entry.created_at or now
        eff_conf = compute_effective_confidence(
            entry.confidence_level,
            entry.entry_type,
            anchor,
            now,
            last_accessed=entry.last_accessed,
        )

        if not query.include_stale and eff_conf < 0.3:
            continue

        warning = staleness_warning(eff_conf, entry.entry_type)
        results.append(
            SearchResult(
                entry=entry,
                score=0.0,
                effective_confidence=eff_conf,
                staleness_warning=warning,
                match_source="filter",
            )
        )

    return results


async def _record_search_event(
    db: Database,
    query_text: str,
    result_count: int,
    top_score: float | None,
    match_source: str,
    contributor: str | None = None,
) -> None:
    """Record a search event for telemetry. Never raises."""
    try:
        now = datetime.now(UTC).isoformat()
        await db.execute(
            "INSERT INTO search_events"
            " (query_text, result_count, top_score, match_source, contributor, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (query_text, result_count, top_score, match_source, contributor, now),
        )
        await db.commit()
    except Exception:
        logger.warning("Failed to record search event", exc_info=True)
