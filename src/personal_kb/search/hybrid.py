"""Reciprocal Rank Fusion (RRF) hybrid ranking."""

import logging
from datetime import UTC, datetime

import aiosqlite

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.db.queries import get_entry
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.search.fts import fts_search
from personal_kb.search.vector import vector_search

logger = logging.getLogger(__name__)

# RRF constant — standard value from the literature
RRF_K = 60


async def hybrid_search(
    db: aiosqlite.Connection,
    embedder: EmbeddingClient | None,
    query: SearchQuery,
) -> list[SearchResult]:
    """Execute hybrid search combining FTS5 and vector similarity via RRF.

    Falls back to FTS-only when embeddings are unavailable.
    """
    fetch_limit = query.limit * 3  # Over-fetch for re-ranking

    # Always run FTS
    fts_results = await fts_search(
        db,
        query.query,
        limit=fetch_limit,
        project_ref=query.project_ref,
        entry_type=query.entry_type.value if query.entry_type else None,
        tags=query.tags,
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

    # Build results
    now = datetime.now(UTC)
    results: list[SearchResult] = []
    for entry_id in sorted_ids[: query.limit]:
        entry = await get_entry(db, entry_id)
        if entry is None:
            continue

        # Apply confidence decay
        created = entry.created_at or now
        eff_conf = compute_effective_confidence(
            entry.confidence_level, entry.entry_type, created, now
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

    return results
