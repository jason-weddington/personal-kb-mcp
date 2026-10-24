"""Ranking must not depend on the requested limit: limit=5 is a prefix of limit=10.

limit=10 is the largest limit whose per-leg pool (max(limit * 3, RRF_CANDIDATE_POOL)) is
still the constant RRF_CANDIDATE_POOL; above it the pool grows with the limit. Dropped rows
(stale, expired, filtered) are not backfilled, so limit=5 may return fewer than 5 ids, but
never different ones."""

from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search


async def test_small_limit_is_prefix_of_larger_limit(eval_kb):
    db, embedder, _title_to_id, queries = eval_kb
    for q in queries:
        kwargs = {
            "query": q["query"],
            "project_ref": q.get("project_ref"),
            "tags": q.get("tags"),
            "include_stale": q.get("include_stale", False),
        }
        small, _ = await hybrid_search(db, embedder, SearchQuery(limit=5, **kwargs))
        large, _ = await hybrid_search(db, embedder, SearchQuery(limit=10, **kwargs))
        small_ids = [r.entry.id for r in small]
        assert [r.entry.id for r in large][: len(small_ids)] == small_ids, q["id"]
