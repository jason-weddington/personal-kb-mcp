"""Tier 1: Abstention hard gate.

Asserts that queries with category=='abstention', expected==[], and
include_stale=False return an EMPTY result set from hybrid_search.

This is a hard correctness gate — an over-retrieval failure on a no-answer
query (returning a deactivated or decayed entry) is scored as 0 here.
The scored metric (correct_rejection_rate) also appears in test_baseline.py
as a separate aggregate key.
"""

import json
from pathlib import Path

import pytest

from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search
from tests.eval.metrics import correct_rejection_rate

_QUERIES = json.loads((Path(__file__).parent / "queries.json").read_text())

# Abstention queries: category=='abstention', expected==[], include_stale=False
# q6x-stale-flag-recovers is EXCLUDED from this gate (it has expected != [] and
# include_stale=True — it is a recall control, not an abstention assertion).
_ABSTENTION_QUERIES = [
    q
    for q in _QUERIES
    if q.get("category") == "abstention"
    and q.get("expected") == []
    and not q.get("include_stale", False)
]


@pytest.mark.eval
class TestAbstention:
    """Hard gate: abstention queries must return empty result sets."""

    @pytest.mark.parametrize(
        "query_def",
        _ABSTENTION_QUERIES,
        ids=[q["id"] for q in _ABSTENTION_QUERIES],
    )
    async def test_abstention_returns_empty(self, eval_kb, query_def):
        """Abstention query must return no results (hard gate)."""
        db, embedder, _title_to_id, _ = eval_kb

        search_query = SearchQuery(
            query=query_def["query"],
            limit=query_def["top_k"],
            project_ref=query_def.get("project_ref"),
            tags=query_def.get("tags"),
            include_stale=query_def.get("include_stale", False),
        )

        results, _filtered = await hybrid_search(db, embedder, search_query)
        result_ids = [r.entry.id for r in results]

        assert result_ids == [], (
            f"Query '{query_def['id']}': expected empty result set "
            f"(abstention query), got {result_ids}"
        )

    async def test_correct_rejection_rate_is_one(self, eval_kb):
        """Aggregate correct_rejection_rate over all abstention queries must be 1.0."""
        db, embedder, _title_to_id, queries = eval_kb

        results_map: dict[str, list[str]] = {}
        for q in _ABSTENTION_QUERIES:
            search_query = SearchQuery(
                query=q["query"],
                limit=q["top_k"],
                project_ref=q.get("project_ref"),
                tags=q.get("tags"),
                include_stale=q.get("include_stale", False),
            )
            results, _filtered = await hybrid_search(db, embedder, search_query)
            results_map[q["id"]] = [r.entry.id for r in results]

        rate = correct_rejection_rate(queries, results_map)
        assert rate == 1.0, (
            f"correct_rejection_rate expected 1.0, got {rate}. Results map: {results_map}"
        )
