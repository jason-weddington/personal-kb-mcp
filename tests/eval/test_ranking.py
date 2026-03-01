"""Tier 1: Search ranking regression tests.

Parametrized over queries.json — each query asserts expected entries
appear in top-k and excluded entries are absent.
"""

import json
from pathlib import Path

import pytest

from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search

_QUERIES = json.loads((Path(__file__).parent / "queries.json").read_text())

# Only run queries that have expected entries (skip decay/sparse-only queries)
_RANKING_QUERIES = [q for q in _QUERIES if q.get("expected")]
_EXCLUSION_QUERIES = [q for q in _QUERIES if q.get("excluded")]


@pytest.mark.eval
class TestExpectedInTopK:
    """Each expected entry must appear in the top-k results."""

    @pytest.mark.parametrize(
        "query_def",
        _RANKING_QUERIES,
        ids=[q["id"] for q in _RANKING_QUERIES],
    )
    async def test_expected_in_top_k(self, eval_kb, query_def):
        db, embedder, title_to_id, _ = eval_kb

        search_query = SearchQuery(
            query=query_def["query"],
            limit=query_def["top_k"],
            project_ref=query_def.get("project_ref"),
            tags=query_def.get("tags"),
            include_stale=query_def.get("include_stale", False),
        )

        results = await hybrid_search(db, embedder, search_query)
        result_ids = [r.entry.id for r in results]

        for expected_title in query_def["expected"]:
            expected_id = title_to_id[expected_title]
            assert expected_id in result_ids, (
                f"Query '{query_def['id']}': expected '{expected_title}' ({expected_id}) "
                f"in top {query_def['top_k']}, got {result_ids}"
            )


@pytest.mark.eval
class TestExcludedAbsent:
    """Excluded entries must not appear in results."""

    @pytest.mark.parametrize(
        "query_def",
        _EXCLUSION_QUERIES,
        ids=[q["id"] for q in _EXCLUSION_QUERIES],
    )
    async def test_excluded_absent(self, eval_kb, query_def):
        db, embedder, title_to_id, _ = eval_kb

        search_query = SearchQuery(
            query=query_def["query"],
            limit=query_def["top_k"],
            project_ref=query_def.get("project_ref"),
            tags=query_def.get("tags"),
            include_stale=query_def.get("include_stale", False),
        )

        results = await hybrid_search(db, embedder, search_query)
        result_ids = [r.entry.id for r in results]

        for excluded_title in query_def["excluded"]:
            excluded_id = title_to_id.get(excluded_title)
            if excluded_id is None:
                # Entry was deactivated and might not appear — skip
                continue
            assert excluded_id not in result_ids, (
                f"Query '{query_def['id']}': '{excluded_title}' ({excluded_id}) "
                f"should NOT appear in results, got {result_ids}"
            )
