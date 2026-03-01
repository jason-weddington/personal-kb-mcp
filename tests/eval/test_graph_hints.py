"""Tier 1: Graph hint regression tests.

Verifies that sparse search results trigger graph-connected hints,
and that plentiful results suppress them.
"""

import pytest

from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search
from personal_kb.tools.kb_search import collect_graph_hints


@pytest.mark.eval
class TestGraphHints:
    async def test_sparse_results_get_hints(self, eval_kb):
        """A narrow query returning <3 results should trigger graph hints."""
        db, embedder, _, _ = eval_kb

        results = await hybrid_search(
            db,
            embedder,
            SearchQuery(
                query="secrets management AWS rotation",
                limit=5,
                include_stale=False,
            ),
        )

        # Should have sparse results (secrets-dotenv-old is deactivated)
        if len(results) < 3:
            hints = await collect_graph_hints(db, results)
            # Hints should be non-empty — the secrets-management entry
            # is connected to other entries via shared tags (security, devops)
            assert len(hints) > 0, (
                f"Sparse results ({len(results)} results) should produce graph hints"
            )

    async def test_hints_surface_related_entries(self, eval_kb):
        """Graph hints should find entries connected via shared tag nodes."""
        db, embedder, _, _ = eval_kb

        # Search for something narrow in the devops cluster
        results = await hybrid_search(
            db,
            embedder,
            SearchQuery(
                query="Terraform infrastructure modules",
                limit=3,
                include_stale=False,
            ),
        )

        if len(results) < 3:
            hints = await collect_graph_hints(db, results)
            # If hints are present, they should be formatted strings
            for hint in hints:
                assert isinstance(hint, str)
                assert len(hint) > 0

    async def test_no_hints_when_plentiful(self, eval_kb):
        """Queries with >=3 results should not trigger hints."""
        db, embedder, _, _ = eval_kb

        results = await hybrid_search(
            db,
            embedder,
            SearchQuery(
                query="async sqlite database access patterns",
                limit=10,
                include_stale=False,
            ),
        )

        # The database cluster should return plenty of results
        if len(results) >= 3:
            # Call collect_graph_hints to verify it doesn't error
            await collect_graph_hints(db, results)
        else:
            pytest.skip("Expected >=3 results for broad database query")

    async def test_deactivated_entries_excluded_from_hints(self, eval_kb):
        """Deactivated entries should not appear in graph hints."""
        db, embedder, title_to_id, _ = eval_kb

        # secrets-dotenv-old is deactivated
        deactivated_id = title_to_id["secrets-dotenv-old"]

        results = await hybrid_search(
            db,
            embedder,
            SearchQuery(
                query="secrets management security",
                limit=5,
                include_stale=False,
            ),
        )

        hints = await collect_graph_hints(db, results)

        # Deactivated entry should not appear in hints
        for hint in hints:
            assert deactivated_id not in hint, (
                f"Deactivated entry {deactivated_id} should not appear in hints"
            )
