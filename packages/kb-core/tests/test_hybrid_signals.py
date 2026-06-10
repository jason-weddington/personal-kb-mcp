"""Tests for per-leg raw relevance signals on SearchResult.

Acceptance criteria:
- SearchResult gains vector_similarity, fts_matched, fts_rank.
- Existing score (RRF) is byte-identical to before (regression guard).
- Coverage: both-legs result, FTS-only result, vector-only result,
  degraded mode (no embedder), and RRF score regression.
"""

from __future__ import annotations

from typing import Any

from kb_core import create_sqlite
from kb_core.models.entry import EntryType
from kb_core.models.search import SearchQuery
from kb_core.search.hybrid import RRF_K, hybrid_search

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ControlledEmbedder:
    """Embedder that returns a pre-scripted list of (entry_id, distance) pairs.

    Allows tests to control which entries appear in the vector leg and with
    what cosine distance, independently of what the FTS leg returns.
    """

    def __init__(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    async def embed(self, text: str) -> list[float] | None:
        """Return a unit-norm dummy vector (content irrelevant — results are scripted)."""
        # 1-hot on dimension 0, rest zero — already unit-norm.
        vec = [0.0] * 1024
        vec[0] = 1.0
        return vec

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 20,
        *,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """Return the pre-scripted results (respects limit)."""
        return self._results[:limit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _store_entry(kb: Any, short_title: str, details: str) -> Any:
    """Thin helper: store an entry with minimal required fields."""
    return await kb.store(
        short_title=short_title,
        long_title=short_title,
        knowledge_details=details,
        entry_type=EntryType.LESSON_LEARNED,
        enrich=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_both_legs_populated(tmp_path: Any) -> None:
    """When an entry appears in both FTS and vector legs, all three signal fields
    are populated: fts_matched=True, fts_rank>=1, vector_similarity in (0, 1]."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        entry = await _store_entry(kb, "python async patterns", "async/await in Python")

        # Vector leg returns this entry with distance 0.2 → similarity 0.8
        embedder = ControlledEmbedder([(entry.id, 0.2)])

        results, _ = await hybrid_search(
            kb.db, embedder, SearchQuery(query="python async patterns")
        )

        hit = next((r for r in results if r.entry.id == entry.id), None)
        assert hit is not None, "Entry expected in results"
        assert hit.fts_matched is True
        assert hit.fts_rank == 1
        assert hit.vector_similarity is not None
        assert abs(hit.vector_similarity - 0.8) < 1e-9
    finally:
        await kb.close()


async def test_fts_only_result(tmp_path: Any) -> None:
    """An entry appearing only in the FTS leg (vector leg omits it) has
    fts_matched=True, fts_rank set, vector_similarity=None."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        e_both = await _store_entry(kb, "alpha search term", "first alpha entry")
        e_fts_only = await _store_entry(kb, "alpha query target", "second alpha entry")

        # Vector leg returns only e_both — e_fts_only is absent.
        embedder = ControlledEmbedder([(e_both.id, 0.1)])

        # min_score_ratio=0 to prevent the relative-threshold filter from
        # dropping the FTS-only entry (which naturally scores lower than the
        # entry that appears in both legs).
        results, _ = await hybrid_search(
            kb.db, embedder, SearchQuery(query="alpha", limit=10, min_score_ratio=0.0)
        )

        by_id = {r.entry.id: r for r in results}

        # e_both: in both legs
        assert e_both.id in by_id
        r_both = by_id[e_both.id]
        assert r_both.fts_matched is True
        assert r_both.fts_rank is not None
        assert r_both.vector_similarity is not None
        assert abs(r_both.vector_similarity - 0.9) < 1e-9  # 1.0 - 0.1

        # e_fts_only: FTS-only — vector_similarity must be None
        assert e_fts_only.id in by_id
        r_fts = by_id[e_fts_only.id]
        assert r_fts.fts_matched is True
        assert r_fts.fts_rank is not None
        assert r_fts.vector_similarity is None
    finally:
        await kb.close()


async def test_vector_only_result(tmp_path: Any) -> None:
    """An entry appearing only in the vector leg (no FTS match) has
    fts_matched=False, fts_rank=None, vector_similarity populated."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        e_fts = await _store_entry(kb, "alpha term present", "text mentioning alpha for FTS")
        # This entry's text does NOT contain "alpha" so FTS won't match it
        # (but the controlled embedder returns it in the vector leg).
        e_vec_only = await _store_entry(
            kb, "completely different subject", "unrelated knowledge detail here"
        )

        # Vector leg returns both; FTS will only match e_fts for query "alpha".
        # min_score_ratio=0 prevents the relative-threshold filter from dropping
        # the vector-only entry (which scores lower than the hybrid entry).
        embedder = ControlledEmbedder([(e_fts.id, 0.3), (e_vec_only.id, 0.2)])

        results, _ = await hybrid_search(
            kb.db, embedder, SearchQuery(query="alpha", limit=10, min_score_ratio=0.0)
        )

        by_id = {r.entry.id: r for r in results}

        # e_fts: in both legs
        assert e_fts.id in by_id
        r_fts = by_id[e_fts.id]
        assert r_fts.fts_matched is True
        assert r_fts.fts_rank == 1
        assert r_fts.vector_similarity is not None
        assert abs(r_fts.vector_similarity - 0.7) < 1e-9  # 1.0 - 0.3

        # e_vec_only: vector-only — fts_matched must be False
        assert e_vec_only.id in by_id
        r_vec = by_id[e_vec_only.id]
        assert r_vec.fts_matched is False
        assert r_vec.fts_rank is None
        assert r_vec.vector_similarity is not None
        assert abs(r_vec.vector_similarity - 0.8) < 1e-9  # 1.0 - 0.2
    finally:
        await kb.close()


async def test_degraded_mode_no_embedder(tmp_path: Any) -> None:
    """With embedder=None (FTS-only degraded mode), all results have
    vector_similarity=None, fts_matched=True, and fts_rank populated."""
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        await _store_entry(kb, "rrf degraded test", "content for degraded mode test")

        # No embedder → vector leg produces no results
        results, _ = await hybrid_search(kb.db, None, SearchQuery(query="rrf degraded test"))

        assert len(results) >= 1
        for r in results:
            assert r.vector_similarity is None
            assert r.fts_matched is True
            assert r.fts_rank is not None
            assert r.fts_rank >= 1
    finally:
        await kb.close()


async def test_rrf_scores_unchanged_by_new_fields(tmp_path: Any) -> None:
    """Adding per-leg signal fields must not change existing RRF score values.

    Verifies byte-identical RRF arithmetic for a fixed fixture with a
    controlled embedder producing known distances.
    """
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        e1 = await _store_entry(kb, "rrf regression alpha", "alpha content regression")
        e2 = await _store_entry(kb, "rrf regression beta", "beta content regression")

        # Both entries appear in the vector leg at known distances.
        embedder = ControlledEmbedder([(e1.id, 0.1), (e2.id, 0.3)])

        query = SearchQuery(query="rrf regression alpha beta", limit=10)
        results, _ = await hybrid_search(kb.db, embedder, query)

        by_id = {r.entry.id: r for r in results}

        # Manually compute expected RRF scores based on actual ranks.
        # FTS ranks depend on BM25 — both entries contain "rrf regression".
        # Vector ranks are deterministic: e1 rank 0 (distance 0.1), e2 rank 1 (distance 0.3).
        # We can't predict exact FTS rank order, so we verify by reconstructing
        # from the actual fts_rank values in the results.
        for entry_id, r in by_id.items():
            expected = 0.0
            if r.fts_rank is not None:
                expected += 1.0 / (RRF_K + r.fts_rank)
            if r.vector_similarity is not None:
                # vector rank 0 → e1 (distance 0.1), rank 1 → e2 (distance 0.3)
                vec_rank = [e1.id, e2.id].index(entry_id)
                expected += 1.0 / (RRF_K + vec_rank + 1)
            assert abs(r.score - expected) < 1e-12, (
                f"RRF score mismatch for {entry_id}: got {r.score}, expected {expected}"
            )

        # Sanity: results are ordered by descending score.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    finally:
        await kb.close()


async def test_search_result_defaults_backward_compatible() -> None:
    """SearchResult can be constructed without the new fields (backward compat)."""
    from kb_core.models.entry import KnowledgeEntry

    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="t",
        long_title="t",
        knowledge_details="d",
        entry_type=EntryType.LESSON_LEARNED,
        confidence_level=0.9,
        tags=[],
        hints={},
        created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        updated_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        version=1,
    )
    from kb_core.models.search import SearchResult

    result = SearchResult(
        entry=entry,
        score=0.03,
        effective_confidence=0.9,
        match_source="fts",
    )
    assert result.vector_similarity is None
    assert result.fts_matched is False
    assert result.fts_rank is None
