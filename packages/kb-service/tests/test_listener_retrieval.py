"""Hermetic unit tests for the listener's detail-match candidate retrieval.

Companion to ``test_listener_routes.py`` (which locks in the full HTTP
response shape) and ``test_listener_decisions.py`` (which locks in the
telemetry sink). This file drives :func:`_retrieve_candidate_maps` and
:func:`_owning_active_maps` (``kb_service.routes.listener_routes``) directly
— no TestClient, no HTTP layer — to pin the GTD bf40d4f1 fix:

  * detail-corpus search (entry_type='mental_map' EXCLUDED, nothing else
    restricted), top N=20 hits considered
  * many-to-many owning-map resolution via the 'references' graph edge
  * deduplication, best-rank aggregation, deterministic id tie-break
  * cap at 5 candidate maps
  * fallback to direct mental_map search when detail-matching yields zero
    candidates

All tests are hermetic — no live Postgres, Ollama, or network. ``FakeKbDb``
(``tests/conftest.py``) fakes the kb-core ``Database`` handle; SQL shape is
asserted directly off ``fake_kb.db.calls`` rather than re-executing it.
"""

from typing import Any

from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchResult

from kb_service.routes.listener_routes import (
    _DETAIL_SEARCH_FETCH_LIMIT,
    _DETAIL_TOP_N,
    _MAP_CANDIDATE_CAP,
    _owning_active_maps,
    _retrieve_candidate_maps,
)
from tests.conftest import FakeKnowledgeBase

# ─── helpers ─────────────────────────────────────────────────────────────────


def _detail_entry(entry_id: str, **kwargs: Any) -> KnowledgeEntry:
    """A non-map (detail) entry — the default EntryType.FACTUAL_REFERENCE."""
    defaults: dict[str, Any] = {
        "id": entry_id,
        "short_title": f"Detail {entry_id}",
        "long_title": f"Detail {entry_id} (long)",
        "knowledge_details": "Some chunky detail with paths and identifiers.",
        "entry_type": EntryType.FACTUAL_REFERENCE,
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _map_entry(entry_id: str, **kwargs: Any) -> KnowledgeEntry:
    """An active mental_map entry."""
    defaults: dict[str, Any] = {
        "id": entry_id,
        "short_title": f"Map {entry_id}",
        "long_title": f"Map {entry_id} (long)",
        "knowledge_details": "Orientation map body.",
        "entry_type": EntryType.MENTAL_MAP,
        "is_active": True,
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def _detail_results(*entries: KnowledgeEntry) -> list[SearchResult]:
    """Wrap KnowledgeEntry objects as SearchResult (detail-search hits)."""
    return [
        SearchResult(
            entry=e,
            score=1.0 / (60 + i + 1),
            effective_confidence=0.9,
            staleness_warning=None,
            match_source="fts",
        )
        for i, e in enumerate(entries)
    ]


def _make_kb(entries: list[KnowledgeEntry]) -> FakeKnowledgeBase:
    """A FakeKnowledgeBase seeded with detail-search results as fake_kb.results."""
    kb = FakeKnowledgeBase(results=_detail_results(*entries), filtered_count=0)
    return kb


# ─── _retrieve_candidate_maps: primary (detail-match) path ─────────────────


async def test_single_detail_resolves_single_map_no_fallback() -> None:
    """One detail hit -> one owning map: succeeds without falling back."""
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = [("kb-10001", "kb-20001")]
    kb.entries["kb-20001"] = _map_entry("kb-20001")

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(
        kb, "rsync raw-pairs"
    )

    assert used_fallback is False
    assert [c.id for c in candidates] == ["kb-20001"]
    # Only the primary detail search ran — no fallback map search.
    assert len(kb.search_calls) == 1


async def test_primary_search_shape_excludes_nothing_but_maps() -> None:
    """The detail search has no entry_type filter and uses the over-fetch limit."""
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = [("kb-10001", "kb-20001")]
    kb.entries["kb-20001"] = _map_entry("kb-20001")

    await _retrieve_candidate_maps(kb, "some query text")

    assert len(kb.search_calls) == 1
    query, _ = kb.search_calls[0]
    assert query.query == "some query text"
    assert query.entry_type is None
    assert query.limit == _DETAIL_SEARCH_FETCH_LIMIT == 50


async def test_dedupe_two_details_same_map_appears_once() -> None:
    """Two detail hits resolving to the SAME map yield that map exactly once."""
    kb = _make_kb([_detail_entry("kb-10001"), _detail_entry("kb-10002")])
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20001"),
        ("kb-10002", "kb-20001"),
    ]
    kb.entries["kb-20001"] = _map_entry("kb-20001")

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is False
    assert [c.id for c in candidates] == ["kb-20001"]


async def test_many_to_many_one_detail_owned_by_two_maps() -> None:
    """One detail hit owned by two different maps: both surface as candidates."""
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20002"),
        ("kb-10001", "kb-20001"),
    ]
    kb.entries["kb-20001"] = _map_entry("kb-20001")
    kb.entries["kb-20002"] = _map_entry("kb-20002")

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is False
    # Both maps tie at best-rank=1 (owned by the same, single detail hit) ->
    # deterministic tie-break by map id ascending.
    assert [c.id for c in candidates] == ["kb-20001", "kb-20002"]


async def test_ranking_uses_best_rank_not_hit_count() -> None:
    """A map's rank is its BEST (smallest) detail rank, not how often it's hit."""
    kb = _make_kb(
        [
            _detail_entry("kb-10001"),  # rank 1 -> map B only
            _detail_entry("kb-10002"),  # rank 2 -> map A only
            _detail_entry("kb-10003"),  # rank 3 -> map A again (doesn't help A)
        ]
    )
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20002"),  # map B: best rank 1
        ("kb-10002", "kb-20001"),  # map A: best rank 2
        ("kb-10003", "kb-20001"),  # map A hit again at rank 3 (no-op on best rank)
    ]
    kb.entries["kb-20001"] = _map_entry("kb-20001")
    kb.entries["kb-20002"] = _map_entry("kb-20002")

    candidates, _, _evidence = await _retrieve_candidate_maps(kb, "text")

    # Map B (rank 1) ranks ahead of map A (rank 2) despite A being hit twice.
    assert [c.id for c in candidates] == ["kb-20002", "kb-20001"]


async def test_tie_break_is_map_id_ascending_regardless_of_edge_order() -> None:
    """Equal best-rank ties break by map id ascending, not insertion order."""
    kb = _make_kb([_detail_entry("kb-10001")])
    # Edge insertion order deliberately reversed vs. expected output order.
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20099"),
        ("kb-10001", "kb-20001"),
        ("kb-10001", "kb-20050"),
    ]
    for mid in ("kb-20099", "kb-20001", "kb-20050"):
        kb.entries[mid] = _map_entry(mid)

    candidates, _, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert [c.id for c in candidates] == ["kb-20001", "kb-20050", "kb-20099"]


async def test_cap_at_five_candidate_maps() -> None:
    """Six distinct resolvable maps are capped to the pinned literal 5."""
    details = [_detail_entry(f"kb-1{i:04d}") for i in range(6)]
    kb = _make_kb(details)
    edges = [(f"kb-1{i:04d}", f"kb-2{i:04d}") for i in range(6)]
    kb.db.rows_for["graph_edges"] = edges
    for i in range(6):
        kb.entries[f"kb-2{i:04d}"] = _map_entry(f"kb-2{i:04d}")

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is False
    assert len(candidates) == _MAP_CANDIDATE_CAP == 5
    # The lowest-ranked (rank 1..5) maps win the cap, in rank order.
    assert [c.id for c in candidates] == [f"kb-2{i:04d}" for i in range(5)]


async def test_top_n_20_detail_hits_only() -> None:
    """Only the top N=20 detail hits are considered for owning-map resolution.

    A 21st-ranked detail hit's edge must NOT contribute a candidate map, even
    though it would otherwise resolve cleanly.
    """
    details = [_detail_entry(f"kb-1{i:04d}") for i in range(25)]
    kb = _make_kb(details)
    # The 21st hit (0-indexed 20, rank 21) points at a map no other detail
    # reaches.
    beyond_n_detail_id = "kb-10020"
    beyond_n_map_id = "kb-29999"
    kb.db.rows_for["graph_edges"] = [(beyond_n_detail_id, beyond_n_map_id)]
    kb.entries[beyond_n_map_id] = _map_entry(beyond_n_map_id)

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    # Zero candidate maps resolve (the only edge is beyond top-20) -> fallback.
    assert used_fallback is True
    assert beyond_n_map_id not in [c.id for c in candidates]

    # The reverse-join query itself was called with at most 20 detail ids.
    graph_calls = [c for c in kb.db.calls if "graph_edges" in c[0]]
    assert len(graph_calls) == 1
    _, params = graph_calls[0]
    assert len(params) == _DETAIL_TOP_N == 20


# ─── fallback path ───────────────────────────────────────────────────────────


async def test_fallback_when_no_detail_hits_at_all() -> None:
    """Zero detail hits -> straight to the fallback direct mental_map search."""
    kb = _make_kb([])  # empty search results

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is True
    assert candidates == []
    assert len(kb.search_calls) == 2
    fallback_query, _ = kb.search_calls[1]
    assert fallback_query.entry_type == EntryType.MENTAL_MAP
    assert fallback_query.limit == 5


async def test_fallback_when_detail_hits_have_no_owning_edges() -> None:
    """Detail hits exist but no graph edge resolves them -> falls back.

    Covers the two active maps (per GTD bf40d4f1 evidence) with zero
    outbound 'references' edges: unreachable by detail-matching, reachable
    only via this fallback.
    """
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = []  # no owning maps at all

    fallback_map = _map_entry("kb-20001")
    # The fallback direct search returns the fallback map via the SAME
    # fake `results` attribute (FakeKnowledgeBase.search ignores query
    # params) -- reassign it here to model the second call's response.
    kb.results = _detail_results(fallback_map)

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is True
    assert [c.id for c in candidates] == ["kb-20001"]
    assert len(kb.search_calls) == 2


async def test_fallback_when_resolved_map_id_is_stale_or_missing() -> None:
    """Defense-in-depth: a graph-edge map id that no longer resolves via kb.get()
    (deleted, or a race with deactivation) contributes nothing -> falls back."""
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = [("kb-10001", "kb-20001")]
    # Deliberately do NOT register kb-20001 in kb.entries -> kb.get() -> None.

    _candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is True
    # Fallback search reuses fake_kb.results (still the original detail hit,
    # not a mental_map) -- the point of this test is that we DID fall back,
    # not what the fallback returned.
    assert len(kb.search_calls) == 2


# ─── _owning_active_maps query shape ────────────────────────────────────────


async def test_owning_active_maps_empty_input_short_circuits() -> None:
    """No detail ids -> no DB call at all."""
    kb = _make_kb([])
    result = await _owning_active_maps(kb.db, [])
    assert result == []
    assert kb.db.calls == []


# ─── counter plumbing (GTD 268e2af3) ────────────────────────────────────────
#
# The listener route derives its `n_retrieved` telemetry counter as
# `len(candidates)` straight off this function's return value (see
# `listener_routes.py::listener`) — captured BEFORE rule A / rule B ever
# touch `candidates`. These tests pin that len(...) contract at the
# retrieval-function boundary: whatever this function returns IS the
# pre-rule-A pool the route records into `listener_decisions.candidate_ids`.


async def test_retrieved_candidate_count_feeds_n_retrieved_contract() -> None:
    """Multiple resolved candidates: len(candidates) is what n_retrieved reads.

    Mirrors the many-to-many case (one detail hit owned by two maps) but
    exists specifically to pin the count the route treats as n_retrieved —
    a regression here would silently under/over-count per-stage attrition.
    """
    kb = _make_kb([_detail_entry("kb-10001")])
    kb.db.rows_for["graph_edges"] = [
        ("kb-10001", "kb-20001"),
        ("kb-10001", "kb-20002"),
    ]
    kb.entries["kb-20001"] = _map_entry("kb-20001")
    kb.entries["kb-20002"] = _map_entry("kb-20002")

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is False
    n_retrieved = len(candidates)
    assert n_retrieved == 2
    # The route captures candidate ids from exactly this list, pre-rule-A.
    assert [c.id for c in candidates] == ["kb-20001", "kb-20002"]


async def test_fallback_path_candidate_count_also_feeds_n_retrieved_contract() -> None:
    """Same len(...) contract holds on the fallback retrieval path."""
    kb = _make_kb([])  # no detail hits -> fallback
    fallback_maps = [_map_entry("kb-30001"), _map_entry("kb-30002")]
    kb.results = _detail_results(*fallback_maps)

    candidates, used_fallback, _evidence = await _retrieve_candidate_maps(kb, "text")

    assert used_fallback is True
    assert len(candidates) == 2


async def test_owning_active_maps_query_shape_and_params() -> None:
    """The reverse-join query filters edge_type/entry_type/is_active and binds
    detail ids as parameters (portable '?' placeholders, no f-string data)."""
    kb = _make_kb([])
    kb.db.rows_for["graph_edges"] = [("kb-10001", "kb-20001")]

    result = await _owning_active_maps(kb.db, ["kb-10001", "kb-10002"])

    assert result == [("kb-10001", "kb-20001")]
    assert len(kb.db.calls) == 1
    sql, params = kb.db.calls[0]
    assert "graph_edges" in sql
    assert "edge_type = 'references'" in sql
    assert "entry_type = 'mental_map'" in sql
    assert "is_active = 1" in sql
    assert "?" in sql
    assert params == ["kb-10001", "kb-10002"]
