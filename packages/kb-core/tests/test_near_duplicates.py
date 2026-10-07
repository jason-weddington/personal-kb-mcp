"""Tests for the near-duplicate lookup, its facade, and the distinct_from edge."""

from __future__ import annotations

import math
from typing import Any

import pytest

from kb_core import create_sqlite
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.near_duplicates import find_near_duplicates

DIM = 1024


def _vec(theta: float) -> list[float]:
    """Unit vector at cosine similarity cos(theta) to e0."""
    v = [0.0] * DIM
    v[0] = math.cos(theta)
    v[1] = math.sin(theta)
    return v


def _theta(sim: float) -> float:
    return math.acos(sim)


E0 = _vec(0.0)


class _Embedder:
    def __init__(self, vec: list[float] | None) -> None:
        self._vec = vec

    async def embed(self, text: str) -> list[float] | None:
        return self._vec


async def _put(
    kb: Any,
    title: str,
    sim: float,
    *,
    project: str = "p",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
) -> str:
    details = "see kb-00001" if entry_type is EntryType.MENTAL_MAP else f"details {title}"
    entry = await kb.store(
        short_title=title,
        long_title=title,
        knowledge_details=details,
        entry_type=entry_type,
        project_ref=project,
        enrich=False,
    )
    await kb.db.vector_store(entry.id, _vec(_theta(sim)))
    await kb.db.commit()
    return str(entry.id)


@pytest.fixture
async def seeded(tmp_path: Any) -> Any:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=_Embedder(E0))
    ids = {
        "A": await _put(kb, "alpha", 0.95),
        "B": await _put(kb, "bravo", 0.80),
        "C": await _put(kb, "charlie", 0.99, project="q"),
        "D": await _put(kb, "delta", 0.99, entry_type=EntryType.MENTAL_MAP),
        "E": await _put(kb, "echo", 0.99),
        "F": await _put(kb, "foxtrot", 0.99),
    }
    await kb.db.execute("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", (ids["E"],))
    await kb.db.execute(
        "UPDATE knowledge_entries SET superseded_by = 'kb-09999' WHERE id = ?", (ids["F"],)
    )
    await kb.db.commit()
    try:
        yield kb, ids
    finally:
        await kb.close()


def test_compose_embedding_text_matches_property() -> None:
    e = KnowledgeEntry(
        id="kb-00001",
        short_title="s",
        long_title="l",
        knowledge_details="d",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert (
        KnowledgeEntry.compose_embedding_text(e.short_title, e.long_title, e.knowledge_details)
        == e.embedding_text
    )


async def test_floor_filters_and_eligibility(seeded: Any) -> None:
    kb, ids = seeded
    res = await find_near_duplicates(kb.db, E0, project_ref="p", floor=0.88)
    assert res.status == "checked"
    assert [c.id for c in res.candidates] == [ids["A"]]
    assert abs(res.candidates[0].similarity - 0.95) < 1e-4
    assert res.top_similarity is not None and abs(res.top_similarity - 0.95) < 1e-4
    assert res.eligible_count == 2
    assert res.raw_hits == 4


async def test_lower_floor_orders_by_similarity(seeded: Any) -> None:
    kb, ids = seeded
    res = await find_near_duplicates(kb.db, E0, project_ref="p", floor=0.79)
    assert [c.id for c in res.candidates] == [ids["A"], ids["B"]]


async def test_floor_above_everything(seeded: Any) -> None:
    kb, _ = seeded
    res = await find_near_duplicates(kb.db, E0, project_ref="p", floor=0.96)
    assert res.candidates == ()
    assert res.top_similarity is not None and abs(res.top_similarity - 0.95) < 1e-4


async def test_limit(seeded: Any) -> None:
    kb, ids = seeded
    await _put(kb, "golf", 0.93)
    res = await find_near_duplicates(kb.db, E0, project_ref="p", floor=0.5, limit=1)
    assert [c.id for c in res.candidates] == [ids["A"]]


async def test_empty_project(seeded: Any) -> None:
    kb, _ = seeded
    res = await find_near_duplicates(kb.db, E0, project_ref="z", floor=0.5)
    assert res.status == "checked"
    assert res.candidates == ()
    assert res.top_similarity is None
    assert res.raw_hits == 0


async def test_facade_no_embedder(tmp_path: Any) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=None)
    try:
        res = await kb.find_near_duplicates(
            short_title="a", long_title="b", knowledge_details="c", project_ref="p", floor=0.88
        )
        assert res.status == "embedder_unavailable"
        assert res.candidates == ()
    finally:
        await kb.close()


async def test_facade_embedder_returns_none(tmp_path: Any) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=_Embedder(None))
    try:
        res = await kb.find_near_duplicates(
            short_title="a", long_title="b", knowledge_details="c", project_ref="p", floor=0.88
        )
        assert res.status == "embedder_unavailable"
    finally:
        await kb.close()


async def test_facade_search_failed(seeded: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    kb, _ = seeded

    async def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("boom")

    monkeypatch.setattr(kb.db, "vector_search", boom)
    res = await kb.find_near_duplicates(
        short_title="a", long_title="b", knowledge_details="c", project_ref="p", floor=0.88
    )
    assert res.status == "search_failed"
    assert res.candidates == ()
    assert res.top_similarity is None


async def test_facade_checked_sets_timings(seeded: Any) -> None:
    kb, ids = seeded
    res = await kb.find_near_duplicates(
        short_title="a", long_title="b", knowledge_details="c", project_ref="p", floor=0.88
    )
    assert res.status == "checked"
    assert [c.id for c in res.candidates] == [ids["A"]]
    assert res.embed_ms >= 0 and res.search_ms >= 0


async def test_record_audit_event(tmp_path: Any) -> None:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=None)
    try:
        await kb.record_audit_event(
            "near_duplicate_checked", entry_id=None, contributor="x", detail="{}"
        )
        cur = await kb.db.execute(
            "SELECT entry_id, contributor FROM audit_events WHERE event_type = ?",
            ("near_duplicate_checked",),
        )
        row = await cur.fetchone()
        assert row is not None and row["entry_id"] is None and row["contributor"] == "x"
    finally:
        await kb.close()


# ── distinct_from edge ────────────────────────────────────────────────────


async def _edges(kb: Any, source: str) -> list[tuple[str, str]]:
    cur = await kb.db.execute(
        "SELECT target, edge_type FROM graph_edges WHERE source = ? AND edge_type = ?",
        (source, "distinct_from"),
    )
    return [(r["target"], r["edge_type"]) for r in await cur.fetchall()]


async def _store(kb: Any, hints: dict[str, Any] | None, **kw: Any) -> Any:
    return await kb.store(
        short_title="new",
        long_title="new",
        knowledge_details="new details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="p",
        hints=hints,
        enrich=False,
        **kw,
    )


@pytest.fixture
async def kb1(tmp_path: Any) -> Any:
    kb = await create_sqlite(tmp_path / "kb.db", embedder=None)
    await _put(kb, "first", 0.5)
    try:
        yield kb
    finally:
        await kb.close()


async def test_distinct_from_edge_created_and_idempotent(kb1: Any) -> None:
    e = await _store(kb1, {"distinct_from": ["kb-00001"]})
    assert await _edges(kb1, e.id) == [("kb-00001", "distinct_from")]
    from kb_core.graph.builder import GraphBuilder

    await GraphBuilder(kb1.db).build_for_entry(e)
    assert await _edges(kb1, e.id) == [("kb-00001", "distinct_from")]


async def test_distinct_from_invalid_ignored(kb1: Any) -> None:
    e = await _store(kb1, {"distinct_from": ["bogus"]})
    assert await _edges(kb1, e.id) == []


async def test_distinct_from_missing_target_ignored(kb1: Any) -> None:
    e = await _store(kb1, {"distinct_from": ["kb-09999"]})
    assert await _edges(kb1, e.id) == []
    cur = await kb1.db.execute("SELECT 1 FROM graph_nodes WHERE node_id = ?", ("kb-09999",))
    assert await cur.fetchone() is None


async def test_distinct_from_self_ignored(kb1: Any) -> None:
    e = await _store(kb1, None)
    await kb1.update(e.id, hints={"distinct_from": [e.id]}, change_reason="x")
    assert await _edges(kb1, e.id) == []


async def test_related_entities_distinct_from_skipped(kb1: Any) -> None:
    e = await _store(kb1, {"related_entities": [{"id": "kb-00001", "edge_type": "distinct_from"}]})
    assert await _edges(kb1, e.id) == []
