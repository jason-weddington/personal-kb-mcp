"""SQLite suite for the supersession invariant (``kb_core.supersession``).

Covers: recompute (newest created_at, tie-break, audit row, no-op), the
GraphBuilder integration (retraction, step-5 removal, step-7 skip), the
``queries.update_entry`` lost-update guard, deactivate/reactivate, the
reconcile and its idempotence, and every ``check_supersedes_targets`` rule.
The Postgres parity leg lives in ``test_postgres_backend.py``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from kb_core import create_sqlite
from kb_core.db import queries
from kb_core.models.entry import EntryType
from kb_core.supersession import (
    check_supersedes_targets,
    norm_supersedes,
    recompute_superseded_by,
    reconcile_supersession,
)
from supersession_fixture import (
    EXPECTED_CLEARED_COUNT,
    EXPECTED_EDGES_ADDED,
    EXPECTED_EDGES_ADDED_IDS,
    EXPECTED_SET_COUNT,
    EXPECTED_SUPERSEDED_BY,
    seed_corpus,
    superseded_by_map,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from kb_core.knowledge_base import KnowledgeBase


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[KnowledgeBase]:
    """A hermetic SQLite KB: no embedder, no LLMs."""
    instance = await create_sqlite(
        tmp_path / "kb.db",
        embedder=None,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
    )
    try:
        yield instance
    finally:
        await instance.close()


async def _store(
    kb: KnowledgeBase,
    title: str,
    *,
    hints: dict[str, object] | None = None,
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    details: str | None = None,
) -> str:
    entry = await kb.store(
        short_title=title,
        long_title=f"{title} (long)",
        knowledge_details=details or f"details of {title}",
        entry_type=entry_type,
        project_ref="p",
        hints=hints,
        enrich=False,
    )
    return entry.id


async def _set_created_at(kb: KnowledgeBase, entry_id: str, ts: str) -> None:
    await kb.db.execute("UPDATE knowledge_entries SET created_at = ? WHERE id = ?", (ts, entry_id))
    await kb.db.commit()


async def _superseded_by(kb: KnowledgeBase, entry_id: str) -> str | None:
    cursor = await kb.db.execute(
        "SELECT superseded_by FROM knowledge_entries WHERE id = ?", (entry_id,)
    )
    row = await cursor.fetchone()
    assert row is not None
    value: str | None = row["superseded_by"]
    return value


async def _audit(kb: KnowledgeBase, entry_id: str, event_type: str) -> list[dict[str, Any]]:
    cursor = await kb.db.execute(
        "SELECT detail FROM audit_events WHERE entry_id = ? AND event_type = ? ORDER BY id",
        (entry_id, event_type),
    )
    return [json.loads(r["detail"]) for r in await cursor.fetchall()]


async def _add_edge(kb: KnowledgeBase, source: str, target: str, props: str = "{}") -> None:
    await kb.db.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, 'supersedes', ?, '2026-01-01T00:00:00+00:00')",
        (source, target, props),
    )
    await kb.db.commit()


async def _edges(kb: KnowledgeBase, source: str, edge_type: str) -> list[str]:
    cursor = await kb.db.execute(
        "SELECT target FROM graph_edges WHERE source = ? AND edge_type = ?",
        (source, edge_type),
    )
    return sorted(r["target"] for r in await cursor.fetchall())


# ─── recompute ────────────────────────────────────────────────────────────────


async def test_recompute_picks_newest_created_at_and_leaves_row_metadata(
    kb: KnowledgeBase,
) -> None:
    t = await _store(kb, "T")
    s1 = await _store(kb, "S1", hints={"supersedes": [t]})
    s2 = await _store(kb, "S2", hints={"supersedes": [t]})
    await _set_created_at(kb, s1, "2026-01-01T00:00:00+00:00")
    await _set_created_at(kb, s2, "2026-02-01T00:00:00")  # naive -> UTC
    # Start from a NULL column so the first recompute is a real write.
    await kb.db.execute("UPDATE knowledge_entries SET superseded_by = NULL WHERE id = ?", (t,))
    await kb.db.commit()
    before = await kb.get(t)
    assert before is not None
    cursor = await kb.db.execute("SELECT COUNT(*) FROM entry_versions WHERE entry_id = ?", (t,))
    versions_before = (await cursor.fetchone())[0]

    async with kb.db.transaction():
        counts = await recompute_superseded_by(kb.db, [t], trigger="build")

    after = await kb.get(t)
    assert after is not None
    assert after.superseded_by == s2
    assert after.updated_at == before.updated_at
    assert after.version == before.version
    cursor = await kb.db.execute("SELECT COUNT(*) FROM entry_versions WHERE entry_id = ?", (t,))
    assert (await cursor.fetchone())[0] == versions_before
    assert counts.set_count == 1
    assert counts.cleared_count == 0
    assert counts.changed == ((t, None, s2),)

    # Newest superseder goes inactive -> falls back to S1, one audit row.
    await kb.db.execute("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", (s2,))
    async with kb.db.transaction():
        await recompute_superseded_by(kb.db, [t], trigger="deactivate")
    assert await _superseded_by(kb, t) == s1
    rows = await _audit(kb, t, "superseded_by_changed")
    assert rows[-1] == {"old": s2, "new": s1, "candidates": [s1], "trigger": "deactivate"}
    assert len([r for r in rows if r["trigger"] == "deactivate"]) == 1

    # Editing S1 bumps its updated_at but never its created_at: pointer stays.
    await kb.update(s1, knowledge_details="x", change_reason="edit", enrich=False)
    async with kb.db.transaction():
        await recompute_superseded_by(kb.db, [t], trigger="build")
    assert await _superseded_by(kb, t) == s1


async def test_recompute_no_edges_is_null_and_noop_writes_no_audit(kb: KnowledgeBase) -> None:
    t = await _store(kb, "T")
    counts = await recompute_superseded_by(kb.db, [t], trigger="reconcile")
    assert await _superseded_by(kb, t) is None
    assert counts.changed == ()
    assert await _audit(kb, t, "superseded_by_changed") == []


async def test_recompute_ignores_mental_map_and_llm_edges(kb: KnowledgeBase) -> None:
    t = await _store(kb, "T")
    m = await _store(kb, "M", entry_type=EntryType.MENTAL_MAP, details=f"see {t}")
    s = await _store(kb, "S")
    await _add_edge(kb, m, t)
    await _add_edge(kb, s, t, json.dumps({"source": "llm"}))
    await recompute_superseded_by(kb.db, [t], trigger="reconcile")
    assert await _superseded_by(kb, t) is None


# ─── queries.update_entry lost-update guard ───────────────────────────────────


async def test_update_entry_does_not_overwrite_superseded_by(kb: KnowledgeBase) -> None:
    t = await _store(kb, "T")
    stale = await queries.get_entry(kb.db, t)
    assert stale is not None and stale.superseded_by is None
    await kb.db.execute(
        "UPDATE knowledge_entries SET superseded_by = 'kb-00002' WHERE id = ?", (t,)
    )
    await kb.db.commit()
    await queries.update_entry(kb.db, stale.model_copy(update={"knowledge_details": "x"}))
    assert await _superseded_by(kb, t) == "kb-00002"


# ─── GraphBuilder integration ─────────────────────────────────────────────────


async def test_store_with_supersedes_sets_target_and_rebuild_retracts(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    b = await _store(kb, "B", hints={"supersedes": [a]})
    assert await _superseded_by(kb, a) == b

    entry_b = await kb.get(b)
    assert entry_b is not None
    await kb.graph_builder.build_for_entry(entry_b.model_copy(update={"hints": {"supersedes": []}}))
    assert await _superseded_by(kb, a) is None


async def test_rebuild_writes_no_edge_from_superseded_by(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    entry_a = await kb.get(a)
    assert entry_a is not None
    await kb.graph_builder.build_for_entry(entry_a.model_copy(update={"superseded_by": "kb-00099"}))
    assert await _edges(kb, "kb-00099", "supersedes") == []
    assert await _superseded_by(kb, a) is None


async def test_related_entities_supersedes_is_ignored(
    kb: KnowledgeBase, caplog: pytest.LogCaptureFixture
) -> None:
    a = await _store(kb, "A")
    b = await _store(kb, "B", hints={"related_entities": [{"id": a, "edge_type": "supersedes"}]})
    assert await _edges(kb, b, "supersedes") == []
    assert await _superseded_by(kb, a) is None
    assert "ignoring related_entities supersedes edge" in caplog.text


# ─── deactivate / reactivate ──────────────────────────────────────────────────


async def test_deactivating_superseder_clears_target(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    b = await _store(kb, "B", hints={"supersedes": [a]})
    assert await _superseded_by(kb, a) == b
    await kb.deactivate(b, change_reason="wrong")
    assert await _superseded_by(kb, a) is None
    cursor = await kb.db.execute(
        "SELECT detail FROM audit_events WHERE entry_id = ? AND event_type = 'entry_deactivated'",
        (b,),
    )
    assert (await cursor.fetchone())["detail"] == "wrong"

    # Reactivate + rebuild restores the pointer.
    entry_b = await kb.reactivate(b)
    await kb.graph_builder.build_for_entry(entry_b)
    assert await _superseded_by(kb, a) == b


async def test_deactivate_with_superseded_by(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    b = await _store(kb, "B")
    entry = await kb.deactivate(a, change_reason="replaced", superseded_by=b)
    assert entry.superseded_by == b
    assert await _superseded_by(kb, a) == b
    entry_b = await kb.get(b)
    assert entry_b is not None
    assert a in norm_supersedes(entry_b.hints.get("supersedes"))
    assert entry_b.version == 1
    assert await _audit(kb, b, "supersedes_hint_appended") == [{"added": a, "via": "deactivate"}]

    await kb.update(b, knowledge_details="newer", change_reason="edit", enrich=False)
    assert await _superseded_by(kb, a) == b


async def test_store_deactivate_entry_default_audit_detail_is_short_title(
    kb: KnowledgeBase,
) -> None:
    a = await _store(kb, "Alpha")
    await kb.knowledge_store.deactivate_entry(a)
    cursor = await kb.db.execute(
        "SELECT detail FROM audit_events WHERE entry_id = ? AND event_type = 'entry_deactivated'",
        (a,),
    )
    assert (await cursor.fetchone())["detail"] == "Alpha"


# ─── reconcile ────────────────────────────────────────────────────────────────


async def test_reconcile_fixture_and_idempotence(kb: KnowledgeBase) -> None:
    await seed_corpus(kb.db)
    report = await reconcile_supersession(kb.db)
    assert await superseded_by_map(kb.db) == EXPECTED_SUPERSEDED_BY
    assert report.edges_added == EXPECTED_EDGES_ADDED
    assert report.edges_added_ids == EXPECTED_EDGES_ADDED_IDS
    assert report.set_count == EXPECTED_SET_COUNT
    assert report.cleared_count == EXPECTED_CLEARED_COUNT

    again = await kb.reconcile_supersession()
    assert (again.edges_added, again.set_count, again.cleared_count) == (0, 0, 0)
    assert again.changed == ()


async def test_reconcile_clears_stale_pointer(kb: KnowledgeBase) -> None:
    t = await _store(kb, "T")
    await kb.db.execute(
        "UPDATE knowledge_entries SET superseded_by = 'kb-00777' WHERE id = ?", (t,)
    )
    await kb.db.commit()
    report = await reconcile_supersession(kb.db)
    assert report.cleared_count == 1
    assert report.changed == ((t, "kb-00777", None),)
    assert await _superseded_by(kb, t) is None


# ─── check_supersedes_targets ─────────────────────────────────────────────────


def test_norm_supersedes() -> None:
    assert norm_supersedes(None) == []
    assert norm_supersedes("kb-00001") == ["kb-00001"]
    assert norm_supersedes(["a", 1]) == ["a", 1]


async def test_check_map_writer_rejected(kb: KnowledgeBase) -> None:
    problems = await check_supersedes_targets(
        kb.db, ["kb-00001", "bad"], writer_id=None, writer_entry_type=EntryType.MENTAL_MAP
    )
    assert problems == ["a mental_map cannot supersede entries"]


async def test_check_invalid_id(kb: KnowledgeBase) -> None:
    problems = await check_supersedes_targets(
        kb.db, ["kb-1", 7], writer_id=None, writer_entry_type=EntryType.DECISION
    )
    assert problems == ["kb-1 is not a valid entry id", "7 is not a valid entry id"]


async def test_check_not_found(kb: KnowledgeBase) -> None:
    problems = await check_supersedes_targets(
        kb.db, ["kb-99999"], writer_id=None, writer_entry_type=EntryType.DECISION
    )
    assert problems == ["kb-99999 not found"]


async def test_check_inactive(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    await kb.deactivate(a, change_reason="gone")
    problems = await check_supersedes_targets(
        kb.db, [a], writer_id=None, writer_entry_type=EntryType.DECISION
    )
    assert problems == [f"{a} is inactive"]


async def test_check_self(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    problems = await check_supersedes_targets(
        kb.db, [a], writer_id=a, writer_entry_type=EntryType.DECISION
    )
    assert problems == ["an entry cannot supersede itself"]


async def test_check_map_target(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    m = await _store(kb, "M", entry_type=EntryType.MENTAL_MAP, details=f"see {a}")
    problems = await check_supersedes_targets(
        kb.db, [m], writer_id=None, writer_entry_type=EntryType.DECISION
    )
    assert problems == [f"{m} is a mental_map; maps are deleted, not superseded"]


async def test_check_cycle(kb: KnowledgeBase) -> None:
    a = await _store(kb, "A")
    b = await _store(kb, "B", hints={"supersedes": a})
    problems = await check_supersedes_targets(
        kb.db, [b], writer_id=a, writer_entry_type=EntryType.DECISION
    )
    assert problems == [f"{b} already supersedes {a} (cycle)"]


async def test_check_cross_project_allowed(kb: KnowledgeBase) -> None:
    other = await kb.store(
        short_title="Other",
        long_title="Other project",
        knowledge_details="elsewhere",
        project_ref="other-project",
        enrich=False,
    )
    problems = await kb.check_supersedes(
        [other.id], writer_id=None, writer_entry_type=EntryType.DECISION
    )
    assert problems == []
