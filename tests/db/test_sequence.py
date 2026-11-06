"""Tests for atomic entry ID generation via next_sequence_value."""

import sqlite3

import pytest

from personal_kb.db.queries import next_entry_id
from personal_kb.models.entry import EntryType


@pytest.mark.asyncio
async def test_next_sequence_value_returns_int(db):
    """next_sequence_value returns the current value and increments."""
    val = await db.next_sequence_value()
    assert val == 1
    val2 = await db.next_sequence_value()
    assert val2 == 2


@pytest.mark.asyncio
async def test_next_entry_id_uses_sequence(db):
    """next_entry_id delegates to next_sequence_value."""
    eid = await next_entry_id(db)
    assert eid == "kb-00001"
    eid2 = await next_entry_id(db)
    assert eid2 == "kb-00002"


@pytest.mark.asyncio
async def test_sequence_survives_commit(db):
    """Sequence value persists across commits."""
    await db.next_sequence_value()
    await db.commit()
    val = await db.next_sequence_value()
    assert val == 2


# ---------------------------------------------------------------------------
# Regression suite: the allocator must never hand out an id that already
# exists in knowledge_entries (kb-05000/kb-05001 explicit-id rows wedged every
# store once the bare counter reached 5000 — see kb-05002).
# ---------------------------------------------------------------------------

# 7 NOT-NULL-no-default columns; every other column has a default. Written as
# adjacent literals so no physical line exceeds the 100-column limit.
_SEED_ENTRY_SQL = (
    "INSERT INTO knowledge_entries (id, short_title, long_title, knowledge_details,"
    " entry_type, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)"
)

_TS = "2026-01-01T00:00:00+00:00"


@pytest.mark.asyncio
async def test_allocation_skips_existing_ids(db, store, caplog):
    """A counter seeded behind explicit-id rows allocates past them."""
    for eid in ("kb-05000", "kb-05001"):
        await db.execute(
            _SEED_ENTRY_SQL,
            (eid, eid, eid, "seed body", "factual_reference", _TS, _TS),
        )
    await db.execute("UPDATE entry_id_seq SET next_id = 5000")
    # Legacy transaction control: the seed MUST be committed before the first
    # create_entry, otherwise the failed create's ROLLBACK discards it and the
    # wedge is not reproduced.
    await db.commit()

    with caplog.at_level("WARNING"):
        first = await store.create_entry(
            short_title="collision seed 1",
            long_title="collision seed 1",
            knowledge_details="collision seed body",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )
    assert first.id == "kb-05002"

    second = await store.create_entry(
        short_title="collision seed 2",
        long_title="collision seed 2",
        knowledge_details="collision seed body",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert second.id == "kb-05003"

    assert (
        "entry-id-alloc tripwire=counter_behind backend=sqlite counter=5000"
        " floor=5002 allocated=5002 behind=2" in caplog.text
    )


@pytest.mark.asyncio
async def test_allocation_self_heals_after_rolled_back_allocation(db):
    """A lost allocation self-heals because the id is derived from the table."""
    with pytest.raises(RuntimeError, match="simulated"):
        async with db.transaction():
            assert await next_entry_id(db) == "kb-00001"
            raise RuntimeError("simulated")

    # Simulate an explicit-id import landing after the lost allocation.
    await db.execute(
        _SEED_ENTRY_SQL,
        (
            "kb-00001",
            "self-heal seed",
            "self-heal seed",
            "self-heal seed",
            "factual_reference",
            _TS,
            _TS,
        ),
    )
    await db.commit()

    assert await next_entry_id(db) == "kb-00002"


@pytest.mark.asyncio
async def test_allocation_ignores_non_kb_ids(db, caplog):
    """Non-``kb-<1-18 digits>`` ids neither shift the counter nor raise."""
    for eid in ("e1", "not-an-entry", "kb-0a123", "kb-99999999999999999999"):
        await db.execute(
            _SEED_ENTRY_SQL,
            (eid, eid, eid, "seed body", "factual_reference", _TS, _TS),
        )
    await db.execute("UPDATE entry_id_seq SET next_id = 1")
    await db.commit()

    with caplog.at_level("WARNING"):
        assert await next_entry_id(db) == "kb-00001"
        assert await next_entry_id(db) == "kb-00002"

    assert "entry-id-alloc" not in caplog.text


@pytest.mark.asyncio
async def test_allocation_counts_soft_deleted_ids(db):
    """The max scan must not filter ``is_active``: a deleted row keeps its id."""
    await db.execute(
        _SEED_ENTRY_SQL,
        ("kb-05000", "deleted seed", "deleted seed", "seed body", "factual_reference", _TS, _TS),
    )
    await db.execute("UPDATE knowledge_entries SET is_active = 0 WHERE id = ?", ("kb-05000",))
    await db.execute("UPDATE entry_id_seq SET next_id = 5000")
    await db.commit()

    assert await next_entry_id(db) == "kb-05001"


@pytest.mark.asyncio
async def test_allocation_uses_max_of_counter_and_table(db):
    """The team-KB shape: counter 122, explicit rows up to kb-05002."""
    for eid in ("kb-05000", "kb-05001", "kb-05002"):
        await db.execute(
            _SEED_ENTRY_SQL,
            (eid, eid, eid, "seed body", "factual_reference", _TS, _TS),
        )
    await db.execute("UPDATE entry_id_seq SET next_id = 122")
    await db.commit()

    assert await next_entry_id(db) == "kb-05003"


@pytest.mark.asyncio
async def test_allocation_raises_on_empty_sequence_table(db):
    """Zero rows updated means zero RETURNING rows — surface that branch."""
    await db.execute("DELETE FROM entry_id_seq")
    await db.commit()

    with pytest.raises(RuntimeError, match="entry_id_seq table is empty"):
        await next_entry_id(db)


@pytest.mark.asyncio
async def test_create_entry_logs_insert_failed_marker(db, store, monkeypatch, caplog):
    """A failed insert is logged with the entry-id-alloc marker and re-raised."""
    await db.execute(
        _SEED_ENTRY_SQL,
        (
            "kb-00001",
            "forced collision",
            "forced collision",
            "forced body",
            "factual_reference",
            _TS,
            _TS,
        ),
    )
    await db.commit()

    async def _stub(_db):
        return "kb-00001"

    monkeypatch.setattr("kb_core.store.knowledge_store.next_entry_id", _stub)

    with caplog.at_level("ERROR"), pytest.raises(sqlite3.IntegrityError):
        await store.create_entry(
            short_title="forced collision",
            long_title="forced collision",
            knowledge_details="forced collision body",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )

    assert "entry-id-alloc op=insert_failed id=kb-00001 error=IntegrityError" in caplog.text

    cursor = await db.execute("SELECT COUNT(*) FROM knowledge_entries")
    count_row = await cursor.fetchone()
    assert count_row[0] == 1
