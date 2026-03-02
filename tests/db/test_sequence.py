"""Tests for atomic entry ID generation via next_sequence_value."""

import pytest

from personal_kb.db.queries import next_entry_id


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
