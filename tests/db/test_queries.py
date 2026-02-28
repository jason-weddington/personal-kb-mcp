"""Tests for database query helpers."""

import asyncio

import pytest

from personal_kb.db.queries import get_entry, touch_accessed
from personal_kb.models.entry import EntryType


@pytest.mark.asyncio
async def test_touch_accessed_sets_timestamp(db, store):
    """touch_accessed should set last_accessed for given entries."""
    await store.create_entry(
        short_title="Entry A",
        long_title="Entry A",
        knowledge_details="Details A",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.create_entry(
        short_title="Entry B",
        long_title="Entry B",
        knowledge_details="Details B",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Both should have NULL last_accessed initially
    a = await get_entry(db, "kb-00001")
    b = await get_entry(db, "kb-00002")
    assert a is not None and a.last_accessed is None
    assert b is not None and b.last_accessed is None

    # Touch only entry A
    await touch_accessed(db, ["kb-00001"])

    a = await get_entry(db, "kb-00001")
    b = await get_entry(db, "kb-00002")
    assert a is not None and a.last_accessed is not None
    assert b is not None and b.last_accessed is None


@pytest.mark.asyncio
async def test_touch_accessed_empty_list(db):
    """touch_accessed with empty list should be a no-op."""
    await touch_accessed(db, [])


@pytest.mark.asyncio
async def test_touch_accessed_updates_existing(db, store):
    """Calling touch_accessed again should update the timestamp."""
    await store.create_entry(
        short_title="Entry",
        long_title="Entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    await touch_accessed(db, ["kb-00001"])
    entry1 = await get_entry(db, "kb-00001")
    assert entry1 is not None and entry1.last_accessed is not None
    first_access = entry1.last_accessed

    # Small delay to ensure timestamp differs
    await asyncio.sleep(0.01)

    await touch_accessed(db, ["kb-00001"])
    entry2 = await get_entry(db, "kb-00001")
    assert entry2 is not None and entry2.last_accessed is not None
    assert entry2.last_accessed >= first_access
