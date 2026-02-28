"""Tests for the kb_get MCP tool."""

import pytest

from personal_kb.db.queries import get_entry, touch_accessed
from personal_kb.models.entry import EntryType
from personal_kb.tools.formatters import format_entry_full, format_result_list
from personal_kb.tools.kb_get import _MAX_IDS


async def _kb_get_logic(db, ids: list[str]) -> str:
    """Replicate kb_get logic for testing without MCP context."""
    if len(ids) > _MAX_IDS:
        return f"Error: Maximum {_MAX_IDS} IDs per request (got {len(ids)})."

    formatted: list[str] = []
    accessed_ids: list[str] = []
    for eid in ids:
        entry = await get_entry(db, eid)
        if entry is None or not entry.is_active:
            formatted.append(f"[{eid}] not found")
        else:
            formatted.append(format_entry_full(entry))
            accessed_ids.append(eid)

    if accessed_ids:
        await touch_accessed(db, accessed_ids)

    return format_result_list(formatted)


@pytest.mark.asyncio
async def test_get_single_entry(db, store):
    """Retrieve a single entry by ID."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Full details here",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
        project_ref="my-proj",
    )

    result = await _kb_get_logic(db, [entry.id])
    assert entry.id in result
    assert "Full details here" in result
    assert "#python" in result
    assert "my-proj" in result


@pytest.mark.asyncio
async def test_get_multiple_entries(db, store):
    """Retrieve multiple entries at once."""
    e1 = await store.create_entry(
        short_title="First",
        long_title="First entry",
        knowledge_details="First details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    e2 = await store.create_entry(
        short_title="Second",
        long_title="Second entry",
        knowledge_details="Second details",
        entry_type=EntryType.DECISION,
    )

    result = await _kb_get_logic(db, [e1.id, e2.id])
    assert e1.id in result
    assert e2.id in result
    assert "First details" in result
    assert "Second details" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_missing_entry(db):
    """Missing IDs show 'not found'."""
    result = await _kb_get_logic(db, ["kb-99999"])
    assert "kb-99999" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_get_mixed_found_and_missing(db, store):
    """Mix of found and missing entries."""
    entry = await store.create_entry(
        short_title="Exists",
        long_title="Existing entry",
        knowledge_details="Real content",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    result = await _kb_get_logic(db, [entry.id, "kb-99999"])
    assert entry.id in result
    assert "Real content" in result
    assert "kb-99999" in result
    assert "not found" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_inactive_entry_skipped(db, store):
    """Inactive entries are treated as not found."""
    entry = await store.create_entry(
        short_title="Soon gone",
        long_title="Will be deactivated",
        knowledge_details="Should not appear after deactivation",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    result = await _kb_get_logic(db, [entry.id])
    assert "not found" in result
    assert "Should not appear" not in result


@pytest.mark.asyncio
async def test_get_cap_at_20(db):
    """Exceeding 20 IDs returns an error."""
    ids = [f"kb-{i:05d}" for i in range(1, 22)]
    result = await _kb_get_logic(db, ids)
    assert "Maximum 20" in result


@pytest.mark.asyncio
async def test_get_updates_last_accessed(db, store):
    """kb_get should update last_accessed — explicit retrieval resets decay clock."""
    entry = await store.create_entry(
        short_title="Decay test",
        long_title="Access-aware decay",
        knowledge_details="Explicit retrieval should reset the decay clock.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    # Initially NULL
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is None

    # kb_get should touch last_accessed
    await _kb_get_logic(db, [entry.id])

    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is not None


@pytest.mark.asyncio
async def test_get_does_not_touch_missing_entries(db, store):
    """kb_get should not touch last_accessed for missing/inactive entries."""
    entry = await store.create_entry(
        short_title="Will deactivate",
        long_title="Inactive entry",
        knowledge_details="Should not get accessed.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    await _kb_get_logic(db, [entry.id, "kb-99999"])

    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.last_accessed is None
