"""Tests for the kb_store MCP tool logic."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.tools.kb_store import format_store_result


@pytest.mark.asyncio
async def test_format_store_result_create(store):
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="my-project",
        tags=["tag1", "tag2"],
    )
    result = format_store_result(entry, is_update=False)
    assert "Created entry kb-00001" in result
    assert "my-project" in result
    assert "tag1, tag2" in result


@pytest.mark.asyncio
async def test_format_store_result_update(store):
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="New details",
        change_reason="Updated",
    )
    result = format_store_result(updated, is_update=True)
    assert "Updated entry kb-00001 (v2)" in result


@pytest.mark.asyncio
async def test_deactivate_entry(store):
    """Deactivate removes entry from active set."""
    entry = await store.create_entry(
        short_title="Wrong fact",
        long_title="An incorrect fact",
        knowledge_details="This is wrong",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    deactivated = await store.deactivate_entry(entry.id)
    assert deactivated.is_active is False

    reloaded = await store.get_entry(entry.id)
    assert reloaded is not None
    assert reloaded.is_active is False


@pytest.mark.asyncio
async def test_deactivate_nonexistent_entry(store):
    """Deactivating a nonexistent entry raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        await store.deactivate_entry("kb-99999")


@pytest.mark.asyncio
async def test_deactivate_already_inactive(store):
    """Deactivating an already-inactive entry raises ValueError."""
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)
    with pytest.raises(ValueError, match="already inactive"):
        await store.deactivate_entry(entry.id)
