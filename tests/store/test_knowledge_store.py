"""Tests for the knowledge store."""

import pytest

from personal_kb.models.entry import EntryType


@pytest.mark.asyncio
async def test_create_entry(store):
    entry = await store.create_entry(
        short_title="Test entry",
        long_title="A test knowledge entry",
        knowledge_details="This is a test entry with some details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="test-project",
        tags=["test", "example"],
    )
    assert entry.id == "kb-00001"
    assert entry.short_title == "Test entry"
    assert entry.entry_type == EntryType.FACTUAL_REFERENCE
    assert entry.version == 1
    assert entry.is_active is True
    assert entry.tags == ["test", "example"]


@pytest.mark.asyncio
async def test_create_multiple_entries(store):
    e1 = await store.create_entry(
        short_title="First",
        long_title="First entry",
        knowledge_details="Details 1",
        entry_type=EntryType.DECISION,
    )
    e2 = await store.create_entry(
        short_title="Second",
        long_title="Second entry",
        knowledge_details="Details 2",
        entry_type=EntryType.LESSON_LEARNED,
    )
    assert e1.id == "kb-00001"
    assert e2.id == "kb-00002"


@pytest.mark.asyncio
async def test_update_entry(store):
    entry = await store.create_entry(
        short_title="Original",
        long_title="Original entry",
        knowledge_details="Original details",
        entry_type=EntryType.DECISION,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Updated details with new info",
        change_reason="Added new information",
        confidence_level=0.95,
    )
    assert updated.version == 2
    assert updated.knowledge_details == "Updated details with new info"
    assert updated.confidence_level == 0.95


@pytest.mark.asyncio
async def test_update_nonexistent_entry(store):
    with pytest.raises(ValueError, match="not found"):
        await store.update_entry(
            entry_id="kb-99999",
            knowledge_details="Won't work",
        )


@pytest.mark.asyncio
async def test_update_preserves_tags(store):
    entry = await store.create_entry(
        short_title="Tagged",
        long_title="Tagged entry",
        knowledge_details="Some details",
        entry_type=EntryType.PATTERN_CONVENTION,
        tags=["python", "patterns"],
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Updated details",
    )
    assert updated.tags == ["python", "patterns"]


@pytest.mark.asyncio
async def test_update_merges_hints(store):
    entry = await store.create_entry(
        short_title="Hinted",
        long_title="Entry with hints",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
        hints={"supersedes": "kb-00000", "context": "original"},
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="New details",
        hints={"new_key": "new_value"},
    )
    assert updated.hints["supersedes"] == "kb-00000"
    assert updated.hints["new_key"] == "new_value"


@pytest.mark.asyncio
async def test_get_entry(store):
    created = await store.create_entry(
        short_title="Fetchable",
        long_title="Fetchable entry",
        knowledge_details="Fetch me",
        entry_type=EntryType.LESSON_LEARNED,
    )
    fetched = await store.get_entry(created.id)
    assert fetched is not None
    assert fetched.id == created.id
    assert fetched.short_title == "Fetchable"


@pytest.mark.asyncio
async def test_get_nonexistent_entry(store):
    result = await store.get_entry("kb-99999")
    assert result is None


@pytest.mark.asyncio
async def test_has_embedding_flag(store):
    entry = await store.create_entry(
        short_title="Embed test",
        long_title="Embedding test entry",
        knowledge_details="Test",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert entry.has_embedding is False

    await store.mark_embedding(entry.id, True)
    fetched = await store.get_entry(entry.id)
    assert fetched.has_embedding is True

    # Update resets has_embedding
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Updated",
    )
    assert updated.has_embedding is False
