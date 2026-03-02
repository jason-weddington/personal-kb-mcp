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
async def test_deactivate_entry(store):
    entry = await store.create_entry(
        short_title="To deactivate",
        long_title="Entry to deactivate",
        knowledge_details="Will be deactivated",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    deactivated = await store.deactivate_entry(entry.id)
    assert deactivated.is_active is False

    fetched = await store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.is_active is False


@pytest.mark.asyncio
async def test_deactivate_nonexistent_raises(store):
    with pytest.raises(ValueError, match="not found"):
        await store.deactivate_entry("kb-99999")


@pytest.mark.asyncio
async def test_reactivate_entry(store):
    entry = await store.create_entry(
        short_title="To reactivate",
        long_title="Entry to reactivate",
        knowledge_details="Will be reactivated",
        entry_type=EntryType.DECISION,
    )
    await store.deactivate_entry(entry.id)
    reactivated = await store.reactivate_entry(entry.id)
    assert reactivated.is_active is True


@pytest.mark.asyncio
async def test_reactivate_active_entry_raises(store):
    entry = await store.create_entry(
        short_title="Already active",
        long_title="Already active entry",
        knowledge_details="Already active",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    with pytest.raises(ValueError, match="already active"):
        await store.reactivate_entry(entry.id)


@pytest.mark.asyncio
async def test_create_entry_with_sensitivity(store):
    """Sensitivity field should persist through create/read."""
    entry = await store.create_entry(
        short_title="Sensitive",
        long_title="Sensitive entry",
        knowledge_details="Restricted content",
        entry_type=EntryType.FACTUAL_REFERENCE,
        sensitivity="restricted",
    )
    assert entry.sensitivity == "restricted"

    fetched = await store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.sensitivity == "restricted"


@pytest.mark.asyncio
async def test_update_entry_with_sensitivity(store):
    """Sensitivity can be set on update."""
    entry = await store.create_entry(
        short_title="Public",
        long_title="Public entry",
        knowledge_details="Public content",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert entry.sensitivity is None

    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Now restricted",
        sensitivity="internal",
    )
    assert updated.sensitivity == "internal"

    fetched = await store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.sensitivity == "internal"


@pytest.mark.asyncio
async def test_audit_events_on_create(db, store):
    """Creating an entry should produce an audit event."""
    await store.create_entry(
        short_title="Audit test",
        long_title="Audit test entry",
        knowledge_details="Testing audit",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="jason",
    )

    cursor = await db.execute("SELECT * FROM audit_events WHERE event_type = 'entry_created'")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["entry_id"] == "kb-00001"
    assert rows[0]["contributor"] == "jason"


@pytest.mark.asyncio
async def test_audit_events_on_deactivate(db, store):
    """Deactivating an entry should produce an audit event."""
    entry = await store.create_entry(
        short_title="To deactivate",
        long_title="Deactivation audit test",
        knowledge_details="Will be deactivated",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)

    cursor = await db.execute("SELECT * FROM audit_events WHERE event_type = 'entry_deactivated'")
    rows = await cursor.fetchall()
    assert len(rows) == 1
    assert rows[0]["entry_id"] == entry.id


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
