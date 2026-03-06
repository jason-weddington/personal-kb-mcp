"""Tests for metadata-only updates (no knowledge_details rewrite)."""

import pytest

from personal_kb.db.queries import get_entry
from personal_kb.models.entry import EntryType


@pytest.mark.asyncio
async def test_metadata_only_update_tags(db, store):
    """Update tags without rewriting knowledge_details."""
    entry = await store.create_entry(
        short_title="Original",
        long_title="Original entry",
        knowledge_details="Important details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["old-tag"],
    )
    await store.mark_embedding(entry.id, True)

    updated = await store.update_entry(
        entry_id=entry.id,
        tags=["new-tag", "added"],
    )
    assert updated.knowledge_details == "Important details"
    assert updated.tags == ["new-tag", "added"]
    assert updated.version == 2
    # Embedding should NOT be reset since content didn't change
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.has_embedding is True


@pytest.mark.asyncio
async def test_metadata_only_update_project_ref(store):
    """Update project_ref without rewriting knowledge_details."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        project_ref="new-project",
    )
    assert updated.project_ref == "new-project"
    assert updated.knowledge_details == "Details"


@pytest.mark.asyncio
async def test_metadata_only_update_sensitivity(store):
    """Update sensitivity without rewriting knowledge_details."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Secret stuff",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        sensitivity="restricted",
    )
    assert updated.sensitivity == "restricted"
    assert updated.knowledge_details == "Secret stuff"


@pytest.mark.asyncio
async def test_metadata_only_update_entry_type(store):
    """Update entry_type without rewriting knowledge_details."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        entry_type=EntryType.DECISION,
    )
    assert updated.entry_type == EntryType.DECISION
    assert updated.knowledge_details == "Details"


@pytest.mark.asyncio
async def test_content_update_resets_embedding(db, store):
    """When knowledge_details IS provided, embedding flag should reset."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Original details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.mark_embedding(entry.id, True)

    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="New details",
    )
    assert updated.knowledge_details == "New details"
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.has_embedding is False


@pytest.mark.asyncio
async def test_metadata_update_creates_version(store):
    """Even metadata-only updates create a version record."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        tags=["new"],
        change_reason="Retagged",
    )
    assert updated.version == 2


@pytest.mark.asyncio
async def test_update_inactive_entry_raises(store):
    """Updating an inactive entry raises ValueError."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.deactivate_entry(entry.id)
    with pytest.raises(ValueError, match="inactive"):
        await store.update_entry(entry_id=entry.id, tags=["nope"])


@pytest.mark.asyncio
async def test_multiple_metadata_fields_at_once(store):
    """Update multiple metadata fields in one call."""
    entry = await store.create_entry(
        short_title="Entry",
        long_title="An entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["old"],
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        tags=["new"],
        project_ref="my-project",
        sensitivity="internal",
        confidence_level=0.5,
    )
    assert updated.tags == ["new"]
    assert updated.project_ref == "my-project"
    assert updated.sensitivity == "internal"
    assert updated.confidence_level == 0.5
    assert updated.knowledge_details == "Details"
