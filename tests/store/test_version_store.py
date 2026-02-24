"""Tests for version store."""

import pytest

from personal_kb.models.entry import EntryType
from personal_kb.store.version_store import VersionStore


@pytest.mark.asyncio
async def test_get_versions(db, store):
    entry = await store.create_entry(
        short_title="Versioned",
        long_title="Versioned entry",
        knowledge_details="Version 1",
        entry_type=EntryType.DECISION,
    )
    await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Version 2",
        change_reason="Updated content",
    )

    vs = VersionStore(db)
    versions = await vs.get_versions(entry.id)
    assert len(versions) == 2
    assert versions[0].version_number == 1
    assert versions[0].knowledge_details == "Version 1"
    assert versions[1].version_number == 2
    assert versions[1].knowledge_details == "Version 2"
    assert versions[1].change_reason == "Updated content"


@pytest.mark.asyncio
async def test_get_latest_version(db, store):
    entry = await store.create_entry(
        short_title="Latest",
        long_title="Latest version test",
        knowledge_details="V1",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    await store.update_entry(entry_id=entry.id, knowledge_details="V2")
    await store.update_entry(entry_id=entry.id, knowledge_details="V3")

    vs = VersionStore(db)
    latest = await vs.get_latest_version(entry.id)
    assert latest is not None
    assert latest.version_number == 3
    assert latest.knowledge_details == "V3"


@pytest.mark.asyncio
async def test_get_versions_nonexistent(db):
    vs = VersionStore(db)
    versions = await vs.get_versions("kb-99999")
    assert versions == []
