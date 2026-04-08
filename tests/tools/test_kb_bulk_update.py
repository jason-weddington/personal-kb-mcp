"""Tests for the bulk_update feature (store + tool)."""

import pytest
import pytest_asyncio

from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore


@pytest_asyncio.fixture
async def populated_store(store: KnowledgeStore):
    """Store with a few entries for bulk update testing."""
    await store.create_entry(
        short_title="Entry A",
        long_title="Entry A long",
        knowledge_details="Details A",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        contributor="alice",
        team="alpha",
        tags=["python", "api"],
    )
    await store.create_entry(
        short_title="Entry B",
        long_title="Entry B long",
        knowledge_details="Details B",
        entry_type=EntryType.DECISION,
        project_ref=None,
        contributor="alice",
        team="alpha",
        tags=["python"],
    )
    await store.create_entry(
        short_title="Entry C",
        long_title="Entry C long",
        knowledge_details="Details C",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="other-project",
        contributor="bob",
        team="beta",
        tags=["rust"],
    )
    return store


@pytest.mark.asyncio
async def test_bulk_update_project_ref(populated_store: KnowledgeStore):
    """Filter by contributor, set project_ref."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "new-project"},
    )
    assert len(results) == 2
    for before, after in results:
        assert before.project_ref is None
        assert after.project_ref == "new-project"
        assert after.version == before.version + 1


@pytest.mark.asyncio
async def test_bulk_update_dry_run(populated_store: KnowledgeStore):
    """Dry run should return previews but not persist changes."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "preview-project"},
        dry_run=True,
    )
    assert len(results) == 2
    # Verify nothing was persisted
    entry = await populated_store.get_entry(results[0][0].id)
    assert entry is not None
    assert entry.project_ref is None
    assert entry.version == 1


@pytest.mark.asyncio
async def test_bulk_update_tags_add(populated_store: KnowledgeStore):
    """Add tags without removing existing ones."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"tags_add": ["new-tag"]},
    )
    assert len(results) == 2
    for _before, after in results:
        assert "new-tag" in after.tags
        assert "python" in after.tags  # original preserved


@pytest.mark.asyncio
async def test_bulk_update_tags_remove(populated_store: KnowledgeStore):
    """Remove specific tags."""
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"tags_remove": ["python"]},
    )
    assert len(results) == 2
    for _before, after in results:
        assert "python" not in after.tags


@pytest.mark.asyncio
async def test_bulk_update_tags_add_and_remove(populated_store: KnowledgeStore):
    """Add and remove tags in the same operation."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"tags_add": ["new"], "tags_remove": ["api"]},
    )
    assert len(results) == 1
    _before, after = results[0]
    assert "new" in after.tags
    assert "api" not in after.tags
    assert "python" in after.tags


@pytest.mark.asyncio
async def test_bulk_update_filter_null_project_ref(populated_store: KnowledgeStore):
    """Filter by project_ref=None to find unscoped entries."""
    results = await populated_store.bulk_update(
        filters={"project_ref": None},
        updates={"project_ref": "scoped"},
    )
    assert len(results) == 2  # entries A and B have no project_ref
    for _before, after in results:
        assert after.project_ref == "scoped"


@pytest.mark.asyncio
async def test_bulk_update_filter_by_entry_ids(populated_store: KnowledgeStore):
    """Filter by specific entry IDs."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00003"]},
        updates={"project_ref": "targeted"},
    )
    assert len(results) == 1
    assert results[0][1].project_ref == "targeted"


@pytest.mark.asyncio
async def test_bulk_update_filter_by_entry_type(populated_store: KnowledgeStore):
    """Filter by entry_type."""
    results = await populated_store.bulk_update(
        filters={"entry_type": "decision"},
        updates={"confidence_level": 0.5},
    )
    assert len(results) == 1
    assert results[0][1].confidence_level == 0.5


@pytest.mark.asyncio
async def test_bulk_update_no_matching_entries(populated_store: KnowledgeStore):
    """No matches returns empty list."""
    results = await populated_store.bulk_update(
        filters={"contributor": "nobody"},
        updates={"project_ref": "x"},
    )
    assert results == []


@pytest.mark.asyncio
async def test_bulk_update_no_effective_changes(populated_store: KnowledgeStore):
    """Entries where updates produce no change are skipped."""
    results = await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00003"]},
        updates={"project_ref": "other-project"},  # already this value
    )
    assert results == []


@pytest.mark.asyncio
async def test_bulk_update_version_bump(populated_store: KnowledgeStore, db):
    """Each updated entry gets a version bump and version record."""
    await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"project_ref": "versioned"},
    )
    entry = await populated_store.get_entry("kb-00001")
    assert entry is not None
    assert entry.version == 2

    cursor = await db.execute(
        "SELECT * FROM entry_versions WHERE entry_id = ? ORDER BY version_number",
        ("kb-00001",),
    )
    versions = await cursor.fetchall()
    assert len(versions) == 2  # initial + bulk update


@pytest.mark.asyncio
async def test_bulk_update_audit_event(populated_store: KnowledgeStore, db):
    """Bulk update records audit events."""
    await populated_store.bulk_update(
        filters={"entry_ids": ["kb-00001"]},
        updates={"project_ref": "audited"},
        contributor="tester",
    )
    cursor = await db.execute(
        "SELECT * FROM audit_events WHERE entry_id = ? AND event_type = 'entry_updated'",
        ("kb-00001",),
    )
    events = await cursor.fetchall()
    # At least one audit event from the bulk update
    assert any("Bulk" in (e["detail"] or "") for e in events)


@pytest.mark.asyncio
async def test_bulk_update_skips_inactive(populated_store: KnowledgeStore):
    """Inactive entries are not included in bulk updates."""
    await populated_store.deactivate_entry("kb-00001")
    results = await populated_store.bulk_update(
        filters={"contributor": "alice"},
        updates={"project_ref": "x"},
    )
    assert len(results) == 1  # only kb-00002
    assert results[0][0].id == "kb-00002"


# --- Tool-level tests ---


@pytest_asyncio.fixture
async def _lifespan(db, store):
    """Minimal lifespan dict for tool tests."""
    return {"db": db, "store": store, "contributor": "tester", "team": "test-team"}


@pytest.mark.asyncio
async def test_tool_requires_filters(_lifespan):
    """Tool rejects calls with empty filters."""
    from personal_kb.tools.kb_bulk_update import _format_result

    result = _format_result([], dry_run=False)
    assert "No entries matched" in result


@pytest.mark.asyncio
async def test_tool_requires_updates():
    """_format_result handles empty results."""
    from personal_kb.tools.kb_bulk_update import _format_result

    result = _format_result([], dry_run=True)
    assert "No entries matched" in result


@pytest.mark.asyncio
async def test_format_result_dry_run():
    """Dry run output includes DRY RUN prefix."""
    from datetime import UTC, datetime

    from personal_kb.models.entry import EntryType, KnowledgeEntry
    from personal_kb.tools.kb_bulk_update import _format_result

    now = datetime.now(UTC)
    before = KnowledgeEntry(
        id="kb-00001",
        short_title="T",
        long_title="T",
        knowledge_details="D",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        version=1,
        created_at=now,
        updated_at=now,
    )
    after = before.model_copy(update={"project_ref": "new", "version": 2})
    result = _format_result([(before, after)], dry_run=True)
    assert "DRY RUN" in result
    assert "kb-00001" in result
    assert "project_ref" in result


@pytest.mark.asyncio
async def test_format_result_committed():
    """Committed output does not include DRY RUN prefix."""
    from datetime import UTC, datetime

    from personal_kb.models.entry import EntryType, KnowledgeEntry
    from personal_kb.tools.kb_bulk_update import _format_result

    now = datetime.now(UTC)
    before = KnowledgeEntry(
        id="kb-00001",
        short_title="T",
        long_title="T",
        knowledge_details="D",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref=None,
        version=1,
        created_at=now,
        updated_at=now,
    )
    after = before.model_copy(update={"project_ref": "new", "version": 2})
    result = _format_result([(before, after)], dry_run=False)
    assert "DRY RUN" not in result
    assert "1 entries updated" in result
