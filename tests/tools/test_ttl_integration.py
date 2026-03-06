"""Integration tests for TTL/expiry in store, search, and formatters."""

from datetime import UTC, datetime, timedelta

import pytest

from personal_kb.db.queries import get_entry
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.search import SearchQuery
from personal_kb.search.hybrid import hybrid_search
from personal_kb.tools.formatters import format_entry_meta

# --- KnowledgeStore tests ---


@pytest.mark.asyncio
async def test_create_entry_with_expires_at(store):
    """Create an entry with expires_at set."""
    expires = datetime(2025, 12, 31, tzinfo=UTC)
    entry = await store.create_entry(
        short_title="Status update",
        long_title="Weekly project status",
        knowledge_details="Project is on track.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        expires_at=expires,
    )
    assert entry.expires_at == expires


@pytest.mark.asyncio
async def test_create_entry_without_expires_at(store):
    """Create an entry without expires_at — should be None."""
    entry = await store.create_entry(
        short_title="Permanent fact",
        long_title="A fact that doesn't expire",
        knowledge_details="This is permanent.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert entry.expires_at is None


@pytest.mark.asyncio
async def test_update_preserves_existing_expires_at(store):
    """Update without new expires_at should preserve the existing one."""
    expires = datetime(2025, 12, 31, tzinfo=UTC)
    entry = await store.create_entry(
        short_title="Expiring fact",
        long_title="A fact that expires",
        knowledge_details="Details v1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        expires_at=expires,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Details v2",
    )
    assert updated.expires_at == expires


@pytest.mark.asyncio
async def test_update_changes_expires_at(store):
    """Update with new expires_at should change it."""
    old_expires = datetime(2025, 6, 1, tzinfo=UTC)
    new_expires = datetime(2026, 1, 1, tzinfo=UTC)
    entry = await store.create_entry(
        short_title="Expiring fact",
        long_title="A fact that expires",
        knowledge_details="Details v1",
        entry_type=EntryType.FACTUAL_REFERENCE,
        expires_at=old_expires,
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Details v2",
        expires_at=new_expires,
    )
    assert updated.expires_at == new_expires


# --- Round-trip through DB ---


@pytest.mark.asyncio
async def test_expires_at_round_trip(db, store):
    """expires_at should survive insert → select round-trip."""
    expires = datetime(2025, 9, 15, 14, 30, 0, tzinfo=UTC)
    entry = await store.create_entry(
        short_title="RT test",
        long_title="Round-trip test",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        expires_at=expires,
    )
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.expires_at is not None
    assert fetched.expires_at.replace(tzinfo=UTC) == expires


@pytest.mark.asyncio
async def test_expires_at_none_round_trip(db, store):
    """expires_at=None should survive insert → select round-trip."""
    entry = await store.create_entry(
        short_title="No expiry",
        long_title="No expiry test",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    fetched = await get_entry(db, entry.id)
    assert fetched is not None
    assert fetched.expires_at is None


# --- Hybrid search filter ---


@pytest.mark.asyncio
async def test_expired_entry_filtered_from_search(db, store):
    """An expired entry should not appear in search results."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    await store.create_entry(
        short_title="Old status",
        long_title="Expired project status",
        knowledge_details="This status has expired.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["status"],
        expires_at=past,
    )
    query = SearchQuery(query="status expired")
    results, _ = await hybrid_search(db, None, query)
    assert all(r.entry.short_title != "Old status" for r in results)


@pytest.mark.asyncio
async def test_not_yet_expired_entry_returned(db, store):
    """An entry with future expiry should appear in search results."""
    future = datetime.now(UTC) + timedelta(days=30)
    await store.create_entry(
        short_title="Current status",
        long_title="Current project status",
        knowledge_details="This status is still valid.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["status"],
        expires_at=future,
    )
    query = SearchQuery(query="current status")
    results, _ = await hybrid_search(db, None, query)
    assert any(r.entry.short_title == "Current status" for r in results)


@pytest.mark.asyncio
async def test_include_expired_returns_expired(db, store):
    """include_expired=True should return expired entries."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    await store.create_entry(
        short_title="Expired entry",
        long_title="Expired but requested",
        knowledge_details="This entry has expired but include_expired is set.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["expired"],
        expires_at=past,
    )
    query = SearchQuery(query="expired entry", include_expired=True)
    results, _ = await hybrid_search(db, None, query)
    assert any(r.entry.short_title == "Expired entry" for r in results)


@pytest.mark.asyncio
async def test_expired_independent_of_stale(db, store):
    """An entry can be expired but not stale (high confidence, past expiry)."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    await store.create_entry(
        short_title="High conf expired",
        long_title="High confidence but expired",
        knowledge_details="Confident but time-bounded.",
        entry_type=EntryType.LESSON_LEARNED,  # long half-life
        confidence_level=0.99,
        expires_at=past,
    )
    # Without include_expired: filtered out
    query = SearchQuery(query="high confidence expired", include_stale=True)
    results, _ = await hybrid_search(db, None, query)
    assert all(r.entry.short_title != "High conf expired" for r in results)

    # With include_expired: returned
    query2 = SearchQuery(query="high confidence expired", include_expired=True)
    results2, _ = await hybrid_search(db, None, query2)
    assert any(r.entry.short_title == "High conf expired" for r in results2)


# --- Formatter tests ---


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = {
        "id": "kb-00001",
        "short_title": "Test",
        "long_title": "Test entry",
        "knowledge_details": "Details",
        "entry_type": EntryType.FACTUAL_REFERENCE,
        "tags": [],
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def test_expired_badge():
    """Expired entries get [EXPIRED] badge."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    entry = _make_entry(expires_at=past)
    result = format_entry_meta(entry)
    assert "[EXPIRED]" in result


def test_expires_days_badge():
    """Entries expiring in days get [EXPIRES Nd] badge."""
    future = datetime.now(UTC) + timedelta(days=3, hours=5)
    entry = _make_entry(expires_at=future)
    result = format_entry_meta(entry)
    assert "[EXPIRES 3d]" in result


def test_expires_hours_badge():
    """Entries expiring in <1 day get [EXPIRES Nh] badge."""
    future = datetime.now(UTC) + timedelta(hours=5, minutes=30)
    entry = _make_entry(expires_at=future)
    result = format_entry_meta(entry)
    assert "[EXPIRES 5h]" in result


def test_no_expiry_no_badge():
    """Entries without expires_at get no expiry badge."""
    entry = _make_entry(expires_at=None)
    result = format_entry_meta(entry)
    assert "EXPIRE" not in result
    assert "EXPIRED" not in result


def test_expired_with_sensitivity_and_stale():
    """All badges should appear together in order: sensitivity > expiry > staleness."""
    past = datetime(2020, 1, 1, tzinfo=UTC)
    entry = _make_entry(expires_at=past, sensitivity="restricted")
    result = format_entry_meta(entry, stale_warning="Stale")
    assert "[RESTRICTED]" in result
    assert "[EXPIRED]" in result
    assert "[STALE]" in result
    # Order check: RESTRICTED before EXPIRED before STALE
    ri = result.index("[RESTRICTED]")
    ei = result.index("[EXPIRED]")
    si = result.index("[STALE]")
    assert ri < ei < si
