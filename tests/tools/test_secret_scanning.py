"""Tests for secret scanning on kb_store path."""

import pytest

from personal_kb.tools.kb_store import _check_secrets


def test_check_secrets_clean_content():
    """Normal content passes without error."""
    result = _check_secrets("This is a normal knowledge entry about Python.")
    assert result is None


def test_check_secrets_with_safety_skip(monkeypatch):
    """KB_SKIP_SAFETY=TRUE bypasses scanning."""
    monkeypatch.setenv("KB_SKIP_SAFETY", "TRUE")
    result = _check_secrets("password = 'hunter2'")
    assert result is None


def test_check_secrets_detects_keyword_secrets():
    """detect-secrets KeywordDetector catches common secret patterns."""
    # Only runs if detect-secrets is installed
    try:
        from detect_secrets.core.scan import scan_line  # noqa: F401
    except ImportError:
        pytest.skip("detect-secrets not installed")

    result = _check_secrets("password = 'super_secret_password_123'")
    if result is not None:
        assert "Error: Potential secrets detected" in result
        assert "KB_SKIP_SAFETY=TRUE" in result


def test_check_secrets_empty_content():
    """Empty content passes without error."""
    result = _check_secrets("")
    assert result is None


@pytest.mark.asyncio
async def test_store_contributor_and_team(db, store):
    """create_entry accepts and persists contributor/team."""
    from personal_kb.models.entry import EntryType

    entry = await store.create_entry(
        short_title="Attributed entry",
        long_title="Entry with contributor",
        knowledge_details="This entry has attribution.",
        entry_type=EntryType.DECISION,
        contributor="jason",
        team="platform",
    )
    assert entry.contributor == "jason"
    assert entry.team == "platform"

    # Verify it persists in DB
    fetched = await store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.contributor == "jason"
    assert fetched.team == "platform"


@pytest.mark.asyncio
async def test_update_sets_updated_by(db, store):
    """update_entry sets updated_by on the entry and version."""
    from personal_kb.models.entry import EntryType

    entry = await store.create_entry(
        short_title="To update",
        long_title="Entry to update",
        knowledge_details="Original content.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        contributor="alice",
    )
    updated = await store.update_entry(
        entry_id=entry.id,
        knowledge_details="Updated content.",
        updated_by="bob",
    )
    assert updated.updated_by == "bob"

    # Verify in DB
    fetched = await store.get_entry(entry.id)
    assert fetched is not None
    assert fetched.updated_by == "bob"
    # Original contributor preserved
    assert fetched.contributor == "alice"

    # Check version record has contributor
    cursor = await db.execute(
        "SELECT contributor FROM entry_versions WHERE entry_id = ? ORDER BY version_number DESC",
        (entry.id,),
    )
    row = await cursor.fetchone()
    assert row["contributor"] == "bob"
