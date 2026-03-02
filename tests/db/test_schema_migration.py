"""Tests for schema migrations — multi-user columns."""

import pytest


@pytest.mark.asyncio
async def test_multi_user_columns_exist(db):
    """After schema init, multi-user columns should exist on all tables."""
    # knowledge_entries
    cursor = await db.execute("PRAGMA table_info(knowledge_entries)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "contributor" in cols
    assert "team" in cols
    assert "updated_by" in cols

    # entry_versions
    cursor = await db.execute("PRAGMA table_info(entry_versions)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "contributor" in cols

    # search_events
    cursor = await db.execute("PRAGMA table_info(search_events)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "contributor" in cols

    # agent_feedback
    cursor = await db.execute("PRAGMA table_info(agent_feedback)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "contributor" in cols

    # ingested_files
    cursor = await db.execute("PRAGMA table_info(ingested_files)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "contributor" in cols


@pytest.mark.asyncio
async def test_deployment_config_table_exists(db):
    """deployment_config table should exist after schema init."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='deployment_config'"
    )
    row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_audit_events_table_exists(db):
    """audit_events table should exist after schema init."""
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_events'"
    )
    row = await cursor.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_sensitivity_column_exists(db):
    """sensitivity column should exist on knowledge_entries after migration."""
    cursor = await db.execute("PRAGMA table_info(knowledge_entries)")
    cols = {row[1] for row in await cursor.fetchall()}
    assert "sensitivity" in cols


@pytest.mark.asyncio
async def test_multi_user_columns_nullable(db):
    """Multi-user columns should accept NULL (existing entries have no attribution)."""
    from personal_kb.models.entry import EntryType
    from personal_kb.store.knowledge_store import KnowledgeStore

    store = KnowledgeStore(db)
    entry = await store.create_entry(
        short_title="No attribution",
        long_title="Entry without contributor",
        knowledge_details="Created without contributor or team.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    assert entry.contributor is None
    assert entry.team is None
    assert entry.updated_by is None
