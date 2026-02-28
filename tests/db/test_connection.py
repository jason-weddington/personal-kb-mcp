"""Tests for database connection and schema initialization."""

import pytest

from personal_kb.db.connection import create_connection


@pytest.mark.asyncio
async def test_create_in_memory_connection():
    db = await create_connection(":memory:")
    try:
        # Verify tables exist
        cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in await cursor.fetchall()}
        assert "knowledge_entries" in tables
        assert "entry_versions" in tables
        assert "entry_id_seq" in tables
        assert "schema_version" in tables
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_fts_table_created():
    db = await create_connection(":memory:")
    try:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_fts'"
        )
        row = await cursor.fetchone()
        assert row is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_vec_table_created():
    db = await create_connection(":memory:")
    try:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='knowledge_vec'"
        )
        row = await cursor.fetchone()
        assert row is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_schema_version():
    db = await create_connection(":memory:")
    try:
        cursor = await db.execute("SELECT version FROM schema_version")
        row = await cursor.fetchone()
        assert row[0] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_entry_id_sequence():
    db = await create_connection(":memory:")
    try:
        cursor = await db.execute("SELECT next_id FROM entry_id_seq")
        row = await cursor.fetchone()
        assert row[0] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_last_accessed_column_exists():
    """The last_accessed column should be present after schema migration."""
    db = await create_connection(":memory:")
    try:
        cursor = await db.execute("PRAGMA table_info(knowledge_entries)")
        columns = {row[1] for row in await cursor.fetchall()}
        assert "last_accessed" in columns
    finally:
        await db.close()
