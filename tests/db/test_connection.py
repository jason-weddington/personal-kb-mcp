"""Tests for database connection and schema initialization."""

import ssl
from unittest.mock import AsyncMock, patch

import pytest

from personal_kb.db.connection import _create_postgres, create_connection


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


# -- IAM auth dispatch tests --


@pytest.mark.asyncio
async def test_create_postgres_with_iam_auth(monkeypatch):
    """When KB_PG_IAM_AUTH=TRUE, password and ssl are passed to PostgresBackend.create()."""
    monkeypatch.setenv("KB_PG_IAM_AUTH", "TRUE")
    monkeypatch.setenv("KB_PG_REGION", "eu-west-1")

    mock_db = AsyncMock()
    mock_db.apply_schema = AsyncMock()

    captured_kwargs = {}

    async def fake_create(url, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_db

    with (
        patch(
            "personal_kb.db.postgres_backend.PostgresBackend.create",
            side_effect=fake_create,
        ),
        patch("personal_kb.db.connection._check_deployment_config", new_callable=AsyncMock),
    ):
        await _create_postgres(
            "postgresql://myuser@aurora.cluster-xxx.us-east-1.rds.amazonaws.com:5432/kb"
        )

    assert "password" in captured_kwargs
    assert callable(captured_kwargs["password"])
    assert "ssl" in captured_kwargs
    assert isinstance(captured_kwargs["ssl"], ssl.SSLContext)


@pytest.mark.asyncio
async def test_create_postgres_without_iam_auth(monkeypatch):
    """When KB_PG_IAM_AUTH is not set, no password or ssl kwargs are passed."""
    monkeypatch.delenv("KB_PG_IAM_AUTH", raising=False)

    mock_db = AsyncMock()
    mock_db.apply_schema = AsyncMock()

    captured_kwargs = {}

    async def fake_create(url, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_db

    with (
        patch(
            "personal_kb.db.postgres_backend.PostgresBackend.create",
            side_effect=fake_create,
        ),
        patch("personal_kb.db.connection._check_deployment_config", new_callable=AsyncMock),
    ):
        await _create_postgres("postgresql://myuser@myhost:5432/mydb")

    assert "password" not in captured_kwargs
    assert "ssl" not in captured_kwargs
