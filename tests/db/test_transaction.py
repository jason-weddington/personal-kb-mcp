"""Tests for transaction() context manager on both backends."""

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.db.queries import (
    delete_entry_cascade,
    get_entry,
    insert_entry,
    insert_version,
)
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.version import EntryVersion
from personal_kb.store.knowledge_store import KnowledgeStore


@pytest_asyncio.fixture
async def db():
    """In-memory database with full schema."""
    conn = await create_connection(":memory:")
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def store(db):
    return KnowledgeStore(db)


def _make_entry(entry_id: str = "kb-00001", **kwargs) -> KnowledgeEntry:
    now = datetime.now(UTC)
    defaults = {
        "id": entry_id,
        "short_title": "Test",
        "long_title": "Test entry",
        "knowledge_details": "Details",
        "entry_type": EntryType.FACTUAL_REFERENCE,
        "confidence_level": 0.9,
        "tags": [],
        "hints": {},
        "created_at": now,
        "updated_at": now,
        "version": 1,
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


class TestSQLiteTransaction:
    """Test transaction() on SQLiteBackend."""

    @pytest.mark.asyncio
    async def test_commit_on_success(self, db):
        """Transaction commits all operations atomically."""
        async with db.transaction():
            entry = _make_entry()
            await insert_entry(db, entry)
            version = EntryVersion(
                entry_id="kb-00001",
                version_number=1,
                knowledge_details="Details",
                change_reason="Init",
                confidence_level=0.9,
                created_at=datetime.now(UTC),
            )
            await insert_version(db, version)

        # Both should exist after transaction commits
        result = await get_entry(db, "kb-00001")
        assert result is not None
        assert result.short_title == "Test"

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM entry_versions WHERE entry_id = ?",
            ("kb-00001",),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 1

    @pytest.mark.asyncio
    async def test_rollback_on_exception(self, db):
        """Transaction rolls back all operations on exception."""
        with pytest.raises(ValueError, match="boom"):
            async with db.transaction():
                entry = _make_entry()
                await insert_entry(db, entry)
                raise ValueError("boom")

        # Entry should NOT exist
        result = await get_entry(db, "kb-00001")
        assert result is None

    @pytest.mark.asyncio
    async def test_nested_savepoint_rollback(self, db):
        """Inner transaction rolls back via savepoint without affecting outer."""
        async with db.transaction():
            entry1 = _make_entry("kb-00001", short_title="Outer")
            await insert_entry(db, entry1)

            # Inner transaction fails — should not affect outer
            with pytest.raises(ValueError, match="inner fail"):
                async with db.transaction():
                    entry2 = _make_entry("kb-00002", short_title="Inner")
                    await insert_entry(db, entry2)
                    raise ValueError("inner fail")

        # Outer entry committed, inner rolled back
        assert (await get_entry(db, "kb-00001")) is not None
        assert (await get_entry(db, "kb-00002")) is None

    @pytest.mark.asyncio
    async def test_intermediate_commits_suppressed(self, db):
        """db.commit() inside transaction() is a no-op."""
        with pytest.raises(ValueError, match="after commit"):
            async with db.transaction():
                entry = _make_entry()
                await insert_entry(db, entry)
                # insert_entry calls db.commit() internally — should be suppressed
                await db.commit()  # explicit call also suppressed
                raise ValueError("after commit")

        # Everything rolled back despite intermediate commit calls
        assert (await get_entry(db, "kb-00001")) is None

    @pytest.mark.asyncio
    async def test_code_outside_transaction_still_autocommits(self, db):
        """Outside transaction(), commit() works normally."""
        entry = _make_entry()
        await insert_entry(db, entry)
        # insert_entry calls db.commit() — should work
        assert (await get_entry(db, "kb-00001")) is not None


class TestKnowledgeStoreTransaction:
    """Test that KnowledgeStore methods are transactional."""

    @pytest.mark.asyncio
    async def test_create_entry_is_atomic(self, store, db):
        """create_entry produces entry + version + audit in one transaction."""
        entry = await store.create_entry(
            short_title="Atomic",
            long_title="Atomic test",
            knowledge_details="Details",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )

        # Verify all artifacts exist
        assert (await get_entry(db, entry.id)) is not None

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM entry_versions WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 1

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM audit_events WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 1

    @pytest.mark.asyncio
    async def test_update_entry_is_atomic(self, store, db):
        """update_entry produces update + version + audit in one transaction."""
        entry = await store.create_entry(
            short_title="V1",
            long_title="Version 1",
            knowledge_details="Original",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )
        updated = await store.update_entry(
            entry.id,
            knowledge_details="Updated",
            change_reason="Test update",
        )
        assert updated.version == 2

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM entry_versions WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 2  # v1 + v2

    @pytest.mark.asyncio
    async def test_deactivate_entry_is_atomic(self, store, db):
        """deactivate_entry updates status + audit atomically."""
        entry = await store.create_entry(
            short_title="To deactivate",
            long_title="Will be deactivated",
            knowledge_details="Content",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )
        result = await store.deactivate_entry(entry.id)
        assert not result.is_active

        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM audit_events WHERE entry_id = ? AND event_type = ?",
            (entry.id, "entry_deactivated"),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 1

    @pytest.mark.asyncio
    async def test_delete_cascade_is_atomic(self, store, db):
        """delete_entry_cascade removes all related data atomically."""
        entry = await store.create_entry(
            short_title="To delete",
            long_title="Will be deleted",
            knowledge_details="Content",
            entry_type=EntryType.FACTUAL_REFERENCE,
        )
        await delete_entry_cascade(db, entry.id)

        # Everything gone
        assert (await get_entry(db, entry.id)) is None
        cursor = await db.execute(
            "SELECT COUNT(*) as cnt FROM entry_versions WHERE entry_id = ?",
            (entry.id,),
        )
        row = await cursor.fetchone()
        assert row["cnt"] == 0


class TestPostgresTransactionUnit:
    """Unit tests for PostgresBackend transaction routing — no real DB needed."""

    @pytest.mark.asyncio
    async def test_txn_conn_is_none_outside_transaction(self):
        """ContextVar should be None when not inside a transaction."""
        from personal_kb.db.postgres_backend import _txn_conn

        assert _txn_conn.get(None) is None

    @pytest.mark.asyncio
    async def test_conn_helper_yields_txn_conn_when_set(self):
        """_conn() yields the transaction connection when the ContextVar is set."""
        from unittest.mock import AsyncMock

        from personal_kb.db.postgres_backend import PostgresBackend, _txn_conn

        mock_pool = AsyncMock()
        backend = PostgresBackend(mock_pool)

        sentinel_conn = object()
        token = _txn_conn.set(sentinel_conn)
        try:
            async with backend._conn() as conn:
                assert conn is sentinel_conn
            # Pool should NOT have been acquired
            mock_pool.acquire.assert_not_called()
        finally:
            _txn_conn.reset(token)
