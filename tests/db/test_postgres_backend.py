"""Unit tests for PostgresBackend wrappers — no database needed."""

import ssl
from unittest.mock import AsyncMock, patch

import pytest

from personal_kb.db.postgres_backend import (
    PostgresBackend,
    PostgresCursor,
    PostgresRow,
    _translate_placeholders,
)


class TestTranslatePlaceholders:
    """Test ? → $N placeholder translation."""

    def test_no_placeholders(self):
        assert _translate_placeholders("SELECT 1") == "SELECT 1"

    def test_single_placeholder(self):
        assert _translate_placeholders("SELECT * FROM t WHERE id = ?") == (
            "SELECT * FROM t WHERE id = $1"
        )

    def test_multiple_placeholders(self):
        sql = "INSERT INTO t (a, b, c) VALUES (?, ?, ?)"
        assert _translate_placeholders(sql) == "INSERT INTO t (a, b, c) VALUES ($1, $2, $3)"

    def test_placeholders_in_where(self):
        sql = "UPDATE t SET x = ? WHERE id = ? AND name = ?"
        assert _translate_placeholders(sql) == "UPDATE t SET x = $1 WHERE id = $2 AND name = $3"

    def test_question_mark_in_string_literal_still_replaced(self):
        # This is a known limitation — real SQL shouldn't have ? in string literals
        # since parameterized queries use placeholders, not inline values
        sql = "SELECT '?' FROM t WHERE x = ?"
        result = _translate_placeholders(sql)
        assert result == "SELECT '$1' FROM t WHERE x = $2"

    def test_empty_sql(self):
        assert _translate_placeholders("") == ""


class TestPostgresRow:
    """Test PostgresRow wrapper over asyncpg.Record-like objects."""

    def _make_row(self, data: dict):
        """Create a fake Record-like object."""

        class FakeRecord:
            def __init__(self, d):
                self._data = d
                self._keys = list(d.keys())
                self._values = list(d.values())

            def __getitem__(self, key):
                if isinstance(key, int):
                    return self._values[key]
                return self._data[key]

            def keys(self):
                return self._keys

        return PostgresRow(FakeRecord(data))

    def test_named_access(self):
        row = self._make_row({"id": "kb-00001", "score": 1.5})
        assert row["id"] == "kb-00001"
        assert row["score"] == 1.5

    def test_positional_access(self):
        row = self._make_row({"id": "kb-00001", "score": 1.5})
        assert row[0] == "kb-00001"
        assert row[1] == 1.5

    def test_keys(self):
        row = self._make_row({"a": 1, "b": 2, "c": 3})
        assert row.keys() == ["a", "b", "c"]


class TestPostgresCursor:
    """Test PostgresCursor wrapper."""

    def _make_record(self, data: dict):
        """Create a minimal asyncpg.Record stand-in."""

        class FakeRecord:
            def __init__(self, d):
                self._data = d
                self._keys = list(d.keys())
                self._values = list(d.values())

            def __getitem__(self, key):
                if isinstance(key, int):
                    return self._values[key]
                return self._data[key]

            def keys(self):
                return self._keys

        return FakeRecord(data)

    @pytest.mark.asyncio
    async def test_fetchone_returns_rows(self):
        records = [self._make_record({"id": "a"}), self._make_record({"id": "b"})]
        cursor = PostgresCursor(records)
        row1 = await cursor.fetchone()
        assert row1 is not None
        assert row1["id"] == "a"
        row2 = await cursor.fetchone()
        assert row2 is not None
        assert row2["id"] == "b"
        row3 = await cursor.fetchone()
        assert row3 is None

    @pytest.mark.asyncio
    async def test_fetchall(self):
        records = [self._make_record({"x": 1}), self._make_record({"x": 2})]
        cursor = PostgresCursor(records)
        rows = await cursor.fetchall()
        assert len(rows) == 2
        assert rows[0]["x"] == 1
        assert rows[1]["x"] == 2

    @pytest.mark.asyncio
    async def test_fetchall_after_fetchone(self):
        records = [
            self._make_record({"v": "a"}),
            self._make_record({"v": "b"}),
            self._make_record({"v": "c"}),
        ]
        cursor = PostgresCursor(records)
        await cursor.fetchone()  # consume first
        remaining = await cursor.fetchall()
        assert len(remaining) == 2
        assert remaining[0]["v"] == "b"

    def test_rowcount_insert(self):
        cursor = PostgresCursor([], status="INSERT 0 1")
        assert cursor.rowcount == 1

    def test_rowcount_update(self):
        cursor = PostgresCursor([], status="UPDATE 3")
        assert cursor.rowcount == 3

    def test_rowcount_delete_zero(self):
        cursor = PostgresCursor([], status="DELETE 0")
        assert cursor.rowcount == 0

    def test_rowcount_no_status(self):
        cursor = PostgresCursor([])
        assert cursor.rowcount == -1

    def test_rowcount_unparseable(self):
        cursor = PostgresCursor([], status="SOMETHING")
        assert cursor.rowcount == -1

    @pytest.mark.asyncio
    async def test_empty_cursor(self):
        cursor = PostgresCursor([])
        assert await cursor.fetchone() is None
        assert await cursor.fetchall() == []


class TestPostgresBackendCreate:
    """Test PostgresBackend.create() kwarg forwarding."""

    @pytest.mark.asyncio
    async def test_create_passes_password_and_ssl(self):
        mock_pool = AsyncMock()

        def password_fn() -> str:
            return "token-123"

        ssl_ctx = ssl.create_default_context()

        # asyncpg is imported lazily inside create() as `_asyncpg`
        mock_asyncpg = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            result = await PostgresBackend.create(
                "postgresql://u@h/db",
                pool_min=1,
                pool_max=5,
                password=password_fn,
                ssl=ssl_ctx,
            )

        assert isinstance(result, PostgresBackend)
        mock_asyncpg.create_pool.assert_called_once()
        _, call_kwargs = mock_asyncpg.create_pool.call_args
        assert call_kwargs["password"] is password_fn
        assert call_kwargs["ssl"] is ssl_ctx
        assert call_kwargs["min_size"] == 1
        assert call_kwargs["max_size"] == 5

    @pytest.mark.asyncio
    async def test_create_omits_none_kwargs(self):
        mock_pool = AsyncMock()
        mock_asyncpg = AsyncMock()
        mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)

        with patch.dict("sys.modules", {"asyncpg": mock_asyncpg}):
            await PostgresBackend.create(
                "postgresql://u@h/db",
                pool_min=1,
                pool_max=5,
            )

        _, call_kwargs = mock_asyncpg.create_pool.call_args
        assert "password" not in call_kwargs
        assert "ssl" not in call_kwargs
