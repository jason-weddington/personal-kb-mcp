"""Single source of truth for the expected ``embedding_retry_queue`` shape.

``embedding_retry_queue`` is defined TWICE in production code — SQLite DDL
in ``kb_core.db.schema`` (``EMBEDDING_RETRY_QUEUE_SCHEMA_SQL``) and Postgres
DDL inline in ``kb_core.db.postgres_backend.PostgresBackend._apply_schema_locked``.
Nothing in the production code enforces that the two stay in step — a
column added to one and forgotten in the other would pass every test that
existed before GTD 93ae476e while breaking the hosted Postgres deployments
at runtime.

Both the Postgres-side parity test (``test_postgres_backend.py``) and the
SQLite-side parity test (``test_embedding_retry_queue_sqlite_schema.py``)
import the constants below rather than each hand-rolling their own copy of
the expected shape. Adding, renaming, or retyping a column in either
backend's DDL now forces the author to update this ONE module — and both
backends get checked against the same expectation.

This module is NOT a test module itself (no ``test_`` functions), is not
matched by pytest's ``test_*.py`` / ``*_test.py`` collection glob, and has
no dependency on either the sqlite or postgres backend — it is pure data,
safely importable from both test modules with no circular import.
"""

from __future__ import annotations

# Every column in embedding_retry_queue, on both backends.
EXPECTED_COLUMNS = frozenset(
    {
        "entry_id",
        "attempts",
        "last_error",
        "next_attempt_at",
        "status",
        "created_at",
        "updated_at",
    }
)

# Columns whose backend type is an integer family (SQLite ``INTEGER`` /
# Postgres ``integer``). Every OTHER column in EXPECTED_COLUMNS is expected
# to be a text-family type (SQLite ``TEXT`` / Postgres ``text``).
EXPECTED_INTEGER_COLUMNS = frozenset({"attempts"})

EXPECTED_TEXT_COLUMNS = EXPECTED_COLUMNS - EXPECTED_INTEGER_COLUMNS

EXPECTED_PRIMARY_KEY = "entry_id"

# next_attempt_at is pinned COLLATE "C" on Postgres so ISO-8601 ordering is
# pure byte-wise, independent of the database's default locale (see the DDL
# comment + migration in postgres_backend.py for the full rationale). SQLite
# has no separate collation concept to check here — TEXT columns already
# compare byte-wise (BINARY collation) by default.
EXPECTED_C_COLLATION_COLUMN = "next_attempt_at"

# The compound index that _claim_due's due-row scan and queue_stats' MIN()
# depend on. Order matters (leading column first).
EXPECTED_INDEX_COLUMNS = ("status", "next_attempt_at")
