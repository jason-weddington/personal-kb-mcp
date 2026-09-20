"""Single source of truth for the expected ``map_cluster_ledger`` shape.

``map_cluster_ledger`` is defined TWICE in production code — SQLite DDL in
``kb_core.db.schema`` (``MAP_CLUSTER_LEDGER_SCHEMA_SQL``) and Postgres DDL
inline in
``kb_core.db.postgres_backend.PostgresBackend._apply_schema_locked``.
Nothing in production enforces that the two stay in step — a column added to
one and forgotten in the other would break the hosted Postgres deployments at
runtime while passing every SQLite-only test.

Both the Postgres-side parity test (``test_postgres_backend.py``) and the
SQLite-side parity test (``test_map_cluster_ledger_sqlite_schema.py``) import
the constants below rather than each hand-rolling their own copy of the
expected shape. Adding, renaming, or retyping a column in either backend's
DDL now forces the author to update this ONE module — and both backends get
checked against the same expectation.

This module is NOT a test module itself (no ``test_`` functions), is not
matched by pytest's collection glob, and has no dependency on either the
sqlite or postgres backend — it is pure data, safely importable from both
test modules with no circular import.

Deliberately NO ``EXPECTED_C_COLLATION_COLUMN``: unlike
``embedding_retry_queue.next_attempt_at``, nothing in this feature orders or
range-compares any column of this table (every lookup is ``cluster_key`` PK
equality or ``project_ref`` equality, and both ``list_clusters`` and
``match_clusters`` sort in Python), so no column carries a ``COLLATE "C"``
pin.
"""

from __future__ import annotations

# Every column in map_cluster_ledger, on both backends.
EXPECTED_COLUMNS = frozenset(
    {
        "cluster_key",
        "project_ref",
        "member_entry_ids",
        "last_label",
        "status",
        "sightings",
        "first_seen_at",
        "last_seen_at",
        "declined_member_count",
        "declined_reason",
        "declined_by",
        "declined_at",
    }
)

# Columns whose backend type is an integer family (SQLite ``INTEGER`` /
# Postgres ``integer``). Every OTHER column in EXPECTED_COLUMNS is expected
# to be a text-family type (SQLite ``TEXT`` / Postgres ``text``).
EXPECTED_INTEGER_COLUMNS = frozenset({"sightings", "declined_member_count"})

EXPECTED_TEXT_COLUMNS = EXPECTED_COLUMNS - EXPECTED_INTEGER_COLUMNS

EXPECTED_PRIMARY_KEY = "cluster_key"

# The four ``declined_*`` columns are NULL exactly while the row carries
# ``status = 'proposed'``; a decline sets all four together.
EXPECTED_NULLABLE_COLUMNS = frozenset(
    {"declined_member_count", "declined_reason", "declined_by", "declined_at"}
)

# The PK index only — no secondary index on ``project_ref`` even though the
# read path filters on it: at a few hundred rows (25 eligible projects x
# 10-20 clusters each) a sequential scan per read is free, and keeping the
# two backends' DDL in step matters more than the index. Add the index when
# the table passes ~10,000 rows.
EXPECTED_INDEX_COUNT = 1
