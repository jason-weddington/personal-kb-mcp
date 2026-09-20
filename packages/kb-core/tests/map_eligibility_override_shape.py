"""Single source of truth for the expected ``map_eligibility_override`` shape.

``map_eligibility_override`` is defined TWICE in production code — SQLite DDL
in ``kb_core.db.schema`` (``MAP_ELIGIBILITY_OVERRIDE_SCHEMA_SQL``) and
Postgres DDL inline in
``kb_core.db.postgres_backend.PostgresBackend._apply_schema_locked``.
Nothing in production enforces that the two stay in step — a column added to
one and forgotten in the other would break the hosted Postgres deployments at
runtime while passing every SQLite-only test.

Both the Postgres-side parity test (``test_postgres_backend.py``) and the
SQLite-side parity test (``test_map_eligibility_override_sqlite_schema.py``)
import the constants below rather than each hand-rolling their own copy of
the expected shape. Adding, renaming, or retyping a column in either
backend's DDL now forces the author to update this ONE module — and both
backends get checked against the same expectation.

This module is NOT a test module itself (no ``test_`` functions), is not
matched by pytest's collection glob, and has no dependency on either the
sqlite or postgres backend — it is pure data, safely importable from both
test modules with no circular import.

Deliberately NO ``EXPECTED_C_COLLATION_COLUMN``: unlike
``embedding_retry_queue.next_attempt_at``, nothing in this feature orders or
range-compares any column of this table (the PK is the only access path and
callers sort in Python), so no column carries a ``COLLATE "C"`` pin.
"""

from __future__ import annotations

# Every column in map_eligibility_override, on both backends.
EXPECTED_COLUMNS = frozenset({"project_ref", "eligible", "reason", "set_by", "set_at"})

# Columns whose backend type is an integer family (SQLite ``INTEGER`` /
# Postgres ``integer``). Every OTHER column in EXPECTED_COLUMNS is expected
# to be a text-family type (SQLite ``TEXT`` / Postgres ``text``).
EXPECTED_INTEGER_COLUMNS = frozenset({"eligible"})

EXPECTED_TEXT_COLUMNS = EXPECTED_COLUMNS - EXPECTED_INTEGER_COLUMNS

EXPECTED_PRIMARY_KEY = "project_ref"

# ``set_by`` is the only nullable column — a human decision may have no
# recorded identity on an older row.
EXPECTED_NULLABLE_COLUMNS = frozenset({"set_by"})

# The PK index only — the table holds one row per project_ref and the PK is
# the only access path, so no secondary index exists on either backend.
EXPECTED_INDEX_COUNT = 1
