"""SQLite-side schema parity guard for ``map_eligibility_override``.

``map_eligibility_override`` is hand-defined TWICE in production code —
SQLite DDL in ``kb_core.db.schema`` (``MAP_ELIGIBILITY_OVERRIDE_SCHEMA_SQL``)
and Postgres DDL inline in
``kb_core.db.postgres_backend.PostgresBackend._apply_schema_locked``.
Nothing in production enforces the two stay in step, so a column added to
one and forgotten in the other passes every SQLite-only test while breaking
the hosted Postgres deployments at runtime.

This test introspects a REAL SQLite table (via ``create_sqlite`` -> real
``aiosqlite`` connection -> ``PRAGMA table_info`` / ``PRAGMA index_list``,
never a hand-copied SQL string) and checks it against
``map_eligibility_override_shape.py`` — the SAME shared constants the
Postgres-side counterpart in ``test_postgres_backend.py``
(``test_map_eligibility_override_matches_expected_shape_on_postgres``)
checks against a live Postgres table. Requires no Postgres and no env var.

No collation assertion: unlike ``embedding_retry_queue.next_attempt_at``,
nothing in this feature orders or range-compares any column of this table,
so no column carries a ``COLLATE "C"`` pin on either backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kb_core import create_sqlite
from map_eligibility_override_shape import (
    EXPECTED_COLUMNS,
    EXPECTED_INDEX_COUNT,
    EXPECTED_INTEGER_COLUMNS,
    EXPECTED_NULLABLE_COLUMNS,
    EXPECTED_PRIMARY_KEY,
    EXPECTED_TEXT_COLUMNS,
)

if TYPE_CHECKING:
    from pathlib import Path


async def test_map_eligibility_override_matches_expected_shape_on_sqlite(
    tmp_path: Path,
) -> None:
    """Introspect map_eligibility_override on a live SQLite DB and check its shape.

    Checks, via ``PRAGMA table_info`` and ``PRAGMA index_list``, the SAME
    facts the Postgres-side counterpart checks:

    * the exact column-name set (``EXPECTED_COLUMNS``);
    * ``eligible`` is an integer-affinity column and every other column is
      a text-affinity column;
    * ``project_ref`` is the (single) primary key;
    * ``set_by`` is the only nullable column (observed as the nullable set
      plus the PK column — SQLite's PRAGMA does not report a non-INTEGER
      PRIMARY KEY as NOT NULL; see the quirk note inline);
    * exactly one index exists — the PK's ``sqlite_autoindex`` (the table
      holds one row per project_ref and the PK is the only access path).
    """
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        cursor = await kb.db.execute("PRAGMA table_info(map_eligibility_override)")
        columns = await cursor.fetchall()

        actual_column_names = {row["name"] for row in columns}
        assert actual_column_names == EXPECTED_COLUMNS

        actual_integer_columns = {
            row["name"] for row in columns if row["type"].upper() == "INTEGER"
        }
        actual_text_columns = {row["name"] for row in columns if row["type"].upper() == "TEXT"}
        assert actual_integer_columns == EXPECTED_INTEGER_COLUMNS
        assert actual_text_columns == EXPECTED_TEXT_COLUMNS

        pk_columns = [row["name"] for row in columns if row["pk"]]
        assert pk_columns == [EXPECTED_PRIMARY_KEY]

        # SQLite quirk: a TEXT PRIMARY KEY column in an ordinary rowid table
        # does NOT report notnull=1 in PRAGMA table_info (the NOT NULL
        # constraint is not enforced for non-INTEGER PKs). The observed
        # not-null set is therefore EXPECTED_NULLABLE_COLUMNS plus the PK
        # column; the Postgres-side counterpart checks the real nullability
        # of every column via information_schema.
        assert {row["name"] for row in columns if not row["notnull"]} == (
            EXPECTED_NULLABLE_COLUMNS | {EXPECTED_PRIMARY_KEY}
        )

        index_list_cursor = await kb.db.execute("PRAGMA index_list(map_eligibility_override)")
        index_list = await index_list_cursor.fetchall()
        assert len(index_list) == EXPECTED_INDEX_COUNT
        assert index_list[0]["name"].startswith("sqlite_autoindex")
    finally:
        await kb.close()
