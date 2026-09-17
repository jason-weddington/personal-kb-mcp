"""SQLite-side schema parity guard for ``embedding_retry_queue`` (GTD 93ae476e).

``embedding_retry_queue`` is hand-defined TWICE in production code — SQLite
DDL in ``kb_core.db.schema`` (``EMBEDDING_RETRY_QUEUE_SCHEMA_SQL``) and
Postgres DDL inline in
``kb_core.db.postgres_backend.PostgresBackend._apply_schema_locked``.
Nothing in production enforces the two stay in step, so a column added to
one and forgotten in the other passes every SQLite-only test while breaking
the hosted Postgres deployments at runtime (flagged in adversarial review of
kb-03277, the original embedding-retry-queue feature).

This test introspects a REAL SQLite table (via ``create_sqlite`` -> real
``aiosqlite`` connection -> ``PRAGMA table_info`` / ``PRAGMA index_list`` /
``PRAGMA index_info``, never a hand-copied SQL string) and checks it against
``embedding_retry_queue_shape.py`` — the SAME shared constants the
Postgres-side counterpart in ``test_postgres_backend.py``
(``test_embedding_retry_queue_matches_expected_shape_on_postgres``) checks
against a live Postgres table. Requires no Postgres and no env var — plain
SQLite in a tmp_path file, like every other test in this module's siblings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from embedding_retry_queue_shape import (
    EXPECTED_COLUMNS,
    EXPECTED_INDEX_COLUMNS,
    EXPECTED_INTEGER_COLUMNS,
    EXPECTED_PRIMARY_KEY,
    EXPECTED_TEXT_COLUMNS,
)
from kb_core import create_sqlite

if TYPE_CHECKING:
    from pathlib import Path


async def test_embedding_retry_queue_matches_expected_shape_on_sqlite(tmp_path: Path) -> None:
    """Introspect embedding_retry_queue on a live SQLite DB and check its shape.

    Checks, via ``PRAGMA table_info`` and ``PRAGMA index_list`` /
    ``PRAGMA index_info``, the SAME facts the Postgres-side counterpart
    checks:

    * the exact column-name set (``EXPECTED_COLUMNS``);
    * ``attempts`` is an integer-affinity column and every other column is
      a text-affinity column;
    * ``entry_id`` is the primary key;
    * a two-column index on ``(status, next_attempt_at)`` exists.

    No collation assertion here — SQLite TEXT columns already compare
    byte-wise (``BINARY`` collation) by default, so there is no SQLite
    counterpart to Postgres's explicit ``COLLATE "C"`` pin to check.
    """
    kb = await create_sqlite(tmp_path / "kb.db")
    try:
        cursor = await kb.db.execute("PRAGMA table_info(embedding_retry_queue)")
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

        index_list_cursor = await kb.db.execute("PRAGMA index_list(embedding_retry_queue)")
        index_list = await index_list_cursor.fetchall()

        found_matching_index = False
        for index_row in index_list:
            index_info_cursor = await kb.db.execute(f"PRAGMA index_info({index_row['name']})")
            index_info = await index_info_cursor.fetchall()
            ordered_columns = tuple(
                row["name"] for row in sorted(index_info, key=lambda r: r["seqno"])
            )
            if ordered_columns == EXPECTED_INDEX_COLUMNS:
                found_matching_index = True
                break

        assert found_matching_index, (
            f"no index on {EXPECTED_INDEX_COLUMNS} found; "
            f"index_list={[row['name'] for row in index_list]}"
        )
    finally:
        await kb.close()
