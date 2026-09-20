"""Real-Postgres integration tests for :class:`PostgresBackend`.

Env-var-gated + skip-when-unset (see ``conftest.py``): with
``KB_TEST_DATABASE_URL`` unset these tests SKIP, not error/fail, and the
existing SQLite suites stay green — the no-URL run is hermetic.

Why this suite exists
---------------------
The rest of kb-core's test surface runs on SQLite, which silently accepts
several statements that Postgres rejects.  Real bugs caught by this
suite so far:

* **kb-02915** — ``PostgresBackend.delete_llm_edges`` originally ran
  ``properties->>'source' = 'llm'`` against a ``TEXT`` column,
  producing ``operator does not exist: text ->> unknown`` at runtime.
  The fix (``properties::jsonb->>'source' = 'llm'``, currently at
  ``postgres_backend.py:388-389``) is invisible to any SQLite-only
  test.  The regression test below exercises that path against real
  Postgres and asserts selective deletion.

Beyond kb-02915 we sweep the other Postgres-dialect codepaths on the
backend so a future regression on any of them (tsvector trigger, GIN
FTS, pgvector ``<=>`` cosine + ``::vector`` cast, sequence RETURNING,
``ANALYZE``) also lights up here.

Note on jsonb usage
-------------------
``properties::jsonb->>`` appears in :meth:`PostgresBackend.delete_llm_edges`
and, since kb-017b7606 (see below), in
:meth:`PostgresBackend.delete_deterministic_edges` — the two json-operator
regression cases this file needs to guard.

kb-017b7606 — delete_deterministic_edges selectivity
-----------------------------------------------------
A metadata-only entry update (tags/hints/title, no ``knowledge_details``)
still unconditionally rebuilds deterministic graph edges via
``GraphBuilder._clear_edges_for_source``. That rebuild must clear only the
deterministic edges it re-derives and leave LLM-enriched edges
(``properties.source == "llm"``) untouched, or every enrichment edge on an
entry gets silently and permanently destroyed by the next tags-only update.
The fix adds :meth:`PostgresBackend.delete_deterministic_edges`, the mirror
image of ``delete_llm_edges`` (same ``properties::jsonb->>'source'`` cast,
inverted selection via ``IS DISTINCT FROM`` for NULL-safety). The test below
proves it selectively deletes the non-LLM edge and preserves the LLM one.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from embedding_retry_queue_shape import (
    EXPECTED_C_COLLATION_COLUMN,
    EXPECTED_COLUMNS,
    EXPECTED_INDEX_COLUMNS,
    EXPECTED_INTEGER_COLUMNS,
    EXPECTED_PRIMARY_KEY,
    EXPECTED_TEXT_COLUMNS,
)
from kb_core.embedding_retry import (
    _claim_due,
    _vectorless_unqueued_count,
    backfill,
    enqueue,
    queue_stats,
    resolve,
)
from kb_core.store.knowledge_store import KnowledgeStore
from map_cluster_ledger_shape import (
    EXPECTED_COLUMNS as MCL_EXPECTED_COLUMNS,
)
from map_cluster_ledger_shape import (
    EXPECTED_INDEX_COUNT as MCL_EXPECTED_INDEX_COUNT,
)
from map_cluster_ledger_shape import (
    EXPECTED_INTEGER_COLUMNS as MCL_EXPECTED_INTEGER_COLUMNS,
)
from map_cluster_ledger_shape import (
    EXPECTED_NULLABLE_COLUMNS as MCL_EXPECTED_NULLABLE_COLUMNS,
)
from map_cluster_ledger_shape import (
    EXPECTED_PRIMARY_KEY as MCL_EXPECTED_PRIMARY_KEY,
)
from map_cluster_ledger_shape import (
    EXPECTED_TEXT_COLUMNS as MCL_EXPECTED_TEXT_COLUMNS,
)
from map_eligibility_counts_fixture import (
    EXPECTED_COUNTS,
    SEED_ENTRIES,
    SEED_INGESTED_FILES,
)
from map_eligibility_override_shape import (
    EXPECTED_COLUMNS as ME_EXPECTED_COLUMNS,
)
from map_eligibility_override_shape import (
    EXPECTED_INDEX_COUNT as ME_EXPECTED_INDEX_COUNT,
)
from map_eligibility_override_shape import (
    EXPECTED_INTEGER_COLUMNS as ME_EXPECTED_INTEGER_COLUMNS,
)
from map_eligibility_override_shape import (
    EXPECTED_NULLABLE_COLUMNS,
)
from map_eligibility_override_shape import (
    EXPECTED_PRIMARY_KEY as ME_EXPECTED_PRIMARY_KEY,
)
from map_eligibility_override_shape import (
    EXPECTED_TEXT_COLUMNS as ME_EXPECTED_TEXT_COLUMNS,
)

if TYPE_CHECKING:
    # Only used in ``pg_kb: PostgresBackend`` annotations below (stringified
    # under ``from __future__ import annotations``).
    from kb_core.db.postgres_backend import PostgresBackend

pytestmark = pytest.mark.postgres


# ---------------------------------------------------------------------------
# kb-02915 regression: delete_llm_edges properties::jsonb->> path
# ---------------------------------------------------------------------------


async def test_delete_llm_edges_selective_on_postgres(pg_kb: PostgresBackend) -> None:
    """delete_llm_edges removes only ``source='llm'`` edges (kb-02915 regression).

    Prior to the ``properties::jsonb->>`` cast in delete_llm_edges, this
    statement raised ``operator does not exist: text ->> unknown`` on
    Postgres — the ``properties`` column is ``TEXT`` and ``->>`` is a
    jsonb operator.  This test:

    1. Inserts two ``graph_nodes`` (``src1`` and ``tgt1``) — required by
       the ``graph_edges.source/target`` FKs.
    2. Inserts two ``graph_edges`` both ``src1 → tgt1``, with DISTINCT
       ``edge_type`` values to satisfy the ``UNIQUE(source, target,
       edge_type)`` constraint — one flagged ``"source": "llm"``, one
       flagged ``"source": "manual"``.
    3. Calls :meth:`PostgresBackend.delete_llm_edges` for ``src1``.
    4. Asserts the llm-flagged edge is gone and the manual edge remains.

    Every insert supplies every NOT-NULL-no-default column
    explicitly (``created_at`` is NOT NULL and has no default on both
    tables, so it must be supplied even though the rest of the app
    always stamps it).
    """
    ts = "2026-01-01T00:00:00Z"

    # 1) Nodes required by the source/target FKs on graph_edges.
    await pg_kb.execute(
        "INSERT INTO graph_nodes (node_id, node_type, created_at) VALUES (?, ?, ?)",
        ("src1", "entry", ts),
    )
    await pg_kb.execute(
        "INSERT INTO graph_nodes (node_id, node_type, created_at) VALUES (?, ?, ?)",
        ("tgt1", "entry", ts),
    )

    # 2) Two edges src1 -> tgt1 with DISTINCT edge_type so the UNIQUE
    #    (source, target, edge_type) constraint is satisfied.  One tagged
    #    source=llm, one tagged source=manual — the delete must be selective.
    await pg_kb.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        ("src1", "tgt1", "relates_to", '{"source": "llm"}', ts),
    )
    await pg_kb.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        ("src1", "tgt1", "references", '{"source": "manual"}', ts),
    )

    # 3) Run the kb-02915 codepath.  Before the fix this raised
    #    ``operator does not exist: text ->> unknown``.
    await pg_kb.delete_llm_edges("src1")

    # 4) Only the manual edge should remain.
    cursor = await pg_kb.execute(
        "SELECT edge_type, properties FROM graph_edges WHERE source = ?",
        ("src1",),
    )
    rows = await cursor.fetchall()
    remaining = [(row["edge_type"], row["properties"]) for row in rows]

    assert len(remaining) == 1
    assert remaining[0][0] == "references"
    assert "manual" in remaining[0][1]


# ---------------------------------------------------------------------------
# kb-017b7606 regression: delete_deterministic_edges properties::jsonb->> path
# ---------------------------------------------------------------------------


async def test_delete_deterministic_edges_selective_on_postgres(pg_kb: PostgresBackend) -> None:
    """delete_deterministic_edges removes only non-``source='llm'`` edges.

    Mirror image of :func:`test_delete_llm_edges_selective_on_postgres`:
    same jsonb cast, inverted selection. This is what
    ``GraphBuilder._clear_edges_for_source`` now calls ahead of every
    rebuild so a metadata-only update can't destroy enrichment edges.

    1. Inserts two ``graph_nodes`` (``src2`` and ``tgt2``).
    2. Inserts two ``graph_edges`` both ``src2 -> tgt2`` with distinct
       ``edge_type`` values — one flagged ``"source": "llm"``, one with
       the deterministic-builder default properties ``'{}'`` (no
       ``source`` key at all, exercising the NULL-safety of
       ``IS DISTINCT FROM``).
    3. Calls :meth:`PostgresBackend.delete_deterministic_edges` for ``src2``.
    4. Asserts the llm-flagged edge survives and the deterministic edge is gone.
    """
    ts = "2026-01-01T00:00:00Z"

    await pg_kb.execute(
        "INSERT INTO graph_nodes (node_id, node_type, created_at) VALUES (?, ?, ?)",
        ("src2", "entry", ts),
    )
    await pg_kb.execute(
        "INSERT INTO graph_nodes (node_id, node_type, created_at) VALUES (?, ?, ?)",
        ("tgt2", "entry", ts),
    )

    await pg_kb.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        ("src2", "tgt2", "relates_to", '{"source": "llm"}', ts),
    )
    await pg_kb.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        ("src2", "tgt2", "has_tag", "{}", ts),
    )

    await pg_kb.delete_deterministic_edges("src2")

    cursor = await pg_kb.execute(
        "SELECT edge_type, properties FROM graph_edges WHERE source = ?",
        ("src2",),
    )
    rows = await cursor.fetchall()
    remaining = [(row["edge_type"], row["properties"]) for row in rows]

    assert len(remaining) == 1
    assert remaining[0][0] == "relates_to"
    assert "llm" in remaining[0][1]


# ---------------------------------------------------------------------------
# Postgres-dialect SQL sweep — every PG-only path exercised against real PG
# ---------------------------------------------------------------------------


async def test_fts_search_uses_tsvector_trigger(pg_kb: PostgresBackend) -> None:
    """fts_search returns tsvector hits — exercises the trigger + GIN index.

    Inserting a ``knowledge_entries`` row fires the BEFORE INSERT
    ``tsvector_update`` trigger, which populates ``search_vector`` from
    ``short_title / long_title / knowledge_details / tags``.  A
    subsequent :meth:`PostgresBackend.fts_search` query then runs
    ``plainto_tsquery`` against that column with an ``is_active = 1``
    filter (the column's default).
    """
    ts = "2026-01-01T00:00:00Z"
    entry_id = "kb-00001"

    # 7 NOT-NULL-no-default columns: id, short_title, long_title,
    # knowledge_details, entry_type, created_at, updated_at.  Everything
    # else (confidence_level / tags / hints / is_active / has_embedding /
    # version) has a default and can be omitted.  The BEFORE INSERT
    # trigger populates search_vector from short_title (weight A) etc.
    await pg_kb.execute(
        "INSERT INTO knowledge_entries"
        " (id, short_title, long_title, knowledge_details, entry_type,"
        "  created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            entry_id,
            "zorptastic marker word",
            "A distinctive title",
            "Body text.",
            "note",
            ts,
            ts,
        ),
    )

    results = await pg_kb.fts_search("zorptastic")

    assert results, "expected at least one FTS hit for the unique marker word"
    first_id, first_score = results[0]
    assert first_id == entry_id
    assert isinstance(first_score, float)


async def test_vector_store_and_search_round_trip(pg_kb: PostgresBackend) -> None:
    """vector_store + vector_search round-trip a 1024-dim vector via pgvector.

    Exercises three Postgres-only things at once: the ``knowledge_vec``
    pgvector column, the ``$N::vector`` cast, and the ``<=>`` cosine
    distance operator.  We call ``vector_search`` with NO filter kwargs
    so we hit the ``knowledge_vec``-only branch (no join against
    ``knowledge_entries``).  Same vector in and out → cosine distance
    approximately 0.
    """
    # Deterministic unit-length 1024-d vector — matches the ``vector(1024)``
    # column produced by ``apply_schema(embedding_dim=1024)`` in the fixture.
    embedding = [0.0] * 1024
    embedding[0] = 1.0

    await pg_kb.vector_store("e1", embedding)
    results = await pg_kb.vector_search(embedding)

    assert results, "expected at least one vector hit"
    first_id, first_distance = results[0]
    assert first_id == "e1"
    assert abs(first_distance) < 1e-6


async def test_next_sequence_value_increments_atomically(pg_kb: PostgresBackend) -> None:
    """next_sequence_value returns ints and second == first + 1."""
    first = await pg_kb.next_sequence_value()
    second = await pg_kb.next_sequence_value()

    assert isinstance(first, int)
    assert isinstance(second, int)
    assert second == first + 1


async def test_vacuum_returns_pinned_status_string(pg_kb: PostgresBackend) -> None:
    """vacuum() runs ``ANALYZE`` and returns the pinned status string."""
    result = await pg_kb.vacuum()
    assert result == "Vacuum complete (ANALYZE)."


# ---------------------------------------------------------------------------
# embedding_retry_queue — the ONLY place the ?->$N translation and the various
# ON CONFLICT / dynamic-IN / NOT-EXISTS shapes kb_core.embedding_retry issues
# get exercised against real PG (GTD 735a7e1d; follow-up hardening f9fef4f9).
#
# Every test below calls the REAL kb_core.embedding_retry functions
# (enqueue/backfill/resolve/_claim_due/queue_stats/_vectorless_unqueued_count)
# against the asyncpg-backed ``pg_kb`` fixture — never a hand-copied SQL
# string — so a future edit to that module's SQL is exercised here
# automatically instead of silently drifting from a frozen snapshot.
# ---------------------------------------------------------------------------


async def _insert_vectorless_entry(pg_kb: PostgresBackend, entry_id: str) -> None:
    """Insert a minimal active, vectorless ``knowledge_entries`` row.

    Vectorless + active is exactly the source-of-truth predicate
    :meth:`KnowledgeStore.get_entries_without_embeddings` (and therefore
    :func:`kb_core.embedding_retry.backfill`) selects on: ``has_embedding = 0
    AND is_active = 1`` (both column defaults, so neither needs to be
    supplied explicitly).
    """
    ts = "2026-01-01T00:00:00Z"
    await pg_kb.execute(
        "INSERT INTO knowledge_entries"
        " (id, short_title, long_title, knowledge_details, entry_type,"
        "  created_at, updated_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (entry_id, entry_id, entry_id, "Body text.", "note", ts, ts),
    )


async def test_embedding_retry_queue_table_exists_after_apply_schema(
    pg_kb: PostgresBackend,
) -> None:
    """apply_schema() creates embedding_retry_queue on Postgres too."""
    cursor = await pg_kb.execute(
        "SELECT table_name FROM information_schema.tables"
        " WHERE table_name = 'embedding_retry_queue'"
    )
    assert await cursor.fetchone() is not None


async def test_embedding_retry_queue_next_attempt_at_is_c_collation(
    pg_kb: PostgresBackend,
) -> None:
    """``next_attempt_at`` is pinned ``COLLATE "C"`` (byte-wise ordering).

    ``_claim_due``'s ``ORDER BY next_attempt_at`` / ``next_attempt_at <= ?``
    and ``queue_stats``'s ``MIN(next_attempt_at)`` all depend on this column
    comparing as pure byte-wise (C-locale) text, not the database's default
    collation — which under a glibc locale can apply punctuation-insensitive
    comparison rules that misorder the exact ISO-8601 shapes ``_iso()``
    emits (with vs. without a microseconds component; see the DDL comment
    in ``postgres_backend.py`` for the full rationale).
    """
    cursor = await pg_kb.execute(
        "SELECT collation_name FROM information_schema.columns"
        " WHERE table_name = 'embedding_retry_queue' AND column_name = 'next_attempt_at'"
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row["collation_name"] == "C"


async def test_embedding_retry_enqueue_and_claim_due_round_trip_on_asyncpg(
    pg_kb: PostgresBackend,
) -> None:
    """enqueue() and _claim_due() round-trip via the REAL functions on asyncpg.

    Exercises enqueue()'s ``INSERT ... ON CONFLICT(entry_id) DO UPDATE ...
    CASE WHEN`` upsert (the fresh-insert path) and ``_claim_due``'s
    compare-and-swap claim ``UPDATE ... WHERE next_attempt_at = ? AND status
    = 'pending'``. A claimed row is pushed past its due window by the lease,
    so calling ``_claim_due`` again immediately claims nothing — proving the
    CAS protects a claimed row from being claimed twice, via the actual
    ``?``->``$N``-translated statement (not a hand-copied one).

    NOTE: enqueue()'s ``DO UPDATE`` has no ``WHERE`` clause on the conflict
    action itself (the exhausted-vs-pending branching happens inside ``CASE
    WHEN`` in the ``SET`` list, unconditionally applied). The WHERE-qualified
    conflict action lives in :func:`kb_core.embedding_retry.backfill` — see
    ``test_backfill_revives_only_exhausted_rows_via_do_update_where_on_asyncpg``
    below for that coverage.
    """
    entry_id = "kb-77001"
    now = datetime(2026, 1, 1, tzinfo=UTC)

    await enqueue(pg_kb, entry_id, error="embed returned None", now=now)

    cursor = await pg_kb.execute(
        "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?",
        (entry_id,),
    )
    row = await cursor.fetchone()
    assert row["attempts"] == 0
    assert row["status"] == "pending"
    # now + BACKOFF_SCHEDULE_SECONDS[0] (60s) — enqueue()'s fresh-row offset.
    assert row["next_attempt_at"] == "2026-01-01T00:01:00+00:00"

    claim_now = now + timedelta(seconds=61)
    first_claim = await _claim_due(pg_kb, now=claim_now, batch_size=10, lease_seconds=600)
    assert [c["entry_id"] for c in first_claim] == [entry_id]
    assert first_claim[0]["attempts"] == 0

    second_claim = await _claim_due(pg_kb, now=claim_now, batch_size=10, lease_seconds=600)
    assert second_claim == []


async def test_backfill_revives_only_exhausted_rows_via_do_update_where_on_asyncpg(
    pg_kb: PostgresBackend,
) -> None:
    """backfill()'s ``ON CONFLICT(entry_id) DO UPDATE SET ... WHERE status =
    'exhausted'`` clause round-trips on asyncpg — never exercised against
    real Postgres before this test (the original suite only exercised
    enqueue()'s unconditional CASE-WHEN upsert, not this WHERE-qualified
    conflict action). Proves two things at once, both required for the
    clause to be doing its job:

    * An ``'exhausted'`` row among the vectorless/active entries IS revived
      (status -> 'pending', attempts -> 0, next_attempt_at -> now).
    * A ``'pending'`` row among the same set is left COMPLETELY untouched —
      the WHERE clause is the only thing stopping backfill from clobbering
      a row that's already mid-backoff.
    """
    exhausted_id = "kb-88001"
    pending_id = "kb-88002"
    await _insert_vectorless_entry(pg_kb, exhausted_id)
    await _insert_vectorless_entry(pg_kb, pending_id)

    early = datetime(2025, 1, 1, tzinfo=UTC)
    await enqueue(pg_kb, exhausted_id, error="x", now=early)
    await pg_kb.execute(
        "UPDATE embedding_retry_queue SET status = 'exhausted', attempts = 6 WHERE entry_id = ?",
        (exhausted_id,),
    )
    await enqueue(pg_kb, pending_id, error="x", now=early)
    cursor = await pg_kb.execute(
        "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?",
        (pending_id,),
    )
    row = await cursor.fetchone()
    pending_before = (row["attempts"], row["status"], row["next_attempt_at"])

    store = KnowledgeStore(pg_kb)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    n = await backfill(pg_kb, store, now=now)
    assert n == 2

    cursor = await pg_kb.execute(
        "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?",
        (exhausted_id,),
    )
    revived = await cursor.fetchone()
    assert revived["status"] == "pending"
    assert revived["attempts"] == 0
    assert revived["next_attempt_at"] == "2026-01-01T00:00:00+00:00"

    cursor = await pg_kb.execute(
        "SELECT attempts, status, next_attempt_at FROM embedding_retry_queue WHERE entry_id = ?",
        (pending_id,),
    )
    row = await cursor.fetchone()
    assert (row["attempts"], row["status"], row["next_attempt_at"]) == pending_before


async def test_resolve_dynamic_in_clause_on_asyncpg(pg_kb: PostgresBackend) -> None:
    """resolve()'s dynamically-built ``DELETE ... WHERE entry_id IN (?,...)``
    round-trips on asyncpg for a multi-id batch — never exercised against
    real Postgres before this test. Also proves the empty-list no-op path
    never issues a statement (would be invalid SQL: ``IN ()``).
    """
    ids = ["kb-99001", "kb-99002", "kb-99003"]
    now = datetime(2026, 1, 1, tzinfo=UTC)
    for entry_id in ids:
        await enqueue(pg_kb, entry_id, error="x", now=now)

    await resolve(pg_kb, [])  # no-op — must not raise on an empty IN clause

    await resolve(pg_kb, [ids[0], ids[1]])

    cursor = await pg_kb.execute("SELECT entry_id FROM embedding_retry_queue")
    remaining = {r["entry_id"] for r in await cursor.fetchall()}
    assert remaining == {ids[2]}


async def test_vectorless_unqueued_count_not_exists_subquery_on_asyncpg(
    pg_kb: PostgresBackend,
) -> None:
    """_vectorless_unqueued_count()'s ``NOT EXISTS`` correlated subquery
    round-trips on asyncpg — never exercised against real Postgres before
    this test. Exercised both directly and through :func:`queue_stats`
    (the operator-facing surface that calls it), against a mix of a
    vectorless-and-queued entry (must NOT count) and a vectorless-and-
    unqueued entry (must count) plus a vectored entry (excluded by
    ``has_embedding``, never reaches the subquery either way).
    """
    queued_id = "kb-77101"
    unqueued_id = "kb-77102"
    vectored_id = "kb-77103"
    await _insert_vectorless_entry(pg_kb, queued_id)
    await _insert_vectorless_entry(pg_kb, unqueued_id)
    await _insert_vectorless_entry(pg_kb, vectored_id)
    await pg_kb.execute(
        "UPDATE knowledge_entries SET has_embedding = 1 WHERE id = ?", (vectored_id,)
    )

    now = datetime(2026, 1, 1, tzinfo=UTC)
    await enqueue(pg_kb, queued_id, error="x", now=now)

    assert await _vectorless_unqueued_count(pg_kb) == 1

    stats = await queue_stats(pg_kb, now=now)
    assert stats["vectorless_unqueued"] == 1


# ---------------------------------------------------------------------------
# Schema parity guard (GTD 93ae476e) — embedding_retry_queue is hand-defined
# TWICE (SQLite in kb_core.db.schema, Postgres inline in
# PostgresBackend._apply_schema_locked) with nothing enforcing the two stay
# in step. This test introspects the LIVE Postgres table shape and checks it
# against the single shared expectation in embedding_retry_queue_shape.py —
# the SQLite counterpart (test_embedding_retry_queue_sqlite_schema.py)
# checks the same constants against a live SQLite table. A column added to
# one backend and forgotten in the other fails whichever side's constant it
# broke.
# ---------------------------------------------------------------------------


async def test_embedding_retry_queue_matches_expected_shape_on_postgres(
    pg_kb: PostgresBackend,
) -> None:
    """Introspect embedding_retry_queue on live Postgres and check its shape.

    Checks, via ``information_schema.columns`` and ``pg_indexes``:

    * the exact column-name set (``EXPECTED_COLUMNS``);
    * ``attempts`` is an integer type and every other column is a text type;
    * ``entry_id`` is the primary key;
    * ``next_attempt_at`` carries ``COLLATE "C"`` (``collation_name = 'C'``);
    * an index on ``(status, next_attempt_at)`` exists.
    """
    cursor = await pg_kb.execute(
        "SELECT column_name, data_type, collation_name FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = 'embedding_retry_queue'"
    )
    columns = await cursor.fetchall()

    actual_column_names = {row["column_name"] for row in columns}
    assert actual_column_names == EXPECTED_COLUMNS

    actual_integer_columns = {
        row["column_name"] for row in columns if row["data_type"] == "integer"
    }
    actual_text_columns = {row["column_name"] for row in columns if row["data_type"] == "text"}
    assert actual_integer_columns == EXPECTED_INTEGER_COLUMNS
    assert actual_text_columns == EXPECTED_TEXT_COLUMNS

    collation_by_column = {row["column_name"]: row["collation_name"] for row in columns}
    assert collation_by_column[EXPECTED_C_COLLATION_COLUMN] == "C"

    pk_cursor = await pg_kb.execute(
        "SELECT kcu.column_name FROM information_schema.table_constraints tc"
        " JOIN information_schema.key_column_usage kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "   AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = current_schema()"
        "   AND tc.table_name = 'embedding_retry_queue'"
        "   AND tc.constraint_type = 'PRIMARY KEY'"
    )
    pk_rows = await pk_cursor.fetchall()
    assert [row["column_name"] for row in pk_rows] == [EXPECTED_PRIMARY_KEY]

    idx_cursor = await pg_kb.execute(
        "SELECT indexdef FROM pg_indexes"
        " WHERE schemaname = current_schema() AND tablename = 'embedding_retry_queue'"
    )
    indexdefs = [row["indexdef"] for row in await idx_cursor.fetchall()]
    expected_cols_pattern = (
        r"\(\s*" + r"\s*,\s*".join(re.escape(col) for col in EXPECTED_INDEX_COLUMNS) + r"\s*\)"
    )
    assert any(
        re.search(expected_cols_pattern, indexdef, re.IGNORECASE) for indexdef in indexdefs
    ), f"no index on {EXPECTED_INDEX_COLUMNS} found; indexdefs={indexdefs}"


# ---------------------------------------------------------------------------
# Map eligibility (nightly map maintenance, Loop 1) — shape parity + counts.
# The override table is hand-defined TWICE (SQLite in kb_core.db.schema,
# Postgres inline in _apply_schema_locked); the counts query diverges by
# design (split_part / jsonb_array_elements_text / COLLATE "C"). Both tests
# below check against the SAME shared expectation modules the SQLite suite
# uses (map_eligibility_override_shape.py / map_eligibility_counts_fixture.py),
# so a dialect drift fails on whichever side runs.
# ---------------------------------------------------------------------------


async def test_map_eligibility_override_matches_expected_shape_on_postgres(
    pg_kb: PostgresBackend,
) -> None:
    """Introspect map_eligibility_override on live Postgres and check its shape.

    Checks, via ``information_schema.columns``, ``information_schema.
    key_column_usage`` and ``pg_indexes``:

    * the exact column-name set (``ME_EXPECTED_COLUMNS``);
    * ``eligible`` is an integer type and every other column is a text type;
    * ``project_ref`` is the primary key;
    * ``set_by`` is the only nullable column;
    * exactly one index exists — the PK index (no secondary index; the
      table holds one row per project_ref and the PK is the only access
      path).

    No collation assertion: nothing in this feature orders or range-compares
    any column of this table, so no column carries a ``COLLATE "C"`` pin
    (unlike embedding_retry_queue.next_attempt_at).
    """
    cursor = await pg_kb.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = 'map_eligibility_override'"
    )
    columns = await cursor.fetchall()

    actual_column_names = {row["column_name"] for row in columns}
    assert actual_column_names == ME_EXPECTED_COLUMNS

    actual_integer_columns = {
        row["column_name"] for row in columns if row["data_type"] == "integer"
    }
    actual_text_columns = {row["column_name"] for row in columns if row["data_type"] == "text"}
    assert actual_integer_columns == ME_EXPECTED_INTEGER_COLUMNS
    assert actual_text_columns == ME_EXPECTED_TEXT_COLUMNS

    assert {
        row["column_name"] for row in columns if row["is_nullable"] == "YES"
    } == EXPECTED_NULLABLE_COLUMNS

    pk_cursor = await pg_kb.execute(
        "SELECT kcu.column_name FROM information_schema.table_constraints tc"
        " JOIN information_schema.key_column_usage kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "   AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = current_schema()"
        "   AND tc.table_name = 'map_eligibility_override'"
        "   AND tc.constraint_type = 'PRIMARY KEY'"
    )
    pk_rows = await pk_cursor.fetchall()
    assert [row["column_name"] for row in pk_rows] == [ME_EXPECTED_PRIMARY_KEY]

    idx_cursor = await pg_kb.execute(
        "SELECT indexname FROM pg_indexes"
        " WHERE schemaname = current_schema() AND tablename = 'map_eligibility_override'"
    )
    indexdefs = [row["indexname"] for row in await idx_cursor.fetchall()]
    assert len(indexdefs) == ME_EXPECTED_INDEX_COUNT


async def test_map_cluster_ledger_matches_expected_shape_on_postgres(
    pg_kb: PostgresBackend,
) -> None:
    """Introspect map_cluster_ledger on live Postgres and check its shape.

    Checks, via ``information_schema.columns``, ``information_schema.
    key_column_usage`` and ``pg_indexes``:

    * the exact column-name set (``MCL_EXPECTED_COLUMNS``);
    * ``sightings`` and ``declined_member_count`` are integer types and
      every other column is a text type;
    * ``cluster_key`` is the primary key;
    * the four ``declined_*`` columns are the only nullable columns;
    * exactly one index exists — the PK index (no secondary index on
      ``project_ref``: the table reaches a few hundred rows and the
      measured retune trigger for adding one is ~10,000 rows).

    No collation assertion: nothing in this feature orders or range-compares
    any column of this table, so no column carries a ``COLLATE "C"`` pin
    (unlike embedding_retry_queue.next_attempt_at).

    The ``MCL_`` aliases are a NAME COLLISION guard, not style: the import
    block above already imports ``EXPECTED_NULLABLE_COLUMNS`` from
    ``map_eligibility_override_shape`` UNALIASED and the map_eligibility
    parity test asserts against that bare name — a second unaliased import
    would shadow it and quietly re-point a currently-green assertion at
    this table's nullable set.
    """
    cursor = await pg_kb.execute(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns"
        " WHERE table_schema = current_schema() AND table_name = 'map_cluster_ledger'"
    )
    columns = await cursor.fetchall()

    actual_column_names = {row["column_name"] for row in columns}
    assert actual_column_names == MCL_EXPECTED_COLUMNS

    actual_integer_columns = {
        row["column_name"] for row in columns if row["data_type"] == "integer"
    }
    actual_text_columns = {row["column_name"] for row in columns if row["data_type"] == "text"}
    assert actual_integer_columns == MCL_EXPECTED_INTEGER_COLUMNS
    assert actual_text_columns == MCL_EXPECTED_TEXT_COLUMNS

    assert {
        row["column_name"] for row in columns if row["is_nullable"] == "YES"
    } == MCL_EXPECTED_NULLABLE_COLUMNS

    pk_cursor = await pg_kb.execute(
        "SELECT kcu.column_name FROM information_schema.table_constraints tc"
        " JOIN information_schema.key_column_usage kcu"
        "   ON tc.constraint_name = kcu.constraint_name"
        "   AND tc.table_schema = kcu.table_schema"
        " WHERE tc.table_schema = current_schema()"
        "   AND tc.table_name = 'map_cluster_ledger'"
        "   AND tc.constraint_type = 'PRIMARY KEY'"
    )
    pk_rows = await pk_cursor.fetchall()
    assert [row["column_name"] for row in pk_rows] == [MCL_EXPECTED_PRIMARY_KEY]

    idx_cursor = await pg_kb.execute(
        "SELECT indexname FROM pg_indexes"
        " WHERE schemaname = current_schema() AND tablename = 'map_cluster_ledger'"
    )
    indexdefs = [row["indexname"] for row in await idx_cursor.fetchall()]
    assert len(indexdefs) == MCL_EXPECTED_INDEX_COUNT


async def test_map_eligibility_counts_on_postgres(pg_kb: PostgresBackend) -> None:
    """The Postgres counts query returns the SAME EXPECTED_COUNTS as SQLite.

    Seeds the throwaway DB from the shared ``SEED_ENTRIES`` /
    ``SEED_INGESTED_FILES`` corpus and asserts
    ``await pg_kb.map_eligibility_counts()`` equals the shared
    ``EXPECTED_COUNTS`` — one expectation for both dialects, so the
    instr/substr vs split_part, json_each vs jsonb_array_elements_text, and
    COLLATE "C" tie-break divergences cannot drift apart silently.
    """
    for row in SEED_ENTRIES:
        await pg_kb.execute(
            "INSERT INTO knowledge_entries"
            " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
            " created_at, updated_at, is_active)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )
    for row in SEED_INGESTED_FILES:
        await pg_kb.execute(
            "INSERT INTO ingested_files"
            " (relative_path, content_hash, note_node_id, entry_ids, summary, file_size,"
            " file_extension, ingested_at, updated_at, is_active)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            row,
        )

    assert await pg_kb.map_eligibility_counts() == list(EXPECTED_COUNTS)
