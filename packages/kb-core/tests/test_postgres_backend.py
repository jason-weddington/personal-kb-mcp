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
``properties::jsonb->>`` in :meth:`PostgresBackend.delete_llm_edges` is
the ONLY jsonb-operator usage in ``postgres_backend.py``, so the
delete_llm_edges regression is the sole json-operator regression case
this file needs to guard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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
