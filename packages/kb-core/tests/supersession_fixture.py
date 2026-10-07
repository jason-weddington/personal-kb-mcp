"""Shared seed corpus + expected values for the supersession reconcile tests.

`kb_core.supersession.reconcile_supersession` runs on BOTH backends.
The SQLite suite is `test_supersession.py`; the Postgres one is `test_postgres_backend.py`.
Both seed THIS corpus and both assert the SAME expected superseded_by map and counts.

The pairs, one per rule the invariant names:

* S1 -> T1 and S1b -> T1 share one created_at; S1b has the higher id, so the
  id-descending tie-break resolves T1 to S1b.
* S2 carries hints.supersedes=[T2] but NO edge — the kb-00299 -> kb-00295
  shape. The reconcile backfills exactly this one edge.
* S3 -> T3 where T3 is inactive: the invariant holds for inactive targets too.
* mental_map M -> T4 (edge inserted directly, plus hints.supersedes=[T4]):
  a map never qualifies as a superseder, and the backfill skips map rows.
* S5 -> T5 whose edge properties carry {"source": "llm"}: not a qualifying edge.

Rows are seeded via direct SQL INSERTs, never `kb.store()`.
`graph_nodes` rows are seeded for every entry, because Postgres enforces the
`graph_edges` REFERENCES clauses.
This module is NOT a test module and carries no assertions.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kb_core.db.backend import Database

TS_OLD = "2026-03-01T12:00:00+00:00"
TS_NEW = "2026-03-02T12:00:00+00:00"

T1, S1, S1B = "kb-80001", "kb-80011", "kb-80012"
T2, S2 = "kb-80002", "kb-80021"
T3, S3 = "kb-80003", "kb-80031"
T4, MAP_M = "kb-80004", "kb-80041"
T5, S5 = "kb-80005", "kb-80051"

# (id, entry_type, hints, created_at, is_active)
_ROWS: tuple[tuple[str, str, dict[str, object], str, int], ...] = (
    (T1, "factual_reference", {}, TS_OLD, 1),
    (T2, "factual_reference", {}, TS_OLD, 1),
    (T3, "factual_reference", {}, TS_OLD, 0),
    (T4, "factual_reference", {}, TS_OLD, 1),
    (T5, "factual_reference", {}, TS_OLD, 1),
    (S1, "decision", {"supersedes": [T1]}, TS_NEW, 1),
    (S1B, "decision", {"supersedes": [T1]}, TS_NEW, 1),
    (S2, "decision", {"supersedes": [T2]}, TS_NEW, 1),
    (S3, "decision", {"supersedes": [T3]}, TS_NEW, 1),
    (MAP_M, "mental_map", {"supersedes": [T4]}, TS_NEW, 1),
    (S5, "decision", {}, TS_NEW, 1),
)

# (source, target, properties JSON)
_EDGES: tuple[tuple[str, str, str], ...] = (
    (S1, T1, "{}"),
    (S1B, T1, "{}"),
    (S3, T3, "{}"),
    (MAP_M, T4, "{}"),
    (S5, T5, json.dumps({"source": "llm"})),
)

EXPECTED_SUPERSEDED_BY: dict[str, str | None] = {
    T1: S1B,
    T2: S2,
    T3: S3,
    T4: None,
    T5: None,
}
EXPECTED_EDGES_ADDED = 1
EXPECTED_EDGES_ADDED_IDS = ((S2, T2),)
EXPECTED_SET_COUNT = 3
EXPECTED_CLEARED_COUNT = 0

_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries"
    " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
    " hints, created_at, updated_at, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)
_NODE_INSERT_SQL = (
    "INSERT INTO graph_nodes (node_id, node_type, properties, created_at) VALUES (?, ?, ?, ?)"
)
_EDGE_INSERT_SQL = (
    "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
    " VALUES (?, ?, 'supersedes', ?, ?)"
)


async def seed_corpus(db: Database) -> None:
    """Insert the supersession corpus (entries, nodes, edges)."""
    for entry_id, entry_type, hints, created_at, is_active in _ROWS:
        await db.execute(
            _ENTRY_INSERT_SQL,
            (
                entry_id,
                "p-sup",
                f"title {entry_id}",
                f"long title {entry_id}",
                f"details for {entry_id}",
                entry_type,
                json.dumps(hints),
                created_at,
                created_at,
                is_active,
            ),
        )
        await db.execute(_NODE_INSERT_SQL, (entry_id, "entry", "{}", created_at))
    for source, target, props in _EDGES:
        await db.execute(_EDGE_INSERT_SQL, (source, target, props, TS_NEW))
    await db.commit()


async def superseded_by_map(db: Database) -> dict[str, str | None]:
    """Read ``superseded_by`` for every target the corpus names."""
    out: dict[str, str | None] = {}
    for target in EXPECTED_SUPERSEDED_BY:
        cursor = await db.execute(
            "SELECT superseded_by FROM knowledge_entries WHERE id = ?", (target,)
        )
        row = await cursor.fetchone()
        out[target] = row["superseded_by"] if row is not None else None
    return out
