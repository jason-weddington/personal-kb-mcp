"""Shared seed corpus + expected record for the map hard-delete tests.

`kb_core.map_delete.delete_map` runs on BOTH backends.
The only dialect question it raises is whether its statements parse at all.
Every predicate is equality matching, so no collation pin appears on either leg.
The SQLite suite is `test_map_delete.py`; the Postgres one is `test_postgres_backend.py`.
Both seed THIS corpus and both assert the SAME `EXPECTED_RECORD`.
Both also assert the SAME leftover rows, so a dialect drift fails on whichever side runs.
On a host with no `KB_TEST_DATABASE_URL` the Postgres leg ships unexercised.
This module is what makes the drift detectable wherever Postgres DOES run.

The corpus is built around one map, `kb-70001`, with exactly three outbound edges.
It has exactly one inbound edge — the two counts the acceptance criteria name.
The inbound edge's source, `kb-70002`, is a SURVIVOR map whose body names `kb-70001` in prose.
That is the dangling-text case `inbound_referrer_ids` exists to repair.
`kb-70003` is the bare map: a body with pointers, but no seeded edges in either direction.
Both of its edge counts come back zero while its `pointer_ids` is non-empty.
`kb-70021` is a `factual_reference`, the entry type a hard delete must refuse.
`kb-70011 -> kb-70012` is an unrelated edge between two survivors.
The deletion must not touch it.

Rows are seeded via direct SQL INSERTs, never `kb.store()`.
The store touches the embedder, and these tests are about DELETE statements.
`graph_nodes` rows are seeded for every entry.
`graph_edges.source` / `graph_edges.target` carry REFERENCES clauses that Postgres enforces.
This module is NOT a test module itself and is not matched by pytest's collection glob.
It carries no assertions — pure data plus the seed/read helpers both suites share.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kb_core.map_delete import DeletedMapRecord

if TYPE_CHECKING:
    from kb_core.db.backend import Database

# One instant for every row — nothing here is about time.
TS = "2026-03-01T12:00:00+00:00"

# The map under deletion.
MAP_ID = "kb-70001"

# The survivor map whose body names MAP_ID in prose and whose edge points at it.
REFERRER_MAP_ID = "kb-70002"

# The bare map: pointers in its body, but zero edges in either direction.
BARE_MAP_ID = "kb-70003"

# A factual entry — the type a hard delete must refuse.
FACTUAL_ID = "kb-70021"

# The three detail entries the deleted map pointed at, in ascending id order.
POINTER_IDS = ("kb-70011", "kb-70012", "kb-70013")

MAP_SHORT_TITLE = "VPN orientation"
MAP_LONG_TITLE = "Where the VPN and DNS knowledge lives"

# The FULL body, exactly as stored and exactly as the record must return it.
MAP_BODY = (
    "Start with the tunnel setup in kb-70011, then the split-DNS notes in kb-70012."
    " The kill-switch failure mode is written up in kb-70013."
)

REFERRER_BODY = "The tunnel map is kb-70001; the split-DNS detail it points at is kb-70012."

BARE_MAP_BODY = "The split-DNS detail is kb-70011; this map has no graph edges yet."

# The edge counts the acceptance criteria name for MAP_ID.
OUTBOUND_EDGE_COUNT = 3
INBOUND_EDGE_COUNT = 1

# The ids that pointed AT the deleted map, captured before the deletion.
INBOUND_REFERRER_IDS = (REFERRER_MAP_ID,)

# The record both backends must return for MAP_ID.
EXPECTED_RECORD = DeletedMapRecord(
    entry_id=MAP_ID,
    project_ref="p-vpn",
    short_title=MAP_SHORT_TITLE,
    long_title=MAP_LONG_TITLE,
    knowledge_details=MAP_BODY,
    pointer_ids=list(POINTER_IDS),
    outbound_edges_deleted=OUTBOUND_EDGE_COUNT,
    inbound_edges_deleted=INBOUND_EDGE_COUNT,
    inbound_referrer_ids=list(INBOUND_REFERRER_IDS),
)

# What the corpus must look like AFTER deleting MAP_ID.
EXPECTED_SURVIVING_ENTRY_IDS = (
    "kb-70002",
    "kb-70003",
    "kb-70011",
    "kb-70012",
    "kb-70013",
    "kb-70021",
)
EXPECTED_SURVIVING_EDGES = (("kb-70011", "kb-70012", "related_to"),)
EXPECTED_SURVIVING_VERSIONS = (("kb-70002", 1),)
# The audit rows referring to the deleted map are the ones that must REMAIN.
EXPECTED_MAP_AUDIT_EVENT_TYPES = ("map_created", "map_pointer_added")

# The ten knowledge_entries columns both suites' INSERT lists name.
_ENTRY_INSERT_SQL = (
    "INSERT INTO knowledge_entries"
    " (id, project_ref, short_title, long_title, knowledge_details, entry_type,"
    " contributor, created_at, updated_at, is_active)"
    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)

_NODE_INSERT_SQL = (
    "INSERT INTO graph_nodes (node_id, node_type, properties, created_at) VALUES (?, ?, ?, ?)"
)

_EDGE_INSERT_SQL = (
    "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
    " VALUES (?, ?, ?, ?, ?)"
)

_VERSION_INSERT_SQL = (
    "INSERT INTO entry_versions"
    " (entry_id, version_number, knowledge_details, change_reason, confidence_level, created_at)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)

_AUDIT_INSERT_SQL = (
    "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
    " VALUES (?, ?, ?, ?, ?)"
)

# Rows in the order the INSERT lists above bind them.
SEED_ENTRIES = (
    (MAP_ID, "p-vpn", MAP_SHORT_TITLE, MAP_LONG_TITLE, MAP_BODY, "mental_map", "somnus", TS, TS, 1),
    (
        REFERRER_MAP_ID,
        "p-vpn",
        "Referrer map",
        "The map that points back at the VPN map",
        REFERRER_BODY,
        "mental_map",
        "somnus",
        TS,
        TS,
        1,
    ),
    (
        BARE_MAP_ID,
        "p-dns",
        "Bare map",
        "The map with no graph edges",
        BARE_MAP_BODY,
        "mental_map",
        "somnus",
        TS,
        TS,
        1,
    ),
    (
        "kb-70011",
        "p-vpn",
        "Tunnel setup",
        "How the tunnel is brought up",
        "the tunnel setup detail",
        "factual_reference",
        "jason",
        TS,
        TS,
        1,
    ),
    (
        "kb-70012",
        "p-vpn",
        "Split DNS",
        "How split DNS is resolved",
        "the split-DNS detail",
        "factual_reference",
        "jason",
        TS,
        TS,
        1,
    ),
    (
        "kb-70013",
        "p-vpn",
        "Kill switch",
        "The kill-switch failure mode",
        "the kill-switch detail",
        "factual_reference",
        "jason",
        TS,
        TS,
        1,
    ),
    (
        FACTUAL_ID,
        "p-vpn",
        "Tunnel MTU",
        "The measured MTU that fits the tunnel",
        "mtu 1380",
        "factual_reference",
        "jason",
        TS,
        TS,
        1,
    ),
)

# graph_nodes rows for every entry id — the FKs graph_edges carries require them.
SEED_NODES = tuple((entry_id, "entry", "{}", TS) for entry_id, *_ in SEED_ENTRIES)

# Three outbound edges, one inbound edge, one unrelated survivor edge.
SEED_EDGES = (
    (MAP_ID, "kb-70011", "references", "{}", TS),
    (MAP_ID, "kb-70012", "references", "{}", TS),
    (MAP_ID, "kb-70013", "references", "{}", TS),
    (REFERRER_MAP_ID, MAP_ID, "references", "{}", TS),
    ("kb-70011", "kb-70012", "related_to", "{}", TS),
)

# Two version rows on the deleted map, one on a survivor — only the former may go.
SEED_VERSIONS = (
    (MAP_ID, 1, MAP_BODY, "initial write", 0.9, TS),
    (MAP_ID, 2, MAP_BODY, "pointer added", 0.9, TS),
    (REFERRER_MAP_ID, 1, REFERRER_BODY, "initial write", 0.9, TS),
)

# Audit rows for the deleted map plus one for a survivor.
SEED_AUDIT_EVENTS = (
    ("map_created", MAP_ID, "somnus", "created by the nightly loop", TS),
    ("map_pointer_added", MAP_ID, "somnus", "kb-70013", TS),
    ("map_created", REFERRER_MAP_ID, "somnus", "created by the nightly loop", TS),
)


async def seed_corpus(db: Database) -> None:
    """Insert the whole corpus, entries first (the versions table carries an FK to it)."""
    for row in SEED_ENTRIES:
        await db.execute(_ENTRY_INSERT_SQL, row)
    for row in SEED_NODES:
        await db.execute(_NODE_INSERT_SQL, row)
    for row in SEED_EDGES:
        await db.execute(_EDGE_INSERT_SQL, row)
    for row in SEED_VERSIONS:
        await db.execute(_VERSION_INSERT_SQL, row)
    for row in SEED_AUDIT_EVENTS:
        await db.execute(_AUDIT_INSERT_SQL, row)
    await db.commit()


async def entry_ids(db: Database) -> list[str]:
    """Every knowledge_entries id, sorted in Python so the order is backend-independent."""
    cursor = await db.execute("SELECT id FROM knowledge_entries")
    rows = await cursor.fetchall()
    return sorted(str(row["id"]) for row in rows)


async def edge_rows(db: Database) -> list[tuple[str, str, str]]:
    """Every (source, target, edge_type), sorted in Python — no SQL ORDER BY anywhere."""
    cursor = await db.execute("SELECT source, target, edge_type FROM graph_edges")
    rows = await cursor.fetchall()
    return sorted((str(r["source"]), str(r["target"]), str(r["edge_type"])) for r in rows)


async def version_rows(db: Database) -> list[tuple[str, int]]:
    """Every (entry_id, version_number), sorted in Python."""
    cursor = await db.execute("SELECT entry_id, version_number FROM entry_versions")
    rows = await cursor.fetchall()
    return sorted((str(r["entry_id"]), int(r["version_number"])) for r in rows)


async def node_ids(db: Database) -> list[str]:
    """Every graph_nodes node_id, sorted in Python."""
    cursor = await db.execute("SELECT node_id FROM graph_nodes")
    rows = await cursor.fetchall()
    return sorted(str(row["node_id"]) for row in rows)


async def audit_event_types_for(db: Database, entry_id: str) -> list[str]:
    """Every audit_events event_type for one entry, in insertion order."""
    cursor = await db.execute(
        "SELECT event_type FROM audit_events WHERE entry_id = ? ORDER BY id", (entry_id,)
    )
    rows = await cursor.fetchall()
    return [str(row["event_type"]) for row in rows]
