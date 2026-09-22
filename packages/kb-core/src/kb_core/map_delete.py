"""Hard-delete one mental_map and its graph edges — the primitive self-healing needs.

The nightly map loop can create maps and add pointers, but had no way to remove a bad one.
A map graph that can only grow is not self-healing, and a KB nobody can hand-repair rots.
The ruling this module implements (2026-09-21): a mental_map holds no facts.
It is a directory card pointing at the entries that hold them, so a bad map is DELETED.
``deactivate`` is for real knowledge entries, whose CONTENT is worth recovering.
There is nothing to recover in a map, and its edges must go with it.
Orphaning edges behind a soft delete is the code smell that made the old refusal look right.

The refusal this replaces: the service's ``_MAP_DEACTIVATE_BLOCKED`` refused maps outright.
Its reasoning was that deactivating a map stripped its outbound edges.
The answer is to delete the edges properly, not to forbid removal.
The HTTP endpoint is a separate item: the service resolves kb-core from ``rev = main``.
It cannot see feature-branch code, so the kb-core primitive lands first.

Column-type audit, read from ``kb_core.db.schema`` before any WHERE clause was written.
``knowledge_entries.id``, ``project_ref``, ``short_title`` and ``long_title`` are TEXT.
So are ``knowledge_details``, ``entry_type``, ``graph_edges.source`` and ``graph_edges.target``.
So are ``entry_versions.entry_id`` and ``audit_events.entry_id``.
The version rows are ``entry_versions`` rows keyed by ``entry_id``; there is no ``versions`` table.
The edge table is ``graph_edges``.
Both names were verified against the schema, not assumed from this item's prose.
Every predicate this module writes is equality matching on those TEXT columns.
No TEXT column is ordered or range-compared, so NO ``COLLATE "C"`` pin is needed.
The pin becomes required the day one of these columns is ordered or range-compared.
The right move then is ``_byte_ordered_text`` in ``map_caps.py``, never a second mechanism.
Both lists on the returned record (``pointer_ids``, ``inbound_referrer_ids``) are sorted in PYTHON.
No SQL ``ORDER BY`` means no collation dependency at all.

What one call deletes, in ONE transaction:
The ``knowledge_entries`` row.
Every ``graph_edges`` row where the map is the SOURCE — its outbound pointers.
Every ``graph_edges`` row where the map is the TARGET.
An edge to a nonexistent node is garbage, not a record.
The map's own ``entry_versions`` rows.
The map's ``graph_nodes`` row goes too.
The house hard-delete (``db/queries.py:delete_entry_cascade``) already treats it as entry-owned.
Leaving it behind would park a node that points at nothing.
The edges are deleted BEFORE the node row.
``graph_edges.source`` / ``graph_edges.target`` are REFERENCES-checked on Postgres.
The version rows go before the entry row for the same reason.

What one call NEVER deletes: the ``audit_events`` rows referring to the map.
Those are the durable record of what happened and the only thing that survives the map.
Deleting them would remove the evidence of the deletion itself.
Do not "complete" this cleanup later.
The map's ``knowledge_vec`` embedding row is also left alone.
``PostgresBackend.vector_delete`` takes its own pooled connection, so it escapes the transaction.
That would break the one-transaction guarantee above.
A vector whose entry row is gone can only surface as an id with no entry behind it.
It is never content, and the hybrid search path resolves ids against ``knowledge_entries``.

This module writes no audit event of its own: the caller can name who deleted the map and why.
The HTTP item owns that event's shape.
What it returns instead is the whole map — :class:`DeletedMapRecord`.
A caller can log a line that reconstructs it from logs alone.
That record is the reason deletion is survivable: ``inbound_referrer_ids`` is load-bearing.
Another map's BODY may contain this map's id as text.
Deleting the row leaves that text dangling.
The edge is gone with the row, but the text is not, and only the referrer list says where to look.
The caller needs it to repair those bodies, which is what makes the deletion recoverable.

All SQL uses ``?`` placeholders only; the Postgres backend rewrites them to ``$N`` at execute time.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from kb_core.map_lint import map_pointer_ids

if TYPE_CHECKING:
    from kb_core.db.backend import Cursor, Database

# Greppable marker, mirroring kb_core.cluster_ledger.CLUSTER_LEDGER_MARKER.
MAP_DELETE_MARKER = "map-delete"

# The only entry type this primitive will remove; see the module docstring for
# why a mental_map is deletable and a knowledge entry is not.
DELETABLE_ENTRY_TYPE = "mental_map"


@dataclasses.dataclass(frozen=True)
class DeletedMapRecord:
    """Everything needed to reconstruct one deleted map from logs alone.

    Deletion is the loop's only irreversible operation, so this record IS the
    recovery path: ``knowledge_details`` is the FULL body, never a summary,
    and ``pointer_ids`` is the id set the body carried, derived by
    :func:`kb_core.map_lint.map_pointer_ids` rather than a second regex.

    ``outbound_edges_deleted`` / ``inbound_edges_deleted`` are edge ROW
    counts, not distinct-neighbour counts: two edges to the same target are
    two rows, and only the ``UNIQUE(source, target, edge_type)`` constraint
    stops a corpus from holding them.

    ``inbound_referrer_ids`` is the list captured BEFORE the deletion — the
    ids that pointed AT this map. It is not a restatement of
    ``inbound_edges_deleted``: it is distinct SOURCES, and its value is that
    each referrer's BODY may still name this map's id as text. The row is
    gone, so that text now dangles even though the edge is gone with it, and
    the caller needs this list to find and repair those bodies.
    """

    entry_id: str
    project_ref: str | None
    short_title: str
    long_title: str
    knowledge_details: str
    pointer_ids: list[str]
    outbound_edges_deleted: int
    inbound_edges_deleted: int
    inbound_referrer_ids: list[str]


def _deleted_count(cursor: Cursor) -> int:
    """The row count of a DELETE, 0 if the backend reports no count.

    Both backends parse a DELETE status into a real count; this guard only
    normalizes the ``-1`` the protocol allows so a count can never surface as
    ``-1`` in a log line.
    """
    count = cursor.rowcount
    return count if count > 0 else 0


async def delete_map(db: Database, entry_id: str) -> DeletedMapRecord:
    """Hard-delete one ``mental_map`` with its edges and version rows.

    Refuses anything else. A ``factual_reference`` — or any other entry type
    — raises ``ValueError`` naming the ACTUAL type: hard-deleting a real
    knowledge entry must stay impossible through this path, because
    ``deactivate`` exists precisely so its content can be recovered, and a
    map holds nothing to recover.

    An id that matches no row raises ``ValueError`` naming the id, never a
    silent no-op: a silent success would make a caller's retry
    indistinguishable from a caller's typo.

    Everything the deletion removes goes in ONE transaction — the entry row,
    the outbound edges, the inbound edges, the map's ``graph_nodes`` row and
    its ``entry_versions`` rows — so a failure mid-delete leaves the map
    graph exactly as it was. See the module docstring for the two things it
    deliberately does NOT remove: ``audit_events`` rows (the durable record)
    and the ``knowledge_vec`` embedding row (it cannot join the transaction).

    All SQL here is equality matching on TEXT columns; no ordering or range
    comparison appears, so no ``COLLATE "C"`` pin is needed. Both lists on
    the returned record are sorted in Python, so no backend's default
    collation can make the two legs disagree.

    Args:
        db: Any :class:`kb_core.db.backend.Database` — SQLite or Postgres.
        entry_id: The ``knowledge_entries.id`` of the map to delete.

    Returns:
        A :class:`DeletedMapRecord` capturing the map's titles, its FULL
        body, its pointer ids, both edge counts and the ids that pointed at
        it — everything a caller needs to log a line that reconstructs the
        map.

    Raises:
        ValueError: If ``entry_id`` matches no row, or matches a row whose
            ``entry_type`` is not ``mental_map``.
    """
    cursor = await db.execute(
        "SELECT project_ref, short_title, long_title, knowledge_details, entry_type"
        " FROM knowledge_entries WHERE id = ?",
        (entry_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        msg = f"{MAP_DELETE_MARKER}: no entry with id {entry_id!r} — refusing to silently no-op"
        raise ValueError(msg)

    entry_type = str(row["entry_type"])
    if entry_type != DELETABLE_ENTRY_TYPE:
        msg = (
            f"{MAP_DELETE_MARKER}: {entry_id!r} is a {entry_type}, not a {DELETABLE_ENTRY_TYPE}"
            " — hard delete is for maps; deactivate a knowledge entry instead"
        )
        raise ValueError(msg)

    short_title = str(row["short_title"])
    long_title = str(row["long_title"])
    knowledge_details = str(row["knowledge_details"])
    project_ref = None if row["project_ref"] is None else str(row["project_ref"])
    pointer_ids = sorted(map_pointer_ids(knowledge_details))

    async with db.transaction():
        # Captured BEFORE the deletes and inside the transaction, so the list
        # cannot drift from the rows that are about to go.
        referrer_cursor = await db.execute(
            "SELECT DISTINCT source FROM graph_edges WHERE target = ?",
            (entry_id,),
        )
        referrer_rows = await referrer_cursor.fetchall()
        inbound_referrer_ids = sorted({str(r["source"]) for r in referrer_rows})

        # Outbound first, then inbound: an edge can be neither double-counted
        # nor missed, and both statements must run before the node row goes.
        outbound_cursor = await db.execute(
            "DELETE FROM graph_edges WHERE source = ?",
            (entry_id,),
        )
        outbound_edges_deleted = _deleted_count(outbound_cursor)

        inbound_cursor = await db.execute(
            "DELETE FROM graph_edges WHERE target = ?",
            (entry_id,),
        )
        inbound_edges_deleted = _deleted_count(inbound_cursor)

        await db.execute("DELETE FROM graph_nodes WHERE node_id = ?", (entry_id,))
        # Version rows before the entry row: entry_versions.entry_id carries a
        # REFERENCES clause Postgres enforces against knowledge_entries.id.
        await db.execute("DELETE FROM entry_versions WHERE entry_id = ?", (entry_id,))
        await db.execute("DELETE FROM knowledge_entries WHERE id = ?", (entry_id,))

    return DeletedMapRecord(
        entry_id=entry_id,
        project_ref=project_ref,
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        pointer_ids=pointer_ids,
        outbound_edges_deleted=outbound_edges_deleted,
        inbound_edges_deleted=inbound_edges_deleted,
        inbound_referrer_ids=inbound_referrer_ids,
    )
