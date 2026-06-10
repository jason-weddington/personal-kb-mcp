"""Graph visualisation data export for the full-graph explorer endpoint.

``extract_graph_data`` runs four SQL queries against the kb-core data DB and
applies the same filter rules as the old personal_kb explorer:

- Inactive entry nodes are excluded.
- Non-entry nodes with no edges (orphans) are excluded.
- Edges touching any excluded node are excluded.
"""

import json
from typing import Any

from kb_core.db.backend import Database


def _parse_json(value: Any) -> dict[str, Any]:
    """Parse a JSON field from a database row into a plain dict.

    Returns ``{}`` on ``None``, empty / falsy input, non-dict result,
    ``JSONDecodeError``, or ``TypeError`` — never raises.

    Args:
        value: Raw value from the database (typically a JSON string or None).

    Returns:
        Parsed dict, or ``{}`` on any failure.
    """
    if not value:
        return {}
    try:
        result = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(result, dict):
        return {}
    return result


async def extract_graph_data(db: Database) -> dict[str, Any]:
    """Extract full graph data for visualisation.

    Runs four SQL queries in sequence and applies filter rules to produce a
    node/edge list suitable for a force-directed graph renderer.

    SQL queries (verbatim):

    1. ``SELECT id FROM knowledge_entries WHERE is_active = 0`` —
       inactive entry IDs to exclude.
    2. ``SELECT n.node_id, n.node_type, n.properties, … AS conn_count
       FROM graph_nodes n`` — all nodes with connection counts.
    3. ``SELECT source, target, edge_type, properties FROM graph_edges`` —
       all edges.
    4. ``SELECT id, short_title, … FROM knowledge_entries WHERE is_active = 1``
       — metadata for active entries.

    Args:
        db: The kb-core ``Database`` instance (``KnowledgeBase.db``).

    Returns:
        Dict with keys ``nodes`` (list), ``edges`` (list), and ``stats``
        (``{node_count, edge_count}``).
    """
    # Query 1 — inactive entry IDs
    cursor = await db.execute("SELECT id FROM knowledge_entries WHERE is_active = 0")
    inactive_rows = await cursor.fetchall()
    inactive_ids: set[str] = {row[0] for row in inactive_rows}

    # Query 2 — all graph nodes with connection counts
    cursor = await db.execute(
        "SELECT n.node_id, n.node_type, n.properties,"
        " (SELECT COUNT(*) FROM graph_edges"
        " WHERE source = n.node_id OR target = n.node_id) AS conn_count"
        " FROM graph_nodes n"
    )
    node_rows = await cursor.fetchall()

    # Query 3 — all graph edges
    cursor = await db.execute(
        "SELECT source, target, edge_type, properties FROM graph_edges"
    )
    edge_rows = await cursor.fetchall()

    # Query 4 — active entry metadata
    cursor = await db.execute(
        "SELECT id, short_title, long_title, entry_type, tags,"
        " confidence_level, contributor, project_ref"
        " FROM knowledge_entries WHERE is_active = 1"
    )
    entry_rows = await cursor.fetchall()

    # Build entry-metadata lookup (active entries only)
    entry_meta: dict[str, dict[str, Any]] = {}
    for row in entry_rows:
        entry_meta[row[0]] = {
            "short_title": row[1],
            "long_title": row[2],
            "entry_type": row[3],
            "tags": row[4],
            "confidence_level": row[5],
            "contributor": row[6],
            "project_ref": row[7],
        }

    # Build node list applying filter rules
    nodes: list[dict[str, Any]] = []
    included_ids: set[str] = set()

    for row in node_rows:
        node_id: str = row[0]
        node_type: str = row[1]
        properties_raw: Any = row[2]
        conn_count: int = row[3]

        # Skip inactive entry nodes
        if node_type == "entry" and node_id in inactive_ids:
            continue

        # Skip orphan non-entry nodes (no connections at all)
        if node_type != "entry" and conn_count == 0:
            continue

        props = _parse_json(properties_raw)

        if node_type == "entry" and node_id in entry_meta:
            meta = entry_meta[node_id]
            label: str = meta["short_title"] or node_id
            props.update(meta)
        elif ":" in node_id:
            label = node_id.split(":", 1)[1]
        else:
            label = node_id

        nodes.append(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "val": max(conn_count, 1),
                "properties": props,
            }
        )
        included_ids.add(node_id)

    # Build edge list, skipping edges to/from excluded nodes
    edges: list[dict[str, Any]] = []
    for row in edge_rows:
        source: str = row[0]
        target: str = row[1]
        edge_type: str = row[2]
        edge_props_raw: Any = row[3]

        if source not in included_ids or target not in included_ids:
            continue

        edges.append(
            {
                "source": source,
                "target": target,
                "type": edge_type,
                "properties": _parse_json(edge_props_raw),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }
