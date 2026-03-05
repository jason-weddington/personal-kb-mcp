"""Extract full knowledge graph data for visualization."""

import json
import logging
from typing import Any

from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


async def extract_graph_data(db: Database) -> dict[str, Any]:
    """Extract all graph nodes, edges, and entry metadata for visualization.

    Returns a dict with keys: nodes, edges, stats.
    """
    # 1. All graph nodes with connection counts
    cursor = await db.execute(
        "SELECT n.node_id, n.node_type, n.properties, "
        "(SELECT COUNT(*) FROM graph_edges WHERE source = n.node_id"
        " OR target = n.node_id) AS conn_count "
        "FROM graph_nodes n"
    )
    raw_nodes = await cursor.fetchall()

    # 2. All graph edges
    cursor = await db.execute("SELECT source, target, edge_type, properties FROM graph_edges")
    raw_edges = await cursor.fetchall()

    # 3. Active entries for enrichment (entry nodes get metadata)
    cursor = await db.execute(
        "SELECT id, short_title, long_title, entry_type, tags, "
        "confidence_level, contributor, project_ref "
        "FROM knowledge_entries WHERE is_active = 1"
    )
    raw_entries = await cursor.fetchall()

    entry_meta: dict[str, dict[str, Any]] = {}
    for row in raw_entries:
        entry_meta[row[0]] = {
            "short_title": row[1],
            "long_title": row[2],
            "entry_type": row[3],
            "tags": row[4],
            "confidence_level": row[5],
            "contributor": row[6],
            "project_ref": row[7],
        }

    # Build node list
    nodes: list[dict[str, Any]] = []
    for row in raw_nodes:
        node_id: str = row[0]
        node_type: str = row[1]
        props_raw = row[2]
        conn_count: int = row[3]

        props = _parse_json(props_raw)

        # Label: strip type prefix for entity nodes, use short_title for entries
        if node_type == "entry" and node_id in entry_meta:
            meta = entry_meta[node_id]
            label = meta["short_title"] or node_id
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

    # Build edge list
    edges: list[dict[str, Any]] = []
    for row in raw_edges:
        props = _parse_json(row[3])
        edges.append(
            {
                "source": row[0],
                "target": row[1],
                "type": row[2],
                "properties": props,
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


def _parse_json(raw: str | None) -> dict[str, Any]:
    """Parse JSON string, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        result = json.loads(raw)
        return result if isinstance(result, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
