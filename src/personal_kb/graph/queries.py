"""Graph traversal queries for kb_ask."""

import logging
import re
from collections import deque

from personal_kb.db.backend import Database
from personal_kb.models.entry import EntryType

logger = logging.getLogger(__name__)

_KB_ID_RE = re.compile(r"^kb-\d{5}$")
_ENTRY_TYPES = {t.value for t in EntryType}


async def get_neighbors(
    db: Database,
    node_id: str,
    edge_types: list[str] | None = None,
    direction: str = "both",
    limit: int = 50,
) -> list[tuple[str, str, str]]:
    """Get neighbors of a node.

    Returns list of (neighbor_id, edge_type, direction) tuples.
    direction is "outgoing" or "incoming" indicating the edge direction.
    """
    results: list[tuple[str, str, str]] = []

    if direction in ("both", "outgoing"):
        query = "SELECT target, edge_type FROM graph_edges WHERE source = ?"
        params: list[object] = [node_id]
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            query += f" AND edge_type IN ({placeholders})"
            params.extend(edge_types)
        query += " LIMIT ?"
        params.append(limit)
        cursor = await db.execute(query, params)
        for row in await cursor.fetchall():
            results.append((row[0], row[1], "outgoing"))

    if direction in ("both", "incoming"):
        remaining = limit - len(results)
        if remaining <= 0:
            return results
        query = "SELECT source, edge_type FROM graph_edges WHERE target = ?"
        params = [node_id]
        if edge_types:
            placeholders = ",".join("?" for _ in edge_types)
            query += f" AND edge_type IN ({placeholders})"
            params.extend(edge_types)
        query += " LIMIT ?"
        params.append(remaining)
        cursor = await db.execute(query, params)
        for row in await cursor.fetchall():
            results.append((row[0], row[1], "incoming"))

    return results


async def bfs_entries(
    db: Database,
    start_node: str,
    max_depth: int = 2,
    edge_types: list[str] | None = None,
    limit: int = 20,
) -> list[tuple[str, int, list[str]]]:
    """BFS from start_node, collecting entry nodes reached.

    Returns list of (entry_id, depth, path) tuples, sorted by depth.
    path is the list of node IDs from start to entry (inclusive).
    """
    visited: set[str] = {start_node}
    # (node_id, depth, path)
    queue: deque[tuple[str, int, list[str]]] = deque([(start_node, 0, [start_node])])
    results: list[tuple[str, int, list[str]]] = []

    while queue and len(results) < limit:
        node, depth, path = queue.popleft()

        # Collect entry nodes (but not the start node itself)
        if depth > 0 and _KB_ID_RE.match(node):
            results.append((node, depth, path))
            if len(results) >= limit:
                break

        if depth >= max_depth:
            continue

        neighbors = await get_neighbors(db, node, edge_types=edge_types)
        for neighbor_id, _edge_type, _direction in neighbors:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, depth + 1, [*path, neighbor_id]))

    return results


async def find_path(
    db: Database,
    source: str,
    target: str,
    max_depth: int = 4,
) -> list[tuple[str, str, str]] | None:
    """Find shortest path between two nodes via BFS.

    Returns list of (node, edge_type, next_node) triples forming the path,
    or None if no path exists within max_depth.
    """
    if source == target:
        return []

    visited: set[str] = {source}
    # (current_node, path_of_triples)
    queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque([(source, [])])

    while queue:
        node, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        neighbors = await get_neighbors(db, node)
        for neighbor_id, edge_type, direction in neighbors:
            if neighbor_id in visited:
                continue
            visited.add(neighbor_id)

            if direction == "outgoing":
                step = (node, edge_type, neighbor_id)
            else:
                step = (neighbor_id, edge_type, node)

            new_path = [*path, step]

            if neighbor_id == target:
                return new_path

            queue.append((neighbor_id, new_path))

    return None


def _parse_scope(scope: str) -> tuple[str, str]:
    """Parse a scope string into (scope_type, value).

    Supported formats:
    - "project:X" -> ("project", "X")
    - "tag:X" -> ("tag", "X")
    - "person:X" -> ("person", "X")
    - "tool:X" -> ("tool", "X")
    - "kb-XXXXX" -> ("entry", "kb-XXXXX")
    - "decision", "factual_reference", etc -> ("entry_type", value)
    """
    if _KB_ID_RE.match(scope):
        return ("entry", scope)

    for prefix in ("project:", "tag:", "person:", "tool:"):
        if scope.startswith(prefix):
            kind = prefix.rstrip(":")
            value = scope[len(prefix) :]
            return (kind, value)

    if scope in _ENTRY_TYPES:
        return ("entry_type", scope)

    # Fall back to treating it as a generic node ID
    return ("node", scope)


async def entries_for_scope(
    db: Database,
    scope: str,
    entry_type: str | None = None,
    order_by: str = "created_at",
) -> list[str]:
    """Get entry IDs matching a scope string.

    Interprets scope as project, tag, person, tool, entry ID, or entry type.
    """
    scope_type, value = _parse_scope(scope)

    if scope_type == "entry":
        return [value]

    if scope_type == "entry_type":
        query = "SELECT id FROM knowledge_entries WHERE entry_type = ? AND is_active = 1"
        params: list[object] = [value]
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
        query += f" ORDER BY {_safe_order(order_by)}"
        cursor = await db.execute(query, params)
        return [row[0] for row in await cursor.fetchall()]

    if scope_type == "project":
        query = "SELECT id FROM knowledge_entries WHERE project_ref = ? AND is_active = 1"
        params = [value]
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
        query += f" ORDER BY {_safe_order(order_by)}"
        cursor = await db.execute(query, params)
        return [row[0] for row in await cursor.fetchall()]

    # For tag/person/tool: find entries via graph edges
    if scope_type == "tag":
        node_id = f"tag:{value}"
        edge_type = "has_tag"
    elif scope_type == "person":
        node_id = f"person:{value}"
        edge_type = "mentions_person"
    elif scope_type == "tool":
        node_id = f"tool:{value}"
        edge_type = "uses_tool"
    else:
        # Generic node — find connected entries
        node_id = value
        edge_type = None

    query = "SELECT source FROM graph_edges WHERE target = ?"
    params = [node_id]
    if edge_type:
        query += " AND edge_type = ?"
        params.append(edge_type)
    cursor = await db.execute(query, params)
    entry_ids = [row[0] for row in await cursor.fetchall() if _KB_ID_RE.match(row[0])]

    # Sort by created_at if we have entries
    if entry_ids and order_by:
        entry_ids = await _sort_entries(db, entry_ids, entry_type=entry_type, order_by=order_by)

    return entry_ids


async def supersedes_chain(
    db: Database,
    entry_id: str,
) -> list[str]:
    """Build the full supersedes chain containing entry_id, oldest first.

    Follows supersedes edges in both directions to find the complete chain.
    """
    chain_set: set[str] = {entry_id}
    chain_ordered: list[str] = [entry_id]

    # Walk backwards (find what this entry supersedes)
    current = entry_id
    while True:
        cursor = await db.execute(
            "SELECT target FROM graph_edges WHERE source = ? AND edge_type = 'supersedes'",
            (current,),
        )
        row = await cursor.fetchone()
        if not row:
            break
        target = row[0]
        if target in chain_set:
            break
        chain_set.add(target)
        chain_ordered.insert(0, target)
        current = target

    # Walk forwards (find what supersedes this entry)
    current = entry_id
    while True:
        cursor = await db.execute(
            "SELECT source FROM graph_edges WHERE target = ? AND edge_type = 'supersedes'",
            (current,),
        )
        row = await cursor.fetchone()
        if not row:
            break
        source = row[0]
        if source in chain_set:
            break
        chain_set.add(source)
        chain_ordered.append(source)
        current = source

    return chain_ordered


async def get_graph_vocabulary(
    db: Database,
    max_nodes: int = 200,
) -> dict[str, list[str]]:
    """Return non-entry node IDs grouped by type, ordered by connection count.

    Returns a dict like {"tag": ["python", "sqlite"], "tool": ["aiosqlite"], ...}
    with node names stripped of their type prefix. Capped at max_nodes total.
    """
    # Count connections per non-entry node, ordered by most connected first
    query = (
        "SELECT n.node_id, n.node_type, "
        "(SELECT COUNT(*) FROM graph_edges WHERE source = n.node_id"
        " OR target = n.node_id) AS conn_count "
        "FROM graph_nodes n "
        "WHERE n.node_type != 'entry' "
        "ORDER BY conn_count DESC "
        "LIMIT ?"
    )
    cursor = await db.execute(query, (max_nodes,))
    rows = await cursor.fetchall()

    vocab: dict[str, list[str]] = {}
    for row in rows:
        node_id: str = row[0]
        node_type: str = row[1]
        # Strip type prefix (e.g., "tag:python" -> "python")
        prefix = node_type + ":"
        name = node_id[len(prefix) :] if node_id.startswith(prefix) else node_id
        vocab.setdefault(node_type, []).append(name)

    return vocab


async def _sort_entries(
    db: Database,
    entry_ids: list[str],
    entry_type: str | None = None,
    order_by: str = "created_at",
) -> list[str]:
    """Sort entry IDs by a column, optionally filtering by type."""
    placeholders = ",".join("?" for _ in entry_ids)
    params: list[object] = list(entry_ids)
    # Build query — placeholders are safe (only ? characters)
    base = "SELECT id FROM knowledge_entries WHERE id IN ("
    query = base + placeholders + ") AND is_active = 1"
    if entry_type:
        query += " AND entry_type = ?"
        params.append(entry_type)
    query += f" ORDER BY {_safe_order(order_by)}"
    cursor = await db.execute(query, params)
    return [row[0] for row in await cursor.fetchall()]


def _safe_order(order_by: str) -> str:
    """Whitelist order by column to prevent SQL injection."""
    allowed = {"created_at", "updated_at", "confidence_level", "short_title"}
    if order_by in allowed:
        return order_by
    return "created_at"
