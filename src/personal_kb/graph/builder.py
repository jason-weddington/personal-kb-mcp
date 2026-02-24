"""Build knowledge graph edges from entry data."""

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime

import aiosqlite

from personal_kb.models.entry import KnowledgeEntry

logger = logging.getLogger(__name__)

_KB_ID_RE = re.compile(r"kb-\d{5}")


class GraphBuilder:
    """Deterministic graph builder that derives nodes and edges from entry data."""

    def __init__(self, db: aiosqlite.Connection) -> None:
        """Initialize with an aiosqlite connection."""
        self._db = db

    async def build_for_entry(self, entry: KnowledgeEntry) -> None:
        """Rebuild all outgoing graph edges for an entry.

        Deletes existing outgoing edges, then re-derives nodes and edges
        from the entry's tags, project_ref, hints, and text references.
        """
        await self._clear_edges_for_source(entry.id)

        # 1. Upsert entry node
        props = {"short_title": entry.short_title, "entry_type": entry.entry_type.value}
        await self._ensure_node(entry.id, "entry", props)

        # 2. Tags → tag nodes + has_tag edges
        for tag in entry.tags:
            node_id = f"tag:{tag}"
            await self._ensure_node(node_id, "tag")
            await self._add_edge(entry.id, node_id, "has_tag")

        # 3. Project → project node + in_project edge
        if entry.project_ref:
            node_id = f"project:{entry.project_ref}"
            await self._ensure_node(node_id, "project")
            await self._add_edge(entry.id, node_id, "in_project")

        hints = entry.hints or {}

        # 4. Supersedes (from hints)
        for target in _as_list(hints.get("supersedes")):
            if isinstance(target, str) and target:
                await self._ensure_node(target, "entry")
                await self._add_edge(entry.id, target, "supersedes")

        # 5. Superseded_by (reversed — superseder→this entry)
        if entry.superseded_by:
            await self._ensure_node(entry.superseded_by, "entry")
            await self._add_edge(entry.superseded_by, entry.id, "supersedes")

        # 6. Text references (kb-XXXXX patterns in knowledge_details)
        seen_refs: set[str] = set()
        for match in _KB_ID_RE.finditer(entry.knowledge_details):
            ref_id = match.group(0)
            if ref_id != entry.id and ref_id not in seen_refs:
                seen_refs.add(ref_id)
                await self._ensure_node(ref_id, "entry")
                await self._add_edge(entry.id, ref_id, "references")

        # 7. Related entities (from hints)
        for rel in _as_list(hints.get("related_entities")):
            if isinstance(rel, dict):
                target = rel.get("id") or rel.get("target")
                edge_type = rel.get("edge_type") or rel.get("type") or "related_to"
                if isinstance(target, str) and target:
                    await self._ensure_node(target, "entry")
                    await self._add_edge(entry.id, target, str(edge_type))
            elif isinstance(rel, str) and rel:
                await self._ensure_node(rel, "entry")
                await self._add_edge(entry.id, rel, "related_to")

        # 8. Person hints
        for person in _as_list(hints.get("person")):
            if isinstance(person, str) and person:
                node_id = f"person:{person.lower()}"
                await self._ensure_node(node_id, "person")
                await self._add_edge(entry.id, node_id, "mentions_person")

        # 9. Tool hints
        for tool in _as_list(hints.get("tool")):
            if isinstance(tool, str) and tool:
                node_id = f"tool:{tool.lower()}"
                await self._ensure_node(node_id, "tool")
                await self._add_edge(entry.id, node_id, "uses_tool")

        await self._db.commit()

    async def _ensure_node(
        self,
        node_id: str,
        node_type: str,
        properties: Mapping[str, object] | None = None,
    ) -> None:
        """Insert a node or update its properties if it already exists."""
        now = datetime.now(UTC).isoformat()
        props_json = json.dumps(properties) if properties else "{}"
        await self._db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(node_id) DO UPDATE SET
                   properties = excluded.properties
            """,
            (node_id, node_type, props_json, now),
        )

    async def _add_edge(self, source: str, target: str, edge_type: str) -> None:
        """Insert an edge, ignoring duplicates."""
        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            """INSERT OR IGNORE INTO graph_edges (source, target, edge_type, properties, created_at)
               VALUES (?, ?, ?, '{}', ?)""",
            (source, target, edge_type, now),
        )

    async def _clear_edges_for_source(self, source: str) -> None:
        """Delete all outgoing edges for a given source node."""
        await self._db.execute("DELETE FROM graph_edges WHERE source = ?", (source,))


def _as_list(value: object) -> list[object]:
    """Coerce a value to a list (single string → [string], None → [])."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
