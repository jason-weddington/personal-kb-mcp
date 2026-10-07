"""Build knowledge graph edges from entry data."""

import json
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime

from kb_core.db.backend import Database
from kb_core.models.entry import KnowledgeEntry
from kb_core.supersession import outgoing_supersedes_targets, recompute_superseded_by

logger = logging.getLogger(__name__)

_KB_ID_RE = re.compile(r"kb-\d{5}")


class GraphBuilder:
    """Deterministic graph builder that derives nodes and edges from entry data."""

    def __init__(self, db: Database) -> None:
        """Initialize with an aiosqlite connection."""
        self._db = db

    async def build_for_entry(self, entry: KnowledgeEntry) -> None:
        """Rebuild all outgoing graph edges for an entry.

        Deletes existing outgoing edges, then re-derives nodes and edges
        from the entry's tags, project_ref, hints, and text references.

        Supersession: ``hints.supersedes`` is the ONLY channel that writes a
        ``supersedes`` edge (a related_entities item typed ``supersedes`` is
        ignored with a warning). The last statement recomputes
        ``superseded_by`` for every old and new supersedes target plus the
        entry itself, so a retracted target is cleared in the same
        transaction (see :mod:`kb_core.supersession`).
        """
        async with self._db.transaction():
            old_targets = await outgoing_supersedes_targets(self._db, entry.id)
            new_targets: set[str] = set()
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
                    if not _KB_ID_RE.fullmatch(target):
                        logger.warning(
                            "Ignoring invalid supersedes target %r (expected kb-XXXXX)", target
                        )
                        continue
                    await self._ensure_node(target, "entry")
                    await self._add_edge(entry.id, target, "supersedes")
                    new_targets.add(target)

            # 5. (removed) superseded_by is DERIVED from supersedes edges by
            # recompute_superseded_by below; it never writes an edge itself.

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
                    if str(edge_type) == "supersedes":
                        logger.warning(
                            "supersession: ignoring related_entities supersedes edge"
                            " %s->%s; use supersedes",
                            entry.id,
                            target,
                        )
                        continue
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

            # 10. Supersession invariant — always the LAST statement.
            await recompute_superseded_by(
                self._db, old_targets | new_targets | {entry.id}, trigger="build"
            )

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
            """INSERT INTO graph_edges (source, target, edge_type, properties, created_at)
               VALUES (?, ?, ?, '{}', ?)
               ON CONFLICT (source, target, edge_type) DO NOTHING""",
            (source, target, edge_type, now),
        )

    async def _clear_edges_for_source(self, source: str) -> None:
        """Delete the deterministic outgoing edges for a given source node.

        Only clears edges this builder itself re-derives (has_tag,
        in_project, supersedes, references, mentions_person, uses_tool,
        related_to/custom hint edges). LLM-enriched edges (marked with
        ``properties.source == "llm"`` by :class:`GraphEnricher`) are left
        untouched so that a metadata-only update — one that never runs
        enrichment because ``content_changed`` is False — doesn't
        permanently destroy the enrichment graph. Enrichment edges are only
        cleared by the enricher itself, immediately before it repopulates
        them (see ``GraphEnricher._clear_enrichment_edges``).
        """
        await self._db.delete_deterministic_edges(source)


def _as_list(value: object) -> list[object]:
    """Coerce a value to a list (single string → [string], None → [])."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
