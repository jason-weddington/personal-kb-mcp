"""LLM-based graph enrichment — extracts entity relationships from entries."""

import json
import logging
import re

import aiosqlite

from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import KnowledgeEntry

logger = logging.getLogger(__name__)

_VALID_ENTITY_TYPES = {"person", "tool", "concept", "technology"}

_MAX_RELATIONSHIPS = 8

_MAX_BATCH_CONTENT = 500

_BATCH_SYSTEM_PROMPT = """\
You are a knowledge graph builder. Given multiple knowledge entries, extract \
entities and their relationships for EACH entry.

Return ONLY a JSON object keyed by entry ID. Each value is an array of \
relationship objects with:
- "entity": entity name (lowercase, hyphens for spaces)
- "entity_type": one of: person, tool, concept, technology
- "relationship": how the entry relates to the entity

Good entities are SPECIFIC enough to connect related entries:
- "thread-safety", "connection-pooling", "dependency-injection" (good concepts)
- "error", "problem", "pattern" (too vague — avoid these)

Rules:
- Extract 2-6 entities per entry. Use [] for entries that are too generic.
- Skip tags and project references (already captured separately).
- entity_type MUST be one of: person, tool, concept, technology.

Example output:
{
  "kb-00001": [
    {"entity": "fastapi", "entity_type": "tool", "relationship": "uses"}
  ],
  "kb-00002": [
    {"entity": "redis", "entity_type": "technology", "relationship": "depends_on"}
  ]
}\
"""

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """\
You are a knowledge graph builder. Given a knowledge entry, extract entities \
and their relationships to this entry.

Return ONLY a JSON array. Each object has:
- "entity": entity name (lowercase, hyphens for spaces)
- "entity_type": one of: person, tool, concept, technology
- "relationship": how the entry relates to the entity

Good entities are SPECIFIC enough to connect related entries:
- "thread-safety", "connection-pooling", "dependency-injection" (good concepts)
- "error", "problem", "pattern" (too vague — avoid these)
- "postgresql", "redis", "aiosqlite" (good tools/technologies)

Good relationships describe HOW, not just that a link exists:
- uses, depends_on, implements, solves, replaces, configures, learned_from, caused_by

Rules:
- Extract 2-6 entities. Return [] if the entry is too generic.
- Skip tags and project references (already captured separately).
- entity_type MUST be one of: person, tool, concept, technology.

Example input:
  Title: Chose FastAPI over Flask for the new service
  Type: decision
  Content: We chose FastAPI because we need async support and automatic OpenAPI docs.

Example output:
[
  {"entity": "fastapi", "entity_type": "tool", "relationship": "uses"},
  {"entity": "flask", "entity_type": "tool", "relationship": "replaces"},
  {"entity": "openapi", "entity_type": "technology", "relationship": "depends_on"},
  {"entity": "async-http", "entity_type": "concept", "relationship": "implements"}
]\
"""

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class GraphEnricher:
    """Uses an LLM to extract entity relationships and add them as graph edges."""

    def __init__(self, db: aiosqlite.Connection, llm: LLMProvider) -> None:
        """Initialize with a database connection and LLM provider."""
        self._db = db
        self._llm = llm

    async def enrich_entry(self, entry: KnowledgeEntry) -> int:
        """Extract relationships from an entry via LLM and add as graph edges.

        Returns the number of edges added. Never raises — logs warnings on failure.
        """
        if not await self._llm.is_available():
            return 0

        prompt = self._build_prompt(entry)
        raw = await self._llm.generate(prompt, system=_SYSTEM_PROMPT)
        if raw is None:
            return 0

        relationships = self._parse_relationships(raw)

        await self._ensure_entry_node(entry)
        await self._clear_enrichment_edges(entry.id)

        added = 0
        for rel in relationships:
            added += await self._add_enrichment_edge(entry.id, rel)

        await self._db.commit()

        return added

    async def enrich_batch(self, entries: list[KnowledgeEntry]) -> int:
        """Enrich multiple entries with a single LLM call.

        Returns total edges added. Falls back to per-entry on parse failure.
        """
        if not entries:
            return 0
        if not await self._llm.is_available():
            return 0

        prompt = self._build_batch_prompt(entries)
        raw = await self._llm.generate(prompt, system=_BATCH_SYSTEM_PROMPT)
        if raw is None:
            return 0

        batch_rels = self._parse_batch_relationships(raw, [e.id for e in entries])

        # Fallback: if batch parse returned nothing, try per-entry
        if batch_rels is None:
            logger.warning("Batch parse failed, falling back to per-entry enrichment")
            total = 0
            for entry in entries:
                try:
                    total += await self.enrich_entry(entry)
                except Exception:
                    logger.warning("Fallback enrich failed for %s", entry.id, exc_info=True)
            return total

        total = 0
        for entry in entries:
            rels = batch_rels.get(entry.id, [])
            await self._ensure_entry_node(entry)
            await self._clear_enrichment_edges(entry.id)
            for rel in rels:
                total += await self._add_enrichment_edge(entry.id, rel)

        await self._db.commit()
        return total

    def _build_batch_prompt(self, entries: list[KnowledgeEntry]) -> str:
        """Build a prompt containing all entries for batch enrichment."""
        parts: list[str] = []
        for entry in entries:
            content = entry.knowledge_details[:_MAX_BATCH_CONTENT]
            parts.append(f"[{entry.id}] {entry.short_title} ({entry.entry_type.value}): {content}")
        return "\n\n".join(parts)

    def _parse_batch_relationships(
        self, raw: str, entry_ids: list[str]
    ) -> dict[str, list[dict[str, str]]] | None:
        """Parse batch LLM response into per-entry relationship dicts.

        Returns None if the JSON object cannot be parsed (triggers fallback).
        """
        # Strip markdown fences if present
        fence_match = _FENCE_RE.search(raw)
        if fence_match:
            raw = fence_match.group(1)

        # Find JSON object in response
        obj_match = _JSON_OBJECT_RE.search(raw)
        if not obj_match:
            logger.warning("No JSON object found in batch LLM response")
            return None

        try:
            data = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Malformed JSON in batch LLM response")
            return None

        if not isinstance(data, dict):
            return None

        result: dict[str, list[dict[str, str]]] = {}
        valid_ids = set(entry_ids)
        for eid, rels in data.items():
            if eid not in valid_ids:
                continue
            if not isinstance(rels, list):
                continue
            parsed = self._parse_relationships(json.dumps(rels))
            result[eid] = parsed

        return result

    async def enrich_all(self, entries: list[KnowledgeEntry]) -> tuple[int, int]:
        """Enrich multiple entries. Returns (succeeded, failed) counts."""
        succeeded = 0
        failed = 0
        for entry in entries:
            try:
                edges = await self.enrich_entry(entry)
                if edges >= 0:
                    succeeded += 1
                else:
                    failed += 1
            except Exception:
                logger.warning("Failed to enrich %s", entry.id, exc_info=True)
                failed += 1
        return succeeded, failed

    def _build_prompt(self, entry: KnowledgeEntry) -> str:
        parts = [
            f"Title: {entry.short_title}",
            f"Full title: {entry.long_title}",
            f"Type: {entry.entry_type.value}",
        ]
        if entry.tags:
            parts.append(f"Tags: {', '.join(entry.tags)}")
        if entry.project_ref:
            parts.append(f"Project: {entry.project_ref}")
        parts.append(f"\nContent:\n{entry.knowledge_details}")
        return "\n".join(parts)

    def _parse_relationships(self, raw: str) -> list[dict[str, str]]:
        """Parse LLM response into validated relationship dicts."""
        # Strip markdown fences if present
        fence_match = _FENCE_RE.search(raw)
        if fence_match:
            raw = fence_match.group(1)

        # Find JSON array in response
        array_match = _JSON_ARRAY_RE.search(raw)
        if not array_match:
            logger.warning("No JSON array found in LLM response")
            return []

        try:
            data = json.loads(array_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Malformed JSON in LLM response")
            return []

        if not isinstance(data, list):
            return []

        results: list[dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            entity = item.get("entity")
            entity_type = item.get("entity_type")
            relationship = item.get("relationship")
            if not (
                isinstance(entity, str)
                and isinstance(entity_type, str)
                and isinstance(relationship, str)
            ):
                continue
            if entity_type not in _VALID_ENTITY_TYPES:
                continue
            results.append(
                {"entity": entity, "entity_type": entity_type, "relationship": relationship}
            )
            if len(results) >= _MAX_RELATIONSHIPS:
                break

        return results

    async def _ensure_entry_node(self, entry: KnowledgeEntry) -> None:
        """Ensure the entry node exists so edges can reference it."""
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        props = json.dumps({"short_title": entry.short_title, "entry_type": entry.entry_type.value})
        await self._db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(node_id) DO NOTHING""",
            (entry.id, "entry", props, now),
        )

    async def _clear_enrichment_edges(self, entry_id: str) -> None:
        """Remove all LLM-derived edges for a given source entry."""
        await self._db.execute(
            "DELETE FROM graph_edges"
            " WHERE source = ? AND json_extract(properties, '$.source') = 'llm'",
            (entry_id,),
        )

    async def _add_enrichment_edge(self, entry_id: str, rel: dict[str, str]) -> int:
        """Add a single LLM-derived edge. Returns 1 if added, 0 if duplicate."""
        from datetime import UTC, datetime

        node_id = f"{rel['entity_type']}:{rel['entity']}"
        now = datetime.now(UTC).isoformat()

        # Ensure target node exists (don't overwrite deterministic nodes)
        await self._db.execute(
            """INSERT INTO graph_nodes (node_id, node_type, properties, created_at)
               VALUES (?, ?, '{}', ?)
               ON CONFLICT(node_id) DO NOTHING""",
            (node_id, rel["entity_type"], now),
        )

        # Insert edge with LLM source marker
        cursor = await self._db.execute(
            """INSERT OR IGNORE INTO graph_edges (source, target, edge_type, properties, created_at)
               VALUES (?, ?, ?, '{"source": "llm"}', ?)""",
            (entry_id, node_id, rel["relationship"], now),
        )
        return cursor.rowcount
