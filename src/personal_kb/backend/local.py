"""LocalBackend — wraps a KnowledgeBase facade for in-process KB access."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from pathlib import Path

    from kb_core.ingest.ingester import FileResult
    from kb_core.knowledge_base import KnowledgeBase
    from kb_core.models.entry import EntryType, KnowledgeEntry
    from kb_core.models.search import SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class LocalBackend:
    """Backend backed by a local :class:`~kb_core.knowledge_base.KnowledgeBase`.

    Delegates every operation to the facade or to kb_core helpers directly.
    The tool layer receives the same objects (KnowledgeEntry, SearchResult,
    FileResult) regardless of whether a local or remote backend is active.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        """Wrap *kb* as a local backend."""
        self._kb = kb

    @property
    def is_remote(self) -> bool:
        """Always False — this backend is in-process."""
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _compute_pointer_rot(self, entry: KnowledgeEntry) -> list[tuple[str, str | None]]:
        """Compute pointer-rot pairs for a mental_map entry.

        Mirrors the logic in ``kb_get._pointer_rot_note`` but returns raw
        ``(target_id, superseded_by)`` pairs instead of a formatted string so
        the result can flow through the Protocol and be formatted by the tool.
        """
        from kb_core.models.entry import EntryType

        from personal_kb.graph.queries import _KB_ID_RE, get_neighbors

        if entry.entry_type != EntryType.MENTAL_MAP or entry.id is None:
            return []

        from personal_kb.db.queries import get_entry

        neighbors = await get_neighbors(self._kb.db, entry.id, direction="outgoing")
        seen: set[str] = set()
        rotted: list[tuple[str, str | None]] = []
        for target_id, _edge_type, _direction in neighbors:
            if target_id in seen:
                continue
            if not _KB_ID_RE.match(target_id):
                continue
            seen.add(target_id)
            target = await get_entry(self._kb.db, target_id)
            if target is None:
                continue
            if target.superseded_by is not None:
                rotted.append((target_id, target.superseded_by))
            elif not target.is_active:
                rotted.append((target_id, None))

        rotted.sort(key=lambda p: p[0])
        return rotted

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        contributor: str | None = None,
    ) -> tuple[list[SearchResult], int]:
        """Delegate to KnowledgeBase.search()."""
        return await self._kb.search(query, contributor=contributor)

    async def vector_search_available(self) -> bool:
        """Return True when the embedder is configured and reachable."""
        embedder = self._kb.embedder
        is_available = getattr(embedder, "is_available", None)
        if embedder is None or is_available is None:
            return False
        return cast("bool", await is_available())

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_entries(
        self,
        ids: list[str],
    ) -> list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]]:
        """Fetch entries by ID, filtering inactive ones and touching last_accessed."""
        from personal_kb.db.queries import touch_accessed

        results: list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]] = []
        accessed_ids: list[str] = []

        for eid in ids:
            entry = await self._kb.get(eid)
            # Inactive entries are soft-deleted — treat as not found (mirrors
            # the original kb_get tool: ``if entry is None or not entry.is_active``)
            if entry is None or not entry.is_active:
                results.append((eid, None, []))
            else:
                rot = await self._compute_pointer_rot(entry)
                results.append((eid, entry, rot))
                accessed_ids.append(eid)

        if accessed_ids:
            await touch_accessed(self._kb.db, accessed_ids)

        return results

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def store(
        self,
        *,
        short_title: str = "",
        long_title: str = "",
        knowledge_details: str = "",
        entry_type: EntryType | None = None,
        project_ref: str | None = None,
        source_context: str | None = None,
        confidence_level: float = 0.9,
        tags: list[str] | None = None,
        hints: dict[str, object] | None = None,
        sensitivity: str | None = None,
        ttl: str | None = None,
        update_entry_id: str | None = None,
        change_reason: str | None = None,
    ) -> tuple[Literal["created", "updated"], KnowledgeEntry]:
        """Create or update an entry via the local KnowledgeBase facade."""
        from kb_core.models.entry import EntryType as _EntryType

        expires_at = None
        if ttl:
            from personal_kb.tools.ttl import compute_expires_at

            expires_at = compute_expires_at(ttl)

        if update_entry_id:
            entry = await self._kb.update(
                update_entry_id,
                knowledge_details=knowledge_details or None,
                change_reason=change_reason,
                confidence_level=confidence_level,
                tags=tags,
                hints=hints,
                updated_by=self._kb.config.attribution.contributor,
                sensitivity=sensitivity,
                expires_at=expires_at,
                short_title=short_title or None,
                long_title=long_title or None,
                entry_type=entry_type,
                project_ref=project_ref,
                source_context=source_context,
            )
            # Re-fetch to get the fully hydrated entry (mirrors kb_store.py)
            entry = await self._kb.get(entry.id) or entry
            return ("updated", entry)

        # Create path
        resolved_type = entry_type if entry_type is not None else _EntryType.FACTUAL_REFERENCE
        entry = await self._kb.store(
            short_title=short_title,
            long_title=long_title,
            knowledge_details=knowledge_details,
            entry_type=resolved_type,
            project_ref=project_ref,
            source_context=source_context,
            confidence_level=confidence_level,
            tags=tags,
            hints=hints,
            sensitivity=sensitivity,
            expires_at=expires_at,
        )
        # Re-fetch to get the fully hydrated entry (mirrors kb_store.py)
        entry = await self._kb.get(entry.id) or entry
        return ("created", entry)

    async def deactivate(self, entry_id: str) -> KnowledgeEntry:
        """Soft-delete an entry and prune its outgoing graph edges."""
        entry = await self._kb.deactivate(entry_id)
        # Delete outgoing graph edges (mirrors kb_store.py:250-253)
        await self._kb.db.execute(
            "DELETE FROM graph_edges WHERE source = ?",
            (entry_id,),
        )
        await self._kb.db.commit()
        return entry

    async def reactivate(self, entry_id: str) -> KnowledgeEntry:
        """Undo a soft-delete.  Delegate to KnowledgeBase.reactivate()."""
        return await self._kb.reactivate(entry_id)

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[KnowledgeEntry], list[tuple[int, str, str]]]:
        """Per-entry store loop preserving individual failure detail.

        Mirrors the core of ``batch_store_entries`` from the old
        ``kb_store_batch`` tool so per-entry failures (e.g. DB errors) are
        surfaced in the returned *failed* list, keeping local-mode output
        byte-identical to before this refactor.
        """
        from personal_kb.tools.ttl import compute_expires_at

        store = self._kb.knowledge_store
        embedder = self._kb.embedder
        graph_builder = self._kb.graph_builder
        graph_enricher = self._kb.graph_enricher
        db = self._kb.db
        contributor = self._kb.config.attribution.contributor
        team = self._kb.config.attribution.team

        from kb_core.models.entry import EntryType

        created: list[KnowledgeEntry] = []
        failed: list[tuple[int, str, str]] = []

        for i, entry_dict in enumerate(entries):
            entry_type = EntryType(entry_dict.get("entry_type", "factual_reference"))
            confidence = float(entry_dict.get("confidence_level", 0.9))
            tags = entry_dict.get("tags")
            hints = entry_dict.get("hints")
            sensitivity = entry_dict.get("sensitivity")

            expires_at = None
            raw_ttl = entry_dict.get("ttl")
            if raw_ttl:
                try:
                    expires_at = compute_expires_at(str(raw_ttl))
                except ValueError as exc:
                    title = str(entry_dict.get("short_title", f"entry {i}"))
                    failed.append((i, title, str(exc)))
                    logger.warning("Invalid TTL for entry %d (%s): %s", i, title, exc)
                    continue

            try:
                entry = await store.create_entry(
                    short_title=entry_dict["short_title"],
                    long_title=entry_dict["long_title"],
                    knowledge_details=entry_dict["knowledge_details"],
                    entry_type=entry_type,
                    project_ref=entry_dict.get("project_ref"),
                    source_context=entry_dict.get("source_context"),
                    confidence_level=confidence,
                    tags=list(tags) if tags else None,
                    hints=dict(hints) if hints else None,
                    contributor=contributor,
                    team=team,
                    sensitivity=str(sensitivity) if sensitivity else None,  # type: ignore[arg-type]
                    expires_at=expires_at,
                )
            except Exception as exc:
                title = str(entry_dict.get("short_title", f"entry {i}"))
                failed.append((i, title, str(exc)))
                logger.warning("Failed to create entry %d (%s): %s", i, title, exc)
                continue

            # Embed
            if embedder:
                try:
                    embedding = await embedder.embed(entry.embedding_text)
                    if embedding is not None:
                        store_emb = getattr(embedder, "store_embedding", None)
                        if callable(store_emb):
                            await store_emb(entry.id, embedding)
                            await store.mark_embedding(entry.id, True)
                except Exception:
                    logger.warning("Failed to embed entry %s", entry.id, exc_info=True)

            # Build deterministic graph
            try:
                await graph_builder.build_for_entry(entry)
            except Exception:
                logger.warning("Failed to build graph for %s", entry.id, exc_info=True)

            created.append(entry)

        # Batch enrichment
        if graph_enricher and created:
            try:
                await graph_enricher.enrich_batch(created)
            except Exception:
                logger.warning("Batch enrichment failed", exc_info=True)

        # Refresh maps index for mental_map entries
        from kb_core.models.entry import EntryType

        from personal_kb.maps_index_writer import write_project_maps

        map_projects: set[str] = {
            e.project_ref for e in created if e.entry_type == EntryType.MENTAL_MAP and e.project_ref
        }
        for project_ref in map_projects:
            try:
                await write_project_maps(db, project_ref, team=team)
            except Exception:
                logger.warning(
                    "Failed to refresh maps index for project %s", project_ref, exc_info=True
                )
            try:
                await db.notify_maps_changed(project_ref)
            except Exception:
                logger.warning(
                    "Failed to NOTIFY kb_maps_changed for project %s",
                    project_ref,
                    exc_info=True,
                )

        return created, failed

    async def bulk_update(
        self,
        filters: dict[str, Any],
        updates: dict[str, Any],
        dry_run: bool,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Delegate bulk metadata updates to KnowledgeBase.bulk_update()."""
        return await self._kb.bulk_update(filters=filters, updates=updates, dry_run=dry_run)

    async def feedback(
        self,
        feedback_type: str,
        tool_name: str | None = None,
        query_or_params: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Persist agent feedback into the local agent_feedback table."""
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        contributor = self._kb.config.attribution.contributor
        team = self._kb.config.attribution.team
        await self._kb.db.execute(
            "INSERT INTO agent_feedback"
            " (feedback_type, tool_name, query_or_params, detail, contributor, team, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (feedback_type, tool_name, query_or_params, detail, contributor, team, now),
        )
        await self._kb.db.commit()

    # ------------------------------------------------------------------
    # Ask / Summarize
    # ------------------------------------------------------------------

    async def ask_auto(
        self,
        question: str,
        scope: str | None,
        include_graph_context: bool,
        limit: int,
    ) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
        """Retrieve entries for *question* using the local agentic loop."""
        from personal_kb.tools.kb_ask import retrieve_entries

        return await retrieve_entries(
            self._kb.db,
            self._kb.embedder,
            self._kb.query_llm,
            question,
            scope,
            include_graph_context,
            limit,
        )

    async def summarize(
        self,
        question: str,
        scope: str | None,
        limit: int,
    ) -> str:
        """Synthesise a natural-language answer via KnowledgeBase.summarize()."""
        from personal_kb.config import (
            get_agentic_max_tool_calls,
            is_agentic_query,
            is_agentic_synthesis,
        )

        return await self._kb.summarize(
            question,
            scope=scope,
            limit=limit,
            agentic=is_agentic_query(),
            agentic_synthesis=is_agentic_synthesis(),
            max_tool_calls=get_agentic_max_tool_calls(),
        )

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def supersedes_chain(self, entry_id: str) -> list[str]:
        """Return the supersedes chain for *entry_id*, oldest first."""
        from personal_kb.graph.queries import supersedes_chain as _chain

        return await _chain(self._kb.db, entry_id)

    async def bfs_entries(
        self,
        start: str,
        max_depth: int,
        limit: int,
    ) -> list[tuple[str, int, list[str]]]:
        """BFS from *start*.  Returns ``(entry_id, depth, path)`` tuples."""
        from personal_kb.graph.queries import bfs_entries as _bfs

        return await _bfs(self._kb.db, start, max_depth=max_depth, limit=limit)

    async def find_path(
        self,
        source: str,
        target: str,
        max_depth: int,
    ) -> list[tuple[str, str, str]] | None:
        """Find a path between *source* and *target* in the graph."""
        from personal_kb.graph.queries import find_path as _fp

        return await _fp(self._kb.db, source, target, max_depth=max_depth)

    async def entries_for_scope(
        self,
        scope: str,
        entry_type: str | None = None,
        order_by: str = "updated_at",
    ) -> list[str]:
        """Return entry IDs whose scope matches *scope*."""
        from personal_kb.graph.queries import entries_for_scope as _efs

        return await _efs(self._kb.db, scope, entry_type=entry_type, order_by=order_by)

    async def decision_search(self, query: str, limit: int) -> list[str]:
        """FTS search restricted to entry_type='decision'.  Returns IDs."""
        from personal_kb.search.fts import fts_search

        results = await fts_search(self._kb.db, query, limit=limit, entry_type="decision")
        return [eid for eid, _score in results]

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 10,
    ) -> list[tuple[str, str, str]]:
        """Return graph neighbours of *node_id* as ``(neighbor_id, edge_type, direction)``."""
        from personal_kb.graph.queries import get_neighbors

        return await get_neighbors(
            self._kb.db,
            node_id,
            edge_types=edge_types if edge_types else None,
            direction=direction,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self, project_ref: str, since: str | None) -> str:
        """Return a project context primer string.  *since* is a TTL string or None."""
        since_td = None
        if since is not None:
            from personal_kb.tools.ttl import parse_ttl

            since_td = parse_ttl(since)
        return await self._kb.preflight(project_ref, since=since_td)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        path: Path,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """Delegate file ingestion to KnowledgeBase.ingest_file()."""
        return await self._kb.ingest_file(path, project_ref=project_ref, dry_run=dry_run)

    async def ingest_url(
        self,
        url: str,
        content: str | None,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """Ingest a URL, optionally with pre-fetched *content*."""
        if content is not None:
            return await self._kb.ingest_url_content(
                content, url, project_ref=project_ref, dry_run=dry_run
            )
        return await self._kb.ingest_url(url, project_ref=project_ref, dry_run=dry_run)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def list_projects(self) -> list[tuple[str, int]]:
        """Return ``(project_ref, entry_count)`` pairs, count desc."""
        cursor = await self._kb.db.execute(
            "SELECT project_ref, COUNT(*) as cnt FROM knowledge_entries"
            " WHERE is_active = 1 AND project_ref IS NOT NULL"
            " GROUP BY project_ref ORDER BY cnt DESC"
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def list_contributors(self) -> list[tuple[str, int]]:
        """Return ``(contributor, entry_count)`` pairs, count desc."""
        cursor = await self._kb.db.execute(
            "SELECT contributor, COUNT(*) as cnt FROM knowledge_entries"
            " WHERE is_active = 1 AND contributor IS NOT NULL"
            " GROUP BY contributor ORDER BY cnt DESC"
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def list_teams(self) -> list[tuple[str, int]]:
        """Return ``(team, entry_count)`` pairs, count desc."""
        cursor = await self._kb.db.execute(
            "SELECT team, COUNT(*) as cnt FROM knowledge_entries"
            " WHERE is_active = 1 AND team IS NOT NULL"
            " GROUP BY team ORDER BY cnt DESC"
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]
