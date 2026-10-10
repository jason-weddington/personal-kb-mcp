"""Backend Protocol — abstract interface over the KB engine.

All 18 MCP tools obtain their data exclusively through this Protocol so
the same tool code runs against the remote KB service (via HttpBackend).

The Protocol is intentionally narrow: it only enumerates the operations the
18 tools actually need, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from kb_core.ingest.ingester import FileResult
    from kb_core.models.entry import EntryType, KnowledgeEntry
    from kb_core.models.search import SearchQuery, SearchResult


@dataclass(frozen=True)
class QueuedStore:
    """A create the KB service queued as a candidate instead of writing.

    Returned by ``Backend.store`` when the server's write policy routed a
    headless or autonomous ``kb_store`` to the candidate pipeline.
    """

    candidate_id: int
    surface: str
    capture_mode: str


@dataclass(frozen=True)
class QueuedBatch:
    """A batch the KB service queued as candidates instead of writing."""

    candidate_ids: tuple[int, ...]
    surface: str
    capture_mode: str


class Backend(Protocol):
    """Abstract backend — implemented by HttpBackend."""

    @property
    def is_remote(self) -> bool:
        """True for HttpBackend."""
        ...

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query: SearchQuery,
        contributor: str | None = None,
    ) -> tuple[list[SearchResult], int]:
        """Hybrid FTS + vector search.  Returns (results, filtered_count)."""
        ...

    async def vector_search_available(self) -> bool:
        """True when vector (embedding) search is available."""
        ...

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def get_entries(
        self,
        ids: list[str],
    ) -> list[tuple[str, KnowledgeEntry | None, list[tuple[str, str | None]]]]:
        """Fetch entries by ID with pointer-rot and touch-accessed side effect.

        Returns list of (id, entry_or_None, pointer_rot_pairs) where
        ``pointer_rot_pairs`` is a list of ``(target_id, superseded_by)``
        computed for ``mental_map`` entries only.  Inactive or missing entries
        map to ``None``; pointer_rot is ``[]`` in that case.
        """
        ...

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
        supersedes: list[str] | Literal["none"] | None = None,
        distinct_from: list[str] | None = None,
    ) -> tuple[Literal["created", "updated"], KnowledgeEntry, list[str] | None] | QueuedStore:
        """Create or update an entry.

        When *update_entry_id* is set the entry is updated and the action
        returned is ``'updated'``; otherwise a new entry is created and the
        action is ``'created'``.  The third element is the server-reported
        ``superseded_ids`` (None when the server did not report the key).
        A :class:`QueuedStore` means the server's write policy queued the
        create as a candidate instead of writing it.
        """
        ...

    async def deactivate(
        self,
        entry_id: str,
        *,
        change_reason: str | None = None,
        superseded_by: str | None = None,
    ) -> KnowledgeEntry:
        """Soft-delete an entry.  Returns the deactivated entry.

        ``change_reason`` is required by the server; ``superseded_by`` names
        the newer entry that replaces this one, when there is one.
        """
        ...

    async def reactivate(self, entry_id: str) -> KnowledgeEntry:
        """Undo a soft-delete.  Returns the reactivated entry.  Admin-only over HTTP."""
        ...

    async def store_batch(
        self,
        entries: list[dict[str, Any]],
    ) -> tuple[list[KnowledgeEntry], list[tuple[int, str, str]], list[list[str]]] | QueuedBatch:
        """Create multiple entries.

        Returns ``(created, failed, superseded_ids)`` where *superseded_ids*
        is aligned with *created* (empty from servers that do not report it)
        and *failed* is a list of
        ``(index, short_title, error)`` tuples for per-entry failures.
        HttpBackend returns an empty *failed* list; the tool renders the
        aggregate failure count from ``len(created) < len(entries)``.
        A :class:`QueuedBatch` means the server queued every entry.
        """
        ...

    async def bulk_update(
        self,
        filters: dict[str, Any],
        updates: dict[str, Any],
        dry_run: bool,
    ) -> list[tuple[KnowledgeEntry, KnowledgeEntry]]:
        """Apply metadata updates matching *filters*.  Admin-only over HTTP."""
        ...

    async def reconcile_supersession(self) -> dict[str, Any]:
        """Run the supersession reconcile.  Admin-only over HTTP."""
        ...

    async def feedback(
        self,
        feedback_type: str,
        tool_name: str | None = None,
        query_or_params: str | None = None,
        detail: str | None = None,
    ) -> None:
        """Record agent feedback."""
        ...

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
        """Retrieve entries for *question*.

        Returns ``(entries_with_context, agent_turns_used)``.  In HTTP mode
        the server's own agentic config governs agentic behavior; the
        QueryPlanner / agentic_query loop are bypassed client-side.
        """
        ...

    async def summarize(
        self,
        question: str,
        scope: str | None,
        limit: int,
    ) -> str:
        """Synthesised natural-language answer with citations."""
        ...

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    async def supersedes_chain(self, entry_id: str) -> list[str]:
        """Return the supersedes chain for *entry_id*, oldest first."""
        ...

    async def bfs_entries(
        self,
        start: str,
        max_depth: int,
        limit: int,
    ) -> list[tuple[str, int, list[str]]]:
        """BFS from *start*.  Returns ``(entry_id, depth, path)`` tuples."""
        ...

    async def find_path(
        self,
        source: str,
        target: str,
        max_depth: int,
    ) -> list[tuple[str, str, str]] | None:
        """Find a path between *source* and *target*.

        Returns list of ``(src, edge_type, tgt)`` hops, empty list when
        source == target, ``None`` when no path exists.
        """
        ...

    async def entries_for_scope(
        self,
        scope: str,
        entry_type: str | None = None,
        order_by: str = "updated_at",
    ) -> list[str]:
        """Entry IDs matching *scope*."""
        ...

    async def decision_search(self, query: str, limit: int) -> list[str]:
        """FTS / hybrid search limited to ``entry_type='decision'``.

        Returns entry IDs ordered by relevance.  HTTP mode uses the service's
        hybrid-RRF ranking; local mode uses raw FTS — ordering may differ
        (accepted named divergence).
        """
        ...

    async def neighbors(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        direction: str = "both",
        limit: int = 10,
    ) -> list[tuple[str, str, str]]:
        """Return graph neighbours of *node_id*.

        Each element is ``(neighbor_id, edge_type, direction)`` where
        direction is ``'outgoing'`` or ``'incoming'``.
        """
        ...

    # ------------------------------------------------------------------
    # Preflight
    # ------------------------------------------------------------------

    async def preflight(self, project_ref: str, since: str | None) -> str:
        """Project context primer.

        *since* is a raw TTL string (e.g. ``'7d'``) or ``None``.  HttpBackend
        sends it verbatim as a query param.
        """
        ...

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    async def ingest_file(
        self,
        path: Path,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """Ingest a file.  HTTP mode uploads via multipart POST."""
        ...

    async def ingest_url(
        self,
        url: str,
        content: str | None,
        project_ref: str | None,
        dry_run: bool,
    ) -> FileResult:
        """Ingest a URL (optionally with pre-fetched content)."""
        ...

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    async def list_projects(self) -> list[tuple[str, int]]:
        """Return ``(project_ref, entry_count)`` pairs, count desc."""
        ...

    async def list_contributors(self) -> list[tuple[str, int]]:
        """Return ``(contributor, entry_count)`` pairs, count desc."""
        ...

    async def list_teams(self) -> list[tuple[str, int]]:
        """Return ``(team, entry_count)`` pairs, count desc."""
        ...

    # ------------------------------------------------------------------
    # Map eligibility
    # ------------------------------------------------------------------

    async def map_eligibility(self) -> list[dict[str, Any]]:
        """``GET /api/kb/map-eligibility``.

        Returns the ``projects`` list from the ``{"projects": [...]}``
        envelope — one verdict dict per project, passed through unparsed.
        """
        ...

    async def set_map_eligibility_override(
        self,
        project_ref: str,
        *,
        eligible: bool,
        reason: str,
    ) -> dict[str, Any]:
        """``POST /api/kb/map-eligibility/override``.

        Returns the service's full two-key response envelope
        ``{"changed": bool, "verdict": <verdict> | None}`` with the nested
        verdict passed through untouched.
        """
        ...

    async def clear_map_eligibility_override(self, project_ref: str) -> bool:
        """``POST /api/kb/map-eligibility/override/clear`` (a POST, not a DELETE).

        Returns the response's ``changed`` flag as a bool. Because neither
        write endpoint ever 404s for an absent row, a 404 reaching the caller
        can only mean the endpoint does not exist.
        """
        ...
