"""kb_search MCP tool — hybrid FTS + vector search."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.db.backend import Database
from personal_kb.models.entry import EntryType
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.tools.formatters import format_entry_compact, format_graph_hint, format_result_list

logger = logging.getLogger(__name__)

_SPARSE_THRESHOLD = 3
_MAX_HINTS = 3


async def collect_graph_hints(
    db: Database,
    results: list[SearchResult],
    max_hints: int = _MAX_HINTS,
) -> list[str]:
    """Collect graph-connected entries as hints when results are sparse.

    For each search result, does a 1-hop neighbor lookup to find connected
    entries not already in the result set. Returns formatted hint strings.
    """
    from personal_kb.db.queries import get_entry
    from personal_kb.graph.queries import get_neighbors

    seen_ids = {r.entry.id for r in results}
    hints: list[str] = []

    for r in results:
        neighbors = await get_neighbors(db, r.entry.id, limit=10)
        for neighbor_id, edge_type, _direction in neighbors:
            if not neighbor_id.startswith("kb-"):
                # Intermediate node (tag, concept, etc.) — look one more hop
                # to find entries connected through this node
                via_node = neighbor_id
                second_hop = await get_neighbors(db, neighbor_id, limit=10)
                for entry_id, _edge_type, _dir in second_hop:
                    if entry_id in seen_ids or not entry_id.startswith("kb-"):
                        continue
                    entry = await get_entry(db, entry_id)
                    if entry and entry.is_active:
                        seen_ids.add(entry_id)
                        hints.append(format_graph_hint(entry, via_node))
                        if len(hints) >= max_hints:
                            return hints
            else:
                if neighbor_id in seen_ids:
                    continue
                entry = await get_entry(db, neighbor_id)
                if entry and entry.is_active:
                    seen_ids.add(neighbor_id)
                    # Find the shared intermediate node for context
                    hints.append(format_graph_hint(entry, f"{edge_type} from {r.entry.id}"))
                    if len(hints) >= max_hints:
                        return hints

    return hints


def format_search_results(
    results: list[SearchResult],
    match_source_note: str | None = None,
    graph_hints: list[str] | None = None,
) -> str:
    """Format search results as compact entries (no details)."""
    entries = [
        format_entry_compact(r.entry, r.effective_confidence, r.staleness_warning) for r in results
    ]
    return format_result_list(entries, note=match_source_note, hints=graph_hints)


def register_kb_search(mcp: FastMCP) -> None:
    """Register the kb_search tool with the MCP server."""

    @mcp.tool()
    async def kb_search(
        query: Annotated[str, Field(description="Search query (natural language or keywords)")],
        project_ref: Annotated[
            str | None, Field(description="Filter to a specific project")
        ] = None,
        entry_type: Annotated[
            EntryType | None,
            Field(description="Filter by entry type (e.g. factual_reference, decision)"),
        ] = None,
        tags: Annotated[
            list[str] | None, Field(description="Filter by tags (all must match)")
        ] = None,
        limit: Annotated[
            int, Field(description="Maximum results to return (1-50)", ge=1, le=50)
        ] = 10,
        include_stale: Annotated[
            bool, Field(description="Include entries with very low confidence")
        ] = False,
        ctx: Context | None = None,
    ) -> str:
        """Search the personal knowledge base using hybrid semantic + keyword search.

        Combines BM25 full-text search with vector similarity (when Ollama is available)
        using Reciprocal Rank Fusion. Results include confidence decay — older entries
        are flagged with staleness warnings.

        Best for quick lookups: checking if an entry exists, finding by keywords,
        filtering by tags/project/type. For exploring related knowledge, use kb_ask.
        For a synthesized answer to a question, use kb_summarize.
        """
        from personal_kb.search.hybrid import hybrid_search

        if ctx is None:
            raise RuntimeError("Context not injected")
        search_query = SearchQuery(
            query=query,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
            limit=limit,
            include_stale=include_stale,
        )

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        embedder = lifespan["embedder"]

        results = await hybrid_search(db, embedder, search_query)

        # Add a note if vector search was unavailable
        note = None
        if embedder is None or not await embedder.is_available():
            note = "Vector search unavailable (Ollama offline). Results are FTS-only."

        # Collect graph hints when results are sparse
        hints = None
        if len(results) < _SPARSE_THRESHOLD:
            hints = await collect_graph_hints(db, results)

        return format_search_results(results, note, graph_hints=hints)
