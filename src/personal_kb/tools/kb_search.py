"""kb_search MCP tool — hybrid FTS + vector search."""

import logging
from typing import TYPE_CHECKING, Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.models.entry import EntryType
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.tools.formatters import format_entry_compact, format_graph_hint, format_result_list

if TYPE_CHECKING:
    from personal_kb.backend.protocol import Backend

logger = logging.getLogger(__name__)

_SPARSE_THRESHOLD = 3
_MAX_HINTS = 3


async def collect_graph_hints(
    backend: "Backend",
    results: list[SearchResult],
    max_hints: int = _MAX_HINTS,
) -> list[str]:
    """Collect graph-connected entries as hints when results are sparse.

    For each search result, does a 1-hop neighbour lookup to find connected
    entries not already in the result set. Returns formatted hint strings.
    """
    seen_ids = {r.entry.id for r in results}
    hints: list[str] = []

    for r in results:
        neighbours = await backend.neighbors(r.entry.id, limit=10)
        for neighbor_id, edge_type, _direction in neighbours:
            if not neighbor_id.startswith("kb-"):
                # Intermediate node (tag, concept, etc.) — look one more hop
                via_node = neighbor_id
                second_hop = await backend.neighbors(neighbor_id, limit=10)
                for entry_id, _edge_type, _dir in second_hop:
                    if entry_id in seen_ids or not entry_id.startswith("kb-"):
                        continue
                    entries_data = await backend.get_entries([entry_id])
                    _, entry, _ = entries_data[0]
                    if entry is not None:
                        seen_ids.add(entry_id)
                        hints.append(format_graph_hint(entry, via_node))
                        if len(hints) >= max_hints:
                            return hints
            else:
                if neighbor_id in seen_ids:
                    continue
                entries_data = await backend.get_entries([neighbor_id])
                _, entry, _ = entries_data[0]
                if entry is not None:
                    seen_ids.add(neighbor_id)
                    hints.append(format_graph_hint(entry, f"{edge_type} from {r.entry.id}"))
                    if len(hints) >= max_hints:
                        return hints

    return hints


def format_search_results(
    results: list[SearchResult],
    match_source_note: str | None = None,
    graph_hints: list[str] | None = None,
    filtered_count: int = 0,
) -> str:
    """Format search results as compact entries (no details)."""
    entries = [
        format_entry_compact(r.entry, r.effective_confidence, r.staleness_warning) for r in results
    ]
    return format_result_list(
        entries, note=match_source_note, hints=graph_hints, filtered_count=filtered_count
    )


def _search_description(prefix: str) -> str:
    """Build kb_search description with correct tool name cross-references."""
    return (
        "Search the personal knowledge base using hybrid semantic + keyword search.\n\n"
        "Combines BM25 full-text search with vector similarity (when Ollama is available) "
        "using Reciprocal Rank Fusion. Results include confidence decay — older entries "
        "are flagged with staleness warnings.\n\n"
        "Best for quick lookups: checking if an entry exists, finding by keywords, "
        f"filtering by tags/project/type. For exploring related knowledge, use {prefix}ask. "
        f"For a synthesized answer to a question, use {prefix}summarize.\n\n"
        "The query parameter is optional. To list all entries for a project, "
        "omit query and pass project_ref. Filters (project_ref, entry_type, tags, "
        "contributor, team) can be combined with or without a text query.\n\n"
        "Returns compact summaries (titles + metadata, no knowledge_details). "
        f"Use {prefix}get with entry IDs to read the full content of interesting results."
    )


def register_kb_search(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_search tool with the MCP server."""

    @mcp.tool(name=f"{prefix}search", description=_search_description(prefix))
    async def kb_search(
        query: Annotated[
            str,
            Field(
                description="Search query (natural language or keywords). "
                "Optional — omit or leave empty to list entries by filters alone "
                "(e.g. all entries for a project)."
            ),
        ] = "",
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
        ] = 5,
        include_stale: Annotated[
            bool, Field(description="Include entries with very low confidence")
        ] = False,
        include_expired: Annotated[
            bool, Field(description="Include entries past their TTL expiry")
        ] = False,
        contributor: Annotated[str | None, Field(description="Filter by contributor name")] = None,
        team: Annotated[str | None, Field(description="Filter by team name")] = None,
        ctx: Context | None = None,
    ) -> str:
        """Search the knowledge base using hybrid semantic + keyword search."""
        from personal_kb.tools._lifespan import backend_from_lifespan

        if ctx is None:
            raise RuntimeError("Context not injected")

        backend = backend_from_lifespan(ctx.lifespan_context)

        # HTTP mode: contributor/team filters are not supported
        if backend.is_remote and (contributor is not None or team is not None):
            return "Error: contributor/team filters are not supported in HTTP mode."

        search_query = SearchQuery(
            query=query,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
            contributor=contributor,
            team=team,
            limit=limit,
            include_stale=include_stale,
            include_expired=include_expired,
        )

        # For local mode: obtain contributor for telemetry from the KB config.
        # For HTTP mode: the service handles attribution; contributor param above
        # is always None (checked above).
        telemetry_contributor: str | None = None
        if not backend.is_remote:
            from personal_kb.tools._lifespan import kb_from_lifespan

            kb = kb_from_lifespan(ctx.lifespan_context)
            telemetry_contributor = kb.config.attribution.contributor

        results, filtered_count = await backend.search(
            search_query, contributor=telemetry_contributor
        )

        # Add a note if vector search was unavailable.
        note = None
        if not await backend.vector_search_available():
            note = "Vector search unavailable (Ollama offline). Results are FTS-only."

        # Collect graph hints when results are sparse
        hints = None
        if len(results) < _SPARSE_THRESHOLD:
            hints = await collect_graph_hints(backend, results)

        return format_search_results(
            results, note, graph_hints=hints, filtered_count=filtered_count
        )
