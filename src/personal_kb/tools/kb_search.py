"""kb_search MCP tool — hybrid FTS + vector search."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.models.entry import EntryType
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.tools.formatters import format_entry_compact, format_result_list

logger = logging.getLogger(__name__)


def format_search_results(results: list[SearchResult], match_source_note: str | None = None) -> str:
    """Format search results as compact entries (no details)."""
    entries = [
        format_entry_compact(r.entry, r.effective_confidence, r.staleness_warning) for r in results
    ]
    return format_result_list(entries, note=match_source_note)


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

        return format_search_results(results, note)
