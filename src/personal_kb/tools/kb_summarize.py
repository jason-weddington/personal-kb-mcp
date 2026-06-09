"""kb_summarize MCP tool — synthesized answers with citations.

The retrieval + synthesis pipeline lives in :mod:`kb_core.query`; this
file is the FastMCP channel:

* :func:`register_kb_summarize` registers the ``@tool``-decorated entry
  point, unpacks the lifespan context, and reads the agentic env flags.
* :func:`summarize_question` is the historical channel-side helper used
  by the web routes (``personal_kb.web.routes``) and the test suite. It
  forwards to :func:`kb_core.query.synthesize_answer` with explicit
  ``agentic`` / ``agentic_synthesis`` / ``max_tool_calls`` kwargs filled
  in from the env so legacy callers don't have to know about them.
* The synthesis prompt, ``_synthesize``, ``_merge_entries``, and the
  no-LLM fallback formatter are re-exported from :mod:`kb_core.query` so
  existing imports of ``personal_kb.tools.kb_summarize._synthesize`` etc.
  keep working.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.server.context import Context

# Re-exports from kb_core.query: tests and any external caller importing
# `_synthesize` / `_merge_entries` / `_format_entries_fallback` from this
# module keep working transparently.
from kb_core.query import (
    _format_entries_fallback,
    _merge_entries,
    _synthesize,
)
from kb_core.query import synthesize_answer as _kb_core_synthesize_answer
from pydantic import Field

from personal_kb.db.backend import Database
from personal_kb.llm.provider import LLMProvider
from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

__all__ = [
    "_format_entries_fallback",
    "_merge_entries",
    "_synthesize",
    "register_kb_summarize",
    "summarize_question",
]


def _summarize_description(prefix: str) -> str:
    """Build kb_summarize description with correct tool name cross-references."""
    return (
        "Answer a question with a synthesized natural language response.\n\n"
        "Best for answering user questions directly — produces a final, "
        "readable answer with [kb-XXXXX] citations, not raw search results. "
        "Retrieves relevant entries via graph+search, then synthesizes with an LLM. "
        "Falls back to raw results if LLM is unavailable.\n\n"
        "Use this for user-facing answers. For your own research or exploration, "
        f"prefer {prefix}search or {prefix}ask — they're cheaper (no synthesis LLM call)."
    )


def register_kb_summarize(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_summarize tool with the MCP server."""

    @mcp.tool(name=f"{prefix}summarize", description=_summarize_description(prefix))
    async def kb_summarize(
        question: Annotated[str, Field(description="Natural language question")],
        scope: Annotated[
            str | None,
            Field(description="Optional filter (project:X, tag:Y, etc.)"),
        ] = None,
        limit: Annotated[int, Field(description="Max entries to retrieve", ge=1, le=50)] = 20,
        ctx: Context | None = None,
    ) -> str:
        """Answer a question with a synthesized natural language response."""
        from personal_kb.config import (
            get_agentic_max_tool_calls,
            is_agentic_query,
            is_agentic_synthesis,
        )
        from personal_kb.tools._lifespan import kb_from_lifespan

        if ctx is None:
            raise RuntimeError("Context not injected")

        kb = kb_from_lifespan(ctx.lifespan_context)
        return await kb.summarize(
            question,
            scope=scope,
            limit=limit,
            agentic=is_agentic_query(),
            agentic_synthesis=is_agentic_synthesis(),
            max_tool_calls=get_agentic_max_tool_calls(),
        )


async def summarize_question(
    db: Database,
    embedder: EmbeddingClient | None,
    query_llm: LLMProvider | None,
    question: str,
    scope: str | None = None,
    limit: int = 20,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    synthesis_llm: LLMProvider | None = None,
) -> str:
    """Channel-side wrapper: fills agentic params from env, then delegates.

    The real implementation lives in
    :func:`kb_core.query.synthesize_answer`, which takes ``agentic``,
    ``agentic_synthesis``, and ``max_tool_calls`` as explicit keyword
    args (kb_core reads no environment). This wrapper preserves the
    historical signature for callers that don't pass those kwargs.
    """
    from personal_kb.config import (
        get_agentic_max_tool_calls,
        is_agentic_query,
        is_agentic_synthesis,
    )

    return await _kb_core_synthesize_answer(
        db,
        embedder,
        query_llm,
        question,
        scope,
        limit,
        event_callback,
        synthesis_llm,
        agentic=is_agentic_query(),
        agentic_synthesis=is_agentic_synthesis(),
        max_tool_calls=get_agentic_max_tool_calls(),
    )
