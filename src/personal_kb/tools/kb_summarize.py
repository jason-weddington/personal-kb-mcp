"""kb_summarize MCP tool — synthesized answers with citations."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.db.backend import Database
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.tools.formatters import format_entry_full

logger = logging.getLogger(__name__)

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a knowledge base assistant. Given a question and a set of retrieved \
knowledge entries, synthesize a clear, concise answer.

Rules:
- Answer ONLY from the provided entries. Do not use outside knowledge.
- Cite entry IDs in [kb-XXXXX] format when referencing specific entries.
- If entries contain conflicting information, note the conflict and cite both.
- If no entries are relevant to the question, say so clearly.
- Be concise. Prefer bullet points for multi-part answers.
- Do not repeat the question back.\
"""


def register_kb_summarize(mcp: FastMCP) -> None:
    """Register the kb_summarize tool with the MCP server."""

    @mcp.tool()
    async def kb_summarize(
        question: Annotated[str, Field(description="Natural language question")],
        scope: Annotated[
            str | None,
            Field(description="Optional filter (project:X, tag:Y, etc.)"),
        ] = None,
        limit: Annotated[int, Field(description="Max entries to retrieve", ge=1, le=50)] = 20,
        ctx: Context | None = None,
    ) -> str:
        """Answer a question with a synthesized natural language response.

        Best for answering user questions directly — produces a final,
        readable answer with [kb-XXXXX] citations, not raw search results.
        Retrieves relevant entries via graph+search, then synthesizes with an LLM.
        Falls back to raw results if LLM is unavailable.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        embedder = lifespan["embedder"]
        query_llm = lifespan.get("query_llm")

        return await summarize_question(
            db,
            embedder,
            query_llm,
            question,
            scope,
            limit,
        )


async def summarize_question(
    db: Database,
    embedder: EmbeddingClient | None,
    query_llm: object | None,
    question: str,
    scope: str | None = None,
    limit: int = 20,
) -> str:
    """Core summarize logic, testable without FastMCP context."""
    from personal_kb.config import is_agentic_synthesis
    from personal_kb.tools.kb_ask import _auto_search_entries, retrieve_entries

    # Retrieve entries via agentic or single-shot path
    entries, agent_turns = await retrieve_entries(
        db,
        embedder,
        query_llm,
        question,
        scope,
        include_graph_context=True,
        limit=limit,
    )

    if not entries:
        return "No entries found matching your question."

    # Coverage check: only when agentic synthesis enabled, LLM available,
    # and retrieval wasn't fast-path (agent_turns > 0 means agent did work)
    if (
        is_agentic_synthesis()
        and query_llm is not None
        and isinstance(query_llm, LLMProvider)
        and agent_turns > 0
    ):
        from personal_kb.tools.coverage import assess_coverage

        coverage = await assess_coverage(query_llm, question, entries)
        if coverage.has_gaps and coverage.suggested_query:
            extra = await _auto_search_entries(
                db,
                embedder,
                coverage.suggested_query,
                scope,
                True,
                limit,
            )
            entries = _merge_entries(entries, extra)

    # Synthesize with LLM
    if query_llm is not None and isinstance(query_llm, LLMProvider):
        synthesis = await _synthesize(query_llm, question, entries)
        if synthesis is not None:
            return synthesis

        fallback = _format_entries_fallback(entries)
        return f"(LLM synthesis failed — showing raw results)\n\n{fallback}"

    fallback = _format_entries_fallback(entries)
    return f"(LLM unavailable — showing raw results)\n\n{fallback}"


async def _synthesize(
    llm: LLMProvider,
    question: str,
    entries: list[tuple[KnowledgeEntry, str]],
) -> str | None:
    """Synthesize an answer from structured entries using the LLM."""
    # Build rich prompt with full knowledge_details
    entry_blocks = []
    for entry, context in entries:
        tags_str = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""
        block = f"[{entry.id}] {entry.short_title} {tags_str}"
        if context:
            block += f"\n  Context: {context}"
        block += f"\n  {entry.knowledge_details}"
        entry_blocks.append(block)

    entries_text = "\n\n".join(entry_blocks)
    prompt = f"Question: {question}\n\nRetrieved entries:\n{entries_text}"
    return await llm.generate(prompt, system=_SYNTHESIS_SYSTEM_PROMPT)


def _merge_entries(
    original: list[tuple[KnowledgeEntry, str]],
    extra: list[tuple[KnowledgeEntry, str]],
) -> list[tuple[KnowledgeEntry, str]]:
    """Merge extra entries into original, deduplicating by entry ID."""
    seen = {entry.id for entry, _ in original}
    merged = list(original)
    for entry, ctx in extra:
        if entry.id not in seen:
            seen.add(entry.id)
            merged.append((entry, ctx))
    return merged


def _format_entries_fallback(
    entries: list[tuple[KnowledgeEntry, str]],
) -> str:
    """Format entries for the no-LLM fallback path."""
    formatted = [format_entry_full(entry, context=ctx) for entry, ctx in entries]
    return "\n\n".join(formatted)
