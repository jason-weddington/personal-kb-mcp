"""Lifted retrieval + synthesis pipelines.

Pure engine code: no FastMCP Context, no env reads. Channels
(``personal_kb.tools.kb_ask``, ``personal_kb.tools.kb_summarize``) pass
the agentic flags + ``max_tool_calls`` explicitly.

What lives here:

* :func:`retrieve_entries` — agentic-or-planner retrieval that returns
  structured ``(entry, context_string)`` tuples plus the agent turn
  count. The historical inline ``is_agentic_query()`` /
  ``get_agentic_max_tool_calls()`` reads are replaced by the explicit
  ``agentic`` and ``max_tool_calls`` keyword params.
* :func:`_auto_search_entries` — hybrid search + graph neighbor
  expansion. Already pure (no env), lifted as-is so the synthesis
  pipeline can call it without crossing the channel boundary.
* :func:`synthesize_answer` — the body of the old
  ``personal_kb.tools.kb_summarize.summarize_question``: retrieve →
  coverage check → re-search → synthesize. The historical inline
  ``is_agentic_synthesis()`` read is replaced by the explicit
  ``agentic_synthesis`` param; the agentic-query knobs flow through to
  :func:`retrieve_entries`.
* :func:`_synthesize`, :func:`_merge_entries`,
  :func:`_format_entries_fallback` — pure helpers, lifted unchanged.

The ``@tool`` wrappers, Context unpacking, and env reads stay in
``personal_kb.tools``; that channel layer calls the functions here with
fully resolved arguments.
"""

import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from kb_core.coverage import assess_coverage
from kb_core.db.backend import Database
from kb_core.db.queries import get_entry
from kb_core.formatting import format_entry_full
from kb_core.graph.agent import AgentResult, agentic_query
from kb_core.graph.planner import QueryPlanner
from kb_core.graph.queries import _parse_scope, get_neighbors
from kb_core.llm.provider import LLMProvider
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery
from kb_core.search.embedder_protocol import Embedder
from kb_core.search.hybrid import SUPERSESSION_READ_MARKER, hybrid_search

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


async def retrieve_entries(
    db: Database,
    embedder: Embedder | None,
    query_llm: LLMProvider | None,
    question: str,
    scope: str | None = None,
    include_graph_context: bool = True,
    limit: int = 20,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    *,
    agentic: bool,
    max_tool_calls: int,
) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
    """Retrieve entries via agentic or single-shot path.

    Returns ``(entries_with_context, agent_turns_used)``. ``agent_turns_used``
    is ``0`` for the single-shot (planner) path and for the agent's
    fast-path; it is ``> 0`` only when the ReAct loop took at least one
    LLM turn.

    Used by both kb_ask (formatted output) and the kb_summarize synthesis
    pipeline (structured entries).
    """
    # --- Agentic path ---
    if query_llm is not None and agentic:
        agent_result = await agentic_query(
            db,
            embedder,
            query_llm,
            question,
            max_tool_calls=max_tool_calls,
            event_callback=event_callback,
        )
        if isinstance(agent_result, AgentResult) and agent_result.entries:
            entries: list[tuple[KnowledgeEntry, str]] = []
            for entry_id, context in agent_result.entries[:limit]:
                entry = await get_entry(db, entry_id)
                if entry:
                    entries.append((entry, context))
            return entries, agent_result.turns_used

        return [], 0

    # --- Single-shot planner path ---
    # Use planner to refine query, but always retrieve via auto search
    # (non-auto strategies produce formatted strings, not structured entries).
    search_query = question
    if query_llm is not None:
        planner = QueryPlanner(db, query_llm)
        plan = await planner.plan(question)
        logger.debug("Query plan: %s", plan)
        if plan is not None and plan.search_query:
            search_query = plan.search_query

    entries = await _auto_search_entries(
        db,
        embedder,
        search_query,
        scope,
        include_graph_context,
        limit,
    )
    return entries, 0


async def _auto_search_entries(
    db: Database,
    embedder: Embedder | None,
    question: str,
    scope: str | None,
    include_graph_context: bool,
    limit: int,
) -> list[tuple[KnowledgeEntry, str]]:
    """Hybrid search + graph expansion, returning structured entries."""
    # Parse scope into SearchQuery filter fields
    project_ref = None
    entry_type = None
    tags = None
    if scope:
        scope_type, scope_value = _parse_scope(scope)
        if scope_type == "project":
            project_ref = scope_value
        elif scope_type == "entry_type":
            with contextlib.suppress(ValueError):
                entry_type = EntryType(scope_value)
        elif scope_type == "tag":
            tags = [scope_value]

    search_query = SearchQuery(
        query=question,
        project_ref=project_ref,
        entry_type=entry_type,
        tags=tags,
        limit=limit,
        include_stale=False,
    )

    results, _filtered_count = await hybrid_search(db, embedder, search_query)

    # Collect search result entries
    seen_ids: set[str] = set()
    entries_with_context: list[tuple[KnowledgeEntry, str]] = []

    for r in results:
        seen_ids.add(r.entry.id)
        entries_with_context.append((r.entry, f"search match (score: {r.score:.4f})"))

    # Expand via graph neighbors
    skipped: list[tuple[str, str]] = []
    if include_graph_context and results:
        for r in results:
            neighbors = await get_neighbors(db, r.entry.id, limit=10)
            for neighbor_id, edge_type, direction in neighbors:
                if neighbor_id in seen_ids:
                    continue
                if not neighbor_id.startswith("kb-"):
                    continue
                entry = await get_entry(db, neighbor_id)
                if entry and entry.is_active and entry.superseded_by is not None:
                    skipped.append((entry.id, entry.superseded_by))
                    continue
                if entry and entry.is_active:
                    seen_ids.add(neighbor_id)
                    if direction == "outgoing":
                        ctx_str = f"linked from {r.entry.id} via {edge_type}"
                    else:
                        ctx_str = f"links to {r.entry.id} via {edge_type}"
                    entries_with_context.append((entry, ctx_str))
                    if len(entries_with_context) >= limit:
                        break
            if len(entries_with_context) >= limit:
                break

    if skipped:
        logger.info("%s op=ask_expand skipped=%r", SUPERSESSION_READ_MARKER, skipped)

    return entries_with_context


async def synthesize_answer(
    db: Database,
    embedder: Embedder | None,
    query_llm: LLMProvider | None,
    question: str,
    scope: str | None = None,
    limit: int = 20,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    synthesis_llm: LLMProvider | None = None,
    *,
    agentic: bool,
    agentic_synthesis: bool,
    max_tool_calls: int,
) -> str:
    """Core summarize logic, testable without FastMCP context.

    Pipeline:

    1. Retrieve via :func:`retrieve_entries` (agentic or single-shot path).
    2. When ``agentic_synthesis`` is on AND the agent did real work
       (``agent_turns > 0``, i.e. not a fast-path hit), ask the query
       LLM whether the retrieved set covers the question; on a flagged
       gap, run an extra :func:`_auto_search_entries` and merge.
    3. Synthesize with the LLM (prefer ``synthesis_llm`` over
       ``query_llm`` when supplied — channels use a beefier Sonnet for
       the human-facing answer and Haiku for retrieval/planning).
    4. Fall back to formatted raw entries when no LLM is available or
       synthesis returns ``None``.
    """

    async def _emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            await event_callback(event)

    # Retrieve entries via agentic or single-shot path
    entries, agent_turns = await retrieve_entries(
        db,
        embedder,
        query_llm,
        question,
        scope,
        include_graph_context=True,
        limit=limit,
        event_callback=event_callback,
        agentic=agentic,
        max_tool_calls=max_tool_calls,
    )

    if not entries:
        return "No entries found matching your question."

    # Coverage check: only when agentic synthesis enabled, LLM available,
    # and retrieval wasn't fast-path (agent_turns > 0 means agent did work)
    if agentic_synthesis and query_llm is not None and agent_turns > 0:
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

    # Synthesize with LLM — prefer synthesis_llm (Sonnet) when available
    synth_provider = synthesis_llm if synthesis_llm is not None else query_llm
    if synth_provider is not None:
        await _emit({"type": "synthesis_started", "entry_count": len(entries)})
        synthesis = await _synthesize(synth_provider, question, entries)
        if synthesis is not None:
            await _emit({"type": "synthesis_done"})
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
        if entry.superseded_by:
            block += f"\n  [SUPERSEDED by {entry.superseded_by}]"
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
