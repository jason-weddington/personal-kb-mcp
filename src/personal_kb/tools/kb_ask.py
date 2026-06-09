"""kb_ask MCP tool — graph traversal queries."""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.server.context import Context

# Re-exports from kb_core.query keep historical import paths working for
# tests and external callers (web/routes.py, tests/tools/test_kb_ask_tool.py).
# These functions live in kb_core now; the wrappers below add the env-driven
# defaults for `agentic` / `max_tool_calls` so old call sites that don't
# pass those kwargs still get the historical behavior.
from kb_core.query import _auto_search_entries
from kb_core.query import retrieve_entries as _kb_core_retrieve_entries
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.db.backend import Database
from personal_kb.db.queries import get_entry
from personal_kb.graph.planner import QueryPlanner
from personal_kb.graph.queries import (
    bfs_entries,
    entries_for_scope,
    find_path,
    supersedes_chain,
)
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import KnowledgeEntry
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.tools.formatters import format_entry_compact, format_entry_full, format_result_list

__all__ = [
    "_auto_search_entries",
    "register_kb_ask",
    "retrieve_entries",
]

logger = logging.getLogger(__name__)

Strategy = Literal["auto", "decision_trace", "timeline", "related", "connection"]


def register_kb_ask(mcp: FastMCP, prefix: str = "kb_") -> None:
    """Register the kb_ask tool with the MCP server."""

    @mcp.tool(name=f"{prefix}ask")
    async def kb_ask(
        question: Annotated[str, Field(description="Natural language or keywords")],
        strategy: Annotated[
            Strategy,
            Field(
                description="Query strategy: auto, decision_trace, timeline, related, connection",
            ),
        ] = "auto",
        scope: Annotated[
            str | None,
            Field(
                description=('Filter: "project:X", "tag:Y", entry ID, or node ID'),
            ),
        ] = None,
        target: Annotated[
            str | None,
            Field(description="Second node for 'connection' strategy"),
        ] = None,
        include_graph_context: Annotated[
            bool,
            Field(description="Expand results with graph neighbors"),
        ] = True,
        limit: Annotated[int, Field(description="Max results", ge=1, le=50)] = 20,
        ctx: Context | None = None,
    ) -> str:
        """Answer questions by traversing the knowledge graph and combining with search.

        Best for discovery and exploration — when you need to find connections,
        trace history, or understand how knowledge relates.

        Strategies (prefer specific strategies over auto when intent is clear):
        - auto: Hybrid search + graph expansion. Good default for open-ended queries.
        - decision_trace: Follow supersedes chains to see how a decision evolved over
          time. Use for "why did we switch from X to Y?" or "what was the original
          rationale for Z?"
        - timeline: Chronological view of entries in a scope. Use for "what happened
          in project X?" or "recent changes to tag:auth".
        - related: BFS from a starting node — finds everything connected to a concept.
          Use for "what touches tag:python?" or "what depends on kb-00042?"
        - connection: Find paths between two nodes. Use for "how are X and Y related?"
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        embedder = lifespan["embedder"]
        query_llm = lifespan.get("query_llm")

        if strategy == "auto":
            return await _strategy_auto_with_planner(
                db,
                embedder,
                query_llm,
                question,
                scope,
                include_graph_context,
                limit,
            )
        elif strategy == "decision_trace":
            return await _strategy_decision_trace(db, question, scope, limit)
        elif strategy == "timeline":
            return await _strategy_timeline(db, scope, limit)
        elif strategy == "related":
            return await _strategy_related(db, scope, limit)
        elif strategy == "connection":
            return await _strategy_connection(db, scope, target)

        return "Strategy not implemented."


async def retrieve_entries(
    db: Database,
    embedder: EmbeddingClient | None,
    query_llm: LLMProvider | None,
    question: str,
    scope: str | None = None,
    include_graph_context: bool = True,
    limit: int = 20,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[list[tuple[KnowledgeEntry, str]], int]:
    """Channel-side wrapper: fills agentic params from env, then delegates.

    The real implementation lives in :func:`kb_core.query.retrieve_entries`
    and takes ``agentic`` / ``max_tool_calls`` as explicit keyword args
    (kb_core reads no environment). This wrapper preserves the historical
    signature — callers (the MCP tool, the web routes, the test suite)
    don't have to know about the env knobs.
    """
    from personal_kb.config import get_agentic_max_tool_calls, is_agentic_query

    return await _kb_core_retrieve_entries(
        db,
        embedder,
        query_llm,
        question,
        scope,
        include_graph_context,
        limit,
        event_callback,
        agentic=is_agentic_query(),
        max_tool_calls=get_agentic_max_tool_calls(),
    )


async def _strategy_auto_with_planner(
    db: Database,
    embedder: EmbeddingClient | None,
    query_llm: LLMProvider | None,
    question: str,
    scope: str | None,
    include_graph_context: bool,
    limit: int,
) -> str:
    """Auto strategy with optional LLM query planner.

    When agentic query is enabled and an LLM is available, delegates to the
    ReAct agent loop which can plan, execute, evaluate, and retry.  Falls back
    to the single-shot planner when agentic query is disabled.
    """
    from personal_kb.config import get_agentic_max_tool_calls, is_agentic_query

    # --- Agentic path ---
    if query_llm is not None and is_agentic_query():
        from personal_kb.graph.agent import agentic_query

        agent_result = await agentic_query(
            db,
            embedder,
            query_llm,
            question,
            max_tool_calls=get_agentic_max_tool_calls(),
        )
        return await _format_agent_result_full(agent_result, db, question, limit)

    # --- Single-shot planner path ---
    plan = None
    if query_llm is not None:
        planner = QueryPlanner(db, query_llm)
        plan = await planner.plan(question)
        logger.debug("Query plan: %s", plan)

    if plan is not None and plan.strategy != "auto":
        # Dispatch to the planned strategy
        header = f"[Planned: {plan.strategy}]"
        if plan.reasoning:
            header += f" {plan.reasoning}"
        header += "\n\n"

        if plan.strategy == "decision_trace":
            result = await _strategy_decision_trace(
                db,
                plan.search_query or question,
                plan.scope or scope,
                limit,
            )
        elif plan.strategy == "timeline":
            result = await _strategy_timeline(db, plan.scope or scope, limit)
        elif plan.strategy == "related":
            result = await _strategy_related(db, plan.scope or scope, limit)
        elif plan.strategy == "connection":
            result = await _strategy_connection(db, plan.scope or scope, plan.target)
        else:
            result = await _strategy_auto(
                db,
                embedder,
                plan.search_query or question,
                scope,
                include_graph_context,
                limit,
            )
        return header + result

    # Fall through: use auto strategy, optionally with refined search query
    search_query = question
    if plan is not None and plan.search_query:
        search_query = plan.search_query
    return await _strategy_auto(db, embedder, search_query, scope, include_graph_context, limit)


async def _strategy_auto(
    db: Database,
    embedder: EmbeddingClient | None,
    question: str,
    scope: str | None,
    include_graph_context: bool,
    limit: int,
) -> str:
    """Hybrid search + expand results via graph neighbors."""
    entries_with_context = await _auto_search_entries(
        db,
        embedder,
        question,
        scope,
        include_graph_context,
        limit,
    )

    if not entries_with_context:
        return "No results found."

    return _format_entries(entries_with_context, f"Auto search: {question}")


async def _strategy_decision_trace(
    db: Database,
    question: str,
    scope: str | None,
    limit: int,
) -> str:
    """Find decision entries and follow supersedes chains."""
    from personal_kb.search.fts import fts_search

    # Find decision entries matching the question
    fts_results = await fts_search(db, question, limit=limit, entry_type="decision")

    if not fts_results and scope:
        # Try scope-based lookup
        entry_ids = await entries_for_scope(db, scope, entry_type="decision")
        fts_results = [(eid, 0.0) for eid in entry_ids[:limit]]

    if not fts_results:
        return "No decision entries found matching the query."

    # For each decision, build the supersedes chain
    seen_chains: set[str] = set()  # avoid duplicate chains
    entries_with_context: list[tuple[KnowledgeEntry, str]] = []

    for entry_id, _score in fts_results:
        if entry_id in seen_chains:
            continue

        chain = await supersedes_chain(db, entry_id)
        for cid in chain:
            seen_chains.add(cid)

        for i, cid in enumerate(chain):
            entry = await get_entry(db, cid)
            if not entry:
                continue

            if len(chain) == 1:
                ctx_str = "current decision"
            elif i == 0:
                ctx_str = "original decision"
            elif i == len(chain) - 1:
                ctx_str = f"current (supersedes {chain[i - 1]})"
            else:
                ctx_str = f"supersedes {chain[i - 1]}"

            entries_with_context.append((entry, ctx_str))
            if len(entries_with_context) >= limit:
                break

        if len(entries_with_context) >= limit:
            break

    if not entries_with_context:
        return "No decision entries found matching the query."

    return _format_entries(entries_with_context, f"Decision trace: {question}")


async def _strategy_timeline(
    db: Database,
    scope: str | None,
    limit: int,
) -> str:
    """Chronological entries for a scope."""
    if not scope:
        return "Timeline strategy requires a scope (e.g. project:X, tag:Y, decision)."

    entry_ids = await entries_for_scope(db, scope, order_by="created_at")

    if not entry_ids:
        return f"No entries found for scope: {scope}"

    entries_with_context: list[tuple[KnowledgeEntry, str]] = []
    for eid in entry_ids[:limit]:
        entry = await get_entry(db, eid)
        if entry and entry.is_active:
            date_str = entry.created_at.strftime("%Y-%m-%d") if entry.created_at else "unknown"
            entries_with_context.append((entry, f"created {date_str}"))

    if not entries_with_context:
        return f"No active entries found for scope: {scope}"

    return _format_entries(entries_with_context, f"Timeline: {scope}")


async def _strategy_related(
    db: Database,
    scope: str | None,
    limit: int,
) -> str:
    """BFS from a starting entry/concept through graph edges."""
    if not scope:
        return "Related strategy requires a scope (entry ID or node ID like tag:python)."

    results = await bfs_entries(db, scope, max_depth=2, limit=limit)

    if not results:
        return f"No related entries found from: {scope}"

    entries_with_context: list[tuple[KnowledgeEntry, str]] = []
    for entry_id, depth, path in results:
        entry = await get_entry(db, entry_id)
        if entry and entry.is_active:
            if depth == 1:
                ctx_str = "directly connected"
            else:
                intermediates = [n for n in path[1:-1] if not n.startswith("kb-")]
                if intermediates:
                    ctx_str = f"connected via {', '.join(intermediates)}"
                else:
                    ctx_str = f"connected (depth {depth})"
            entries_with_context.append((entry, ctx_str))

    if not entries_with_context:
        return f"No related entries found from: {scope}"

    return _format_entries(entries_with_context, f"Related to: {scope}")


async def _strategy_connection(
    db: Database,
    scope: str | None,
    target: str | None,
) -> str:
    """Find paths between two nodes."""
    if not scope or not target:
        return "Connection strategy requires both scope and target parameters."

    path = await find_path(db, scope, target, max_depth=4)

    if path is None:
        return f"No connection found between {scope} and {target} (max depth: 4)."

    if not path:
        return f"{scope} and {target} are the same node."

    # Format path
    lines = [f"Connection: {scope} -> {target}\n"]
    lines.append("Path:")
    for i, (src, edge_type, tgt) in enumerate(path):
        lines.append(f"  {i + 1}. {src} --[{edge_type}]--> {tgt}")

    # Collect entries along the path
    entry_ids: set[str] = set()
    for src, _et, tgt in path:
        if src.startswith("kb-"):
            entry_ids.add(src)
        if tgt.startswith("kb-"):
            entry_ids.add(tgt)

    if entry_ids:
        lines.append("\nEntries along the path:")
        now = datetime.now(UTC)
        for eid in sorted(entry_ids):
            entry = await get_entry(db, eid)
            if entry:
                eff = compute_effective_confidence(
                    entry.confidence_level,
                    entry.entry_type,
                    entry.updated_at or entry.created_at or now,
                    now,
                    last_accessed=entry.last_accessed,
                )
                warn = staleness_warning(eff, entry.entry_type)
                lines.append(f"  {format_entry_compact(entry, eff, warn)}")

    return "\n".join(lines)


async def _format_agent_result_full(
    result: object,
    db: Database,
    question: str,
    limit: int,
) -> str:
    """Format an AgentResult with full entry details."""
    from personal_kb.graph.agent import AgentResult

    if not isinstance(result, AgentResult):
        return "No results found."
    if not result.entries:
        header = f"[Agent: {result.turns_used} tool calls] No results found."
        if result.reasoning:
            header += f"\n{result.reasoning}"
        return header

    entries_with_context: list[tuple[KnowledgeEntry, str]] = []
    for entry_id, context in result.entries[:limit]:
        entry = await get_entry(db, entry_id)
        if entry:
            entries_with_context.append((entry, context))

    header = f"[Agent: {result.turns_used} tool calls]"
    if result.reasoning:
        header += f" {result.reasoning}"

    formatted = [format_entry_full(entry, context=ctx) for entry, ctx in entries_with_context]
    return format_result_list(formatted, header=header)


def _format_entries(
    entries_with_context: list[tuple[KnowledgeEntry, str]],
    header: str,
) -> str:
    """Format a list of (entry, context_string) tuples for output."""
    formatted = [
        format_entry_full(entry, context=ctx_str) for entry, ctx_str in entries_with_context
    ]
    return format_result_list(formatted, header=header)
