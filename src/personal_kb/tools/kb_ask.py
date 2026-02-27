"""kb_ask MCP tool — graph traversal queries."""

import logging
from datetime import UTC, datetime
from typing import Annotated

import aiosqlite
from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.db.queries import get_entry
from personal_kb.graph.planner import QueryPlanner
from personal_kb.graph.queries import (
    bfs_entries,
    entries_for_scope,
    find_path,
    get_neighbors,
    supersedes_chain,
)
from personal_kb.models.entry import KnowledgeEntry
from personal_kb.models.search import SearchQuery
from personal_kb.search.embeddings import EmbeddingClient

logger = logging.getLogger(__name__)

_STRATEGIES = {"auto", "decision_trace", "timeline", "related", "connection"}


def register_kb_ask(mcp: FastMCP) -> None:
    """Register the kb_ask tool with the MCP server."""

    @mcp.tool()
    async def kb_ask(
        question: Annotated[str, Field(description="Natural language or keywords")],
        strategy: Annotated[
            str,
            Field(
                description=("Query strategy: auto, decision_trace, timeline, related, connection"),
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

        Strategies:
        - auto: Hybrid search + expand via graph neighbors
        - decision_trace: Follow supersedes chains ("how did decision X evolve?")
        - timeline: Chronological entries for a scope ("what happened in project X?")
        - related: BFS from a starting entry/concept ("what connects to tag:python?")
        - connection: Find paths between two nodes ("how are X and Y related?")
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        if strategy not in _STRATEGIES:
            return f"Unknown strategy '{strategy}'. Use: {', '.join(sorted(_STRATEGIES))}"

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


async def _strategy_auto_with_planner(
    db: aiosqlite.Connection,
    embedder: EmbeddingClient | None,
    query_llm: object | None,
    question: str,
    scope: str | None,
    include_graph_context: bool,
    limit: int,
) -> str:
    """Auto strategy with optional LLM query planner."""
    from personal_kb.llm.provider import LLMProvider

    plan = None
    if query_llm is not None and isinstance(query_llm, LLMProvider):
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
    db: aiosqlite.Connection,
    embedder: EmbeddingClient | None,
    question: str,
    scope: str | None,
    include_graph_context: bool,
    limit: int,
) -> str:
    """Hybrid search + expand results via graph neighbors."""
    from personal_kb.search.hybrid import hybrid_search

    search_query = SearchQuery(
        query=question,
        project_ref=None,
        entry_type=None,
        tags=None,
        limit=limit,
        include_stale=False,
    )

    results = await hybrid_search(db, embedder, search_query)

    # Collect search result entries
    seen_ids: set[str] = set()
    entries_with_context: list[tuple[KnowledgeEntry, str]] = []

    for r in results:
        seen_ids.add(r.entry.id)
        entries_with_context.append((r.entry, f"search match (score: {r.score:.4f})"))

    # Expand via graph neighbors
    if include_graph_context and results:
        for r in results:
            neighbors = await get_neighbors(db, r.entry.id, limit=10)
            for neighbor_id, edge_type, direction in neighbors:
                if neighbor_id in seen_ids:
                    continue
                if not neighbor_id.startswith("kb-"):
                    continue
                entry = await get_entry(db, neighbor_id)
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

    if not entries_with_context:
        return "No results found."

    return _format_entries(entries_with_context, f"Auto search: {question}")


async def _strategy_decision_trace(
    db: aiosqlite.Connection,
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
    db: aiosqlite.Connection,
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
    db: aiosqlite.Connection,
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
    db: aiosqlite.Connection,
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
                )
                lines.append(
                    f"  [{entry.id}] {entry.entry_type.value}: {entry.short_title} ({eff:.0%})"
                )

    return "\n".join(lines)


def _format_entries(
    entries_with_context: list[tuple[KnowledgeEntry, str]],
    header: str,
) -> str:
    """Format a list of (entry, context_string) tuples for output."""
    now = datetime.now(UTC)
    lines = [f"{header}\n", f"Found {len(entries_with_context)} result(s):\n"]

    for entry, ctx_str in entries_with_context:
        anchor = entry.updated_at or entry.created_at or now
        eff = compute_effective_confidence(entry.confidence_level, entry.entry_type, anchor, now)
        warning = staleness_warning(eff, entry.entry_type)

        lines.append(f"[{entry.id}] {entry.entry_type.value}: {entry.short_title} ({eff:.0%})")
        lines.append(f"  \u21b3 {ctx_str}")
        if entry.tags:
            lines.append(f"  Tags: {', '.join(entry.tags)}")
        if warning:
            lines.append(f"  WARNING: {warning}")
        lines.append(f"  {entry.knowledge_details}")
        lines.append("")

    return "\n".join(lines)
