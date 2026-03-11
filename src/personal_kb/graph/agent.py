"""ReAct agent loop for kb_ask — plan, execute, evaluate, retry."""

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.config import get_agentic_max_tool_calls
from personal_kb.db.backend import Database
from personal_kb.db.queries import get_entry
from personal_kb.graph.queries import (
    entries_for_scope,
    get_graph_vocabulary,
    get_neighbors,
    supersedes_chain,
)
from personal_kb.llm.provider import LLMProvider
from personal_kb.models.search import SearchQuery, SearchResult
from personal_kb.search.embeddings import EmbeddingClient
from personal_kb.search.hybrid import hybrid_search
from personal_kb.tools.formatters import format_entry_compact

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ENTRY_ID_RE = re.compile(r"kb-\d{5}")

# If the fast-path top result scores above this, skip the agent loop entirely
_FAST_PATH_THRESHOLD = 0.030


@dataclass
class AgentResult:
    """Result of an agentic query."""

    entries: list[tuple[str, str]]  # (entry_id, context)
    turns_used: int = 0
    reasoning: str = ""


@dataclass
class _ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass
class _FinalAnswer:
    entry_ids: list[str]
    reasoning: str


_AGENT_SYSTEM_PROMPT = """\
You are a knowledge-base retrieval specialist. Your job is to find the most \
relevant entries for the user's question. You do NOT answer questions — you \
find entries and return their IDs.

Available tools (respond with ONE JSON object per turn):

1. hybrid_search — full-text + vector search
   {"tool": "hybrid_search", "args": {"query": "...", "limit": 10}}

2. graph_neighbors — get edges from a node
   {"tool": "graph_neighbors", "args": {"node_id": "kb-00001", "limit": 10}}

3. list_graph_nodes — browse the knowledge graph vocabulary
   {"tool": "list_graph_nodes", "args": {"type_filter": null}}

4. decision_chain — follow supersedes chain for a decision entry
   {"tool": "decision_chain", "args": {"entry_id": "kb-00001"}}

5. scope_entries — get entries for a scope (project:X, tag:Y, etc.)
   {"tool": "scope_entries", "args": {"scope": "project:personal-kb", "limit": 20}}

6. done — return final answer (MUST include at least one entry_id)
   {"tool": "done", "args": {"entry_ids": ["kb-00001", "kb-00002"], "reasoning": "..."}}

Evaluation rules:
- 3+ clear matches → call done immediately
- 1-2 partial matches → try ONE refinement (different query, graph exploration)
- 0 matches → try up to TWO different approaches before giving up

NEVER make more than 4 tool calls. Be efficient.
Output exactly ONE JSON object per turn. No commentary before or after.\
"""


def _format_search_results(results: list[SearchResult]) -> str:
    """Format search results compactly for the agent context."""
    if not results:
        return "No results found."
    now = datetime.now(UTC)
    lines = []
    for r in results:
        anchor = r.entry.updated_at or r.entry.created_at or now
        eff = compute_effective_confidence(
            r.entry.confidence_level,
            r.entry.entry_type,
            anchor,
            now,
            last_accessed=r.entry.last_accessed,
        )
        warn = staleness_warning(eff, r.entry.entry_type)
        lines.append(f"(score={r.score:.4f}) {format_entry_compact(r.entry, eff, warn)}")
    return "\n".join(lines)


def _parse_response(raw: str) -> _ToolCall | _FinalAnswer | None:
    """Parse LLM response into a tool call or final answer."""
    # Strip markdown fences
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1)

    obj_match = _JSON_OBJECT_RE.search(raw)
    if not obj_match:
        return None

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    tool = data.get("tool")
    if not tool:
        return None

    if tool == "done":
        args = data.get("args", {})
        return _FinalAnswer(
            entry_ids=args.get("entry_ids", []),
            reasoning=args.get("reasoning", ""),
        )

    return _ToolCall(tool=tool, args=data.get("args", {}))


async def _dispatch_tool(
    tool_call: _ToolCall,
    db: Database,
    embedder: EmbeddingClient | None,
) -> str:
    """Execute a tool call and return formatted results."""
    name = tool_call.tool
    args = tool_call.args

    if name == "hybrid_search":
        query = SearchQuery(
            query=args.get("query", ""),
            project_ref=args.get("project_ref"),
            entry_type=args.get("entry_type"),
            tags=args.get("tags"),
            limit=min(args.get("limit", 10), 10),
            include_stale=False,
        )
        results, filtered = await hybrid_search(db, embedder, query)
        header = f"hybrid_search({query.query!r}): {len(results)} results"
        if filtered:
            header += f" ({filtered} filtered)"
        return header + "\n" + _format_search_results(results)

    elif name == "graph_neighbors":
        node_id = args.get("node_id", "")
        limit = min(args.get("limit", 10), 20)
        neighbors = await get_neighbors(db, node_id, limit=limit)
        if not neighbors:
            return f"graph_neighbors({node_id!r}): no neighbors found"
        lines = [f"graph_neighbors({node_id!r}): {len(neighbors)} neighbors"]
        for nid, etype, direction in neighbors:
            lines.append(f"  {direction}: {nid} [{etype}]")
        return "\n".join(lines)

    elif name == "list_graph_nodes":
        type_filter = args.get("type_filter")
        vocab = await get_graph_vocabulary(db)
        if type_filter and type_filter in vocab:
            vocab = {type_filter: vocab[type_filter]}
        if not vocab:
            return "list_graph_nodes: empty vocabulary"
        lines = ["list_graph_nodes:"]
        for ntype, names in sorted(vocab.items()):
            lines.append(f"  {ntype}: {', '.join(names[:20])}")
        return "\n".join(lines)

    elif name == "decision_chain":
        entry_id = args.get("entry_id", "")
        chain = await supersedes_chain(db, entry_id)
        if not chain:
            return f"decision_chain({entry_id!r}): no chain found"
        lines = [f"decision_chain({entry_id!r}): {len(chain)} entries"]
        for i, cid in enumerate(chain):
            entry = await get_entry(db, cid)
            if entry:
                if len(chain) == 1:
                    pos = "current"
                elif i == 0:
                    pos = "original"
                elif i == len(chain) - 1:
                    pos = "current"
                else:
                    pos = f"v{i + 1}"
                lines.append(f"  [{pos}] {cid}: {entry.short_title}")
        return "\n".join(lines)

    elif name == "scope_entries":
        scope = args.get("scope", "")
        limit = min(args.get("limit", 20), 20)
        entry_ids = await entries_for_scope(db, scope)
        entry_ids = entry_ids[:limit]
        if not entry_ids:
            return f"scope_entries({scope!r}): no entries found"
        now = datetime.now(UTC)
        lines = [f"scope_entries({scope!r}): {len(entry_ids)} entries"]
        for eid in entry_ids:
            entry = await get_entry(db, eid)
            if entry and entry.is_active:
                anchor = entry.updated_at or entry.created_at or now
                eff = compute_effective_confidence(
                    entry.confidence_level,
                    entry.entry_type,
                    anchor,
                    now,
                    last_accessed=entry.last_accessed,
                )
                warn = staleness_warning(eff, entry.entry_type)
                lines.append(f"  {format_entry_compact(entry, eff, warn)}")
        return "\n".join(lines)

    return f"Unknown tool: {name}"


def _extract_entry_ids(messages: list[dict[str, str]]) -> list[str]:
    """Extract all kb-XXXXX IDs mentioned in conversation messages."""
    ids: list[str] = []
    seen: set[str] = set()
    for msg in messages:
        content = msg.get("content", "")
        for match in _ENTRY_ID_RE.findall(content):
            if match not in seen:
                seen.add(match)
                ids.append(match)
    return ids


async def agentic_query(
    db: Database,
    embedder: EmbeddingClient | None,
    llm: LLMProvider,
    question: str,
    *,
    max_tool_calls: int | None = None,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> AgentResult:
    """Run the agentic query loop.

    1. Fast-path: hybrid_search — if strong results, return immediately.
    2. Seed agent with fast-path results.
    3. Loop: LLM → parse → dispatch tool → append result.
    4. On done or exhaust: return collected entries.
    """
    if max_tool_calls is None:
        max_tool_calls = get_agentic_max_tool_calls()

    async def _emit(event: dict[str, Any]) -> None:
        if event_callback is not None:
            await event_callback(event)

    # --- Fast path: run hybrid_search first ---
    fast_query = SearchQuery(query=question, limit=5, include_stale=False)
    fast_results, _filtered = await hybrid_search(db, embedder, fast_query)

    if fast_results and fast_results[0].score >= _FAST_PATH_THRESHOLD:
        logger.debug("Agent fast-path: top score %.4f >= threshold", fast_results[0].score)
        entry_ids = [r.entry.id for r in fast_results]
        await _emit(
            {
                "type": "fast_path",
                "entry_ids": entry_ids,
                "top_score": fast_results[0].score,
            }
        )
        return AgentResult(
            entries=[(r.entry.id, f"search match (score: {r.score:.4f})") for r in fast_results],
            turns_used=0,
            reasoning="Fast-path: strong hybrid search results",
        )

    # --- Build initial messages ---
    seed = f"Question: {question}"
    if fast_results:
        seed += "\n\nInitial search results (may be weak):\n"
        seed += _format_search_results(fast_results)
    seed += "\n\nFind the best entries to answer this question."

    messages: list[dict[str, str]] = [
        {"role": "user", "content": seed},
    ]

    turns_used = 0
    await _emit(
        {
            "type": "agent_started",
            "seed_results": len(fast_results),
            "max_turns": max_tool_calls,
        }
    )

    # --- Agent loop ---
    seen_calls: set[str] = set()
    for turn_num in range(max_tool_calls):
        # Build prompt from messages
        await _emit({"type": "thinking", "turn": turn_num + 1})
        raw = await llm.generate_chat(messages, system=_AGENT_SYSTEM_PROMPT)

        if raw is None:
            logger.warning("Agent LLM returned None — falling back to fast-path results")
            break

        messages.append({"role": "assistant", "content": raw})
        parsed = _parse_response(raw)

        if parsed is None:
            # Parse failure — inject error and continue
            await _emit({"type": "parse_error", "turn": turn_num + 1})
            turns_used += 1
            messages.append(
                {
                    "role": "user",
                    "content": "ERROR: Could not parse your response as JSON. "
                    "Respond with exactly one JSON object.",
                }
            )
            continue

        if isinstance(parsed, _FinalAnswer):
            turns_used += 1
            entries = []
            for eid in parsed.entry_ids:
                entry = await get_entry(db, eid)
                if entry and entry.is_active:
                    entries.append((eid, "agent selected"))
            await _emit(
                {
                    "type": "agent_done",
                    "entry_ids": [eid for eid, _ in entries],
                    "reasoning": parsed.reasoning,
                    "turns_used": turns_used,
                }
            )
            return AgentResult(
                entries=entries,
                turns_used=turns_used,
                reasoning=parsed.reasoning,
            )

        # Tool call
        turns_used += 1

        # Dedup: skip identical tool calls (same name + args)
        call_key = json.dumps({"tool": parsed.tool, "args": parsed.args}, sort_keys=True)
        if call_key in seen_calls:
            logger.debug("Agent repeated tool call %s — skipping", parsed.tool)
            await _emit(
                {
                    "type": "tool_dedup",
                    "turn": turn_num + 1,
                    "tool": parsed.tool,
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": "ERROR: You already made this exact tool call. "
                    "Use a different query or call done() with your results.",
                }
            )
            continue
        seen_calls.add(call_key)

        await _emit(
            {
                "type": "tool_call",
                "turn": turn_num + 1,
                "tool": parsed.tool,
                "args": parsed.args,
            }
        )
        result_text = await _dispatch_tool(parsed, db, embedder)
        logger.debug("Agent tool %s → %d chars", parsed.tool, len(result_text))
        # Extract entry IDs from result for the event
        result_entry_ids = _ENTRY_ID_RE.findall(result_text)
        await _emit(
            {
                "type": "tool_result",
                "turn": turn_num + 1,
                "tool": parsed.tool,
                "entry_ids": list(dict.fromkeys(result_entry_ids)),
            }
        )
        messages.append({"role": "user", "content": result_text})

    # --- Exhaust: extract best-effort from conversation ---
    logger.debug("Agent exhausted %d turns, extracting best-effort", turns_used)
    all_ids = _extract_entry_ids(messages)
    entries = []
    for eid in all_ids:
        entry = await get_entry(db, eid)
        if entry and entry.is_active:
            entries.append((eid, "agent fallback"))

    # Fall back to fast-path results if nothing found
    if not entries and fast_results:
        entries = [(r.entry.id, f"search match (score: {r.score:.4f})") for r in fast_results]

    await _emit(
        {
            "type": "agent_done",
            "entry_ids": [eid for eid, _ in entries],
            "reasoning": "Exhausted tool call budget",
            "turns_used": turns_used,
        }
    )
    return AgentResult(
        entries=entries,
        turns_used=turns_used,
        reasoning="Exhausted tool call budget",
    )
