"""LLM-based query planner — translates natural language to graph query plans."""

import json
import logging
import re
from dataclasses import dataclass

import aiosqlite

from personal_kb.db.queries import get_db_stats
from personal_kb.graph.queries import get_graph_vocabulary
from personal_kb.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_VALID_STRATEGIES = {"auto", "decision_trace", "timeline", "related", "connection"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_SYSTEM_PROMPT = """\
You are a knowledge graph query planner. Given a natural language question and \
a graph vocabulary, choose the best query strategy and resolve entity references.

Available strategies:
- auto: Hybrid search + graph expansion. Best for general questions or when unsure.
- decision_trace: Follow supersedes chains for decision history. Use when the \
question asks about WHY something was decided or how a decision evolved.
- timeline: Chronological entries for a scope. Use when the question asks about \
history, progression, or "what happened" in a specific area.
- related: BFS from a starting node. Use when the question asks "what relates to X" \
or "what else uses X".
- connection: Find paths between two nodes. Use when the question asks how two \
things are connected.

Node ID formats:
- tag:X (e.g., tag:python, tag:sqlite)
- project:X (e.g., project:personal-kb)
- person:X (e.g., person:jason)
- tool:X (e.g., tool:aiosqlite)
- concept:X (e.g., concept:async-io)
- technology:X (e.g., technology:fastapi)
- kb-XXXXX (entry IDs)

Output a single JSON object:
{
  "strategy": "auto|decision_trace|timeline|related|connection",
  "scope": "resolved node ID or null",
  "target": "second node ID (connection only) or null",
  "search_query": "refined search terms or null",
  "reasoning": "brief explanation of your choice"
}

Rules:
- Choose ONE strategy. When in doubt, use "auto".
- Resolve mentions to exact node IDs from the vocabulary provided.
- For "auto", provide a refined search_query if the original question is verbose.
- For "related" and "timeline", scope is required.
- For "connection", both scope and target are required.
- If you can't resolve a mention to a known node, use "auto" instead.\
"""


@dataclass
class QueryPlan:
    """Result of query planning: a structured graph query."""

    strategy: str
    scope: str | None = None
    target: str | None = None
    search_query: str | None = None
    reasoning: str | None = None


class QueryPlanner:
    """Translates natural language questions into structured query plans."""

    def __init__(self, db: aiosqlite.Connection, llm: LLMProvider) -> None:
        """Initialize with database and LLM provider."""
        self._db = db
        self._llm = llm

    async def plan(self, question: str) -> QueryPlan | None:
        """Generate a query plan for a question. Returns None on failure."""
        context = await self._build_context(question)
        raw = await self._llm.generate(context, system=_SYSTEM_PROMPT)
        if raw is None:
            return None
        return self._parse_plan(raw)

    async def _build_context(self, question: str) -> str:
        """Build the per-request context with graph stats and vocabulary."""
        parts: list[str] = []

        # Graph stats
        stats = await get_db_stats(self._db)
        nodes_by_type = stats.get("graph_nodes_by_type", {})
        edges_by_type = stats.get("graph_edges_by_type", {})
        parts.append("Graph stats:")
        parts.append(f"  Nodes by type: {json.dumps(nodes_by_type)}")
        parts.append(f"  Edges by type: {json.dumps(edges_by_type)}")
        parts.append(f"  Active entries: {stats.get('active_entries', 0)}")

        # Vocabulary
        vocab = await get_graph_vocabulary(self._db)
        if vocab:
            parts.append("\nGraph vocabulary (available node names by type):")
            for node_type, names in sorted(vocab.items()):
                parts.append(f"  {node_type}: {', '.join(names)}")

        parts.append(f"\nQuestion: {question}")
        return "\n".join(parts)

    def _parse_plan(self, raw: str) -> QueryPlan | None:
        """Parse LLM response into a QueryPlan."""
        # Strip markdown fences if present
        fence_match = _FENCE_RE.search(raw)
        if fence_match:
            raw = fence_match.group(1)

        # Find JSON object
        obj_match = _JSON_OBJECT_RE.search(raw)
        if not obj_match:
            logger.warning("No JSON object found in planner response")
            return None

        try:
            data = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            logger.warning("Malformed JSON in planner response")
            return None

        if not isinstance(data, dict):
            return None

        strategy = data.get("strategy", "auto")
        if strategy not in _VALID_STRATEGIES:
            logger.warning("Invalid strategy '%s' from planner, falling back to auto", strategy)
            strategy = "auto"

        return QueryPlan(
            strategy=strategy,
            scope=data.get("scope"),
            target=data.get("target"),
            search_query=data.get("search_query"),
            reasoning=data.get("reasoning"),
        )
