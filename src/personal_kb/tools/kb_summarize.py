"""kb_summarize MCP tool — synthesized answers with citations."""

import logging
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.server.context import Context
from pydantic import Field

from personal_kb.llm.provider import LLMProvider
from personal_kb.tools.kb_ask import _strategy_auto

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

        Citations reference entry IDs [kb-XXXXX].
        Retrieves relevant entries and uses an LLM to synthesize an answer.
        Falls back to raw results if LLM is unavailable.
        """
        if ctx is None:
            raise RuntimeError("Context not injected")

        lifespan = ctx.lifespan_context
        db = lifespan["db"]
        embedder = lifespan["embedder"]
        query_llm = lifespan.get("query_llm")

        # Retrieve entries using the auto strategy
        raw_results = await _strategy_auto(
            db,
            embedder,
            question,
            scope,
            include_graph_context=True,
            limit=limit,
        )

        if raw_results == "No results found.":
            return "No entries found matching your question."

        # Try LLM synthesis
        if query_llm is not None and isinstance(query_llm, LLMProvider):
            synthesis = await _synthesize(query_llm, question, raw_results)
            if synthesis is not None:
                return synthesis

            return f"(LLM synthesis failed — showing raw results)\n\n{raw_results}"

        return f"(LLM unavailable — showing raw results)\n\n{raw_results}"


async def _synthesize(llm: LLMProvider, question: str, raw_results: str) -> str | None:
    """Synthesize an answer from retrieved entries using the LLM."""
    prompt = f"Question: {question}\n\nRetrieved entries:\n{raw_results}"
    return await llm.generate(prompt, system=_SYNTHESIS_SYSTEM_PROMPT)
