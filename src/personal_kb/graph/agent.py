"""Re-export shim — real code moved to ``kb_core.graph.agent``.

Shim: real code moved to kb_core (kb-core extraction wave 3). The
``agentic_query`` wrapper preserves the historical env-default behavior
for any caller that does not yet pass ``max_tool_calls`` explicitly; the
pure ``kb_core`` implementation now takes the budget as an explicit
parameter so the engine reads no environment.

Channel-rewiring wave removes this.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from kb_core.db.backend import Database
from kb_core.graph.agent import AgentResult, _parse_response
from kb_core.graph.agent import agentic_query as _kb_core_agentic_query
from kb_core.llm.provider import LLMProvider
from kb_core.search.embedder_protocol import Embedder

__all__ = ["AgentResult", "_parse_response", "agentic_query"]


async def agentic_query(
    db: Database,
    embedder: Embedder | None,
    llm: LLMProvider,
    question: str,
    *,
    max_tool_calls: int | None = None,
    event_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> AgentResult:
    """Backward-compatible wrapper: fills ``max_tool_calls`` from env when omitted.

    The kb_core implementation requires ``max_tool_calls`` explicitly so
    the engine reads no environment. Older callers (tests, eval scripts)
    that omit it get the historical ``KB_AGENTIC_MAX_CALLS`` default
    here.
    """
    if max_tool_calls is None:
        from personal_kb.config import get_agentic_max_tool_calls

        max_tool_calls = get_agentic_max_tool_calls()
    return await _kb_core_agentic_query(
        db,
        embedder,
        llm,
        question,
        max_tool_calls=max_tool_calls,
        event_callback=event_callback,
    )
