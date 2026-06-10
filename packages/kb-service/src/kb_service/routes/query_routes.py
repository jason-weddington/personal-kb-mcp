"""KB query routes: agentic ask and summarize endpoints.

These endpoints run multi-second agentic LLM loops (retrieval + ReAct +
synthesis; commonly 10-60 s, can exceed 30 s).  uvicorn applies no request
timeout so the server lets them run.  Clients MUST set an HTTP read timeout of
at least 120 seconds (the P5 thin MCP client uses httpx with a 120 s read
timeout).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request

from kb_service.auth import get_current_user
from kb_service.models import (
    AskEntry,
    AskRequest,
    AskResponse,
    SummarizeRequest,
    SummarizeResponse,
    User,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> AskResponse:
    """Agentic ask over the knowledge base (authenticated).

    Runs a ReAct retrieval loop and synthesises a response from the matched
    entries.  Commonly 10-60 s; can exceed 30 s.  Clients must set an HTTP
    read timeout of at least 120 seconds.

    Agentic knobs (``agentic``, ``max_tool_calls``) are resolved from server
    env config (``KB_AGENTIC_QUERY``, ``KB_AGENTIC_MAX_CALLS``) and are NOT
    accepted from the request body.
    """
    entries_with_context, agent_turns_used = await request.app.state.kb.ask(
        body.question,
        scope=body.scope,
        limit=body.limit,
        include_graph_context=body.include_graph_context,
    )
    return AskResponse(
        entries=[
            AskEntry(entry=entry, context=ctx) for entry, ctx in entries_with_context
        ],
        agent_turns_used=agent_turns_used,
    )


@router.post("/summarize", response_model=SummarizeResponse)
async def summarize(
    body: SummarizeRequest,
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> SummarizeResponse:
    """Agentic summarize over the knowledge base (authenticated).

    Runs a ReAct retrieval loop then synthesises a natural-language answer.
    Commonly 10-60 s; can exceed 30 s.  Clients must set an HTTP read timeout
    of at least 120 seconds.

    Agentic knobs (``agentic``, ``agentic_synthesis``, ``max_tool_calls``) are
    resolved from server env config and are NOT accepted from the request body.
    """
    answer: str = await request.app.state.kb.summarize(
        body.question,
        scope=body.scope,
        limit=body.limit,
    )
    return SummarizeResponse(answer=answer)
