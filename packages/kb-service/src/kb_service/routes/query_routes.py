"""KB query routes: agentic ask, summarize, and SSE query-stream endpoints.

The ask/summarize endpoints run multi-second agentic LLM loops (retrieval +
ReAct + synthesis; commonly 10-60 s, can exceed 30 s).  uvicorn applies no
request timeout so the server lets them run.  Clients MUST set an HTTP read
timeout of at least 120 seconds (the P5 thin MCP client uses httpx with a
120 s read timeout).

The ``/query/stream`` endpoint uses Server-Sent Events and requires JWT auth
via the ``?token=`` query parameter (EventSource cannot set Authorization
headers).  API keys are NOT accepted on that endpoint.
"""

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from kb_service.auth import get_current_user, get_current_user_sse
from kb_service.classifier import classify_query
from kb_service.models import (
    AskEntry,
    AskRequest,
    AskResponse,
    QueryStreamRequest,
    SummarizeRequest,
    SummarizeResponse,
    User,
)
from kb_service.sse import event_to_status, sse_event

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


@router.post("/query/stream")
async def query_stream(
    body: QueryStreamRequest,
    request: Request,
    _user: Annotated[User, Depends(get_current_user_sse)],
) -> StreamingResponse:
    """SSE query stream — classify, then ask or summarize with live progress.

    Authenticates via ``?token=<jwt>`` query parameter only.  API keys do NOT
    work on this endpoint (EventSource cannot send Authorization headers).

    Event sequence:

    1. ``classified`` — ``{mode: "explore"|"summarize"}``
    2. Zero or more engine progress events (each optionally followed by a
       ``status`` event with a human-readable message).
    3. On success:
       - summarize mode → ``synthesis_result``
         ``{answer, question, entry_ids: [...]}``.
       - explore mode → ``entries``
         ``{entries: [{id, short_title, entry_type, tags, context}],
         turns_used}``.
    4. On error: ``error`` ``{message: "ExcType: detail"}``.
    5. Always last: ``stream_end`` ``{}``.

    Args:
        body: Request body; ``question`` is required (422 if omitted).
        request: FastAPI request (provides ``app.state.kb``).
        _user: Authenticated user via JWT query param.

    Returns:
        ``StreamingResponse`` with ``text/event-stream`` media type.
    """
    kb = request.app.state.kb
    question = body.question

    async def _generate() -> AsyncGenerator[str]:
        # (a) classify — skip LLM if query_llm is None (explore fallback)
        mode: str
        if kb.query_llm is None:
            mode = "explore"
        else:
            mode = await classify_query(kb.query_llm, question)
        yield sse_event("classified", {"mode": mode})

        # (b) shared queue and entry-id collector
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        collected_entry_ids: list[str] = []

        async def _cb(event: dict[str, Any]) -> None:
            """Forward kb-core event to queue; collect entry IDs on key events."""
            if event.get("type") in ("fast_path", "agent_done"):
                collected_entry_ids.extend(event.get("entry_ids", []))
            await queue.put(event)

        # (c) run ask or summarize as a background task; always sentinel-close
        async def _run() -> Any:
            try:
                if mode == "summarize":
                    return await kb.summarize(question, event_callback=_cb)
                return await kb.ask(question, event_callback=_cb)
            finally:
                await queue.put(None)

        task: asyncio.Task[Any] = asyncio.create_task(_run())

        # (d) drain events until sentinel
        while True:
            ev: dict[str, Any] | None = await queue.get()
            if ev is None:
                break
            yield sse_event(ev["type"], ev)
            status_msg = event_to_status(ev)
            if status_msg:
                yield sse_event("status", {"message": status_msg})

        # (e) collect task result or exception
        try:
            result: Any = await task
        except Exception as exc:
            yield sse_event("error", {"message": f"{type(exc).__name__}: {exc}"})
            yield sse_event("stream_end", {})
            return

        # (f) success — emit mode-specific result
        if mode == "summarize":
            yield sse_event(
                "synthesis_result",
                {
                    "answer": result,
                    "question": question,
                    "entry_ids": collected_entry_ids,
                },
            )
        else:
            entries_with_ctx, turns_used = result
            yield sse_event(
                "entries",
                {
                    "entries": [
                        {
                            "id": e.id,
                            "short_title": e.short_title,
                            "entry_type": (
                                e.entry_type.value if e.entry_type else None
                            ),
                            "tags": e.tags or [],
                            "context": ctx,
                        }
                        for e, ctx in entries_with_ctx
                    ],
                    "turns_used": turns_used,
                },
            )

        # (g) always close the stream
        yield sse_event("stream_end", {})

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
