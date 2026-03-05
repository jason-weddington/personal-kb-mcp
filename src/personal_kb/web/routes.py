"""HTTP routes for the KB Explorer web server."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html
from personal_kb.llm.provider import LLMProvider
from personal_kb.web.events import event_to_status, sse_event

logger = logging.getLogger(__name__)


def register_routes(app: Any) -> None:
    """Register all HTTP routes on the FastAPI app."""
    from fastapi import Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the explorer HTML page."""
        data = await extract_graph_data(request.app.state.db)
        html = render_explorer_html(data)
        return HTMLResponse(content=html)

    @app.get("/api/graph")
    async def api_graph(request: Request) -> JSONResponse:
        """Return full graph data as JSON."""
        data = await extract_graph_data(request.app.state.db)
        return JSONResponse(content=data)

    @app.post("/api/query/stream")
    async def api_query_stream(request: Request) -> StreamingResponse:
        """Stream query events via SSE."""
        body = await request.json()
        question = body.get("question", "")
        db = request.app.state.db
        embedder = request.app.state.embedder
        query_llm = request.app.state.query_llm

        async def event_stream() -> AsyncGenerator[str]:
            # Classify query
            mode = "explore"
            if query_llm is not None and isinstance(query_llm, LLMProvider):
                from personal_kb.web.classifier import classify_query

                mode = await classify_query(query_llm, question)

            yield sse_event("classified", {"mode": mode})

            # Set up event queue for callback bridge
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def event_callback(event: dict[str, Any]) -> None:
                await queue.put(event)

            async def run_query() -> dict[str, Any]:
                """Run the query and return result data."""
                try:
                    if mode == "summarize":
                        from personal_kb.tools.kb_summarize import summarize_question

                        answer = await summarize_question(
                            db,
                            embedder,
                            query_llm,
                            question,
                            event_callback=event_callback,
                        )
                        return {"type": "summarize", "answer": answer}
                    else:
                        from personal_kb.tools.kb_ask import retrieve_entries

                        entries, turns = await retrieve_entries(
                            db,
                            embedder,
                            query_llm,
                            question,
                            event_callback=event_callback,
                        )
                        entry_data = []
                        for entry, context in entries:
                            entry_data.append(
                                {
                                    "id": entry.id,
                                    "short_title": entry.short_title,
                                    "entry_type": entry.entry_type.value
                                    if entry.entry_type
                                    else None,
                                    "tags": entry.tags or [],
                                    "context": context,
                                }
                            )
                        return {
                            "type": "explore",
                            "entries": entry_data,
                            "turns_used": turns,
                        }
                finally:
                    await queue.put(None)  # Signal completion

            # Start query as a task
            task = asyncio.create_task(run_query())

            # Drain events from the queue
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield sse_event(event["type"], event)
                status = event_to_status(event)
                if status:
                    yield sse_event("status", {"message": status})

            # Get final result
            try:
                result = await task
            except Exception:
                logger.exception("Query task failed")
                yield sse_event("error", {"message": "Query failed"})
                yield sse_event("stream_end", {})
                return

            if result["type"] == "summarize":
                yield sse_event(
                    "synthesis_result",
                    {
                        "answer": result["answer"],
                    },
                )
            else:
                yield sse_event(
                    "entries",
                    {
                        "entries": result["entries"],
                        "turns_used": result["turns_used"],
                    },
                )

            yield sse_event("stream_end", {})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
