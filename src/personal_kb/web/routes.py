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

    @app.get("/api/entry/{entry_id}")
    async def api_entry(entry_id: str, request: Request) -> JSONResponse:
        """Return full entry details by ID."""
        from personal_kb.db.queries import get_entry

        entry = await get_entry(request.app.state.db, entry_id)
        if entry is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(
            {
                "id": entry.id,
                "short_title": entry.short_title,
                "long_title": entry.long_title,
                "knowledge_details": entry.knowledge_details,
                "entry_type": entry.entry_type.value if entry.entry_type else None,
                "tags": entry.tags or [],
                "project_ref": entry.project_ref,
                "confidence_level": entry.confidence_level,
            }
        )

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
            collected_entry_ids: list[str] = []

            async def event_callback(event: dict[str, Any]) -> None:
                # Collect entry IDs from agent events for chat seeding
                if event.get("type") in ("fast_path", "agent_done"):
                    collected_entry_ids.extend(event.get("entry_ids", []))
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
                        return {
                            "type": "summarize",
                            "answer": answer,
                            "entry_ids": collected_entry_ids,
                        }
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
            except Exception as exc:
                logger.exception("Query task failed")
                detail = f"{type(exc).__name__}: {exc}"
                yield sse_event("error", {"message": detail})
                yield sse_event("stream_end", {})
                return

            if result["type"] == "summarize":
                yield sse_event(
                    "synthesis_result",
                    {
                        "answer": result["answer"],
                        "question": question,
                        "entry_ids": result.get("entry_ids", []),
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

    @app.post("/api/chat/stream")
    async def api_chat_stream(request: Request) -> StreamingResponse:
        """Stream a follow-up chat response via SSE."""
        from personal_kb.web.chat import get_or_create_session, get_session

        body = await request.json()
        session_id = body.get("session_id")
        message = body.get("message", "")
        # For seeding a new session from a summarize result
        seed_question = body.get("seed_question")
        seed_answer = body.get("seed_answer")
        seed_entry_ids = body.get("seed_entry_ids", [])

        db = request.app.state.db
        embedder = request.app.state.embedder
        query_llm = request.app.state.query_llm

        async def chat_stream() -> AsyncGenerator[str]:
            if query_llm is None or not isinstance(query_llm, LLMProvider):
                yield sse_event("error", {"message": "LLM not available"})
                yield sse_event("stream_end", {})
                return

            # Get or create session
            session = get_session(session_id) if session_id else None

            if session is None:
                session = get_or_create_session(None, db, embedder, query_llm)
                if seed_question and seed_answer:
                    session.seed(seed_question, seed_answer, seed_entry_ids)

            yield sse_event(
                "chat_session",
                {"session_id": session.id},
            )

            # Set up event queue
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

            async def event_callback(event: dict[str, Any]) -> None:
                await queue.put(event)

            async def run_chat() -> str:
                try:
                    return await session.reply(message, event_callback=event_callback)
                finally:
                    await queue.put(None)

            task = asyncio.create_task(run_chat())

            # Drain events
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield sse_event(event.get("type", "status"), event)

            # Get result
            try:
                answer = await task
            except Exception as exc:
                logger.exception("Chat task failed")
                detail = f"{type(exc).__name__}: {exc}"
                yield sse_event("error", {"message": detail})
                yield sse_event("stream_end", {})
                return

            yield sse_event(
                "chat_response",
                {"answer": answer, "session_id": session.id},
            )
            yield sse_event("stream_end", {})

        return StreamingResponse(
            chat_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
