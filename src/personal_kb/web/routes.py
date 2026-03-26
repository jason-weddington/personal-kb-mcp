"""HTTP routes for the KB Explorer web server."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html
from personal_kb.web.events import event_to_status, sse_event

logger = logging.getLogger(__name__)


async def _ingest_binary_file(
    ingester: Any,
    b64_content: str,
    name: str,
    *,
    project_ref: str | None = None,
    progress_callback: Any = None,
) -> Any:
    """Decode a base64-encoded file, write to a temp file, and ingest via ingest_file()."""
    import base64
    import tempfile
    from pathlib import Path

    raw = base64.b64decode(b64_content)
    suffix = Path(name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = Path(tmp.name)
    try:
        result = await ingester.ingest_file(
            tmp_path,
            project_ref=project_ref,
            progress_callback=progress_callback,
        )
        # Replace temp path with original filename in result
        result.path = name
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


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

    @app.get("/api/projects")
    async def api_projects(request: Request) -> JSONResponse:
        """Return distinct project_ref values from active entries."""
        db = request.app.state.db
        cursor = await db.execute(
            "SELECT DISTINCT project_ref FROM knowledge_entries"
            " WHERE is_active = 1 AND project_ref IS NOT NULL"
            " ORDER BY project_ref"
        )
        rows = await cursor.fetchall()
        return JSONResponse([row[0] for row in rows])

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

        # Use synthesis_llm (Sonnet) for human-facing summarization if available
        synthesis_llm = getattr(request.app.state, "synthesis_llm", None)

        async def event_stream() -> AsyncGenerator[str]:
            # Classify query
            mode = "explore"
            if query_llm is not None:
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

                        # Use Sonnet for human-facing synthesis, Haiku for retrieval
                        answer = await summarize_question(
                            db,
                            embedder,
                            query_llm,
                            question,
                            event_callback=event_callback,
                            synthesis_llm=synthesis_llm,
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
        from personal_kb.web.chat import WriteDeps, get_or_create_session, get_session

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
        # Use Sonnet for human-facing chat if available
        chat_llm = getattr(request.app.state, "synthesis_llm", None) or query_llm

        # Build write deps if store is available
        write_deps: WriteDeps | None = None
        store = getattr(request.app.state, "store", None)
        if store is not None:
            write_deps = WriteDeps(
                store=store,
                graph_builder=getattr(request.app.state, "graph_builder", None),
                graph_enricher=getattr(request.app.state, "graph_enricher", None),
                extraction_llm=getattr(request.app.state, "extraction_llm", None),
                contributor=getattr(request.app.state, "contributor", None),
                team=getattr(request.app.state, "team", None),
            )

        async def chat_stream() -> AsyncGenerator[str]:
            if chat_llm is None:
                yield sse_event("error", {"message": "LLM not available"})
                yield sse_event("stream_end", {})
                return

            # Get or create session
            session = get_session(session_id) if session_id else None

            if session is None:
                session = get_or_create_session(None, db, embedder, chat_llm, write_deps=write_deps)
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

    @app.post("/api/ingest_url")
    async def api_ingest_url(request: Request) -> JSONResponse:
        """Ingest a URL into the KB."""
        body = await request.json()
        url = body.get("url", "").strip()
        project_ref = body.get("project_ref")

        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)

        store = getattr(request.app.state, "store", None)
        extraction_llm = getattr(request.app.state, "extraction_llm", None)
        graph_builder = getattr(request.app.state, "graph_builder", None)
        if store is None or extraction_llm is None or graph_builder is None:
            return JSONResponse(
                {"error": "Ingestion not available (missing dependencies)"},
                status_code=503,
            )

        db = request.app.state.db
        embedder = request.app.state.embedder
        graph_enricher = getattr(request.app.state, "graph_enricher", None)
        contributor = getattr(request.app.state, "contributor", None)
        team = getattr(request.app.state, "team", None)

        from personal_kb.ingest.ingester import FileIngester

        ingester = FileIngester(
            db=db,
            store=store,
            embedder=embedder,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            llm=extraction_llm,
            contributor=contributor,
            team=team,
        )

        try:
            result = await ingester.ingest_url(url, project_ref=project_ref)
        except Exception as exc:
            logger.exception("Ingest URL failed: %s", url)
            return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

        return JSONResponse(
            {
                "action": result.action,
                "reason": result.reason,
                "entry_count": result.entry_count,
                "entry_ids": result.entry_ids,
                "summary": result.summary,
            }
        )

    @app.post("/api/ingest/stream", response_model=None)
    async def api_ingest_stream(request: Request) -> StreamingResponse | JSONResponse:
        """Stream ingestion progress for URLs and/or file content via SSE."""
        body = await request.json()
        items = body.get("items", [])
        project_ref = body.get("project_ref")

        if not items:
            return JSONResponse({"error": "items is required"}, status_code=400)

        store = getattr(request.app.state, "store", None)
        extraction_llm = getattr(request.app.state, "extraction_llm", None)
        graph_builder = getattr(request.app.state, "graph_builder", None)
        if store is None or extraction_llm is None or graph_builder is None:
            return JSONResponse(
                {"error": "Ingestion not available (missing dependencies)"},
                status_code=503,
            )

        db = request.app.state.db
        embedder = request.app.state.embedder
        graph_enricher = getattr(request.app.state, "graph_enricher", None)
        contributor = getattr(request.app.state, "contributor", None)
        team = getattr(request.app.state, "team", None)

        from personal_kb.ingest.ingester import FileIngester

        ingester = FileIngester(
            db=db,
            store=store,
            embedder=embedder,
            graph_builder=graph_builder,
            graph_enricher=graph_enricher,
            llm=extraction_llm,
            contributor=contributor,
            team=team,
        )

        async def event_stream() -> AsyncGenerator[str]:
            queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
            all_entry_ids: list[str] = []
            total_entries = 0

            async def progress_cb(event: dict[str, Any]) -> None:
                await queue.put(event)

            async def run_batch() -> None:
                try:
                    for idx, item in enumerate(items):
                        item_type = item.get("type", "")
                        if item_type == "url":
                            source = item.get("value", "").strip()
                        else:
                            source = item.get("name", "file")
                        await queue.put(
                            {
                                "type": "batch_progress",
                                "batch_index": idx,
                                "batch_total": len(items),
                                "source": source,
                            }
                        )
                        try:
                            if item_type == "url":
                                result = await ingester.ingest_url(
                                    source,
                                    project_ref=project_ref,
                                    progress_callback=progress_cb,
                                )
                            elif item_type == "file":
                                if item.get("encoding") == "base64":
                                    result = await _ingest_binary_file(
                                        ingester,
                                        item.get("content", ""),
                                        item.get("name", "file"),
                                        project_ref=project_ref,
                                        progress_callback=progress_cb,
                                    )
                                else:
                                    result = await ingester.ingest_text(
                                        item.get("content", ""),
                                        item.get("name", "file"),
                                        project_ref=project_ref,
                                        progress_callback=progress_cb,
                                    )
                            else:
                                await queue.put(
                                    {
                                        "type": "ingest_error",
                                        "source": source,
                                        "error": f"Unknown item type: {item_type}",
                                    }
                                )
                                continue
                        except Exception as exc:
                            logger.exception("Ingest item failed: %s", source)
                            await queue.put(
                                {
                                    "type": "ingest_error",
                                    "source": source,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                            continue

                        await queue.put(
                            {
                                "type": "item_done",
                                "source": source,
                                "action": result.action,
                                "reason": result.reason,
                                "entry_count": result.entry_count,
                                "entry_ids": result.entry_ids,
                            }
                        )
                        all_entry_ids.extend(result.entry_ids)
                        nonlocal total_entries
                        total_entries += result.entry_count

                    await queue.put(
                        {
                            "type": "batch_done",
                            "total_entries": total_entries,
                            "entry_ids": all_entry_ids,
                        }
                    )
                finally:
                    await queue.put(None)

            task = asyncio.create_task(run_batch())

            while True:
                event = await queue.get()
                if event is None:
                    break
                yield sse_event(event["type"], event)
                status = event_to_status(event)
                if status:
                    yield sse_event("status", {"message": status})

            try:
                await task
            except Exception as exc:
                logger.exception("Ingest batch task failed")
                yield sse_event("error", {"message": f"{type(exc).__name__}: {exc}"})

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
