"""HTTP routes for the KB Explorer web server.

The handlers reach into the :class:`~kb_core.knowledge_base.KnowledgeBase`
on ``app.state.kb`` for every read/write — search, ask, summarize,
ingest, get. The web channel no longer imports from
:mod:`personal_kb.tools` (the historical backdoor): everything routes
through the facade plus a couple of channel-local helpers
(SSE event encoding, chat history persistence).
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html
from personal_kb.web.events import event_to_status, sse_event

logger = logging.getLogger(__name__)


def _build_ingester(kb: Any) -> Any:
    """Build a :class:`~kb_core.ingest.ingester.FileIngester` from a facade.

    Mirrors :meth:`KnowledgeBase._build_ingester` so the streaming routes
    can attach a ``progress_callback`` (the facade's ``ingest_*`` methods
    don't expose one). Uses only the facade's public accessors —
    nothing imported from :mod:`personal_kb.tools`.
    """
    from kb_core.ingest.dedup_agent import DedupAgent
    from kb_core.ingest.ingester import FileIngester

    dedup_agent: Any | None = None
    if kb.config.ingest.agentic_ingest:
        dedup_agent = DedupAgent(
            kb.db,
            kb.embedder,
            kb.extraction_llm,
            threshold=kb.config.ingest.dedup_threshold,
        )
    return FileIngester(
        db=kb.db,
        store=kb.knowledge_store,
        embedder=kb.embedder,
        graph_builder=kb.graph_builder,
        graph_enricher=kb.graph_enricher,
        llm=kb.extraction_llm,
        dedup_agent=dedup_agent,
        contributor=kb.config.attribution.contributor,
        team=kb.config.attribution.team,
        config=kb.config.ingest,
    )


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
            display_name=name,
            project_ref=project_ref,
            progress_callback=progress_callback,
        )
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
        kb = request.app.state.kb
        data = await extract_graph_data(kb.db)
        html = render_explorer_html(data)
        return HTMLResponse(content=html)

    @app.get("/api/graph")
    async def api_graph(request: Request) -> JSONResponse:
        """Return full graph data as JSON."""
        kb = request.app.state.kb
        data = await extract_graph_data(kb.db)
        return JSONResponse(content=data)

    @app.get("/api/projects")
    async def api_projects(request: Request) -> JSONResponse:
        """Return distinct project_ref values from active entries."""
        kb = request.app.state.kb
        cursor = await kb.db.execute(
            "SELECT DISTINCT project_ref FROM knowledge_entries"
            " WHERE is_active = 1 AND project_ref IS NOT NULL"
            " ORDER BY project_ref"
        )
        rows = await cursor.fetchall()
        return JSONResponse([row[0] for row in rows])

    @app.get("/api/entry/{entry_id}")
    async def api_entry(entry_id: str, request: Request) -> JSONResponse:
        """Return full entry details by ID."""
        kb = request.app.state.kb
        entry = await kb.get(entry_id)
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
        kb = request.app.state.kb

        async def event_stream() -> AsyncGenerator[str]:
            # Classify query
            mode = "explore"
            if kb.query_llm is not None:
                from personal_kb.web.classifier import classify_query

                mode = await classify_query(kb.query_llm, question)

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
                        answer = await kb.summarize(
                            question,
                            event_callback=event_callback,
                        )
                        return {
                            "type": "summarize",
                            "answer": answer,
                            "entry_ids": collected_entry_ids,
                        }
                    entries, turns = await kb.ask(
                        question,
                        event_callback=event_callback,
                    )
                    entry_data = []
                    for entry, context in entries:
                        entry_data.append(
                            {
                                "id": entry.id,
                                "short_title": entry.short_title,
                                "entry_type": entry.entry_type.value if entry.entry_type else None,
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
        from personal_kb.web.chat import (
            ChatSession,
            cache_session,
            get_or_create_session,
            get_session,
        )
        from personal_kb.web.chat_history import derive_title

        body = await request.json()
        session_id = body.get("session_id")
        message = body.get("message", "")
        # For seeding a new session from a summarize result
        seed_question = body.get("seed_question")
        seed_answer = body.get("seed_answer")
        seed_entry_ids = body.get("seed_entry_ids", [])
        mode = body.get("mode", "")

        kb = request.app.state.kb
        # Use Sonnet for human-facing chat if available, else fall back to query LLM
        chat_llm = kb.synthesis_llm or kb.query_llm
        ch = getattr(request.app.state, "chat_history", None)

        async def chat_stream() -> AsyncGenerator[str]:
            if chat_llm is None:
                yield sse_event("error", {"message": "LLM not available"})
                yield sse_event("stream_end", {})
                return

            is_new = False
            # Get or create session
            session = get_session(session_id) if session_id else None

            if session is None and session_id and ch and await ch.chat_exists(session_id):
                saved_msgs = await ch.get_messages(session_id)
                session = ChatSession.from_saved(
                    session_id,
                    saved_msgs,
                    kb,
                    chat_llm,
                )
                cache_session(session)

            if session is None:
                is_new = True
                session = get_or_create_session(
                    None,
                    kb,
                    chat_llm,
                )
                if seed_question and seed_answer:
                    session.seed(seed_question, seed_answer, seed_entry_ids)

            yield sse_event(
                "chat_session",
                {"session_id": session.id},
            )

            # Persist new chat to history
            if is_new and ch:
                title = derive_title(seed_question or message)
                try:
                    await ch.create_chat(session.id, title, mode)
                    if seed_question and seed_answer:
                        await ch.save_messages_bulk(
                            session.id,
                            [
                                {"role": "user", "content": seed_question},
                                {"role": "assistant", "content": seed_answer},
                            ],
                        )
                except Exception:
                    logger.warning("Failed to persist new chat", exc_info=True)

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

            # Persist the exchange
            if ch:
                try:
                    await ch.save_message(session.id, "user", message)
                    await ch.save_message(session.id, "assistant", answer)
                except Exception:
                    logger.warning("Failed to persist chat messages", exc_info=True)

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

    @app.post("/api/chat/create")
    async def api_chat_create(request: Request) -> JSONResponse:
        """Create a chat with seed messages (before any follow-up)."""
        from personal_kb.web.chat_history import derive_title

        ch = getattr(request.app.state, "chat_history", None)
        if ch is None:
            return JSONResponse({"error": "Chat history not available"}, status_code=503)

        body = await request.json()
        chat_id = body.get("chat_id", "")
        question = body.get("question", "")
        answer = body.get("answer", "")
        mode = body.get("mode", "")

        if not chat_id or not question:
            return JSONResponse({"error": "chat_id and question required"}, status_code=400)

        # Idempotent — skip if already exists
        if await ch.chat_exists(chat_id):
            return JSONResponse({"ok": True, "id": chat_id})

        title = derive_title(question)
        await ch.create_chat(chat_id, title, mode)
        messages = [{"role": "user", "content": question}]
        if answer:
            messages.append({"role": "assistant", "content": answer})
        await ch.save_messages_bulk(chat_id, messages)
        return JSONResponse({"ok": True, "id": chat_id})

    @app.get("/api/chat/history")
    async def api_chat_history(request: Request) -> JSONResponse:
        """Return recent chat conversations."""
        ch = getattr(request.app.state, "chat_history", None)
        if ch is None:
            return JSONResponse([])
        chats = await ch.list_chats()
        return JSONResponse(chats)

    @app.get("/api/chat/{chat_id}/messages")
    async def api_chat_messages(chat_id: str, request: Request) -> JSONResponse:
        """Return all messages for a saved chat."""
        ch = getattr(request.app.state, "chat_history", None)
        if ch is None:
            return JSONResponse({"error": "Chat history not available"}, status_code=503)
        if not await ch.chat_exists(chat_id):
            return JSONResponse({"error": "Chat not found"}, status_code=404)
        messages = await ch.get_messages(chat_id)
        return JSONResponse(messages)

    @app.delete("/api/chat/{chat_id}")
    async def api_chat_delete(chat_id: str, request: Request) -> JSONResponse:
        """Delete a chat and its messages."""
        from personal_kb.web.chat import _sessions

        ch = getattr(request.app.state, "chat_history", None)
        if ch is None:
            return JSONResponse({"error": "Chat history not available"}, status_code=503)
        deleted = await ch.delete_chat(chat_id)
        if not deleted:
            return JSONResponse({"error": "Chat not found"}, status_code=404)
        _sessions.pop(chat_id, None)
        return JSONResponse({"ok": True})

    @app.post("/api/ingest_url")
    async def api_ingest_url(request: Request) -> JSONResponse:
        """Ingest a URL into the KB."""
        body = await request.json()
        url = body.get("url", "").strip()
        project_ref = body.get("project_ref")

        if not url:
            return JSONResponse({"error": "url is required"}, status_code=400)

        kb = request.app.state.kb
        if kb.extraction_llm is None or kb.embedder is None:
            return JSONResponse(
                {"error": "Ingestion not available (missing dependencies)"},
                status_code=503,
            )

        try:
            result = await kb.ingest_url(url, project_ref=project_ref)
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

        kb = request.app.state.kb
        if kb.extraction_llm is None or kb.embedder is None:
            return JSONResponse(
                {"error": "Ingestion not available (missing dependencies)"},
                status_code=503,
            )

        # Streaming routes want per-item progress events flowing into the
        # SSE queue; the facade's ``ingest_*`` methods don't expose a
        # ``progress_callback``. Build a FileIngester from the facade's
        # public accessors so the streaming UX stays granular.
        ingester = _build_ingester(kb)

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
