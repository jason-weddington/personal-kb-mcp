"""Chat routes: streaming conversation and persistent chat history.

Provides five endpoints under ``/api/chat``:

- ``POST /stream`` — SSE chat stream (JWT ``?token=`` query param, no API keys).
- ``POST /create`` — create/idempotently retrieve a chat session.
- ``GET  /history`` — list the authenticated user's chat sessions.
- ``GET  /{chat_id}/messages`` — retrieve messages for one chat.
- ``DELETE /{chat_id}`` — delete a chat session.

All non-stream endpoints use ``Depends(get_current_user)`` (JWT or API key).
The stream endpoint uses ``Depends(get_current_user_sse)`` (JWT ``?token=``
query param only — EventSource clients cannot set Authorization headers).

Chat history is imported as a MODULE so tests can monkeypatch individual
function attributes (``kb_service.chat_history.create_chat``, etc.) without
importing those names directly.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Annotated, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from kb_service import chat_history
from kb_service.attribution import resolve_attribution
from kb_service.auth import _auth_mode, get_current_user, get_current_user_from_token
from kb_service.chat import ChatSession, cache_session, get_session
from kb_service.chat_history import derive_title
from kb_service.models import (
    ChatCreateRequest,
    ChatListItem,
    ChatMessageItem,
    ChatOkResponse,
    ChatStreamRequest,
    User,
)
from kb_service.sse import sse_event

logger = logging.getLogger(__name__)


async def _block_in_no_auth_mode() -> None:
    """Router-level guard: return 404 for every ``/api/chat/*`` hit in no-auth mode.

    Chat persistence reads/writes the service-auth Postgres pool (``chats`` /
    ``chat_messages`` tables), but ``main.py`` deliberately leaves that pool
    uninitialized when ``KB_AUTH_MODE=none`` — so a direct HTTP hit to any
    chat endpoint would otherwise surface as an opaque 500 from
    ``database.get_db()``. The SPA already hides the chat UI in no-auth mode
    (Sidebar + route redirect, fix 991ce1d), so this is defense-in-depth:
    map a known-broken backend state to a clean ``404 Not Found`` instead
    of an uninitialized-pool stack trace.

    ``_auth_mode()`` is read per-call (not cached) so this stays consistent
    with the rest of the codebase, where ``KB_AUTH_MODE`` can be flipped at
    runtime (e.g. by tests via ``monkeypatch.setenv``).
    """
    if _auth_mode() == "none":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat is not available in single-user no-auth mode",
        )


router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(_block_in_no_auth_mode)],
)


@router.post("/stream")
async def chat_stream(
    body: ChatStreamRequest,
    request: Request,
    token: Annotated[str, Query()],
) -> StreamingResponse:
    """SSE chat stream — authenticate via ``?token=<jwt>`` query param.

    Authenticates via the JWT ``?token=`` query parameter only.  API keys are
    NOT accepted on this endpoint.

    Event sequence:

    1. ``chat_session`` — ``{session_id: "<uuid>"}`` (always first).
    2. ``chat_thinking`` — emitted just before the LLM call.
    3. (Optional) ``chat_tool_result`` — when a tool was dispatched.
    4. ``chat_response`` — ``{answer: "...", session_id: "..."}``.
    5. On error: ``error`` — ``{message: "ExcType: detail"}``.
    6. Always last: ``stream_end`` — ``{}``.

    Args:
        body: Chat stream request body.
        request: FastAPI request (provides ``app.state.kb``).
        token: JWT passed as a ``?token=<jwt>`` query parameter.

    Returns:
        ``StreamingResponse`` with ``text/event-stream`` media type.
    """
    # Authenticate before creating the streaming response so auth failures
    # are returned as proper 401 HTTP errors rather than inside the SSE body.
    user = await get_current_user_from_token(token)

    kb = request.app.state.kb
    chat_llm = getattr(kb, "synthesis_llm", None) or getattr(kb, "query_llm", None)

    async def _generate() -> AsyncGenerator[str]:
        if chat_llm is None:
            yield sse_event("error", {"message": "LLM not available"})
            yield sse_event("stream_end", {})
            return

        attribution = await resolve_attribution(user)

        # ── session resolution ────────────────────────────────────────────────
        is_new = False
        session: ChatSession | None = None

        if body.session_id is not None:
            session = get_session(user.id, body.session_id)
            if session is None and await chat_history.chat_exists(
                body.session_id, user.id
            ):
                msgs = await chat_history.get_messages(body.session_id)
                session = ChatSession.from_saved(
                    body.session_id, msgs, kb, chat_llm, attribution, user.id
                )
                cache_session(session)

        if session is None:
            is_new = True
            session = ChatSession(kb, chat_llm, attribution, user.id)
            cache_session(session)
            if body.seed_question is not None and body.seed_answer is not None:
                session.seed(body.seed_question, body.seed_answer, body.seed_entry_ids)

        # Per-request attribution refresh (team setting can change)
        session.attribution = attribution

        # ── always emit chat_session first ────────────────────────────────────
        yield sse_event("chat_session", {"session_id": session.id})

        # ── persist new session ───────────────────────────────────────────────
        if is_new:
            title = derive_title(body.seed_question or body.message)
            try:
                await chat_history.create_chat(session.id, user.id, title, body.mode)
                if body.seed_question is not None and body.seed_answer is not None:
                    await chat_history.save_messages_bulk(
                        session.id,
                        [
                            {"role": "user", "content": body.seed_question},
                            {"role": "assistant", "content": body.seed_answer},
                        ],
                    )
            except Exception:
                logger.warning(
                    "chat_stream: failed to persist new session %s", session.id
                )

        # ── run reply as a task, drain events via queue ───────────────────────
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def _event_cb(event: dict[str, Any]) -> None:
            await queue.put(event)

        async def _run_reply() -> str:
            try:
                return await session.reply(body.message, event_callback=_event_cb)
            finally:
                await queue.put(None)

        task: asyncio.Task[str] = asyncio.create_task(_run_reply())

        # Drain until sentinel
        while True:
            ev: dict[str, Any] | None = await queue.get()
            if ev is None:
                break
            yield sse_event(ev.get("type", "status"), ev)

        # Await the task result
        try:
            answer = await task
        except Exception as exc:
            yield sse_event("error", {"message": f"{type(exc).__name__}: {exc}"})
            yield sse_event("stream_end", {})
            return

        # Persist user + assistant messages (warn-only)
        try:
            await chat_history.save_message(session.id, "user", body.message)
            await chat_history.save_message(session.id, "assistant", answer)
        except Exception:
            logger.warning(
                "chat_stream: failed to save messages for session %s", session.id
            )

        yield sse_event("chat_response", {"answer": answer, "session_id": session.id})
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


@router.post("/create", response_model=ChatOkResponse)
async def create_chat_endpoint(
    body: ChatCreateRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> ChatOkResponse:
    """Create a new chat session (idempotent).

    If the caller-supplied ``chat_id`` is already owned by the authenticated
    user, returns ``{ok: True, id: chat_id}`` with no side effects.  A
    different user owning the same ``chat_id`` (global PK collision) is mapped
    to **409 Conflict**.

    Args:
        body: Chat creation request with ``chat_id``, ``question``, optional
            ``answer`` and ``mode``.
        _user: Authenticated user.

    Returns:
        ``ChatOkResponse`` with ``ok=True`` and ``id=chat_id``.

    Raises:
        HTTPException: 409 if the ``chat_id`` is already owned by another user.
    """
    if await chat_history.chat_exists(body.chat_id, _user.id):
        return ChatOkResponse(ok=True, id=body.chat_id)

    messages: list[dict[str, str]] = [{"role": "user", "content": body.question}]
    if body.answer:
        messages.append({"role": "assistant", "content": body.answer})

    try:
        await chat_history.create_chat(
            body.chat_id,
            _user.id,
            derive_title(body.question),
            body.mode,
        )
        await chat_history.save_messages_bulk(body.chat_id, messages)
    except asyncpg.UniqueViolationError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Chat ID already exists",
        ) from None

    return ChatOkResponse(ok=True, id=body.chat_id)


@router.get("/history", response_model=list[ChatListItem])
async def list_history(
    _user: Annotated[User, Depends(get_current_user)],
) -> list[ChatListItem]:
    """Return the authenticated user's chat history (most-recent first).

    Args:
        _user: Authenticated user.

    Returns:
        List of ``ChatListItem`` records ordered by ``updated_at`` DESC.
        Empty list when no chats exist.
    """
    rows = await chat_history.list_chats(_user.id)
    return [ChatListItem(**row) for row in rows]


@router.get("/{chat_id}/messages", response_model=list[ChatMessageItem])
async def get_chat_messages(
    chat_id: str,
    _user: Annotated[User, Depends(get_current_user)],
) -> list[ChatMessageItem]:
    """Return messages for a specific chat session.

    Args:
        chat_id: The chat session ID.
        _user: Authenticated user (ownership check).

    Returns:
        Ordered list of ``ChatMessageItem`` records.

    Raises:
        HTTPException: 404 if the chat does not exist or is owned by another user.
    """
    if not await chat_history.chat_exists(chat_id, _user.id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    rows = await chat_history.get_messages(chat_id)
    return [ChatMessageItem(**row) for row in rows]


@router.delete("/{chat_id}", response_model=ChatOkResponse)
async def delete_chat_endpoint(
    chat_id: str,
    _user: Annotated[User, Depends(get_current_user)],
) -> ChatOkResponse:
    """Delete a chat session and its messages.

    Also evicts the in-memory session from the registry.

    Args:
        chat_id: The chat session ID to delete.
        _user: Authenticated user (ownership check).

    Returns:
        ``{ok: True}`` on success.

    Raises:
        HTTPException: 404 if the chat does not exist or is owned by another user.
    """
    from kb_service.chat import _sessions

    deleted = await chat_history.delete_chat(chat_id, _user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found",
        )
    # Evict in-memory session
    _sessions.pop((_user.id, chat_id), None)
    return ChatOkResponse(ok=True)
