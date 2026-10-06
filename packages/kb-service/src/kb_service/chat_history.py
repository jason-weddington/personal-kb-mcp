"""Chat history persistence: per-user sessions stored in the service Postgres.

All functions are module-level async functions over the service-auth pool
(KB_SERVICE_DATABASE_URL).  ``get_db`` is bound at import time — the same
convention used by attribution.py — so tests can monkeypatch
``kb_service.chat_history.get_db`` to inject a fake pool.

Tables used (defined in database.py _SCHEMA_STATEMENTS):
    chats         — one row per chat session (id, user_id, title, mode, ...)
    chat_messages — one row per message, linked to chats by chat_id
"""

from datetime import UTC, datetime

from kb_service.database import get_db


def derive_title(text: str, max_len: int = 80) -> str:
    r"""Derive a short display title from the first line of *text*.

    Algorithm (verbatim from old chat_history.py:31-36):
        ``line = text.replace('\n', ' ').strip()``
        ``return line if len(line) <= max_len else line[:max_len].rstrip() + '...'``

    Args:
        text: Raw text to derive a title from (typically the first user message).
        max_len: Maximum length before truncating with '...' (default 80).

    Returns:
        A short single-line title string.
    """
    line = text.replace("\n", " ").strip()
    return line if len(line) <= max_len else line[:max_len].rstrip() + "..."


async def create_chat(
    chat_id: str, user_id: str, title: str, mode: str = ""
) -> dict[str, str]:
    """Create a new chat session row and return its metadata dict.

    Writes *now* to BOTH ``created_at`` and ``updated_at``.

    Returns:
        ``{'id': chat_id, 'title': title, 'mode': mode, 'updated_at': now}``
    """
    now = datetime.now(UTC).isoformat()
    db = await get_db()
    await db.execute(
        "INSERT INTO chats (id, user_id, title, mode, created_at, updated_at)"
        " VALUES ($1, $2, $3, $4, $5, $6)",
        chat_id,
        user_id,
        title,
        mode,
        now,
        now,
    )
    return {"id": chat_id, "title": title, "mode": mode, "updated_at": now}


async def save_message(chat_id: str, role: str, content: str) -> None:
    """Insert one message into chat_messages and bump chats.updated_at.

    Args:
        chat_id: The owning chat session ID.
        role: Message role — ``'user'`` or ``'assistant'``.
        content: Raw message text.
    """
    now = datetime.now(UTC).isoformat()
    db = await get_db()
    await db.execute(
        "INSERT INTO chat_messages (chat_id, role, content, created_at)"
        " VALUES ($1, $2, $3, $4)",
        chat_id,
        role,
        content,
        now,
    )
    await db.execute(
        "UPDATE chats SET updated_at = $1 WHERE id = $2",
        now,
        chat_id,
    )


async def save_messages_bulk(chat_id: str, messages: list[dict[str, str]]) -> None:
    """Insert multiple messages with the same timestamp and bump chats.updated_at.

    All rows share the same ``created_at`` value so ordering is preserved via
    the auto-increment ``id`` column.

    Args:
        chat_id: The owning chat session ID.
        messages: List of ``{'role': ..., 'content': ...}`` dicts.
    """
    if not messages:
        return
    now = datetime.now(UTC).isoformat()
    db = await get_db()
    for msg in messages:
        await db.execute(
            "INSERT INTO chat_messages (chat_id, role, content, created_at)"
            " VALUES ($1, $2, $3, $4)",
            chat_id,
            msg["role"],
            msg["content"],
            now,
        )
    await db.execute(
        "UPDATE chats SET updated_at = $1 WHERE id = $2",
        now,
        chat_id,
    )


async def list_chats(user_id: str, limit: int = 50) -> list[dict[str, str]]:
    """Return chats for *user_id* ordered by updated_at DESC.

    Args:
        user_id: The owning user's ID.
        limit: Maximum number of rows to return (default 50).

    Returns:
        List of ``{id, title, mode, updated_at}`` dicts.
    """
    db = await get_db()
    rows = await db.fetch(
        "SELECT id, title, mode, updated_at FROM chats"
        " WHERE user_id = $1 ORDER BY updated_at DESC LIMIT $2",
        user_id,
        limit,
    )
    return [dict(row) for row in rows]


async def get_messages(chat_id: str) -> list[dict[str, str]]:
    """Return messages for *chat_id* ordered by id ASC.

    Args:
        chat_id: The chat session ID.

    Returns:
        List of ``{role, content}`` dicts.
    """
    db = await get_db()
    rows = await db.fetch(
        "SELECT role, content FROM chat_messages WHERE chat_id = $1 ORDER BY id ASC",
        chat_id,
    )
    return [dict(row) for row in rows]


async def delete_chat(chat_id: str, user_id: str) -> bool:
    """Delete a chat and return ``True`` iff a row was deleted.

    Ownership is enforced via the WHERE clause (id AND user_id), so a user
    cannot delete another user's chat even if they know the ID.

    Args:
        chat_id: The chat session ID.
        user_id: The requesting user's ID (ownership check).

    Returns:
        ``True`` when one row was deleted; ``False`` when nothing matched.
    """
    db = await get_db()
    status: str = await db.execute(
        "DELETE FROM chats WHERE id = $1 AND user_id = $2",
        chat_id,
        user_id,
    )
    # asyncpg returns 'DELETE n' (e.g. 'DELETE 0' or 'DELETE 1').
    n = int(status.split()[-1])
    return n > 0


async def chat_exists(chat_id: str, user_id: str) -> bool:
    """Return ``True`` iff the chat exists and is owned by *user_id*.

    Args:
        chat_id: The chat session ID.
        user_id: The requesting user's ID (ownership check).
    """
    db = await get_db()
    row = await db.fetchrow(
        "SELECT 1 FROM chats WHERE id = $1 AND user_id = $2",
        chat_id,
        user_id,
    )
    return row is not None
