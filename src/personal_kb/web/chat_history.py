"""Persistent chat history backed by a dedicated SQLite database."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);
"""


def derive_title(text: str, max_len: int = 80) -> str:
    """Derive a chat title from the first user message."""
    line = text.replace("\n", " ").strip()
    if len(line) <= max_len:
        return line
    return line[:max_len].rstrip() + "..."


class ChatHistoryStore:
    """CRUD for persisting chat conversations in a dedicated SQLite DB."""

    def __init__(self, conn: aiosqlite.Connection) -> None:  # noqa: D107
        self._conn = conn

    @classmethod
    async def open(cls, db_path: str | Path | None = None) -> "ChatHistoryStore":
        """Open (or create) the chat history database."""
        if db_path is None:
            from personal_kb.config import get_chat_history_db_path

            db_path = get_chat_history_db_path()

        path = Path(db_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)

        conn = await aiosqlite.connect(str(path))
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA foreign_keys=ON")
        await conn.executescript(_SCHEMA)
        await conn.commit()
        return cls(conn)

    async def close(self) -> None:
        """Close the database connection."""
        await self._conn.close()

    async def create_chat(self, chat_id: str, title: str, mode: str = "") -> dict[str, str]:
        """Create a new chat record. Returns the chat dict."""
        now = datetime.now(UTC).isoformat()
        await self._conn.execute(
            "INSERT INTO chats (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (chat_id, title, mode, now, now),
        )
        await self._conn.commit()
        return {"id": chat_id, "title": title, "mode": mode, "updated_at": now}

    async def save_message(self, chat_id: str, role: str, content: str) -> None:
        """Append a single message and update the chat's updated_at."""
        now = datetime.now(UTC).isoformat()
        await self._conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, now),
        )
        await self._conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )
        await self._conn.commit()

    async def save_messages_bulk(self, chat_id: str, messages: list[dict[str, str]]) -> None:
        """Save multiple messages at once (e.g. seed Q+A)."""
        now = datetime.now(UTC).isoformat()
        rows = [(chat_id, m["role"], m["content"], now) for m in messages]
        await self._conn.executemany(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        await self._conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )
        await self._conn.commit()

    async def list_chats(self, limit: int = 50) -> list[dict[str, str]]:
        """Return recent chats ordered by updated_at DESC."""
        cursor = await self._conn.execute(
            "SELECT id, title, mode, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {"id": r["id"], "title": r["title"], "mode": r["mode"], "updated_at": r["updated_at"]}
            for r in rows
        ]

    async def get_messages(self, chat_id: str) -> list[dict[str, str]]:
        """Return all messages for a chat, ordered chronologically."""
        cursor = await self._conn.execute(
            "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id ASC",
            (chat_id,),
        )
        rows = await cursor.fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    async def delete_chat(self, chat_id: str) -> bool:
        """Delete a chat and its messages. Returns True if found."""
        cursor = await self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
        await self._conn.commit()
        return cursor.rowcount > 0

    async def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists."""
        cursor = await self._conn.execute("SELECT 1 FROM chats WHERE id = ? LIMIT 1", (chat_id,))
        return await cursor.fetchone() is not None
