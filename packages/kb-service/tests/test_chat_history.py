"""Unit tests for kb_service.chat_history module functions.

These tests exercise the functions directly via a StatefulFakeDbPool — no live
Postgres needed.  The ``chat_history.get_db`` module attribute is monkeypatched
to return the fake pool so the function bodies execute fully.
"""

import pytest

import kb_service.chat_history as chat_history_module
from kb_service.chat_history import derive_title
from tests.conftest import StatefulFakeDbPool


@pytest.fixture
def pool(monkeypatch: pytest.MonkeyPatch) -> StatefulFakeDbPool:
    """Fake pool wired into chat_history via module-attribute monkeypatch."""
    p = StatefulFakeDbPool({})

    async def _get_db() -> StatefulFakeDbPool:
        return p

    monkeypatch.setattr(chat_history_module, "get_db", _get_db)
    return p


# ─── derive_title ─────────────────────────────────────────────────────────────


def test_derive_title_short() -> None:
    """Short text is returned as-is."""
    assert derive_title("Hello world") == "Hello world"


def test_derive_title_strips_newlines() -> None:
    """Newlines are replaced with spaces."""
    assert derive_title("line1\nline2") == "line1 line2"


def test_derive_title_truncates() -> None:
    """Text longer than max_len is truncated with '...'."""
    long = "x" * 100
    result = derive_title(long, max_len=10)
    assert result.endswith("...")
    assert len(result) <= 13  # 10 + '...'


def test_derive_title_exact_max() -> None:
    """Text exactly at max_len is returned as-is (no truncation)."""
    text = "a" * 80
    assert derive_title(text) == text


# ─── create_chat ──────────────────────────────────────────────────────────────


async def test_create_chat_returns_metadata(pool: StatefulFakeDbPool) -> None:
    """create_chat inserts a row and returns the metadata dict."""
    result = await chat_history_module.create_chat(
        "chat-1", "user-1", "My title", "explore"
    )
    assert result["id"] == "chat-1"
    assert result["title"] == "My title"
    assert result["mode"] == "explore"
    assert "updated_at" in result
    # Row should be in pool
    assert "chat-1" in pool._chats
    assert pool._chats["chat-1"]["user_id"] == "user-1"


async def test_create_chat_default_mode(pool: StatefulFakeDbPool) -> None:
    """create_chat with no mode argument uses empty-string default."""
    result = await chat_history_module.create_chat("chat-2", "user-1", "Title")
    assert result["mode"] == ""


# ─── save_message ─────────────────────────────────────────────────────────────


async def test_save_message(pool: StatefulFakeDbPool) -> None:
    """save_message appends to chat_messages."""
    await chat_history_module.create_chat("c1", "u1", "Title")
    await chat_history_module.save_message("c1", "user", "Hello!")
    messages = pool._chat_messages.get("c1", [])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello!"


# ─── save_messages_bulk ───────────────────────────────────────────────────────


async def test_save_messages_bulk(pool: StatefulFakeDbPool) -> None:
    """save_messages_bulk appends multiple messages in one call."""
    await chat_history_module.create_chat("c2", "u1", "Title")
    await chat_history_module.save_messages_bulk(
        "c2",
        [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
        ],
    )
    messages = pool._chat_messages.get("c2", [])
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


async def test_save_messages_bulk_empty(pool: StatefulFakeDbPool) -> None:
    """save_messages_bulk with an empty list is a no-op."""
    await chat_history_module.save_messages_bulk("no-chat", [])
    # Should not raise; no messages inserted


# ─── list_chats ───────────────────────────────────────────────────────────────


async def test_list_chats(pool: StatefulFakeDbPool) -> None:
    """list_chats returns only the requesting user's chats."""
    await chat_history_module.create_chat("c-a", "alice", "Chat A")
    await chat_history_module.create_chat("c-b", "bob", "Chat B")
    rows = await chat_history_module.list_chats("alice")
    ids = [r["id"] for r in rows]
    assert "c-a" in ids
    assert "c-b" not in ids


async def test_list_chats_empty(pool: StatefulFakeDbPool) -> None:
    """list_chats returns an empty list when the user has no chats."""
    rows = await chat_history_module.list_chats("nobody")
    assert rows == []


# ─── get_messages ─────────────────────────────────────────────────────────────


async def test_get_messages(pool: StatefulFakeDbPool) -> None:
    """get_messages returns persisted messages in insertion order."""
    await chat_history_module.create_chat("c3", "u1", "Title")
    await chat_history_module.save_messages_bulk(
        "c3",
        [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
    )
    msgs = await chat_history_module.get_messages("c3")
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Q"}
    assert msgs[1] == {"role": "assistant", "content": "A"}


async def test_get_messages_missing_chat(pool: StatefulFakeDbPool) -> None:
    """get_messages returns an empty list for an unknown chat_id."""
    msgs = await chat_history_module.get_messages("no-such-chat")
    assert msgs == []


# ─── chat_exists ──────────────────────────────────────────────────────────────


async def test_chat_exists_true(pool: StatefulFakeDbPool) -> None:
    """chat_exists returns True for an existing owned chat."""
    await chat_history_module.create_chat("c4", "u1", "Title")
    assert await chat_history_module.chat_exists("c4", "u1") is True


async def test_chat_exists_wrong_user(pool: StatefulFakeDbPool) -> None:
    """chat_exists returns False when the chat is owned by another user."""
    await chat_history_module.create_chat("c5", "alice", "Title")
    assert await chat_history_module.chat_exists("c5", "bob") is False


async def test_chat_exists_missing(pool: StatefulFakeDbPool) -> None:
    """chat_exists returns False for a non-existent chat."""
    assert await chat_history_module.chat_exists("no-chat", "u1") is False


# ─── delete_chat ──────────────────────────────────────────────────────────────


async def test_delete_chat_success(pool: StatefulFakeDbPool) -> None:
    """delete_chat returns True and removes the row."""
    await chat_history_module.create_chat("c6", "u1", "Title")
    deleted = await chat_history_module.delete_chat("c6", "u1")
    assert deleted is True
    assert "c6" not in pool._chats


async def test_delete_chat_wrong_user(pool: StatefulFakeDbPool) -> None:
    """delete_chat returns False when the user doesn't own the chat."""
    await chat_history_module.create_chat("c7", "alice", "Title")
    deleted = await chat_history_module.delete_chat("c7", "bob")
    assert deleted is False
    assert "c7" in pool._chats  # not deleted


async def test_delete_chat_missing(pool: StatefulFakeDbPool) -> None:
    """delete_chat returns False for a non-existent chat."""
    deleted = await chat_history_module.delete_chat("no-chat", "u1")
    assert deleted is False
