"""Tests for chat history API endpoints."""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from personal_kb.web.app import create_app_with_deps
from personal_kb.web.chat_history import ChatHistoryStore
from tests.conftest import FakeLLM


@pytest_asyncio.fixture
async def chat_app(web_kb):
    """App with chat history pre-populated."""
    db, embedder, _ids = web_kb
    llm = FakeLLM()
    app = create_app_with_deps(db, embedder, llm)
    # Open an in-memory chat history store and attach it directly
    ch = await ChatHistoryStore.open(db_path=":memory:")
    app.state.chat_history = ch
    yield app
    await ch.close()


@pytest.mark.asyncio
async def test_chat_history_empty(chat_app):
    """Empty history returns empty list."""
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/history")

    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_chat_history_lists_chats(chat_app):
    """History endpoint returns chats from the store."""
    ch = chat_app.state.chat_history
    await ch.create_chat("c1", "First chat", "summarize")
    await ch.create_chat("c2", "Second chat", "explore")

    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/history")

    chats = resp.json()
    assert len(chats) == 2
    assert chats[0]["id"] == "c2"  # most recent first
    assert chats[1]["id"] == "c1"


@pytest.mark.asyncio
async def test_get_messages(chat_app):
    """Fetch messages for a specific chat."""
    ch = chat_app.state.chat_history
    await ch.create_chat("c1", "Msg test", "")
    await ch.save_message("c1", "user", "Question")
    await ch.save_message("c1", "assistant", "Answer")

    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/c1/messages")

    msgs = resp.json()
    assert len(msgs) == 2
    assert msgs[0] == {"role": "user", "content": "Question"}
    assert msgs[1] == {"role": "assistant", "content": "Answer"}


@pytest.mark.asyncio
async def test_get_messages_not_found(chat_app):
    """Fetching messages for nonexistent chat returns 404."""
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/chat/nonexistent/messages")

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_chat(chat_app):
    """Delete removes chat and returns ok."""
    ch = chat_app.state.chat_history
    await ch.create_chat("c1", "To delete", "")
    await ch.save_message("c1", "user", "Hello")

    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/chat/c1")
        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

        # Verify it's gone
        resp = await client.get("/api/chat/history")
        assert resp.json() == []


@pytest.mark.asyncio
async def test_delete_chat_not_found(chat_app):
    """Deleting nonexistent chat returns 404."""
    transport = ASGITransport(app=chat_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.delete("/api/chat/bogus")

    assert resp.status_code == 404
