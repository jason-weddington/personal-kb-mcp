"""Tests for multi-turn chat sessions."""

import pytest
import pytest_asyncio

from tests.conftest import FakeEmbedder, FakeLLM


@pytest_asyncio.fixture
async def db():
    """In-memory database for chat tests."""
    from personal_kb.db.connection import create_connection

    conn = await create_connection(":memory:")
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def embedder(db):
    return FakeEmbedder(db)


@pytest.fixture
def llm():
    return FakeLLM(response="This is a follow-up answer based on [kb-00001].")


@pytest.mark.asyncio
async def test_chat_session_seed_and_reply(db, embedder, llm):
    """Seeding a session and sending a follow-up works."""
    from personal_kb.web.chat import ChatSession

    session = ChatSession(db, embedder, llm)
    session.seed("What is X?", "X is described in [kb-00001].", ["kb-00001"])

    assert len(session.messages) == 2
    assert session.entry_ids == ["kb-00001"]

    reply = await session.reply("Tell me more about X.")
    assert reply is not None
    assert len(session.messages) == 4  # seed Q + A + follow-up Q + A


@pytest.mark.asyncio
async def test_chat_session_accumulates_entries(db, embedder, llm):
    """Follow-up queries should search for and accumulate entry IDs."""
    from personal_kb.web.chat import ChatSession

    session = ChatSession(db, embedder, llm)
    session.seed("Q1", "A1", ["kb-00001"])

    await session.reply("Q2")
    # Even if search returns nothing new, original entries are preserved
    assert "kb-00001" in session.entry_ids


@pytest.mark.asyncio
async def test_chat_session_trim_history(db, embedder, llm):
    """History trimming preserves first pair and latest messages."""
    from personal_kb.web.chat import _MAX_CONVERSATION_CHARS, ChatSession

    session = ChatSession(db, embedder, llm)
    # Seed with short messages
    session.seed("Q", "A", [])

    # Add enough messages to exceed budget
    big_msg = "x" * (_MAX_CONVERSATION_CHARS // 2)
    session.messages.append({"role": "user", "content": big_msg})
    session.messages.append({"role": "assistant", "content": big_msg})
    session.messages.append({"role": "user", "content": "latest question"})

    session._trim_history()

    # First pair preserved
    assert session.messages[0]["content"] == "Q"
    assert session.messages[1]["content"] == "A"
    # Latest message preserved
    assert session.messages[-1]["content"] == "latest question"


@pytest.mark.asyncio
async def test_chat_session_llm_unavailable(db, embedder):
    """Graceful fallback when LLM returns None."""
    from personal_kb.web.chat import ChatSession

    llm = FakeLLM(response=None, available=True)
    session = ChatSession(db, embedder, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Follow up")
    assert "couldn't generate" in reply.lower()


@pytest.mark.asyncio
async def test_get_or_create_session(db, embedder, llm):
    """Session store creates and retrieves sessions."""
    from personal_kb.web.chat import get_or_create_session, get_session

    session = get_or_create_session(None, db, embedder, llm)
    assert session.id is not None

    # Same ID returns same session
    retrieved = get_session(session.id)
    assert retrieved is session

    # Unknown ID returns None
    assert get_session("nonexistent") is None


@pytest.mark.asyncio
async def test_chat_event_callback(db, embedder, llm):
    """Event callback fires chat_thinking and chat_done events."""
    from personal_kb.web.chat import ChatSession

    events = []

    async def capture(event):
        events.append(event)

    session = ChatSession(db, embedder, llm)
    session.seed("Q", "A", [])

    await session.reply("Follow up", event_callback=capture)

    types = [e["type"] for e in events]
    assert "chat_thinking" in types
    assert "chat_done" in types


@pytest.mark.asyncio
async def test_from_saved_reconstructs_session(db, embedder, llm):
    """from_saved restores messages and extracts entry IDs."""
    from personal_kb.web.chat import ChatSession

    saved = [
        {"role": "user", "content": "What is X?"},
        {"role": "assistant", "content": "X is in [kb-00001] and [kb-00002]."},
        {"role": "user", "content": "More about kb-00001"},
        {"role": "assistant", "content": "Details from [kb-00001] and [kb-00003]."},
    ]
    session = ChatSession.from_saved("test-id", saved, db, embedder, llm)

    assert session.id == "test-id"
    assert len(session.messages) == 4
    assert session.entry_ids == ["kb-00001", "kb-00002", "kb-00003"]


@pytest.mark.asyncio
async def test_from_saved_with_no_entries(db, embedder, llm):
    """from_saved handles conversations with no KB references."""
    from personal_kb.web.chat import ChatSession

    saved = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    session = ChatSession.from_saved("test-id-2", saved, db, embedder, llm)
    assert session.entry_ids == []
    assert len(session.messages) == 2
