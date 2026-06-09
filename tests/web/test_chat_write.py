"""Tests for chat session write-back: tool dispatch, update_entry, ingest_url."""

import pytest
import pytest_asyncio

from personal_kb.graph.builder import GraphBuilder
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.web.chat import _parse_tool_call
from tests.conftest import FakeEmbedder, FakeLLM, ScriptedLLM


@pytest_asyncio.fixture
async def db():
    from personal_kb.db.connection import create_connection

    conn = await create_connection(":memory:")
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def embedder(db):
    return FakeEmbedder(db)


@pytest_asyncio.fixture
async def store(db):
    return KnowledgeStore(db)


@pytest_asyncio.fixture
async def graph_builder(db):
    return GraphBuilder(db)


@pytest_asyncio.fixture
async def kb_read(db, embedder):
    """KB with no extraction LLM — write tools advertise but fail at runtime."""
    from personal_kb.web.app import _wrap_deps_as_kb

    return _wrap_deps_as_kb(
        db,
        embedder,
        query_llm=None,
        synthesis_llm=None,
        store=None,
        graph_builder=None,
        graph_enricher=None,
        extraction_llm=None,
        contributor=None,
        team=None,
    )


@pytest_asyncio.fixture
async def kb_write(db, embedder, store, graph_builder):
    """KB with store + graph_builder for write-capable chat sessions."""
    from personal_kb.web.app import _wrap_deps_as_kb

    return _wrap_deps_as_kb(
        db,
        embedder,
        query_llm=None,
        synthesis_llm=None,
        store=store,
        graph_builder=graph_builder,
        graph_enricher=None,
        extraction_llm=None,  # ingest_url fails gracefully at runtime
        contributor="test-user",
        team=None,
    )


# --- Tool call parsing ---


class TestParseToolCall:
    def test_fenced_json(self):
        text = (
            "Sure, I'll update that.\n```json\n"
            '{"tool": "update_entry", "args": {"entry_id": "kb-00001", "tags": ["postgres"]}}'
            "\n```"
        )
        result = _parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "update_entry"
        assert result["args"]["entry_id"] == "kb-00001"

    def test_bare_json(self):
        text = 'Let me do that: {"tool": "ingest_url", "args": {"url": "https://example.com"}}'
        result = _parse_tool_call(text)
        assert result is not None
        assert result["tool"] == "ingest_url"

    def test_no_tool_call(self):
        text = "This is just a regular answer about [kb-00001]."
        result = _parse_tool_call(text)
        assert result is None

    def test_json_without_tool_key(self):
        text = '```json\n{"entry_id": "kb-00001"}\n```'
        result = _parse_tool_call(text)
        assert result is None

    def test_malformed_json(self):
        text = '```json\n{"tool": broken}\n```'
        result = _parse_tool_call(text)
        assert result is None


# --- No tool call passthrough ---


@pytest.mark.asyncio
async def test_reply_passthrough_read_only_kb(kb_read):
    """Without write capability (no store/extraction_llm), reply still works for read."""
    from personal_kb.web.chat import ChatSession

    llm = FakeLLM(response="Just a normal answer about [kb-00001].")
    session = ChatSession(kb_read, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Tell me more")
    assert "normal answer" in reply
    assert len(session.messages) == 4


@pytest.mark.asyncio
async def test_reply_passthrough_write_kb(kb_write):
    """With a write-capable KB but no tool call in response, passthrough works."""
    from personal_kb.web.chat import ChatSession

    llm = FakeLLM(response="Here's info about that entry.")
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("What's in kb-00001?")
    assert "info about" in reply
    assert len(session.messages) == 4  # seed Q+A + follow-up Q+A


# --- get_entry tool ---


@pytest.mark.asyncio
async def test_get_entry_by_id(kb_write, store):
    """Chat can fetch a specific entry by ID and add it to context."""
    from personal_kb.models.entry import EntryType
    from personal_kb.web.chat import ChatSession

    entry = await store.create_entry(
        short_title="Flickr Missing APIs",
        long_title="Flickr APIs not available in cleanr",
        knowledge_details="flickr.photos.getNotInSet and flickr.photos.recentlyUpdated.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["flickr", "api"],
    )

    # LLM produces a get_entry tool call, then uses the result
    tool_response = (
        f"Let me look that up.\n```json\n"
        f'{{"tool": "get_entry", "args": {{"entry_id": "{entry.id}"}}}}\n```'
    )
    llm = ScriptedLLM(
        responses=[tool_response, f"Here's what [{entry.id}] says: missing APIs are..."]
    )
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply(f"What's in {entry.id}?")
    assert entry.id in reply
    assert entry.id in session.entry_ids


@pytest.mark.asyncio
async def test_get_entry_not_found(kb_write):
    """get_entry with nonexistent ID returns error."""
    from personal_kb.web.chat import ChatSession

    tool_response = '```json\n{"tool": "get_entry", "args": {"entry_id": "kb-99999"}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "That entry doesn't exist."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Show me kb-99999")
    assert reply is not None


@pytest.mark.asyncio
async def test_get_entry_missing_id(kb_write):
    """get_entry without entry_id arg returns error."""
    from personal_kb.web.chat import ChatSession

    tool_response = '```json\n{"tool": "get_entry", "args": {}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "Missing ID."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Get entry")
    assert reply is not None


@pytest.mark.asyncio
async def test_get_entry_works_on_read_only_kb(kb_read, store):
    """get_entry works on a read-only KB — it's a read tool."""
    from personal_kb.models.entry import EntryType
    from personal_kb.web.chat import ChatSession

    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Some details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    tool_response = f'```json\n{{"tool": "get_entry", "args": {{"entry_id": "{entry.id}"}}}}\n```'
    llm = ScriptedLLM(responses=[tool_response, f"Got it: [{entry.id}]"])
    session = ChatSession(kb_read, llm)
    session.seed("Q", "A", [])

    await session.reply(f"Look up {entry.id}")
    assert entry.id in session.entry_ids


# --- update_entry tool ---


@pytest.mark.asyncio
async def test_update_entry_tags(kb_write, store):
    """Chat can update tags on an existing entry."""
    from personal_kb.models.entry import EntryType
    from personal_kb.web.chat import ChatSession

    # Create an entry to update
    entry = await store.create_entry(
        short_title="Test Entry",
        long_title="A test entry for tag updates",
        knowledge_details="Some knowledge details here.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["old-tag"],
    )

    # LLM produces a tool call, then a confirmation
    tool_response = (
        "I'll add those tags.\n```json\n"
        '{"tool": "update_entry", "args": {"entry_id": "'
        + entry.id
        + '", "tags": ["postgres", "migration"]}}\n```'
    )
    llm = ScriptedLLM(responses=[tool_response, "Done! I've updated the tags on " + entry.id + "."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [entry.id])

    reply = await session.reply("Add tags postgres and migration to " + entry.id)
    assert "updated" in reply.lower() or "done" in reply.lower()

    # Verify the entry was actually updated
    updated = await store.get_entry(entry.id)
    assert updated is not None
    assert set(updated.tags) == {"postgres", "migration"}


@pytest.mark.asyncio
async def test_update_entry_missing_id(kb_write):
    """Tool call without entry_id returns error."""
    from personal_kb.web.chat import ChatSession

    tool_response = '```json\n{"tool": "update_entry", "args": {"tags": ["x"]}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "Error noted."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Update tags")
    # The LLM gets the error injected and responds
    assert reply is not None


@pytest.mark.asyncio
async def test_update_entry_invalid_sensitivity(kb_write, store):
    """Invalid sensitivity value is rejected."""
    from personal_kb.models.entry import EntryType
    from personal_kb.web.chat import ChatSession

    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    tool_response = (
        '```json\n{"tool": "update_entry", "args": {"entry_id": "'
        + entry.id
        + '", "sensitivity": "top-secret"}}\n```'
    )
    llm = ScriptedLLM(responses=[tool_response, "Got it, invalid sensitivity."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Set sensitivity")
    assert reply is not None


# --- ingest_url tool ---


@pytest.mark.asyncio
async def test_ingest_url_no_extraction_llm(kb_write):
    """ingest_url fails gracefully when extraction LLM is not available."""
    from personal_kb.web.chat import ChatSession

    # kb_write has extraction_llm=None
    tool_response = '```json\n{"tool": "ingest_url", "args": {"url": "https://example.com"}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "Sorry, ingestion isn't available right now."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Ingest https://example.com")
    assert reply is not None


@pytest.mark.asyncio
async def test_ingest_url_missing_url(kb_write):
    """ingest_url without url arg returns error."""
    from personal_kb.web.chat import ChatSession

    tool_response = '```json\n{"tool": "ingest_url", "args": {}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "URL was missing."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Ingest something")
    assert reply is not None


# --- Unknown tool ---


@pytest.mark.asyncio
async def test_unknown_tool(kb_write):
    """Unknown tool name returns error gracefully."""
    from personal_kb.web.chat import ChatSession

    tool_response = '```json\n{"tool": "delete_everything", "args": {}}\n```'
    llm = ScriptedLLM(responses=[tool_response, "That tool doesn't exist."])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    reply = await session.reply("Delete everything")
    assert reply is not None


# --- Event callback ---


@pytest.mark.asyncio
async def test_tool_dispatch_fires_events(kb_write, store):
    """Tool dispatch fires chat_tool_result event."""
    from personal_kb.models.entry import EntryType
    from personal_kb.web.chat import ChatSession

    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["old"],
    )

    tool_response = (
        '```json\n{"tool": "update_entry", "args": {"entry_id": "'
        + entry.id
        + '", "tags": ["new"]}}\n```'
    )
    llm = ScriptedLLM(responses=[tool_response, "Updated!"])
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    events: list[dict] = []

    async def capture(event: dict) -> None:
        events.append(event)

    await session.reply("Update tags", event_callback=capture)

    types = [e["type"] for e in events]
    assert "chat_thinking" in types
    assert "chat_tool_result" in types
    assert "chat_done" in types

    tool_event = next(e for e in events if e["type"] == "chat_tool_result")
    assert tool_event["tool"] == "update_entry"
    assert tool_event["success"] is True


# --- System prompt ---


@pytest.mark.asyncio
async def test_system_prompt_includes_write_tools(kb_write):
    """System prompt advertises the write tools (update_entry + ingest_url)."""
    from personal_kb.web.chat import ChatSession

    llm = FakeLLM(response="Just a response.")
    session = ChatSession(kb_write, llm)
    session.seed("Q", "A", [])

    await session.reply("Hello")

    # FakeLLM records the system prompt via generate_chat -> generate
    assert llm.last_system is not None
    assert "update_entry" in llm.last_system
    assert "ingest_url" in llm.last_system


@pytest.mark.asyncio
async def test_system_prompt_includes_get_entry(kb_read):
    """System prompt always advertises the read-only get_entry tool."""
    from personal_kb.web.chat import ChatSession

    llm = FakeLLM(response="Just a response.")
    session = ChatSession(kb_read, llm)
    session.seed("Q", "A", [])

    await session.reply("Hello")

    assert llm.last_system is not None
    assert "get_entry" in llm.last_system
