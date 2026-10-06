"""Unit tests for chat.py: _trim_history, _parse_tool_call, ChatSession helpers.

These tests exercise the pure-logic paths that SSE route tests don't reach
(e.g. history trimming, from_saved, unknown-tool dispatch).
"""

from kb_core import Attribution

from kb_service.chat import (
    _MAX_CONVERSATION_CHARS,
    ChatSession,
    _parse_tool_call,
    _trim_history,
)
from tests.conftest import FakeKnowledgeBase, FakeLLM, make_search_result


def _make_fake_kb() -> FakeKnowledgeBase:
    return FakeKnowledgeBase(results=[make_search_result()], filtered_count=1)


def _make_attribution() -> Attribution:
    return Attribution(contributor="test@example.com", team=None)


# ─── _trim_history ────────────────────────────────────────────────────────────


def test_trim_history_noop_short() -> None:
    """Short history is not trimmed."""
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    original = list(msgs)
    _trim_history(msgs)
    assert msgs == original


def test_trim_history_noop_few_messages() -> None:
    """History with <= 3 messages is never trimmed regardless of size."""
    big_content = "x" * (_MAX_CONVERSATION_CHARS + 1)
    msgs = [
        {"role": "user", "content": big_content},
        {"role": "assistant", "content": big_content},
        {"role": "user", "content": big_content},
    ]
    _trim_history(msgs)
    # 3 messages — no trimming
    assert len(msgs) == 3


def test_trim_history_trims_middle() -> None:
    """Excess middle messages are removed to satisfy the char budget."""
    # seed messages (always kept)
    seed_q = {"role": "user", "content": "seed question"}
    seed_a = {"role": "assistant", "content": "seed answer"}
    # Build a long history that exceeds the budget
    big = "x" * 30_000
    msgs = [
        seed_q,
        seed_a,
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
        {"role": "user", "content": big},
        {"role": "assistant", "content": big},
    ]
    _trim_history(msgs)
    # seed messages must be preserved
    assert msgs[0] == seed_q
    assert msgs[1] == seed_a
    # total chars should now be <= budget
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    assert total <= _MAX_CONVERSATION_CHARS


# ─── _parse_tool_call ─────────────────────────────────────────────────────────


def test_parse_tool_call_fenced_json() -> None:
    """Fenced JSON block is extracted as a dict."""
    text = '```json\n{"tool": "get_entry", "args": {"entry_id": "kb-00001"}}\n```'
    result = _parse_tool_call(text)
    assert result is not None
    assert result["tool"] == "get_entry"


def test_parse_tool_call_bare_json() -> None:
    """Bare JSON without a fenced block is parsed as fallback."""
    text = '{"tool": "ingest_url", "args": {"url": "https://example.com"}}'
    result = _parse_tool_call(text)
    assert result is not None
    assert result["tool"] == "ingest_url"


def test_parse_tool_call_no_match() -> None:
    """Plain text without a tool call returns None."""
    assert _parse_tool_call("This is a normal response.") is None


def test_parse_tool_call_fenced_without_tool_key() -> None:
    """Fenced JSON without 'tool' key is skipped; bare fallback also fails."""
    text = '```json\n{"not_a_tool": "value"}\n```'
    assert _parse_tool_call(text) is None


# ─── ChatSession.from_saved ───────────────────────────────────────────────────


def test_from_saved_restores_messages() -> None:
    """from_saved rebuilds the message list from persisted rows."""
    saved = [
        {"role": "user", "content": "What is kb-00001?"},
        {"role": "assistant", "content": "It is [kb-00001] a factual entry."},
    ]
    session = ChatSession.from_saved(
        "chat-id",
        saved,
        _make_fake_kb(),
        FakeLLM(),
        _make_attribution(),
        "user-1",
    )
    assert session.id == "chat-id"
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    # KB IDs mentioned in assistant turns are extracted
    assert "kb-00001" in session.entry_ids


def test_from_saved_deduplicates_entry_ids() -> None:
    """Duplicate KB IDs across messages are de-duped."""
    saved = [
        {"role": "user", "content": "Ask about kb-00001"},
        {"role": "assistant", "content": "[kb-00001] mentioned here."},
        {"role": "user", "content": "More about kb-00001?"},
        {"role": "assistant", "content": "[kb-00001] again here."},
    ]
    session = ChatSession.from_saved(
        "chat-2", saved, _make_fake_kb(), FakeLLM(), _make_attribution(), "u1"
    )
    assert session.entry_ids.count("kb-00001") == 1


# ─── ChatSession.seed ─────────────────────────────────────────────────────────


def test_seed_sets_messages_and_entry_ids() -> None:
    """seed() sets the initial conversation turn and entry_ids."""
    session = ChatSession(_make_fake_kb(), FakeLLM(), _make_attribution(), "user-1")
    session.seed("What is kb-00001?", "It is a factual entry.", ["kb-00001"])
    assert len(session.messages) == 2
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.entry_ids == ["kb-00001"]


# ─── ChatSession._dispatch_tool — unknown tool ────────────────────────────────


async def test_dispatch_unknown_tool() -> None:
    """Unknown tool name returns a failure _ToolResult."""
    session = ChatSession(_make_fake_kb(), FakeLLM(), _make_attribution(), "user-1")
    result = await session._dispatch_tool({"tool": "no_such_tool", "args": {}})
    assert result.success is False
    assert "Unknown tool: no_such_tool" in result.message
