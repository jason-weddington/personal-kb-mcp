"""Payload-assertion tests for prompt-caching in AnthropicLLMClient.

Verifies that cache breakpoints are placed on CONTENT BLOCKS, not as a
top-level ``cache_control`` kwarg (which the Anthropic API silently ignores).

Cases covered:
  (a) generate() with system prompt — cache_control nested in the system block
  (b) generate() without system — no system key sent at all; no top-level cache_control
  (c) generate_chat() with system — cache_control in system block AND in last message block
  (d) generate_chat() without system — no system key; rolling breakpoint on last message block
  (e) generate_chat() second agentic turn — rolling breakpoint present on both turns

All tests are hermetic — the Anthropic SDK is never contacted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kb_core.config import AnthropicProviderConfig
from kb_core.llm.anthropic import AnthropicLLMClient

if TYPE_CHECKING:
    from kb_core.llm.provider import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(model: str = "claude-haiku-4-5") -> AnthropicLLMClient:
    """Build an AnthropicLLMClient with an explicit config (no env reads)."""
    return AnthropicLLMClient(AnthropicProviderConfig(model=model, api_key="test-key"))


def _mock_response(text: str = "ok") -> MagicMock:
    """Build a minimal Anthropic response mock."""
    block = MagicMock()
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def _last_message_last_block(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the last content block of the last message in a messages.create call."""
    msgs = call_kwargs["messages"]
    last_content = msgs[-1]["content"]
    assert isinstance(last_content, list), (
        f"Expected last message content to be a list of blocks, got {type(last_content)}"
    )
    return last_content[-1]


# ---------------------------------------------------------------------------
# (a) generate() with system prompt — cache_control nested in system block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_system_block_has_cache_control() -> None:
    """generate() with system= sends system as a content-block array with cache_control nested."""
    response = _mock_response("generated text")
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        result = await llm.generate("Summarize this.", system="You are a summarizer.")

    assert result == "generated text"
    call_kwargs = sdk_client.messages.create.call_args.kwargs

    # Must NOT pass cache_control as a top-level kwarg
    assert "cache_control" not in call_kwargs, (
        "generate() must NOT include cache_control as a top-level kwarg"
    )

    # system must be in array/content-block form
    system_blocks = call_kwargs.get("system")
    assert isinstance(system_blocks, list), (
        f"system should be a list of content blocks, got {type(system_blocks)}"
    )
    assert len(system_blocks) >= 1
    assert system_blocks[0].get("type") == "text"
    assert system_blocks[0].get("text") == "You are a summarizer."
    assert system_blocks[0].get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must be nested inside the system content block"
    )


# ---------------------------------------------------------------------------
# (b) generate() without system — no system key, no top-level cache_control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_without_system_no_cache_control() -> None:
    """generate() without system= sends no system field and no cache_control."""
    response = _mock_response("generated text")
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        result = await llm.generate("Summarize this document.")

    assert result == "generated text"
    call_kwargs = sdk_client.messages.create.call_args.kwargs
    assert "cache_control" not in call_kwargs, "generate() must NOT include top-level cache_control"
    assert "system" not in call_kwargs, "No system field should be sent when system=None"


# ---------------------------------------------------------------------------
# (c) generate_chat() with system — breakpoint on system block + last message block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_chat_system_block_has_cache_control() -> None:
    """generate_chat with system= nests cache_control inside the system content block."""
    response = _mock_response()
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        msgs: list[Message] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        result = await llm.generate_chat(msgs, system="You are helpful.")

    assert result == "ok"
    call_kwargs = sdk_client.messages.create.call_args.kwargs

    # Must NOT pass cache_control as a top-level kwarg
    assert "cache_control" not in call_kwargs, (
        "generate_chat must NOT include cache_control as a top-level kwarg"
    )

    # system block must carry cache_control
    system_blocks = call_kwargs.get("system")
    assert isinstance(system_blocks, list), "system should be a list of content blocks"
    assert system_blocks[0].get("cache_control") == {"type": "ephemeral"}, (
        "cache_control must be nested inside the system content block"
    )


@pytest.mark.asyncio
async def test_generate_chat_last_message_block_has_cache_control() -> None:
    """generate_chat attaches a rolling cache breakpoint to the last message content block."""
    response = _mock_response()
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        msgs: list[Message] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        await llm.generate_chat(msgs, system="You are helpful.")

    call_kwargs = sdk_client.messages.create.call_args.kwargs
    last_block = _last_message_last_block(call_kwargs)
    assert last_block.get("cache_control") == {"type": "ephemeral"}, (
        "Rolling cache breakpoint must be attached to the last content block of the last message"
    )
    assert last_block.get("text") == "how are you?"


@pytest.mark.asyncio
async def test_generate_chat_earlier_messages_not_modified() -> None:
    """generate_chat does not attach cache_control to non-final messages."""
    response = _mock_response()
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        msgs: list[Message] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "how are you?"},
        ]
        await llm.generate_chat(msgs, system="You are helpful.")

    call_kwargs = sdk_client.messages.create.call_args.kwargs
    api_msgs = call_kwargs["messages"]
    # First two messages should be untouched bare strings
    assert api_msgs[0]["content"] == "hello"
    assert api_msgs[1]["content"] == "hi there"


# ---------------------------------------------------------------------------
# (d) generate_chat() without system — rolling breakpoint still present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_chat_without_system_rolling_breakpoint_present() -> None:
    """generate_chat without system= still places rolling breakpoint on last message block."""
    response = _mock_response()
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        msgs: list[Message] = [{"role": "user", "content": "ping"}]
        await llm.generate_chat(msgs)

    call_kwargs = sdk_client.messages.create.call_args.kwargs

    assert "cache_control" not in call_kwargs, "No top-level cache_control allowed"
    assert "system" not in call_kwargs, "No system key when system=None"

    last_block = _last_message_last_block(call_kwargs)
    assert last_block.get("cache_control") == {"type": "ephemeral"}, (
        "Rolling breakpoint must still be present even without a system prompt"
    )


# ---------------------------------------------------------------------------
# (e) Second agentic-loop turn — rolling breakpoint present on both turns
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agentic_loop_second_turn_carries_rolling_breakpoint() -> None:
    """Both turns in an agentic loop carry a rolling cache breakpoint on the last message block.

    The agentic ReAct loop (graph/agent.py) calls
    ``llm.generate_chat(messages, system=_AGENT_SYSTEM_PROMPT)`` on every turn,
    growing the messages list with each tool-call round-trip.  We simulate two
    consecutive turns and assert that BOTH outgoing requests carry a
    ``cache_control`` breakpoint nested in the last message's content block.
    """
    responses_iter = iter([_mock_response("first"), _mock_response("second")])
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()

        async def _side_effect(**kwargs: Any) -> MagicMock:
            return next(responses_iter)

        sdk_client.messages.create = AsyncMock(side_effect=_side_effect)
        mock_get.return_value = sdk_client

        llm = _make_client()
        agent_system = "You are a knowledge-base retrieval specialist."

        # Turn 1 — initial question
        msgs_turn1: list[Message] = [{"role": "user", "content": "Question: find python async"}]
        result1 = await llm.generate_chat(msgs_turn1, system=agent_system)

        # Turn 2 — after tool result appended
        msgs_turn2: list[Message] = [
            {"role": "user", "content": "Question: find python async"},
            {"role": "assistant", "content": '{"tool": "hybrid_search", "args": {}}'},
            {"role": "user", "content": "hybrid_search result: no entries found"},
        ]
        result2 = await llm.generate_chat(msgs_turn2, system=agent_system)

    assert result1 == "first"
    assert result2 == "second"
    assert sdk_client.messages.create.call_count == 2

    for i, call in enumerate(sdk_client.messages.create.call_args_list):
        kw = call.kwargs
        # No top-level cache_control anywhere
        assert "cache_control" not in kw, f"Turn {i + 1}: top-level cache_control must not exist"
        # System block carries breakpoint
        system_blocks = kw.get("system")
        assert isinstance(system_blocks, list), f"Turn {i + 1}: system must be content-block list"
        assert system_blocks[0].get("cache_control") == {"type": "ephemeral"}, (
            f"Turn {i + 1}: system block cache_control missing or wrong"
        )
        # Rolling breakpoint on last message's last block
        last_block = _last_message_last_block(kw)
        assert last_block.get("cache_control") == {"type": "ephemeral"}, (
            f"Turn {i + 1}: rolling breakpoint missing from last message block"
        )
