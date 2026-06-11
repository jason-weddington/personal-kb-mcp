"""Payload-assertion tests for prompt-caching in AnthropicLLMClient.

Three cases from the acceptance criteria:
  (a) generate_chat carries cache_control
  (b) a second agentic-loop turn (graph/agent.py) carries cache_control
  (c) plain generate() does NOT carry cache_control

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


# ---------------------------------------------------------------------------
# (a) generate_chat carries cache_control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_chat_sends_cache_control() -> None:
    """generate_chat passes cache_control={"type": "ephemeral"} to messages.create."""
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
    assert "cache_control" in call_kwargs, "cache_control kwarg missing from generate_chat call"
    assert call_kwargs["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_generate_chat_cache_control_without_system() -> None:
    """cache_control is present even when no system prompt is passed."""
    response = _mock_response()
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        msgs: list[Message] = [{"role": "user", "content": "ping"}]
        await llm.generate_chat(msgs)

    call_kwargs = sdk_client.messages.create.call_args.kwargs
    assert call_kwargs.get("cache_control") == {"type": "ephemeral"}
    assert "system" not in call_kwargs


# ---------------------------------------------------------------------------
# (b) Second agentic-loop turn carries cache_control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agentic_loop_second_turn_carries_cache_control() -> None:
    """The second LLM call in the agentic loop carries cache_control.

    The agentic ReAct loop (graph/agent.py) calls
    ``llm.generate_chat(messages, system=_AGENT_SYSTEM_PROMPT)`` on every turn,
    growing the messages list with each tool-call round-trip.  We simulate two
    consecutive turns and assert that BOTH outgoing requests carry the
    ``cache_control`` kwarg — so the second turn can reuse the prefix cached by
    the first.
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

        # Turn 1 — initial question (mirrors what agentic_query seeds)
        msgs_turn1: list[Message] = [{"role": "user", "content": "Question: find python async"}]
        result1 = await llm.generate_chat(msgs_turn1, system=agent_system)

        # Turn 2 — after tool result appended (mirrors the loop appending assistant + user)
        msgs_turn2: list[Message] = [
            {"role": "user", "content": "Question: find python async"},
            {"role": "assistant", "content": '{"tool": "hybrid_search", "args": {}}'},
            {"role": "user", "content": "hybrid_search result: no entries found"},
        ]
        result2 = await llm.generate_chat(msgs_turn2, system=agent_system)

    assert result1 == "first"
    assert result2 == "second"

    # Both calls must carry cache_control
    assert sdk_client.messages.create.call_count == 2
    for i, call in enumerate(sdk_client.messages.create.call_args_list):
        kw = call.kwargs
        assert "cache_control" in kw, f"Turn {i + 1} missing cache_control"
        assert kw["cache_control"] == {"type": "ephemeral"}, f"Turn {i + 1} wrong cache_control"


# ---------------------------------------------------------------------------
# (c) generate() does NOT carry cache_control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_does_not_send_cache_control() -> None:
    """Single-shot generate() must NOT include cache_control in its request."""
    response = _mock_response("generated text")
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        result = await llm.generate("Summarize this document.")

    assert result == "generated text"
    call_kwargs = sdk_client.messages.create.call_args.kwargs
    assert "cache_control" not in call_kwargs, (
        "generate() must NOT include cache_control — single-shot prompts have no reusable prefix"
    )


@pytest.mark.asyncio
async def test_generate_with_system_does_not_send_cache_control() -> None:
    """Single-shot generate() with system= still must NOT include cache_control."""
    response = _mock_response("answer")
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        sdk_client: Any = AsyncMock()
        sdk_client.messages.create = AsyncMock(return_value=response)
        mock_get.return_value = sdk_client

        llm = _make_client()
        result = await llm.generate("Extract entities.", system="You extract entities.")

    assert result == "answer"
    call_kwargs = sdk_client.messages.create.call_args.kwargs
    assert "cache_control" not in call_kwargs
