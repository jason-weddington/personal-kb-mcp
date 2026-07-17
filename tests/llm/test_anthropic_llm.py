"""Tests for the Anthropic LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kb_core.config import AnthropicProviderConfig

from personal_kb.llm.anthropic import AnthropicLLMClient


def _client(api_key: str | None = None, model: str = "claude-haiku-4-5") -> AnthropicLLMClient:
    """Helper: build a client with an explicit AnthropicProviderConfig."""
    return AnthropicLLMClient(AnthropicProviderConfig(model=model, api_key=api_key))


@pytest.fixture
def mock_response():
    """Create a mock Anthropic response."""
    content_block = MagicMock()
    content_block.text = "Hello from Haiku"
    response = MagicMock()
    response.content = [content_block]
    return response


@pytest.fixture
def mock_anthropic_class(mock_response):
    """Patch AsyncAnthropic and return the mock class."""
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=mock_response)
        client.close = AsyncMock()
        mock_get.return_value = client
        yield client


@pytest.mark.asyncio
async def test_generate_success(mock_anthropic_class, mock_response):
    """Successful generate returns text and sets available."""
    llm = _client()
    result = await llm.generate("test prompt")
    assert result == "Hello from Haiku"
    assert llm._available is True


@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_anthropic_class):
    """System prompt is sent as a content-block array with a cache breakpoint."""
    llm = _client()
    await llm.generate("test prompt", system="You are helpful")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    system_blocks = call_kwargs.kwargs.get("system")
    assert isinstance(system_blocks, list), "system should be a content-block list"
    assert len(system_blocks) == 1
    assert system_blocks[0]["type"] == "text"
    assert system_blocks[0]["text"] == "You are helpful"
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_generate_without_system_prompt(mock_anthropic_class):
    """No system kwarg when system is None."""
    llm = _client()
    await llm.generate("test prompt")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    assert "system" not in call_kwargs.kwargs


@pytest.mark.asyncio
async def test_generate_failure_returns_none():
    """Generate returns None and clears availability on failure."""
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=Exception("API error"))
        mock_get.return_value = client

        llm = _client()
        llm._available = True
        result = await llm.generate("test")
        assert result is None
        assert llm._available is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_client_none():
    """Generate returns None when SDK is not installed."""
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = _client()
        result = await llm.generate("test")
        assert result is None


@pytest.mark.asyncio
async def test_is_available_caches_success(mock_anthropic_class):
    """After successful generate, is_available returns True."""
    llm = _client()
    await llm.generate("test")
    assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_when_sdk_installed():
    """is_available returns True when the SDK is importable.

    The pre-flight env check is gone: kb_core defers credential
    resolution to the SDK. ``api_key=None`` is fine — a failed call
    will clear the cached availability.
    """
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        llm = _client(api_key=None)
        assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_when_sdk_missing():
    """is_available returns False when the SDK is not importable."""
    with patch("kb_core.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = None
        llm = _client(api_key=None)
        assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_close_cleans_up(mock_anthropic_class):
    """Close calls close on the underlying client."""
    llm = _client()
    llm._client = mock_anthropic_class
    await llm.close()
    mock_anthropic_class.close.assert_awaited_once()
    assert llm._client is None


@pytest.mark.asyncio
async def test_close_noop_when_no_client():
    """Close is safe to call when no client exists."""
    llm = _client()
    await llm.close()  # Should not raise


@pytest.mark.asyncio
async def test_model_override(mock_anthropic_class):
    """Model passed via config is used in API calls."""
    llm = _client(model="claude-sonnet-4-6")
    await llm.generate("test")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    assert call_kwargs.kwargs.get("model") == "claude-sonnet-4-6"


@pytest.mark.asyncio
async def test_model_default(mock_anthropic_class):
    """Without explicit model, the config default is used."""
    llm = AnthropicLLMClient(AnthropicProviderConfig())
    await llm.generate("test")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    # The dataclass default is the same as the env default.
    assert call_kwargs.kwargs.get("model") == "claude-haiku-4-5"


@pytest.mark.asyncio
async def test_protocol_conformance():
    """AnthropicLLMClient satisfies LLMProvider protocol."""
    from personal_kb.llm.provider import LLMProvider

    assert isinstance(_client(), LLMProvider)
