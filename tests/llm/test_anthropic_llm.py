"""Tests for the Anthropic LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_kb.llm.anthropic import AnthropicLLMClient


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
    with patch("personal_kb.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        client = AsyncMock()
        client.messages.create = AsyncMock(return_value=mock_response)
        client.close = AsyncMock()
        mock_get.return_value = client
        yield client


@pytest.mark.asyncio
async def test_generate_success(mock_anthropic_class, mock_response):
    """Successful generate returns text and sets available."""
    llm = AnthropicLLMClient()
    result = await llm.generate("test prompt")
    assert result == "Hello from Haiku"
    assert llm._available is True


@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_anthropic_class):
    """System prompt is passed through to the API."""
    llm = AnthropicLLMClient()
    await llm.generate("test prompt", system="You are helpful")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    assert call_kwargs.kwargs.get("system") == "You are helpful"


@pytest.mark.asyncio
async def test_generate_without_system_prompt(mock_anthropic_class):
    """No system kwarg when system is None."""
    llm = AnthropicLLMClient()
    await llm.generate("test prompt")
    call_kwargs = mock_anthropic_class.messages.create.call_args
    assert "system" not in call_kwargs.kwargs


@pytest.mark.asyncio
async def test_generate_failure_returns_none():
    """Generate returns None and clears availability on failure."""
    with patch("personal_kb.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        client = AsyncMock()
        client.messages.create = AsyncMock(side_effect=Exception("API error"))
        mock_get.return_value = client

        llm = AnthropicLLMClient()
        llm._available = True
        result = await llm.generate("test")
        assert result is None
        assert llm._available is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_client_none():
    """Generate returns None when SDK is not installed."""
    with patch("personal_kb.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = AnthropicLLMClient()
        result = await llm.generate("test")
        assert result is None


@pytest.mark.asyncio
async def test_is_available_caches_success(mock_anthropic_class):
    """After successful generate, is_available returns True."""
    llm = AnthropicLLMClient()
    await llm.generate("test")
    assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_api_key():
    """is_available returns True when API key is set (optimistic)."""
    with patch("personal_kb.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}):
            llm = AnthropicLLMClient()
            assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_without_api_key():
    """is_available returns False when no API key set."""
    with patch("personal_kb.llm.anthropic.AnthropicLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            llm = AnthropicLLMClient()
            assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_close_cleans_up(mock_anthropic_class):
    """Close calls close on the underlying client."""
    llm = AnthropicLLMClient()
    llm._client = mock_anthropic_class
    await llm.close()
    mock_anthropic_class.close.assert_awaited_once()
    assert llm._client is None


@pytest.mark.asyncio
async def test_close_noop_when_no_client():
    """Close is safe to call when no client exists."""
    llm = AnthropicLLMClient()
    await llm.close()  # Should not raise


@pytest.mark.asyncio
async def test_protocol_conformance():
    """AnthropicLLMClient satisfies LLMProvider protocol."""
    from personal_kb.llm.provider import LLMProvider

    assert isinstance(AnthropicLLMClient(), LLMProvider)
