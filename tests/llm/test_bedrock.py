"""Tests for the Bedrock LLM client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from personal_kb.llm.bedrock import BedrockLLMClient


@pytest.fixture
def mock_converse_response():
    """Create a mock Bedrock Converse response."""
    content_block = MagicMock()
    content_block.value = "Hello from Bedrock"
    message = MagicMock()
    message.content = [content_block]
    output = MagicMock()
    output.value = message
    response = MagicMock()
    response.output = output
    return response


@pytest.fixture
def mock_bedrock_client(mock_converse_response):
    """Patch _get_client and return the mock client."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        client = MagicMock()
        client.converse = AsyncMock(return_value=mock_converse_response)
        mock_get.return_value = client
        yield client


@pytest.mark.asyncio
async def test_generate_success(mock_bedrock_client):
    """Successful generate returns text and sets available."""
    llm = BedrockLLMClient()
    result = await llm.generate("test prompt")
    assert result == "Hello from Bedrock"
    assert llm._available is True


@pytest.mark.asyncio
async def test_generate_with_system_prompt(mock_bedrock_client):
    """System prompt is passed through to the Converse API."""
    llm = BedrockLLMClient()
    await llm.generate("test prompt", system="You are helpful")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is not None
    assert len(converse_input.system) == 1
    assert converse_input.system[0].value == "You are helpful"


@pytest.mark.asyncio
async def test_generate_without_system_prompt(mock_bedrock_client):
    """No system field when system is None."""
    llm = BedrockLLMClient()
    await llm.generate("test prompt")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    assert converse_input.system is None


@pytest.mark.asyncio
async def test_generate_failure_returns_none():
    """Generate returns None and clears availability on failure."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        client = MagicMock()
        client.converse = AsyncMock(side_effect=Exception("AWS error"))
        mock_get.return_value = client

        llm = BedrockLLMClient()
        llm._available = True
        result = await llm.generate("test")
        assert result is None
        assert llm._available is None


@pytest.mark.asyncio
async def test_generate_returns_none_when_client_none():
    """Generate returns None when SDK is not installed."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = BedrockLLMClient()
        result = await llm.generate("test")
        assert result is None


@pytest.mark.asyncio
async def test_is_available_caches_success(mock_bedrock_client):
    """After successful generate, is_available returns True."""
    llm = BedrockLLMClient()
    await llm.generate("test")
    assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_true_with_aws_key():
    """is_available returns True when AWS credentials are set."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {"AWS_ACCESS_KEY_ID": "AKIA_TEST"}):
            llm = BedrockLLMClient()
            assert await llm.is_available() is True


@pytest.mark.asyncio
async def test_is_available_false_without_aws_key():
    """is_available returns False when no AWS credentials set."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = MagicMock()
        with patch.dict("os.environ", {}, clear=True):
            llm = BedrockLLMClient()
            assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_is_available_false_when_sdk_missing():
    """is_available returns False when SDK is not importable."""
    with patch("personal_kb.llm.bedrock.BedrockLLMClient._get_client") as mock_get:
        mock_get.return_value = None

        llm = BedrockLLMClient()
        assert await llm.is_available() is False


@pytest.mark.asyncio
async def test_generate_escapes_newlines(mock_bedrock_client):
    """Newlines are escaped to work around smithy-json serialization bug."""
    llm = BedrockLLMClient()
    await llm.generate("line 1\nline 2\nline 3", system="rule 1\nrule 2")
    call_args = mock_bedrock_client.converse.call_args
    converse_input = call_args[0][0]
    prompt_value = converse_input.messages[0].content[0].value
    system_value = converse_input.system[0].value
    assert "\n" not in prompt_value
    assert "\\n" in prompt_value
    assert "\n" not in system_value
    assert "\\n" in system_value


@pytest.mark.asyncio
async def test_close_is_noop():
    """Close is safe to call (no-op)."""
    llm = BedrockLLMClient()
    await llm.close()  # Should not raise


@pytest.mark.asyncio
async def test_protocol_conformance():
    """BedrockLLMClient satisfies LLMProvider protocol."""
    from personal_kb.llm.provider import LLMProvider

    assert isinstance(BedrockLLMClient(), LLMProvider)
