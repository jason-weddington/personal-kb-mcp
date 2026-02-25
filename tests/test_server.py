"""Tests for server-level functions."""

from unittest.mock import patch

from personal_kb.llm.anthropic import AnthropicLLMClient
from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.server import _create_llm


def test_create_llm_ollama():
    """Should create OllamaLLMClient for 'ollama' provider."""
    client = _create_llm("ollama")
    assert isinstance(client, OllamaLLMClient)


def test_create_llm_anthropic():
    """Should create AnthropicLLMClient for 'anthropic' provider."""
    client = _create_llm("anthropic")
    assert isinstance(client, AnthropicLLMClient)


def test_create_llm_anthropic_unavailable():
    """Should return None when Anthropic SDK is not installed."""
    with patch("personal_kb.server.AnthropicLLMClient", None):
        client = _create_llm("anthropic")
        assert client is None


def test_create_llm_unknown_provider():
    """Should return None for unknown providers."""
    client = _create_llm("unknown")
    assert client is None


def test_provider_config_defaults():
    """Default providers should both be 'anthropic'."""
    from personal_kb.config import get_extraction_provider, get_query_provider

    with patch.dict("os.environ", {}, clear=True):
        assert get_extraction_provider() == "anthropic"
        assert get_query_provider() == "anthropic"


def test_provider_config_from_env():
    """Provider config should read from environment variables."""
    from personal_kb.config import get_extraction_provider, get_query_provider

    with patch.dict(
        "os.environ",
        {
            "KB_EXTRACTION_PROVIDER": "ollama",
            "KB_QUERY_PROVIDER": "ollama",
        },
    ):
        assert get_extraction_provider() == "ollama"
        assert get_query_provider() == "ollama"


def test_mixed_providers():
    """Different providers for different use cases."""
    from personal_kb.config import get_extraction_provider, get_query_provider

    with patch.dict(
        "os.environ",
        {
            "KB_EXTRACTION_PROVIDER": "ollama",
            "KB_QUERY_PROVIDER": "anthropic",
        },
    ):
        assert get_extraction_provider() == "ollama"
        assert get_query_provider() == "anthropic"
