"""Tests for server-level functions."""

from unittest.mock import patch

from personal_kb.llm.anthropic import AnthropicLLMClient
from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.server import _build_instructions, _create_llm, _get_tool_prefix, create_server


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


# --- Tool prefix tests ---


def test_tool_prefix_default():
    """No role → default kb_ prefix."""
    with patch.dict("os.environ", {}, clear=False):
        # Remove KB_INSTANCE_ROLE if set
        import os

        os.environ.pop("KB_INSTANCE_ROLE", None)
        assert _get_tool_prefix() == "kb_"


def test_tool_prefix_personal():
    """role=personal → kb_ prefix (unchanged)."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "personal"}):
        assert _get_tool_prefix() == "kb_"


def test_tool_prefix_team():
    """role=team → team_kb_ prefix."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "team"}):
        assert _get_tool_prefix() == "team_kb_"


def test_tool_prefix_team_case_insensitive():
    """KB_INSTANCE_ROLE is case-insensitive."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "TEAM"}):
        assert _get_tool_prefix() == "team_kb_"


def test_build_instructions_default_prefix():
    """Default prefix should not alter tool names in instructions."""
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("KB_INSTANCE_ROLE", None)
        text = _build_instructions("kb_")
        assert "kb_search" in text
        assert "kb_store" in text
        assert "team_kb_" not in text


def test_build_instructions_team_prefix():
    """Team prefix should replace kb_ tool names in instructions."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "team"}):
        text = _build_instructions("team_kb_")
        assert "team_kb_search" in text
        assert "team_kb_store" in text
        assert "team_kb_ask" in text
        assert "team_kb_summarize" in text
        assert "team_kb_get" in text
        assert "team_kb_feedback" in text
        assert "team_kb_ingest" in text
        # Entry IDs should NOT be replaced
        assert "kb-00042" in text
        # Role prefix should be present
        assert "TEAM knowledge base" in text


def test_build_instructions_preserves_entry_ids():
    """store_batch replaced before store to avoid partial match."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "team"}):
        text = _build_instructions("team_kb_")
        assert "team_kb_store_batch" in text
        # Ensure no double-replacement like team_kb_store_batch becoming
        # team_kb_team_kb_store_batch
        assert "team_kb_team_kb_" not in text


async def test_create_server_default_tool_names():
    """Default server should register tools with kb_ prefix."""
    with patch.dict("os.environ", {}, clear=False):
        import os

        os.environ.pop("KB_INSTANCE_ROLE", None)
        mcp = create_server()
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "kb_store" in tool_names
        assert "kb_search" in tool_names
        assert "kb_ask" in tool_names


async def test_create_server_team_tool_names():
    """Team server should register tools with team_kb_ prefix."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "team", "KB_MANAGER": "TRUE"}):
        mcp = create_server()
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "team_kb_store" in tool_names
        assert "team_kb_search" in tool_names
        assert "team_kb_ask" in tool_names
        assert "team_kb_get" in tool_names
        assert "team_kb_summarize" in tool_names
        assert "team_kb_store_batch" in tool_names
        assert "team_kb_ingest" in tool_names
        assert "team_kb_feedback" in tool_names
        assert "team_kb_maintain" in tool_names
        # No kb_ tools should exist
        assert "kb_store" not in tool_names
        assert "kb_search" not in tool_names
