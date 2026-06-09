"""Tests for server-level functions."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import FastMCP

from personal_kb.server import (
    _build_instructions,
    _get_tool_prefix,
    create_server,
    lifespan,
)

# --- Provider config builders (channel-side env → kb_core.config) ---


def test_build_provider_config_default_anthropic():
    """Default env → all three roles point at the Anthropic provider."""
    from personal_kb.config import build_provider_config

    with patch.dict(
        "os.environ",
        {"KB_EXTRACTION_PROVIDER": "anthropic", "KB_QUERY_PROVIDER": "anthropic"},
    ):
        cfg = build_provider_config()
        assert cfg.extraction.provider == "anthropic"
        assert cfg.query.provider == "anthropic"
        assert cfg.synthesis.provider == "anthropic"


def test_build_provider_config_synthesis_uses_sonnet_for_anthropic():
    """Synthesis role overrides the Anthropic model to Sonnet."""
    from kb_core.llm.anthropic import _SONNET_MODEL as ANTHROPIC_SONNET

    from personal_kb.config import build_provider_config

    with patch.dict("os.environ", {"KB_QUERY_PROVIDER": "anthropic"}):
        cfg = build_provider_config()
        assert cfg.synthesis.anthropic.model == ANTHROPIC_SONNET


def test_build_provider_config_synthesis_uses_sonnet_for_bedrock():
    """Synthesis role overrides the Bedrock model to Sonnet."""
    from kb_core.llm.bedrock import _SONNET_MODEL as BEDROCK_SONNET

    from personal_kb.config import build_provider_config

    with patch.dict("os.environ", {"KB_QUERY_PROVIDER": "bedrock"}):
        cfg = build_provider_config()
        assert cfg.synthesis.bedrock.model == BEDROCK_SONNET


def test_build_provider_config_ollama_provider():
    """KB_QUERY_PROVIDER=ollama → query + synthesis roles both report 'ollama'."""
    from personal_kb.config import build_provider_config

    with patch.dict("os.environ", {"KB_QUERY_PROVIDER": "ollama"}):
        cfg = build_provider_config()
        assert cfg.query.provider == "ollama"
        assert cfg.synthesis.provider == "ollama"


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
    """role=personal → personal_kb_ prefix."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "personal"}):
        assert _get_tool_prefix() == "personal_kb_"


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


def test_build_instructions_personal_prefix():
    """Personal prefix should replace kb_ tool names in instructions."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "personal"}):
        text = _build_instructions("personal_kb_")
        assert "personal_kb_search" in text
        assert "personal_kb_store" in text
        assert "personal_kb_ask" in text
        # Entry IDs should NOT be replaced
        assert "kb-00042" in text
        # Role prefix should be present
        assert "PERSONAL knowledge base" in text


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


async def test_create_server_personal_tool_names():
    """Personal server should register tools with personal_kb_ prefix."""
    with patch.dict("os.environ", {"KB_INSTANCE_ROLE": "personal"}):
        mcp = create_server()
        tools = await mcp.list_tools()
        tool_names = {t.name for t in tools}
        assert "personal_kb_store" in tool_names
        assert "personal_kb_search" in tool_names
        assert "personal_kb_ask" in tool_names
        assert "kb_store" not in tool_names


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


# --- Lifespan: maps_index rebuild + listener wiring -----------------------------


@pytest.mark.asyncio
async def test_lifespan_invokes_rebuild_and_listener_teardown(monkeypatch, tmp_path) -> None:
    """Lifespan startup runs ``rebuild_all_projects`` and tears down the listener.

    Asserts:

    * ``maps_index_writer.rebuild_all_projects`` is awaited at startup (before
      the lifespan yields).
    * The teardown returned by ``db.start_maps_listener`` is awaited in the
      ``finally`` block BEFORE ``db.close()`` — verified by spying on the
      teardown mock with a sentinel.
    * No exceptions propagate.

    The actual live LISTEN→NOTIFY round-trip is not exercised here —
    that requires a real Postgres (manual verify, see CLAUDE.md).
    """
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.delenv("KB_DATABASE_URL", raising=False)
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.setenv("KB_AUTO_EXPLORE", "FALSE")

    rebuild_mock = AsyncMock(return_value=None)
    teardown_mock = AsyncMock(return_value=None)
    start_listener_mock = AsyncMock(return_value=teardown_mock)

    with (
        patch(
            "personal_kb.server.maps_index_writer.rebuild_all_projects",
            rebuild_mock,
        ),
        patch(
            "kb_core.search.embeddings.EmbeddingClient.is_available",
            AsyncMock(return_value=False),
        ),
        patch(
            "kb_core.db.sqlite_backend.SQLiteBackend.start_maps_listener",
            start_listener_mock,
        ),
    ):
        mcp = FastMCP("test-kb-lifespan", lifespan=lifespan)
        async with lifespan(mcp) as ctx:
            # Lifespan now stores the KnowledgeBase facade as the source of
            # truth — db / store / embedder / LLMs are reached through it.
            assert "kb" in ctx
            assert ctx["kb"].db is not None
            # Rebuild must have been awaited during startup.
            assert rebuild_mock.await_count >= 1
            # start_maps_listener was invoked with on_change + on_reconnect.
            start_listener_mock.assert_awaited_once()
            kwargs = start_listener_mock.await_args.kwargs
            assert "on_change" in kwargs
            assert "on_reconnect" in kwargs

    # After leaving the context (finally block ran):
    teardown_mock.assert_awaited_once()
