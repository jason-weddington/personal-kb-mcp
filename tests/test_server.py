"""Tests for server-level functions."""

from unittest.mock import patch

import pytest

from personal_kb.server import (
    _build_instructions,
    _get_tool_prefix,
    create_server,
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
    """Synthesis role overrides the Anthropic model to Sonnet at effort low."""
    from kb_core.llm.anthropic import _SONNET_MODEL as ANTHROPIC_SONNET

    from personal_kb.config import build_provider_config

    with patch.dict(
        "os.environ",
        {"KB_QUERY_PROVIDER": "anthropic", "KB_SYNTHESIS_MODEL": "", "KB_SYNTHESIS_EFFORT": ""},
    ):
        cfg = build_provider_config()
        assert cfg.synthesis.anthropic.model == ANTHROPIC_SONNET
        assert cfg.synthesis.anthropic.effort == "low"
        assert cfg.synthesis.anthropic.thinking is None


def test_build_provider_config_synthesis_env_override():
    """KB_SYNTHESIS_MODEL/KB_SYNTHESIS_EFFORT reach only the synthesis role."""
    from personal_kb.config import build_provider_config

    with patch.dict(
        "os.environ",
        {"KB_SYNTHESIS_MODEL": "claude-sonnet-4-6", "KB_SYNTHESIS_EFFORT": "medium"},
    ):
        cfg = build_provider_config()
        assert cfg.synthesis.anthropic.model == "claude-sonnet-4-6"
        assert cfg.synthesis.anthropic.effort == "medium"
        assert cfg.extraction.anthropic.model == "claude-haiku-5-5"
        assert cfg.query.anthropic.model == "claude-haiku-5-5"
        assert cfg.extraction.anthropic.thinking == "disabled"
        assert cfg.query.anthropic.thinking == "disabled"
        assert cfg.extraction.anthropic.effort is None
        assert cfg.query.anthropic.effort is None


def test_build_provider_config_synthesis_uses_sonnet_for_bedrock():
    """Synthesis role overrides the Bedrock model to Sonnet."""
    from kb_core.llm.bedrock import _SONNET_MODEL as BEDROCK_SONNET

    from personal_kb.config import build_provider_config

    with patch.dict(
        "os.environ",
        {
            "KB_QUERY_PROVIDER": "bedrock",
            "KB_BEDROCK_SYNTHESIS_MODEL": "",
            "KB_SYNTHESIS_EFFORT": "",
        },
    ):
        cfg = build_provider_config()
        assert cfg.synthesis.bedrock.model == BEDROCK_SONNET == "us.anthropic.claude-sonnet-5-5"
        assert cfg.synthesis.bedrock.effort == "low"
        assert cfg.synthesis.bedrock.thinking is None


def test_build_provider_config_bedrock_mirrors_anthropic_settings(monkeypatch):
    """Bedrock extraction/query get Haiku 5.5 with thinking disabled, like Anthropic."""
    from personal_kb.config import build_provider_config

    for var in ("KB_BEDROCK_MODEL", "KB_ANTHROPIC_THINKING", "KB_BEDROCK_SYNTHESIS_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "bedrock")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "bedrock")
    cfg = build_provider_config()
    for role in (cfg.extraction, cfg.query):
        assert role.bedrock.model == "us.anthropic.claude-haiku-5-5"
        assert role.bedrock.thinking == "disabled"
        assert role.bedrock.effort is None


def test_build_provider_config_bedrock_synthesis_override(monkeypatch):
    """KB_BEDROCK_SYNTHESIS_MODEL reaches only the Bedrock synthesis role."""
    from personal_kb.config import build_provider_config

    monkeypatch.setenv("KB_BEDROCK_SYNTHESIS_MODEL", " us.anthropic.claude-sonnet-4-6 ")
    monkeypatch.setenv("KB_SYNTHESIS_EFFORT", "default")
    monkeypatch.delenv("KB_BEDROCK_MODEL", raising=False)
    cfg = build_provider_config()
    assert cfg.synthesis.bedrock.model == "us.anthropic.claude-sonnet-4-6"
    assert cfg.synthesis.bedrock.effort is None
    assert cfg.query.bedrock.model == "us.anthropic.claude-haiku-5-5"
    assert cfg.synthesis.anthropic.model == "claude-sonnet-5-5"


def test_build_provider_config_ollama_provider():
    """KB_QUERY_PROVIDER=ollama → query + synthesis roles both report 'ollama'."""
    from personal_kb.config import build_provider_config

    with patch.dict("os.environ", {"KB_QUERY_PROVIDER": "ollama"}):
        cfg = build_provider_config()
        assert cfg.query.provider == "ollama"
        assert cfg.synthesis.provider == "ollama"


def test_build_provider_config_thinking_default(monkeypatch, caplog):
    """Env unset → extraction/query thinking disabled, synthesis untouched."""
    import logging

    from personal_kb.config import build_provider_config

    monkeypatch.delenv("KB_ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("KB_ANTHROPIC_THINKING", raising=False)
    with caplog.at_level(logging.INFO, logger="personal_kb.config"):
        cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking == "disabled"
    assert cfg.query.anthropic.thinking == "disabled"
    assert cfg.synthesis.anthropic.thinking is None
    assert cfg.extraction.anthropic.model == "claude-haiku-5-5"
    shape = (
        "llm roles extraction_model=claude-haiku-5-5 extraction_thinking=disabled "
        "query_model=claude-haiku-5-5 query_thinking=disabled "
        "synthesis_model=claude-sonnet-5-5 synthesis_thinking=None synthesis_effort=low"
    )
    assert shape in caplog.text


def test_build_provider_config_thinking_adaptive(monkeypatch):
    """KB_ANTHROPIC_THINKING=adaptive reaches only the extraction/query roles."""
    from personal_kb.config import build_provider_config

    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "adaptive")
    cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking == "adaptive"
    assert cfg.query.anthropic.thinking == "adaptive"
    assert cfg.synthesis.anthropic.thinking is None
    assert cfg.synthesis.anthropic.model == "claude-sonnet-5-5"
    assert cfg.synthesis.anthropic.effort == "low"


def test_build_provider_config_thinking_default_sentinel(monkeypatch):
    """KB_ANTHROPIC_THINKING=default sends no thinking field on any role."""
    from personal_kb.config import build_provider_config

    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "default")
    cfg = build_provider_config()
    assert cfg.extraction.anthropic.thinking is None
    assert cfg.query.anthropic.thinking is None
    assert cfg.synthesis.anthropic.thinking is None


def test_build_provider_config_invalid_thinking_warns_per_call(monkeypatch, caplog):
    """An unknown value warns once per build and falls back to disabled."""
    import logging

    from personal_kb.config import build_provider_config

    monkeypatch.setenv("KB_ANTHROPIC_THINKING", "bogus")
    with caplog.at_level(logging.WARNING, logger="personal_kb.config"):
        cfg = build_provider_config()
        assert cfg.extraction.anthropic.thinking == "disabled"
        assert cfg.query.anthropic.thinking == "disabled"
        assert cfg.synthesis.anthropic.thinking is None
        first = [
            r for r in caplog.records if r.levelno == logging.WARNING and "bogus" in r.getMessage()
        ]
        assert len(first) == 1
        build_provider_config()
        second = [
            r for r in caplog.records if r.levelno == logging.WARNING and "bogus" in r.getMessage()
        ]
        assert len(second) == 2


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
        assert "kb-00003" in text
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
        assert "kb-00003" in text
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


async def test_create_server_registers_map_eligibility_tools_without_manager_mode():
    """The map-eligibility tools register unconditionally, outside the manager gate."""
    import os

    for role, prefix in (("", "kb_"), ("personal", "personal_kb_"), ("team", "team_kb_")):
        env = {"KB_INSTANCE_ROLE": role} if role else {}
        with patch.dict("os.environ", env, clear=False):
            os.environ.pop("KB_MANAGER", None)
            if not role:
                os.environ.pop("KB_INSTANCE_ROLE", None)
            mcp = create_server()
            tools = await mcp.list_tools()
            tool_names = {t.name for t in tools}
            assert f"{prefix}map_eligibility" in tool_names, role
            assert f"{prefix}map_eligibility_override" in tool_names, role
            # Pin AC12's intent: the manager gate still works around these tools.
            assert f"{prefix}maintain" not in tool_names, role


# --- Lifespan: HttpBackend-over-loopback ----------------------------------------
#
# The maps_index rebuild + LISTEN/NOTIFY wiring previously lived in the
# server.py lifespan's local-mode branch.  Under Reading B (kb-01807) the
# local-mode branch was replaced with an HttpBackend-over-loopback path that
# spawns a daemon (see ``test_daemon_spawn.py``).  The maps_index writer and
# the NOTIFY/LISTEN machinery have since been deleted entirely, so there is
# no rebuild/teardown wiring left to test here.


@pytest.mark.parametrize(("role", "name"), [(None, "kb_store"), ("personal", "personal_kb_store")])
async def test_kb_store_schema_requires_supersedes(role, name):
    import os

    from personal_kb.tools.kb_store import SUPERSEDES_DESCRIPTION

    env = {"KB_INSTANCE_ROLE": role} if role else {}
    with patch.dict("os.environ", env, clear=False):
        if not role:
            os.environ.pop("KB_INSTANCE_ROLE", None)
        mcp = create_server()
        tools = {t.name: t for t in await mcp.list_tools()}
    schema = tools[name].parameters
    assert "supersedes" in schema["required"]
    assert "distinct_from" not in schema["required"]
    assert "superseded_by" not in schema["required"]
    assert schema["properties"]["supersedes"]["description"] == SUPERSEDES_DESCRIPTION


async def test_kb_store_call_without_valid_supersedes_raises():
    import os

    with patch.dict("os.environ", {}, clear=False):
        os.environ.pop("KB_INSTANCE_ROLE", None)
        mcp = create_server()
    base = {"short_title": "a", "long_title": "b", "knowledge_details": "c"}
    with pytest.raises(Exception) as excinfo:
        await mcp.call_tool("kb_store", base)
    assert "supersedes" in str(excinfo.value)
    with pytest.raises(Exception) as excinfo2:
        await mcp.call_tool("kb_store", {**base, "supersedes": "kb-00001"})
    assert "supersedes" in str(excinfo2.value)


def test_instructions_supersedes_is_a_parameter_not_a_hint():
    text = _build_instructions("kb_")
    assert '{"supersedes"' not in text
    assert "supersedes is a required kb_store parameter" in text


@pytest.mark.parametrize(("role", "name"), [(None, "kb_store"), ("personal", "personal_kb_store")])
async def test_kb_store_schema_has_optional_resolution(role, name):
    import os

    from personal_kb.tools.kb_store import RESOLUTION_DESCRIPTION

    env = {"KB_INSTANCE_ROLE": role} if role else {}
    with patch.dict("os.environ", env, clear=False):
        if not role:
            os.environ.pop("KB_INSTANCE_ROLE", None)
        mcp = create_server()
        tools = {t.name: t for t in await mcp.list_tools()}
    schema = tools[name].parameters
    assert "resolution" in schema["properties"]
    assert "resolution" not in schema["required"]
    prop = schema["properties"]["resolution"]
    assert prop["description"] == RESOLUTION_DESCRIPTION
    assert {"const": "none", "type": "string"} in prop["anyOf"]


def test_instructions_describe_required_lesson_resolution():
    from kb_service.mcp_server.server import _build_instructions as http_instructions

    for text in (_build_instructions("kb_"), http_instructions("kb_")):
        assert "lesson_learned requires the resolution parameter" in text
        assert "A corrected belief gets" not in text


def test_build_instructions_mentions_resolution_hint():
    """The STORE section documents the resolution parameter."""
    assert "resolution" in _build_instructions("kb_")


# --- stdio deprecation note (MCP over HTTP overlap) --------------------------


def test_stdio_deprecation_note_silent_for_local_daemon():
    from personal_kb.config import LOCAL_KB_API_KEY
    from personal_kb.server import stdio_deprecation_note

    assert stdio_deprecation_note("http://127.0.0.1:8765", api_key=LOCAL_KB_API_KEY) is None


def test_stdio_deprecation_note_for_hosted_kb_on_loopback(monkeypatch):
    """A real key on a loopback URL is a hosted KB (co-located or SSH-tunnelled)."""
    from personal_kb.server import stdio_deprecation_note

    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    note = stdio_deprecation_note("http://127.0.0.1:8765", api_key="secret-k")
    assert note is not None
    assert "http://127.0.0.1:8765/mcp" in note
    assert "secret-k" not in note


def test_stdio_deprecation_note_for_hosted_url(monkeypatch):
    from personal_kb.server import stdio_deprecation_note

    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret-k")
    note = stdio_deprecation_note("https://kb.example.com/", api_key="secret-k")
    assert note is not None
    assert "https://kb.example.com/mcp" in note
    assert "--transport http" in note
    assert "personal-kb" in note
    assert "$PERSONAL_KB_API_KEY" in note
    assert "secret-k" not in note


def test_stdio_deprecation_note_team_server_name(monkeypatch):
    from personal_kb.server import stdio_deprecation_note

    monkeypatch.setenv("KB_INSTANCE_ROLE", "team")
    note = stdio_deprecation_note("https://team.example.com", api_key="k")
    assert note is not None
    assert "claude mcp add --transport http team-kb https://team.example.com/mcp" in note


async def test_lifespan_yields_and_logs_deprecation_note(monkeypatch, tmp_path, caplog):
    import logging
    from unittest.mock import MagicMock

    from personal_kb import version_skew
    from personal_kb.backend.http import HttpBackend
    from personal_kb.server import lifespan, stdio_deprecation_note

    async def _noop(self):
        return None

    async def _no_skew(_url, *, api_key):
        return None

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret-k")
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.setattr(HttpBackend, "open", _noop)
    monkeypatch.setattr(HttpBackend, "close", _noop)
    monkeypatch.setattr(version_skew, "check_version_skew", _no_skew)

    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        with caplog.at_level(logging.WARNING, logger="personal_kb.server"):
            async with lifespan(MagicMock()) as state:
                assert state["deprecation_note"] == stdio_deprecation_note(
                    "https://kb.example.com", api_key="secret-k"
                )
        warnings = [
            r
            for r in caplog.records
            if r.name == "personal_kb.server" and r.levelno == logging.WARNING
        ]
        assert [r.getMessage() for r in warnings] == [state["deprecation_note"]]
    finally:
        for handler in root.handlers:
            if handler not in saved_handlers:
                handler.close()
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_instructions_fit_claude_codes_per_server_cap() -> None:
    """Claude Code keeps only the first 2048 characters of each MCP server's
    instructions before 2.1.296 (4096 from 2.1.296), and weeks-long sessions keep
    the version they started on, so the full text plus the longest role prefix must
    fit in 2048 for both the stdio and HTTP servers."""
    from kb_service.mcp_server import server as http_server

    from personal_kb import server as stdio_server

    cap = 2048
    for module in (stdio_server, http_server):
        longest_prefix = max(len(p) for p in module._ROLE_PREFIXES.values())
        assert len(module._INSTRUCTIONS) + longest_prefix <= cap, module.__name__
