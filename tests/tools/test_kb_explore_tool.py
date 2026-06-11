"""Tests for the kb_explore MCP tool (hosted-only mode)."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from personal_kb.tools.kb_explore import register_kb_explore


def _register_tool() -> Any:
    """Register kb_explore on a mock MCP and return the captured function."""
    tools: dict[str, Any] = {}

    def capture(**_kw):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_explore(mcp)
    return next(iter(tools.values()))


def _make_ctx() -> MagicMock:
    """Return a mock Context with no real lifespan needed."""
    return MagicMock()


@pytest.mark.asyncio
async def test_kb_explore_local_mode_returns_hosted_message(monkeypatch):
    """In local mode (no PERSONAL_KB_URL), kb_explore returns a hosted message."""
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)

    kb_explore = _register_tool()
    ctx = _make_ctx()
    result = await kb_explore(ctx=ctx)

    assert "KB explorer is hosted at" in result
    assert "open it in a browser" in result


@pytest.mark.asyncio
async def test_kb_explore_http_mode_includes_url(monkeypatch):
    """In HTTP mode (PERSONAL_KB_URL set), kb_explore returns the service URL."""
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")

    kb_explore = _register_tool()
    ctx = _make_ctx()
    result = await kb_explore(ctx=ctx)

    assert "KB explorer is hosted at https://kb.example.com" in result
    assert "open it in a browser" in result


@pytest.mark.asyncio
async def test_kb_explore_raises_without_context():
    """kb_explore raises RuntimeError when ctx is None."""
    kb_explore = _register_tool()
    with pytest.raises(RuntimeError, match="Context not injected"):
        await kb_explore(ctx=None)


def test_kb_explore_tool_registered_with_correct_name():
    """register_kb_explore registers the tool with the correct name."""
    registered_names: list[str] = []

    def capture(**kw):
        registered_names.append(kw.get("name", ""))

        def decorator(fn):
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_explore(mcp, prefix="kb_")
    assert registered_names == ["kb_explore"]


def test_kb_explore_tool_registered_with_custom_prefix():
    """register_kb_explore respects a custom prefix."""
    registered_names: list[str] = []

    def capture(**kw):
        registered_names.append(kw.get("name", ""))

        def decorator(fn):
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_explore(mcp, prefix="team_kb_")
    assert registered_names == ["team_kb_explore"]
