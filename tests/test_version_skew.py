"""Client/server version-skew warning."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from personal_kb import version_skew
from personal_kb.version_skew import check_version_skew, compare_versions


@pytest.mark.parametrize(
    ("client", "server", "warns"),
    [
        ("0.68.0", "0.67.0", True),
        ("0.67.0", "0.67.0", False),
        ("0.66.0", "0.67.0", False),
        ("0.67.1.dev3", "0.67.0", True),
        ("0.67.0+local", "0.67.0", True),
        ("0.67.0.dev1", "0.67.0.dev1", False),
    ],
)
def test_compare_versions(client: str, server: str, warns: bool) -> None:
    assert (compare_versions(client, server) is not None) is warns


@pytest.mark.parametrize("server", [None, "", "unknown", 5])
def test_unparseable_server_is_silent(server: Any, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        assert compare_versions("1.0.0", server) is None
    assert not [r for r in caplog.records if r.levelno >= logging.INFO]


def _mock_health(monkeypatch: pytest.MonkeyPatch, body: dict[str, Any] | None) -> None:
    async def fake(_url: str) -> dict[str, Any] | None:
        return body

    monkeypatch.setattr("personal_kb.daemon._fetch_health", fake)


async def test_remote_newer_client_warns(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(version_skew, "client_version", lambda: "0.68.0")
    _mock_health(monkeypatch, {"status": "ok", "version": "0.67.0"})
    with caplog.at_level(logging.WARNING):
        msg = await check_version_skew("https://kb.example.com")
    assert msg and "0.68.0" in msg and "0.67.0" in msg
    assert any(r.levelno == logging.WARNING for r in caplog.records)


async def test_remote_equal_or_older_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version_skew, "client_version", lambda: "0.67.0")
    _mock_health(monkeypatch, {"status": "ok", "version": "0.67.0"})
    assert await check_version_skew("https://kb.example.com") is None
    _mock_health(monkeypatch, {"status": "ok", "version": "0.70.0"})
    assert await check_version_skew("https://kb.example.com") is None


async def test_loopback_skips_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version_skew, "client_version", lambda: "9.9.9")

    async def boom(_url: str) -> None:
        raise AssertionError("health must not be fetched for loopback")

    monkeypatch.setattr("personal_kb.daemon._fetch_health", boom)
    assert await check_version_skew("http://127.0.0.1:8765") is None


async def test_missing_server_version_is_silent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version_skew, "client_version", lambda: "9.9.9")
    _mock_health(monkeypatch, {"status": "ok"})
    assert await check_version_skew("https://kb.example.com") is None
    _mock_health(monkeypatch, None)
    assert await check_version_skew("https://kb.example.com") is None


async def test_preflight_appends_skew_line() -> None:
    from personal_kb.backend.http import HttpBackend
    from personal_kb.tools.kb_preflight import register_kb_preflight

    tools: dict[str, Any] = {}

    def capture(**_kw: Any) -> Any:
        def deco(fn: Any) -> Any:
            tools[fn.__name__] = fn
            return fn

        return deco

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_preflight(mcp)

    backend = HttpBackend(base_url="http://kb.test", api_key="k")
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={"context": "CTX"})),
    )
    ctx = MagicMock()
    ctx.lifespan_context = {"backend": backend, "version_skew_note": "SKEW NOTE"}
    out = await tools["kb_preflight"](project_ref="p", ctx=ctx)
    assert out == "CTX\n\nSKEW NOTE"
    ctx.lifespan_context = {"backend": backend}
    assert await tools["kb_preflight"](project_ref="p", ctx=ctx) == "CTX"
    ctx.lifespan_context = {
        "backend": backend,
        "version_skew_note": "SKEW NOTE",
        "deprecation_note": "DEP",
    }
    assert await tools["kb_preflight"](project_ref="p", ctx=ctx) == "CTX\n\nSKEW NOTE\n\nDEP"
    ctx.lifespan_context = {"backend": backend, "deprecation_note": "DEP"}
    assert await tools["kb_preflight"](project_ref="p", ctx=ctx) == "CTX\n\nDEP"
