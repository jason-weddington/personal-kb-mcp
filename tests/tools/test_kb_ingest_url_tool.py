"""Tests for the kb_ingest_url MCP tool (HTTP-contract mode).

Each test injects an ``HttpBackend`` (backed by ``httpx.MockTransport``)
under the ``"backend"`` key of the lifespan context, which flips
``backend.is_remote`` to ``True``.  Canned JSON matching the
``POST /api/kb/ingest/url`` response shape (a ``FileResult`` dict) drives
the tool, and assertions are made on the returned string.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from personal_kb.backend.http import HttpBackend
from personal_kb.tools.kb_ingest_url import register_kb_ingest_url


def _make_http_backend(handler) -> HttpBackend:
    """Build an HttpBackend backed by a MockTransport sync handler."""
    transport = httpx.MockTransport(handler)
    backend = HttpBackend(base_url="http://kb.test", api_key="testkey")
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        headers={"Authorization": "Bearer testkey"},
        transport=transport,
    )
    return backend


def _make_ctx(handler) -> MagicMock:
    """Return a MagicMock Context with an HttpBackend lifespan."""
    ctx = MagicMock()
    ctx.lifespan_context = {"backend": _make_http_backend(handler)}
    return ctx


def _register() -> Any:
    """Register kb_ingest_url on a mock MCP and return the captured callable."""
    tools: dict[str, Any] = {}

    def capture(**_kw):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_ingest_url(mcp)
    return tools["kb_ingest_url"]


def _file_result_json(**overrides) -> dict[str, Any]:
    """Build a FileResult-shaped response dict with sensible defaults."""
    base = {
        "path": "https://example.com",
        "action": "ingested",
        "reason": None,
        "entry_count": 0,
        "entry_ids": [],
        "summary": None,
        "chunks_processed": 0,
        "chunks_skipped": 0,
        "chunks_flagged": 0,
    }
    base.update(overrides)
    return base


class TestKbIngestUrlTool:
    @pytest.mark.asyncio
    async def test_prefetched_content_ingests(self):
        """When content is provided, the backend ingests it and returns 'ingested'."""

        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return httpx.Response(
                200,
                json=_file_result_json(
                    path="https://intranet.example.com/deploy-guide",
                    action="ingested",
                    entry_count=1,
                    entry_ids=["kb-00001"],
                    summary="Summary of deployment guide.",
                ),
            )

        kb_ingest_url = _register()
        ctx = _make_ctx(handler)

        result = await kb_ingest_url(
            url="https://intranet.example.com/deploy-guide",
            content="# Deployment Guide\n\nDeploy with kubectl apply.",
            project_ref="infra",
            ctx=ctx,
        )
        assert "ingested" in result
        assert "1 entries" in result
        # The pre-fetched content was forwarded to the service.
        assert captured[0]["content"] == "# Deployment Guide\n\nDeploy with kubectl apply."
        assert captured[0]["project_ref"] == "infra"

    @pytest.mark.asyncio
    async def test_prefetched_content_dry_run(self):
        """Dry run with pre-fetched content surfaces the DRY RUN prefix."""

        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return httpx.Response(
                200,
                json=_file_result_json(
                    path="https://wiki.example.com/page",
                    action="ingested",
                    entry_count=1,
                    entry_ids=["kb-00002"],
                    summary="Wiki page summary.",
                ),
            )

        kb_ingest_url = _register()
        ctx = _make_ctx(handler)

        result = await kb_ingest_url(
            url="https://wiki.example.com/page",
            content="Some wiki content.",
            dry_run=True,
            ctx=ctx,
        )
        assert "DRY RUN" in result
        assert captured[0]["dry_run"] is True

    @pytest.mark.asyncio
    async def test_no_content_calls_ingest_url(self):
        """When content is None, the fetch path is taken (no 'content' in body)."""

        captured: list[dict] = []

        def handler(req: httpx.Request) -> httpx.Response:
            captured.append(json.loads(req.content))
            return httpx.Response(
                200,
                json=_file_result_json(
                    path="https://example.com/article",
                    action="ingested",
                    summary="Test",
                ),
            )

        kb_ingest_url = _register()
        ctx = _make_ctx(handler)

        await kb_ingest_url(
            url="https://example.com/article",
            ctx=ctx,
        )
        assert captured[0]["url"] == "https://example.com/article"
        # No pre-fetched content means the fetch path: 'content' is omitted.
        assert "content" not in captured[0]

    @pytest.mark.asyncio
    async def test_empty_url_errors_before_backend(self):
        """An empty URL is rejected client-side before any backend call."""

        called: list[bool] = []

        def handler(req: httpx.Request) -> httpx.Response:
            called.append(True)
            return httpx.Response(200, json=_file_result_json())

        kb_ingest_url = _register()
        ctx = _make_ctx(handler)

        result = await kb_ingest_url(url="", content="Some content", ctx=ctx)
        assert "Error" in result
        assert not called

    @pytest.mark.asyncio
    async def test_backend_error_mapped(self):
        """A non-2xx from the service is mapped to an error string."""

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "Invalid URL"})

        kb_ingest_url = _register()
        ctx = _make_ctx(handler)

        result = await kb_ingest_url(url="https://example.com", ctx=ctx)
        assert "Error" in result
