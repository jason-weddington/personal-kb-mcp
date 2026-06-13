"""Tests for the kb_ingest MCP tool (HTTP-contract mode).

The in-process local backend has been removed, so every test drives the
tool through an :class:`HttpBackend` backed by an ``httpx.MockTransport``.
The tool's ``is_remote`` branch (POST /api/kb/ingest/file via multipart) is
exercised with canned ``FileResult`` JSON, and assertions are made on the
tool's returned STRING.
"""

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from personal_kb.backend.http import HttpBackend
from personal_kb.tools.kb_ingest import register_kb_ingest


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
    """Return a MagicMock Context whose lifespan injects an HttpBackend."""
    ctx = MagicMock()
    ctx.lifespan_context = {"backend": _make_http_backend(handler)}
    return ctx


def _register_and_capture(mcp_mock):
    """Register kb_ingest on a mock MCP and return the captured tools dict."""
    tools: dict[str, Any] = {}

    def capture_tool(**_kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp_mock.tool = capture_tool
    register_kb_ingest(mcp_mock)
    return tools


def _file_result(
    *,
    path: str,
    action: str = "ingested",
    reason: str | None = None,
    entry_count: int = 0,
    entry_ids: list[str] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """Build a FileResult-shaped response dict for the ingest/file route."""
    return {
        "path": path,
        "action": action,
        "reason": reason,
        "entry_count": entry_count,
        "entry_ids": entry_ids or [],
        "summary": summary,
        "chunks_processed": 0,
        "chunks_skipped": 0,
        "chunks_flagged": 0,
    }


class TestKbIngestTool:
    async def test_error_on_empty_path(self):
        # Empty path is rejected client-side before the backend is even resolved.
        called = []

        def handler(req: httpx.Request) -> httpx.Response:
            called.append(True)
            return httpx.Response(200, json={})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path="", ctx=ctx)
        assert "Error" in result
        assert "path is required" in result
        assert not called  # No HTTP call made

    async def test_error_on_nonexistent_path(self, tmp_path):
        # Path-existence is validated client-side before any backend call.
        called = []

        def handler(req: httpx.Request) -> httpx.Response:
            called.append(True)
            return httpx.Response(200, json={})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](
            path=str(tmp_path / "nope" / "file.md"),
            ctx=ctx,
        )
        assert "does not exist" in result
        assert not called

    async def test_ingest_single_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# My Notes\n\nSome knowledge here.")

        def handler(req: httpx.Request) -> httpx.Response:
            assert req.url.path == "/api/kb/ingest/file"
            return httpx.Response(
                200,
                json=_file_result(
                    path=str(f),
                    action="ingested",
                    entry_count=1,
                    entry_ids=["kb-00001"],
                    summary="Summary of notes.",
                ),
            )

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](
            path=str(f),
            project_ref="test-project",
            ctx=ctx,
        )
        assert "ingested" in result
        assert "1 entries" in result

    async def test_dry_run_single_file(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Dry run test")

        captured: list[dict[str, Any]] = []

        def handler(req: httpx.Request) -> httpx.Response:
            # multipart form carries dry_run=true
            captured.append({"content": req.content})
            return httpx.Response(
                200,
                json=_file_result(
                    path=str(f),
                    action="dry_run",
                    entry_count=1,
                    summary="Dry run summary.",
                ),
            )

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](
            path=str(f),
            dry_run=True,
            ctx=ctx,
        )
        assert "DRY RUN" in result
        assert "Summary: Dry run summary." in result
        assert b"true" in captured[0]["content"]  # dry_run flag forwarded

    async def test_directory_rejected_in_http_mode(self, tmp_path):
        # Directory ingest is not supported in HTTP mode.
        (tmp_path / "a.md").write_text("# File A")
        called = []

        def handler(req: httpx.Request) -> httpx.Response:
            called.append(True)
            return httpx.Response(200, json={})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path=str(tmp_path), ctx=ctx)
        assert "Error" in result
        assert "Directory ingest is not supported in HTTP mode" in result
        assert not called

    async def test_glob_pattern_matches_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create .md and .txt files — glob *.md should only match .md
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "data.txt").write_text("plain text")

        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req.url.path)
            return httpx.Response(
                200,
                json=_file_result(path="ignored", action="ingested", entry_count=1),
            )

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path="*.md", ctx=ctx)
        assert "Ingestion complete" in result
        assert "2 ingested" in result
        # Only the two .md files were uploaded.
        assert len(seen) == 2

    async def test_glob_no_matches(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        called = []

        def handler(req: httpx.Request) -> httpx.Response:
            called.append(True)
            return httpx.Response(200, json={})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path="*.nonexistent", ctx=ctx)
        assert "No files matched pattern" in result
        assert not called

    async def test_glob_recursive_pattern(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        # Create nested structure
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("# Top")
        (sub / "nested.md").write_text("# Nested")

        seen: list[str] = []

        def handler(req: httpx.Request) -> httpx.Response:
            seen.append(req.url.path)
            return httpx.Response(
                200,
                json=_file_result(path="ignored", action="ingested", entry_count=1),
            )

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path="**/*.md", ctx=ctx)
        assert "Ingestion complete" in result
        assert "2 ingested" in result
        assert len(seen) == 2

    async def test_single_file_backend_error_mapped(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Notes")

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"detail": "Unauthorized"})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path=str(f), ctx=ctx)
        assert "Error" in result

    async def test_glob_backend_error_recorded(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "notes.md").write_text("# Notes")

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(403, json={"detail": "Admin only"})

        tools = _register_and_capture(MagicMock())
        ctx = _make_ctx(handler)

        result = await tools["kb_ingest"](path="*.md", ctx=ctx)
        # Per-file BackendHttpError is mapped into an error FileResult/tally.
        assert "Ingestion complete" in result
        assert "1 errors" in result


# A standalone async sanity test (matches reference file conventions).
@pytest.mark.asyncio
async def test_single_file_skipped_action(tmp_path):
    """A 'skipped' FileResult is surfaced in the tool's string output."""
    f = tmp_path / "notes.md"
    f.write_text("# Notes")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_file_result(
                path=str(f),
                action="skipped",
                reason="unsupported type",
            ),
        )

    tools = _register_and_capture(MagicMock())
    ctx = _make_ctx(handler)

    result = await tools["kb_ingest"](path=str(f), ctx=ctx)
    assert "skipped" in result
    assert "unsupported type" in result
