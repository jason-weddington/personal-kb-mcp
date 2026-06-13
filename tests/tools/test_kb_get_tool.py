"""Tests for the kb_get MCP tool.

The in-process local backend has been removed, so these tests drive the real
``kb_get`` tool against an :class:`HttpBackend` backed by ``httpx.MockTransport``.
Each test feeds canned ``/api/kb/get`` JSON (matching the service route shape:
``{"results": [{"id", "found", "entry", "pointer_rot"}]}``) and asserts on the
tool's rendered string output. Real-DB persistence (``last_accessed`` touch,
pointer-rot *computation*) is covered at the service + kb_core layers; here we
verify the tool's rendering of the HTTP contract.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from personal_kb.backend.http import HttpBackend
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.formatters import format_entry_full, format_result_list
from personal_kb.tools.kb_get import register_kb_get


def _make_entry(
    entry_id: str,
    short_title: str = "Title",
    long_title: str = "Long title",
    knowledge_details: str = "Details",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    tags: list[str] | None = None,
    project_ref: str | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        tags=tags or [],
        project_ref=project_ref,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _make_http_backend(handler) -> HttpBackend:
    transport = httpx.MockTransport(handler)
    backend = HttpBackend(base_url="http://kb.test", api_key="testkey")
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        headers={"Authorization": "Bearer testkey"},
        transport=transport,
    )
    return backend


def _register() -> Any:
    """Register kb_get on a mock MCP and return the captured tool function."""
    tools: dict[str, Any] = {}

    def capture(**_kw):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_kb_get(mcp)
    return next(iter(tools.values()))


def _ctx_for(results_map: dict[str, dict[str, Any]]) -> MagicMock:
    """Build a ctx whose /api/kb/get returns canned results per requested id.

    ``results_map`` maps an id → a partial result dict (without ``id``); a
    requested id absent from the map is rendered as ``found=False``.
    """

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/get"
        body = json.loads(req.content)
        results = []
        for eid in body.get("ids", []):
            spec = results_map.get(eid)
            if spec is None:
                results.append({"id": eid, "found": False, "entry": None})
            else:
                results.append({"id": eid, **spec})
        return httpx.Response(200, json={"results": results})

    ctx = MagicMock()
    ctx.lifespan_context = {"backend": _make_http_backend(handler)}
    return ctx


def _found(
    entry: KnowledgeEntry, pointer_rot: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "found": True,
        "entry": entry.model_dump(mode="json"),
        "pointer_rot": pointer_rot or [],
    }


@pytest.mark.asyncio
async def test_get_single_entry():
    """Retrieve a single entry by ID."""
    entry = _make_entry(
        "kb-00001",
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Full details here",
        tags=["python"],
        project_ref="my-proj",
    )
    kb_get = _register()
    ctx = _ctx_for({entry.id: _found(entry)})

    result = await kb_get(entry_id=entry.id, ctx=ctx)
    assert entry.id in result
    assert "Full details here" in result
    assert "#python" in result
    assert "my-proj" in result


@pytest.mark.asyncio
async def test_get_multiple_entries():
    """Retrieve multiple entries at once."""
    e1 = _make_entry(
        "kb-00001", short_title="First", long_title="First entry", knowledge_details="First details"
    )
    e2 = _make_entry(
        "kb-00002",
        short_title="Second",
        long_title="Second entry",
        knowledge_details="Second details",
        entry_type=EntryType.DECISION,
    )
    kb_get = _register()
    ctx = _ctx_for({e1.id: _found(e1), e2.id: _found(e2)})

    result = await kb_get(entry_id=[e1.id, e2.id], ctx=ctx)
    assert e1.id in result
    assert e2.id in result
    assert "First details" in result
    assert "Second details" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_missing_entry():
    """Missing IDs show 'not found'."""
    kb_get = _register()
    ctx = _ctx_for({})
    result = await kb_get(entry_id="kb-99999", ctx=ctx)
    assert "kb-99999" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_get_mixed_found_and_missing():
    """Mix of found and missing entries."""
    entry = _make_entry(
        "kb-00001",
        short_title="Exists",
        long_title="Existing entry",
        knowledge_details="Real content",
    )
    kb_get = _register()
    ctx = _ctx_for({entry.id: _found(entry)})

    result = await kb_get(entry_id=[entry.id, "kb-99999"], ctx=ctx)
    assert entry.id in result
    assert "Real content" in result
    assert "kb-99999" in result
    assert "not found" in result
    assert "2 result(s)" in result


@pytest.mark.asyncio
async def test_get_inactive_entry_skipped():
    """Inactive entries are treated as not found by the service."""
    kb_get = _register()
    # Service reports an inactive entry as found=False
    ctx = _ctx_for({"kb-00001": {"found": False, "entry": None}})
    result = await kb_get(entry_id="kb-00001", ctx=ctx)
    assert "not found" in result


@pytest.mark.asyncio
async def test_get_cap_at_20():
    """Exceeding 20 IDs returns an error (before any backend call)."""
    kb_get = _register()
    ctx = _ctx_for({})
    ids = [f"kb-{i:05d}" for i in range(1, 22)]
    result = await kb_get(entry_id=ids, ctx=ctx)
    assert "Maximum 20" in result


# --- Pointer-rot rendering tests for mental_map (§7.4) -----------------------
# The service computes pointer_rot; the tool renders it. These tests feed canned
# pointer_rot pairs and assert the rendered block.


@pytest.mark.asyncio
async def test_mental_map_renders_superseded_pointer():
    """(a) Map with one superseded pointer target renders 'superseded by' line."""
    mmap = _make_entry(
        "kb-00010", short_title="Map", long_title="Mental map", entry_type=EntryType.MENTAL_MAP
    )
    kb_get = _register()
    ctx = _ctx_for(
        {mmap.id: _found(mmap, [{"target_id": "kb-00001", "superseded_by": "kb-00002"}])}
    )

    result = await kb_get(entry_id=mmap.id, ctx=ctx)
    assert "  Pointer-rot:" in result
    assert "    [kb-00001] superseded by [kb-00002]" in result
    assert "deactivated" not in result


@pytest.mark.asyncio
async def test_mental_map_renders_deactivated_pointer():
    """(b) Map with one deactivated (not-superseded) target renders 'deactivated'."""
    mmap = _make_entry(
        "kb-00010", short_title="Map", long_title="Mental map", entry_type=EntryType.MENTAL_MAP
    )
    kb_get = _register()
    ctx = _ctx_for({mmap.id: _found(mmap, [{"target_id": "kb-00001", "superseded_by": None}])})

    result = await kb_get(entry_id=mmap.id, ctx=ctx)
    assert "  Pointer-rot:" in result
    assert "    [kb-00001] deactivated" in result
    assert "superseded by" not in result


@pytest.mark.asyncio
async def test_mental_map_all_healthy_targets_silent():
    """(c) Map with all-healthy targets renders NO rot note."""
    mmap = _make_entry(
        "kb-00010", short_title="Map", long_title="Mental map", entry_type=EntryType.MENTAL_MAP
    )
    kb_get = _register()
    ctx = _ctx_for({mmap.id: _found(mmap, [])})

    result = await kb_get(entry_id=mmap.id, ctx=ctx)
    assert "Pointer-rot" not in result
    assert "superseded by" not in result
    assert "deactivated" not in result


@pytest.mark.asyncio
async def test_non_map_pointing_at_superseded_target_silent():
    """(d) Non-mental_map entry renders NO rot note; byte-identical to baseline.

    The service returns no pointer_rot for non-map entries, so the rendered
    output equals format_entry_full + the format_result_list wrapper.
    """
    non_map = _make_entry(
        "kb-00020",
        short_title="A decision",
        long_title="A decision pointing somewhere",
        knowledge_details="Points at the old fact.",
        entry_type=EntryType.DECISION,
    )
    kb_get = _register()
    ctx = _ctx_for({non_map.id: _found(non_map, [])})

    result = await kb_get(entry_id=non_map.id, ctx=ctx)
    assert "Pointer-rot" not in result
    assert "superseded by" not in result
    assert "deactivated" not in result

    expected = format_result_list([format_entry_full(non_map)])
    assert result == expected


@pytest.mark.asyncio
async def test_mental_map_superseded_and_deactivated_renders_superseded_form():
    """(e) Target both superseded AND deactivated → SUPERSEDED form (precedence).

    The service applies precedence and reports the superseded_by value, so the
    tool renders the superseded form only.
    """
    mmap = _make_entry(
        "kb-00010", short_title="Map", long_title="Mental map", entry_type=EntryType.MENTAL_MAP
    )
    kb_get = _register()
    ctx = _ctx_for(
        {mmap.id: _found(mmap, [{"target_id": "kb-00001", "superseded_by": "kb-00002"}])}
    )

    result = await kb_get(entry_id=mmap.id, ctx=ctx)
    assert "  Pointer-rot:" in result
    assert "    [kb-00001] superseded by [kb-00002]" in result
    assert "    [kb-00001] deactivated" not in result
