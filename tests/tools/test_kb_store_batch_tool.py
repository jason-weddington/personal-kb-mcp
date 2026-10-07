"""Tests for the kb_store_batch MCP tool.

These tests drive the tool through an :class:`HttpBackend` backed by an
``httpx.MockTransport`` (the in-process LocalBackend has been removed).  The
lifespan dict carries a ``"backend"`` key so ``backend_from_lifespan`` returns
the HTTP backend directly (``is_remote=True``).

Route: POST /api/kb/store_batch -> ``{"created": [<entry>, ...]}``.  The
HttpBackend returns ``(created, [])`` — the failed list is always empty in HTTP
mode, so partial failure is simulated by returning *fewer* created entries than
were submitted.  The tool renders the aggregate failure count from
``len(created) < len(submitted)``.

Pre-backend validation tests (empty batch, cap, missing fields, bad
sensitivity) still inject a backend so ``backend_from_lifespan`` succeeds, even
though the MockTransport handler is never hit.
"""

from typing import Any

import httpx
import pytest
from kb_core.models.entry import EntryType, KnowledgeEntry

from personal_kb.backend.http import HttpBackend
from personal_kb.tools.kb_store_batch import batch_store_entries


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


def _lifespan(handler) -> dict[str, Any]:
    """Return a lifespan dict with an HttpBackend injected."""
    return {"backend": _make_http_backend(handler)}


def _entry_dict(**kwargs):
    """Create a minimal valid entry dict with overrides."""
    defaults = {
        "short_title": "Test",
        "long_title": "Test entry",
        "knowledge_details": "Some details",
    }
    defaults.update(kwargs)
    return defaults


def _entry_json(
    entry_id: str,
    short_title: str = "Test",
    knowledge_details: str = "Some details",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
) -> dict[str, Any]:
    """Build a fully-hydrated entry JSON object as the service would return it."""
    entry = KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=f"{short_title} entry",
        knowledge_details=knowledge_details,
        entry_type=entry_type,
    )
    return entry.model_dump(mode="json")


def _created_handler(*entries_json: dict[str, Any]):
    """Return a handler that responds with the given created entries."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"created": list(entries_json)})

    return handler


def _unhit_handler():
    """Return a handler that records if it was called (it should not be)."""
    called: list[bool] = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={"created": []})

    return handler, called


@pytest.mark.asyncio
async def test_batch_store_three_entries():
    """Basic batch creation of 3 entries."""
    handler = _created_handler(
        _entry_json("kb-00001", "First"),
        _entry_json("kb-00002", "Second"),
        _entry_json("kb-00003", "Third", entry_type=EntryType.DECISION),
    )
    ls = _lifespan(handler)

    entries = [
        _entry_dict(short_title="First", long_title="First entry", knowledge_details="D1"),
        _entry_dict(short_title="Second", long_title="Second entry", knowledge_details="D2"),
        _entry_dict(
            short_title="Third",
            long_title="Third entry",
            knowledge_details="D3",
            entry_type="decision",
        ),
    ]

    result = await batch_store_entries(entries, ls)
    assert "3 entries created" in result
    assert "kb-00001" in result
    assert "kb-00002" in result
    assert "kb-00003" in result
    assert "3 result(s)" in result


@pytest.mark.asyncio
async def test_batch_mental_map_advisory_attributed():
    """A batch mental_map with a config value reports created and shows the advisory;
    a sibling clean non-map entry's block does NOT contain the advisory."""
    handler = _created_handler(
        _entry_json(
            "kb-00001",
            "Net map",
            knowledge_details="orients kb-00050; the explorer runs on port 8767",
            entry_type=EntryType.MENTAL_MAP,
        ),
        _entry_json(
            "kb-00002",
            "Plain fact",
            knowledge_details="some clean prose with no values",
            entry_type=EntryType.FACTUAL_REFERENCE,
        ),
    )
    ls = _lifespan(handler)

    entries = [
        _entry_dict(
            short_title="Net map",
            long_title="Network orientation",
            knowledge_details="orients kb-00050; the explorer runs on port 8767",
            entry_type="mental_map",
        ),
        _entry_dict(
            short_title="Plain fact",
            long_title="A plain fact",
            knowledge_details="some clean prose with no values",
            entry_type="factual_reference",
        ),
    ]

    result = await batch_store_entries(entries, ls)
    assert "2 entries created" in result
    # Advisory appears exactly once, attributed to the offending map entry.
    assert result.count("Map lint (advisory):") == 1

    # The advisory is in the map's block (kb-00001), not the sibling's (kb-00002).
    map_idx = result.index("kb-00001")
    sibling_idx = result.index("kb-00002")
    advisory_idx = result.index("Map lint (advisory):")
    assert map_idx < advisory_idx < sibling_idx


@pytest.mark.asyncio
async def test_batch_store_cap_at_10():
    """Exceeding 10 entries returns an error before any backend call."""
    handler, called = _unhit_handler()
    ls = _lifespan(handler)

    entries = [_entry_dict(short_title=f"Entry {i}") for i in range(11)]
    result = await batch_store_entries(entries, ls)
    assert "Maximum 10" in result
    assert not called


@pytest.mark.asyncio
async def test_batch_store_empty():
    """Empty entries list returns an error before any backend call."""
    handler, called = _unhit_handler()
    ls = _lifespan(handler)

    result = await batch_store_entries([], ls)
    assert "empty" in result.lower()
    assert not called


@pytest.mark.asyncio
async def test_batch_store_validation_error():
    """Missing required fields returns an error before any backend call."""
    handler, called = _unhit_handler()
    ls = _lifespan(handler)

    entries = [{"short_title": "Missing fields"}]
    result = await batch_store_entries(entries, ls)
    assert "missing required fields" in result.lower()
    assert "knowledge_details" in result
    assert "long_title" in result
    assert not called


@pytest.mark.asyncio
async def test_batch_store_rejects_invalid_sensitivity():
    """Batch with invalid sensitivity should return an error before any backend call."""
    handler, called = _unhit_handler()
    ls = _lifespan(handler)

    entries = [
        _entry_dict(short_title="Good", sensitivity="internal"),
        _entry_dict(short_title="Bad", sensitivity="banana"),
    ]
    result = await batch_store_entries(entries, ls)
    assert "invalid sensitivity" in result.lower()
    assert '"banana"' in result
    assert "entry 1" in result

    # No backend call should have been made — validation fails first.
    assert not called


@pytest.mark.asyncio
async def test_batch_store_partial_failure():
    """If the service creates fewer entries than submitted, the failure count is reported.

    HTTP mode has no per-entry failure detail, so only the aggregate count is
    surfaced (no "Failed entries (retry these):" section).
    """
    # 3 submitted, only 2 returned as created -> 1 failed.
    handler = _created_handler(
        _entry_json("kb-00001", "First"),
        _entry_json("kb-00003", "Third"),
    )
    ls = _lifespan(handler)

    entries = [
        _entry_dict(short_title="First", knowledge_details="D1"),
        _entry_dict(short_title="Second", knowledge_details="D2"),
        _entry_dict(short_title="Third", knowledge_details="D3"),
    ]

    result = await batch_store_entries(entries, ls)

    assert "2 entries created" in result
    assert "1 failed" in result
    assert "kb-00001" in result
    assert "kb-00003" in result


@pytest.mark.asyncio
async def test_batch_store_all_fail():
    """If every entry fails client-side (bad TTL), return a clear error with details.

    In HTTP mode the only source of per-entry failure detail is client-side TTL
    pre-validation; those failures populate the "all failed" branch.
    """
    handler, called = _unhit_handler()
    ls = _lifespan(handler)

    entries = [
        _entry_dict(short_title="A", knowledge_details="D1", ttl="banana"),
        _entry_dict(short_title="B", knowledge_details="D2", ttl="banana"),
    ]

    result = await batch_store_entries(entries, ls)

    assert "all 2 entries failed" in result.lower()
    assert "Entry 0 (A)" in result
    assert "Entry 1 (B)" in result
    # The TTL ValueError message is surfaced as the failure detail.
    assert "Invalid TTL" in result
    # Backend receives an empty valid_entries list and short-circuits — handler
    # is never actually invoked over the transport.
    assert not called


async def test_batch_409_renders_as_error_string() -> None:
    """A near-duplicate 409 from the service renders as 'Error: ...' (not a raise)."""
    detail = {
        "error": "near_duplicate",
        "message": "entry 0: Cover EVERY listed id with one of: distinct_from=[<id>]",
        "candidates": [{"id": "kb-00001"}],
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": detail})

    result = await batch_store_entries([_entry_dict()], _lifespan(handler))
    assert result.startswith("Error: ")
    assert "kb-00001" in result
    assert "distinct_from" in result
