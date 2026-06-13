"""Tests for the kb_store MCP tool logic.

Tool-routed tests drive the registered ``kb_store`` callable through an
``HttpBackend`` backed by an ``httpx.MockTransport`` (HTTP-contract style):
the backend is injected under ``lifespan["backend"]`` so
``backend_from_lifespan`` returns it and ``backend.is_remote`` is True. The
handler returns canned JSON matching the ``/api/kb/store`` and
``/api/kb/entries/{id}/deactivate`` response shapes. Assertions are made on
the tool's returned string rather than on real DB state.

Pure-function tests (``format_store_result``, ``_mental_map_has_pointer``,
``_validate_sensitivity``) and store-facade tests (deactivate semantics) do
not route through the tool backend and are exercised directly.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from personal_kb.backend.http import HttpBackend
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.tools.kb_store import (
    ORPHAN_MAP_ERROR,
    _mental_map_has_pointer,
    _validate_sensitivity,
    format_store_result,
    register_kb_store,
)

# ---------------------------------------------------------------------------
# HTTP-backend test harness
# ---------------------------------------------------------------------------


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


def _register_and_capture():
    """Register kb_store on a mock MCP and return the captured tool callable."""
    tools: dict[str, Any] = {}

    def capture_tool(**_kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp_mock = MagicMock()
    mcp_mock.tool = capture_tool
    register_kb_store(mcp_mock)
    return tools["kb_store"]


def _entry_json(
    *,
    entry_id: str = "kb-00001",
    short_title: str = "Test",
    long_title: str = "Test entry",
    knowledge_details: str = "Details",
    entry_type: EntryType = EntryType.FACTUAL_REFERENCE,
    project_ref: str | None = None,
    tags: list[str] | None = None,
    version: int = 1,
    is_active: bool = True,
) -> dict[str, Any]:
    """Build an <entry> JSON object as the service would return it."""
    entry = KnowledgeEntry(
        id=entry_id,
        short_title=short_title,
        long_title=long_title,
        knowledge_details=knowledge_details,
        entry_type=entry_type,
        project_ref=project_ref,
        tags=tags or [],
        confidence_level=0.9,
        version=version,
        is_active=is_active,
        has_embedding=True,
    )
    return entry.model_dump(mode="json")


def _store_handler(entry: dict[str, Any], action: str = "created"):
    """Return a MockTransport handler for the /api/kb/store route."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/store"
        body = json.loads(req.content)
        # Echo back 'updated' when the request carries an update_entry_id.
        resolved = "updated" if body.get("update_entry_id") else action
        return httpx.Response(200, json={"action": resolved, "entry": entry})

    return handler


def _never_called_handler():
    """Return a handler that fails if the backend is ever hit."""

    def handler(req: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError(f"backend should not be called, got {req.url}")

    return handler


# ---------------------------------------------------------------------------
# format_store_result — pure function over a constructed entry
# ---------------------------------------------------------------------------


def test_format_store_result_create():
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="my-project",
        tags=["tag1", "tag2"],
        confidence_level=0.9,
        version=1,
        is_active=True,
        has_embedding=True,
    )
    result = format_store_result(entry, is_update=False, include_backend_warning=False)
    assert "Created kb-00001" in result
    assert "my-project" in result
    assert "#tag1 #tag2" in result


def test_format_store_result_update():
    entry = KnowledgeEntry(
        id="kb-00001",
        short_title="Test",
        long_title="Test entry",
        knowledge_details="New details",
        entry_type=EntryType.DECISION,
        confidence_level=0.9,
        version=2,
        is_active=True,
        has_embedding=True,
    )
    result = format_store_result(entry, is_update=True, include_backend_warning=False)
    assert "Updated kb-00001 (v2)" in result


# --- Deactivate path via the registered tool (HTTP contract) ---


@pytest.mark.asyncio
async def test_deactivate_entry_via_tool():
    """deactivate_entry_id routes to backend.deactivate and reports the entry."""
    entry = _entry_json(
        entry_id="kb-00001",
        short_title="Wrong fact",
        long_title="An incorrect fact",
        is_active=False,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/entries/kb-00001/deactivate"
        return httpx.Response(200, json={"entry": entry})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(deactivate_entry_id="kb-00001", ctx=ctx)
    assert "Deactivated entry kb-00001" in result
    assert "Wrong fact" in result


@pytest.mark.asyncio
async def test_deactivate_entry_with_reason():
    """A change_reason is appended to the deactivation message."""
    entry = _entry_json(entry_id="kb-00002", short_title="Old fact", is_active=False)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"entry": entry})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(deactivate_entry_id="kb-00002", change_reason="obsolete", ctx=ctx)
    assert "Deactivated entry kb-00002" in result
    assert "(obsolete)" in result


@pytest.mark.asyncio
async def test_deactivate_nonexistent_entry_maps_error():
    """A 404 from the deactivate route is mapped to an error string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "entry kb-99999 not found"})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(deactivate_entry_id="kb-99999", ctx=ctx)
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_deactivate_already_inactive_maps_error():
    """A 409 (already inactive) is mapped to an error string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "entry kb-00001 already inactive"})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(deactivate_entry_id="kb-00001", ctx=ctx)
    assert "Error" in result
    assert "already inactive" in result


# --- Sensitivity validation ---


def test_validate_sensitivity_valid_values():
    """Valid sensitivity values should pass."""
    assert _validate_sensitivity(None) is None
    assert _validate_sensitivity("internal") is None
    assert _validate_sensitivity("restricted") is None
    assert _validate_sensitivity("public") is None


def test_validate_sensitivity_invalid_value():
    """Invalid sensitivity should return an error string."""
    result = _validate_sensitivity("banana")
    assert result is not None
    assert "Invalid sensitivity" in result
    assert '"banana"' in result
    assert "internal" in result
    assert "public" in result
    assert "restricted" in result


# --- mental_map pointer-detection helper ---


def test_pointer_helper_body_reference():
    assert _mental_map_has_pointer("see kb-00050 for details", None) is True


def test_pointer_helper_supersedes_hint():
    assert _mental_map_has_pointer("no refs", {"supersedes": "kb-00042"}) is True
    # invalid supersedes target is not a pointer
    assert _mental_map_has_pointer("no refs", {"supersedes": "not-an-id"}) is False


def test_pointer_helper_superseded_by():
    assert _mental_map_has_pointer("no refs", None, superseded_by="kb-00099") is True
    assert _mental_map_has_pointer("no refs", None, superseded_by="") is False


def test_pointer_helper_related_entities_dict():
    assert _mental_map_has_pointer("no refs", {"related_entities": [{"id": "kb-00042"}]}) is True
    assert (
        _mental_map_has_pointer("no refs", {"related_entities": [{"target": "kb-00043"}]}) is True
    )


def test_pointer_helper_related_entities_bare_string():
    """A bare-string related_entity produces a real related_to edge → it is a pointer."""
    assert _mental_map_has_pointer("no refs", {"related_entities": ["kb-00044"]}) is True


def test_pointer_helper_non_pointer_hints_dont_count():
    hints = {
        "has_tag": ["python"],
        "in_project": ["kb"],
        "mentions_person": ["jason"],
        "uses_tool": ["sqlite"],
    }
    assert _mental_map_has_pointer("no refs at all", hints) is False


def test_pointer_helper_zero_pointers():
    assert _mental_map_has_pointer("just plain prose, no ids", None) is False
    assert _mental_map_has_pointer("just plain prose", {}) is False


# --- mental_map create path via the registered tool (HTTP contract) ---


@pytest.mark.asyncio
async def test_mental_map_with_body_reference_succeeds():
    """(a) mental_map with in-body kb-XXXXX ref stores and shows un-decayed confidence."""
    entry = _entry_json(
        short_title="Auth map",
        long_title="Auth subsystem orientation",
        knowledge_details="Start at kb-00050 then follow the edges.",
        entry_type=EntryType.MENTAL_MAP,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    result = await kb_store(
        short_title="Auth map",
        long_title="Auth subsystem orientation",
        knowledge_details="Start at kb-00050 then follow the edges.",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Created" in result
    # un-decayed: effective == base (~90%), and no staleness badge
    assert "(90%)" in result
    assert "[STALE]" not in result


@pytest.mark.asyncio
async def test_mental_map_with_related_entity_hint_succeeds():
    """(b) mental_map with a related_entities id hint and no in-body ref stores."""
    entry = _entry_json(
        short_title="Map",
        long_title="A map",
        knowledge_details="No inline references here.",
        entry_type=EntryType.MENTAL_MAP,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    result = await kb_store(
        short_title="Map",
        long_title="A map",
        knowledge_details="No inline references here.",
        entry_type=EntryType.MENTAL_MAP,
        hints={"related_entities": [{"id": "kb-00042"}]},
        ctx=ctx,
    )
    assert "Created" in result


@pytest.mark.asyncio
async def test_mental_map_zero_pointers_rejected():
    """(c) zero-pointer mental_map returns the orphan error before any backend call."""
    kb_store = _register_and_capture()
    # The backend must never be reached — the orphan check runs client-side.
    ctx = _make_ctx(_never_called_handler())
    result = await kb_store(
        short_title="Orphan",
        long_title="Orphan map",
        knowledge_details="Just prose, no pointers at all.",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert result == ORPHAN_MAP_ERROR


@pytest.mark.asyncio
async def test_non_mental_map_zero_pointers_still_succeeds():
    """(d) regression guard: a factual_reference with zero pointers still stores."""
    entry = _entry_json(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="Plain fact with no pointers.",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    result = await kb_store(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="Plain fact with no pointers.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        ctx=ctx,
    )
    assert "Created" in result


# --- advisory mental_map lint at the store call site ---


@pytest.mark.asyncio
async def test_mental_map_create_with_value_surfaces_advisory():
    """A mental_map create with a config value in the body returns Created + advisory."""
    body = "orients kb-00050; the explorer runs on port 8767"
    entry = _entry_json(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details=body,
        entry_type=EntryType.MENTAL_MAP,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    result = await kb_store(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details=body,
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Created kb-" in result
    assert "Map lint (advisory):" in result


@pytest.mark.asyncio
async def test_mental_map_http_mode_suppresses_backend_warning():
    """In HTTP mode the SQLite-fallback warning is suppressed even if configured."""
    body = "orients kb-00050; the explorer runs on port 8767"
    entry = _entry_json(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details=body,
        entry_type=EntryType.MENTAL_MAP,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    with patch(
        "personal_kb.config.get_backend_warning",
        return_value="Backend fallback: using degraded mode",
    ):
        result = await kb_store(
            short_title="Net map",
            long_title="Network orientation",
            knowledge_details=body,
            entry_type=EntryType.MENTAL_MAP,
            ctx=ctx,
        )
    # HTTP mode passes include_backend_warning=False — the fallback warning
    # never appears, but the advisory lint still does.
    assert "Backend fallback" not in result
    assert "Map lint (advisory):" in result


@pytest.mark.asyncio
async def test_mental_map_update_with_body_surfaces_advisory():
    """Update with a new body and NO entry_type param still lints (gate reads persisted type)."""
    clean_entry = _entry_json(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; clean orientation prose",
        entry_type=EntryType.MENTAL_MAP,
    )
    dirty_entry = _entry_json(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        entry_type=EntryType.MENTAL_MAP,
        version=2,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if body.get("update_entry_id"):
            return httpx.Response(200, json={"action": "updated", "entry": dirty_entry})
        return httpx.Response(200, json={"action": "created", "entry": clean_entry})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    created = await kb_store(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; clean orientation prose",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Map lint (advisory):" not in created  # clean body, no advisory
    result = await kb_store(
        update_entry_id="kb-00001",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        ctx=ctx,  # NOTE: no entry_type argument
    )
    assert "Updated kb-00001" in result
    assert "Map lint (advisory):" in result


@pytest.mark.asyncio
async def test_mental_map_metadata_only_update_skips_lint():
    """A metadata-only update (no knowledge_details) returns Updated with NO advisory."""
    entry = _entry_json(
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; the explorer runs on port 8767",
        entry_type=EntryType.MENTAL_MAP,
        version=2,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"action": "updated", "entry": entry})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(
        update_entry_id="kb-00001",
        tags=["new-tag"],
        ctx=ctx,  # no knowledge_details — metadata-only
    )
    assert "Updated kb-00001" in result
    assert "Map lint (advisory):" not in result


@pytest.mark.asyncio
async def test_non_map_create_is_not_linted():
    """A factual_reference with a numeral body is never linted (call-site gating)."""
    entry = _entry_json(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="the explorer runs on port 8767",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_store_handler(entry))
    result = await kb_store(
        short_title="Fact",
        long_title="A fact",
        knowledge_details="the explorer runs on port 8767",
        entry_type=EntryType.FACTUAL_REFERENCE,
        ctx=ctx,
    )
    assert "Created kb-" in result
    assert "Map lint (advisory):" not in result
