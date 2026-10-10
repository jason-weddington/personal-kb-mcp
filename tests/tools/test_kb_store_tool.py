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
    _validate_distinct_from,
    _validate_hints_supersedes_conflict,
    _validate_sensitivity,
    _validate_supersedes,
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
    result = await kb_store(
        supersedes="none", deactivate_entry_id="kb-00001", change_reason="test", ctx=ctx
    )
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
    result = await kb_store(
        supersedes="none", deactivate_entry_id="kb-00002", change_reason="obsolete", ctx=ctx
    )
    assert "Deactivated entry kb-00002" in result
    assert "(obsolete)" in result


@pytest.mark.asyncio
async def test_deactivate_nonexistent_entry_maps_error():
    """A 404 from the deactivate route is mapped to an error string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "entry kb-99999 not found"})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(
        supersedes="none", deactivate_entry_id="kb-99999", change_reason="test", ctx=ctx
    )
    assert "Error" in result
    assert "not found" in result


@pytest.mark.asyncio
async def test_deactivate_already_inactive_maps_error():
    """A 409 (already inactive) is mapped to an error string."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"detail": "entry kb-00001 already inactive"})

    kb_store = _register_and_capture()
    ctx = _make_ctx(handler)
    result = await kb_store(
        supersedes="none", deactivate_entry_id="kb-00001", change_reason="test", ctx=ctx
    )
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
    assert _mental_map_has_pointer("no refs", {"supersedes": "kb-00042"}) is False
    # invalid supersedes target is not a pointer
    assert _mental_map_has_pointer("no refs", {"supersedes": "not-an-id"}) is False


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
        supersedes="none",
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
        supersedes="none",
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
        supersedes="none",
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
        supersedes="none",
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
        supersedes="none",
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
            supersedes="none",
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
        supersedes="none",
        short_title="Net map",
        long_title="Network orientation",
        knowledge_details="orients kb-00050; clean orientation prose",
        entry_type=EntryType.MENTAL_MAP,
        ctx=ctx,
    )
    assert "Map lint (advisory):" not in created  # clean body, no advisory
    result = await kb_store(
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="test",
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
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="test",
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
        supersedes="none",
        short_title="Fact",
        long_title="A fact",
        knowledge_details="the explorer runs on port 8767",
        entry_type=EntryType.FACTUAL_REFERENCE,
        ctx=ctx,
    )
    assert "Created kb-" in result
    assert "Map lint (advisory):" not in result


# ---------------------------------------------------------------------------
# Supersession client-side validation + forwarding
# ---------------------------------------------------------------------------

_CR_ERR = (
    "Error: change_reason is required when updating or deactivating an entry: "
    "say what changed and why."
)


def test_validate_supersedes_branches():
    assert _validate_supersedes("none") is None
    assert _validate_supersedes(["kb-00001", "kb-00002"]) is None
    assert _validate_supersedes([]) == (
        'Error: supersedes=[] is ambiguous; pass "none" when this entry replaces nothing.'
    )
    for bad in (None, "kb-00001", "None", ["nope"], ["kb-00001", 3]):
        assert _validate_supersedes(bad) == (
            'Error: supersedes must be a list of kb-XXXXX ids or the literal "none" (got '
            + repr(bad)
            + ")."
        )


def test_validate_distinct_from_branches():
    assert _validate_distinct_from(None) is None
    assert _validate_distinct_from([]) is None
    assert _validate_distinct_from(["kb-00001"]) is None
    for bad in ("kb-00001", ["x"], 3):
        assert _validate_distinct_from(bad) == (
            "Error: distinct_from must be a list of kb-XXXXX ids (got " + repr(bad) + ")."
        )


def test_validate_hints_conflict_branches():
    err = 'Error: supersedes="none" conflicts with hints.supersedes; list the ids in supersedes.'
    assert _validate_hints_supersedes_conflict("none", {"supersedes": "kb-00001"}) == err
    assert _validate_hints_supersedes_conflict("none", {"supersedes": ["kb-00001"]}) == err
    assert _validate_hints_supersedes_conflict("none", {"supersedes": ""}) is None
    assert _validate_hints_supersedes_conflict("none", None) is None
    assert _validate_hints_supersedes_conflict(["kb-00002"], {"supersedes": "kb-1"}) is None


def _recording_handler(response: httpx.Response, seen: list[httpx.Request]):
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return response

    return handler


@pytest.mark.asyncio
async def test_deactivate_requires_change_reason():
    seen: list[httpx.Request] = []
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(httpx.Response(500), seen))
    result = await kb_store(supersedes="none", deactivate_entry_id="kb-00001", ctx=ctx)
    assert result == _CR_ERR
    assert seen == []


@pytest.mark.asyncio
async def test_deactivate_with_superseded_by_forwards_and_renders():
    seen: list[httpx.Request] = []
    entry = _entry_json(entry_id="kb-00001", short_title="Old", is_active=False)
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(httpx.Response(200, json={"entry": entry}), seen))
    result = await kb_store(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="r",
        superseded_by="kb-00009",
        ctx=ctx,
    )
    body = json.loads(seen[0].content)
    assert body == {"change_reason": "r", "superseded_by": "kb-00009"}
    assert result == "Deactivated entry kb-00001: Old (r); superseded by kb-00009"


@pytest.mark.asyncio
async def test_deactivate_rejects_supersedes_list_and_bad_superseded_by_and_distinct():
    seen: list[httpx.Request] = []
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(httpx.Response(500), seen))
    r = await kb_store(
        supersedes=["kb-00002"], deactivate_entry_id="kb-00001", change_reason="r", ctx=ctx
    )
    assert r.startswith("Error: supersedes does not apply to deactivate_entry_id")
    r = await kb_store(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="r",
        superseded_by="bad",
        ctx=ctx,
    )
    assert r == "Error: superseded_by must be a kb-XXXXX id (got 'bad')."
    r = await kb_store(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="r",
        distinct_from=["kb-00003"],
        ctx=ctx,
    )
    assert r == "Error: distinct_from applies to create only."
    assert seen == []


@pytest.mark.asyncio
async def test_deactivate_422_surfaced_verbatim():
    kb_store = _register_and_capture()
    ctx = _make_ctx(
        lambda req: httpx.Response(422, json={"detail": "superseded_by kb-00009 not found"})
    )
    r = await kb_store(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="r",
        superseded_by="kb-00009",
        ctx=ctx,
    )
    assert r == "Error: KB service returned 422: superseded_by kb-00009 not found"


@pytest.mark.asyncio
async def test_update_requires_change_reason_and_validates():
    seen: list[httpx.Request] = []
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(httpx.Response(500), seen))
    assert await kb_store(supersedes="none", update_entry_id="kb-00001", ctx=ctx) == _CR_ERR
    r = await kb_store(supersedes=[], update_entry_id="kb-00001", change_reason="r", ctx=ctx)
    assert "ambiguous" in r
    r = await kb_store(
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="r",
        hints={"supersedes": "kb-00002"},
        ctx=ctx,
    )
    assert "conflicts with hints.supersedes" in r
    r = await kb_store(
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="r",
        superseded_by="kb-00002",
        ctx=ctx,
    )
    assert r == "Error: superseded_by applies to deactivate_entry_id only."
    r = await kb_store(
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="r",
        distinct_from=["kb-00002"],
        ctx=ctx,
    )
    assert r == "Error: distinct_from applies to create only."
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [["kb-00003"], "none"])
async def test_update_forwards_supersedes_verbatim(value):
    seen: list[httpx.Request] = []
    entry = _entry_json()
    resp = httpx.Response(
        200,
        json={
            "action": "updated",
            "entry": entry,
            "superseded_ids": ["kb-00003"] if value != "none" else [],
        },
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(resp, seen))
    result = await kb_store(
        supersedes=value, update_entry_id="kb-00001", change_reason="r", ctx=ctx
    )
    assert json.loads(seen[0].content)["supersedes"] == value
    assert ("Supersedes: kb-00003" in result) == (value != "none")


@pytest.mark.asyncio
async def test_create_forwards_and_renders_supersedes():
    seen: list[httpx.Request] = []
    resp = httpx.Response(
        200,
        json={"action": "created", "entry": _entry_json(), "superseded_ids": ["kb-00002"]},
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(resp, seen))
    result = await kb_store(
        supersedes=["kb-00002"],
        short_title="a",
        long_title="b",
        knowledge_details="c",
        distinct_from=["kb-00007"],
        ctx=ctx,
    )
    body = json.loads(seen[0].content)
    assert body["supersedes"] == ["kb-00002"]
    assert body["distinct_from"] == ["kb-00007"]
    assert "\nSupersedes: kb-00002" in result


@pytest.mark.asyncio
async def test_create_none_has_no_distinct_from_and_no_supersedes_line():
    seen: list[httpx.Request] = []
    for extra in ({}, {"superseded_ids": []}):
        resp = httpx.Response(200, json={"action": "created", "entry": _entry_json(), **extra})
        kb_store = _register_and_capture()
        ctx = _make_ctx(_recording_handler(resp, seen))
        result = await kb_store(
            supersedes="none", short_title="a", long_title="b", knowledge_details="c", ctx=ctx
        )
        assert "Supersedes:" not in result
    body = json.loads(seen[0].content)
    assert body["supersedes"] == "none"
    assert "distinct_from" not in body


@pytest.mark.asyncio
async def test_create_rejections_make_no_request():
    seen: list[httpx.Request] = []
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(httpx.Response(500), seen))
    base = {"short_title": "a", "long_title": "b", "knowledge_details": "c", "ctx": ctx}
    r = await kb_store(supersedes=[], **base)
    assert "ambiguous" in r
    r = await kb_store(supersedes="none", hints={"supersedes": "kb-00001"}, **base)
    assert "conflicts with hints.supersedes" in r
    r = await kb_store(supersedes="none", superseded_by="kb-00001", **base)
    assert r == "Error: superseded_by applies to deactivate_entry_id only."
    r = await kb_store(supersedes="none", distinct_from=["x"], **base)
    assert r.startswith("Error: distinct_from must be")
    assert seen == []


@pytest.mark.asyncio
async def test_skew_warning_and_client_telemetry(caplog):
    import logging

    caplog.set_level(logging.INFO)
    seen: list[httpx.Request] = []
    resp = httpx.Response(200, json={"action": "created", "entry": _entry_json()})
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(resp, seen))
    await kb_store(
        supersedes=["kb-00002"], short_title="a", long_title="b", knowledge_details="c", ctx=ctx
    )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("supersession-client mismatch" in r.getMessage() for r in warnings)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("outcome=accepted rule=none" in m for m in msgs)
    caplog.clear()
    await kb_store(supersedes=[], short_title="a", long_title="b", knowledge_details="c", ctx=ctx)
    assert any(
        "supersession-client op=store path=create outcome=rejected rule=empty_list"
        in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_store_409_and_422_surfaced_verbatim():
    kb_store = _register_and_capture()
    kw = {"supersedes": "none", "short_title": "a", "long_title": "b", "knowledge_details": "c"}
    ctx = _make_ctx(lambda req: httpx.Response(409, json={"detail": "dup of kb-00001"}))
    assert await kb_store(ctx=ctx, **kw) == "Error: dup of kb-00001"
    detail = {"error": "near_duplicate", "candidates": [{"id": "kb-00001"}]}
    ctx = _make_ctx(lambda req: httpx.Response(409, json={"detail": detail}))
    assert await kb_store(ctx=ctx, **kw) == "Error: " + json.dumps(detail)
    ctx = _make_ctx(
        lambda req: httpx.Response(422, json={"detail": "supersedes rejected: kb-99999 not found"})
    )
    assert await kb_store(ctx=ctx, **kw) == (
        "Error: KB service returned 422: supersedes rejected: kb-99999 not found"
    )


@pytest.mark.asyncio
async def test_skew_warning_silent_on_server_union_of_hints(caplog):
    import logging

    caplog.set_level(logging.INFO)
    resp = httpx.Response(
        200,
        json={
            "action": "created",
            "entry": _entry_json(),
            "superseded_ids": ["kb-00002", "kb-00003"],
        },
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(resp, []))
    await kb_store(
        supersedes=["kb-00002"],
        hints={"supersedes": ["kb-00003"]},
        short_title="a",
        long_title="b",
        knowledge_details="c",
        ctx=ctx,
    )
    assert not [r for r in caplog.records if "supersession-client mismatch" in r.getMessage()]


@pytest.mark.asyncio
async def test_skew_warning_fires_on_real_mismatch_with_hints(caplog):
    import logging

    caplog.set_level(logging.INFO)
    resp = httpx.Response(
        200,
        json={"action": "created", "entry": _entry_json(), "superseded_ids": ["kb-00002"]},
    )
    kb_store = _register_and_capture()
    ctx = _make_ctx(_recording_handler(resp, []))
    await kb_store(
        supersedes=["kb-00002"],
        hints={"supersedes": ["kb-00003"]},
        short_title="a",
        long_title="b",
        knowledge_details="c",
        ctx=ctx,
    )
    assert [r for r in caplog.records if "supersession-client mismatch" in r.getMessage()]


def test_hints_description_documents_resolution() -> None:
    from personal_kb.tools.kb_store import HINTS_DESCRIPTION

    for word in ("resolution", "corrected_fact", "target_class", "global"):
        assert word in HINTS_DESCRIPTION


# ---------------------------------------------------------------------------
# Write policy: queued creates and never-queued updates
# ---------------------------------------------------------------------------

QUEUED_STORE_TEXT = {
    "on": (
        "Queued as candidate 7 for review (write policy: headless surface). Not in"
        " the KB yet: the candidate pipeline's distiller and critic decide whether"
        " it is written."
    ),
    "shadow": (
        "Queued as candidate 7 (write policy: headless surface). Not in the KB:"
        " capture mode is shadow, so it is recorded for audit only."
    ),
    "off": (
        "Queued as candidate 7 (write policy: headless surface). Not in the KB:"
        " capture mode is off, so it is recorded for audit only."
    ),
}


def _queued_handler(mode: str):
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/store"
        return httpx.Response(
            200,
            json={
                "status": "queued",
                "candidate_id": 7,
                "surface": "headless",
                "capture_mode": mode,
            },
        )

    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
async def test_queued_create_renders(mode: str) -> None:
    from personal_kb.backend.protocol import QueuedStore
    from personal_kb.tools.kb_store import format_queued_store

    assert format_queued_store(QueuedStore(7, "headless", mode)) == QUEUED_STORE_TEXT[mode]
    kb_store = _register_and_capture()
    result = await kb_store(
        short_title="t",
        long_title="lt",
        knowledge_details="d",
        supersedes=["kb-00001"],
        ctx=_make_ctx(_queued_handler(mode)),
    )
    assert result == QUEUED_STORE_TEXT[mode]


@pytest.mark.asyncio
async def test_queued_update_is_an_error() -> None:
    kb_store = _register_and_capture()
    result = await kb_store(
        update_entry_id="kb-00001",
        knowledge_details="d",
        change_reason="r",
        supersedes="none",
        ctx=_make_ctx(_queued_handler("on")),
    )
    assert result == (
        "Error: the KB service queued an update as candidate 7; updates are never queued."
    )


@pytest.mark.asyncio
async def test_write_policy_403_renders_detail() -> None:
    detail = (
        "write policy: deactivating an entry requires an interactive surface; this"
        " request is headless."
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": detail})

    kb_store = _register_and_capture()
    result = await kb_store(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="r",
        ctx=_make_ctx(handler),
    )
    assert result == f"Error: {detail}"
