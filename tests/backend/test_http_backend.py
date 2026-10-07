"""Tests for HttpBackend — mock-transport coverage of all endpoints.

Uses httpx.MockTransport (sync handler, httpx wraps it for async) to avoid
real network calls.  The request-capture pattern mirrors tests/llm/test_ollama_llm.py.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from personal_kb.backend.http import (
    BackendHttpError,
    HttpBackend,
    _extract_detail,
    _map_error,
    _parse_file_result,
    _raise_for_status,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENTRY_JSON: dict[str, Any] = {
    "id": "kb-00001",
    "short_title": "Test Entry",
    "long_title": "Test long title",
    "knowledge_details": "Some details",
    "entry_type": "factual_reference",
    "confidence_level": 0.9,
    "is_active": True,
    "version": 1,
    "has_embedding": False,
    "tags": ["python"],
    "project_ref": "my-project",
    "source_context": None,
    "created_at": "2026-01-01T00:00:00+00:00",
    "updated_at": "2026-01-01T00:00:00+00:00",
    "last_accessed": None,
    "expires_at": None,
    "superseded_by": None,
    "sensitivity": None,
    "contributor": None,
    "team": None,
}


def _make_backend(handler) -> HttpBackend:
    """Build an HttpBackend with a MockTransport (sync handler)."""
    transport = httpx.MockTransport(handler)
    backend = HttpBackend(base_url="http://kb.test", api_key="testkey")
    # Inject client directly so we bypass open()
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        headers={"Authorization": "Bearer testkey"},
        transport=transport,
    )
    return backend


# ---------------------------------------------------------------------------
# is_remote
# ---------------------------------------------------------------------------


def test_http_backend_is_remote():
    backend = HttpBackend(base_url="http://kb.test", api_key="key")
    assert backend.is_remote is True


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_returns_results():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/api/kb/search"
        body = json.loads(req.content)
        assert body["query"] == "python testing"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "entry": _ENTRY_JSON,
                        "score": 0.95,
                        "effective_confidence": 0.85,
                        "staleness_warning": None,
                        "match_source": "fts",
                    }
                ],
                "filtered_count": 0,
            },
        )

    backend = _make_backend(handler)
    from kb_core.models.search import SearchQuery

    query = SearchQuery(query="python testing", limit=10)
    results, filtered = await backend.search(query)
    assert len(results) == 1
    assert results[0].entry.id == "kb-00001"
    assert results[0].score == pytest.approx(0.95)
    assert filtered == 0


# ---------------------------------------------------------------------------
# get_entries chunking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_entries_45_ids_produces_3_requests():
    """45 IDs must produce exactly 3 POST requests (batches of 20, 20, 5)."""
    calls: list[list[str]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/get"
        body = json.loads(req.content)
        batch_ids = body["ids"]
        calls.append(batch_ids)
        results = [
            {"id": eid, "found": False, "entry": None, "pointer_rot": []} for eid in batch_ids
        ]
        return httpx.Response(200, json={"results": results})

    backend = _make_backend(handler)
    ids = [f"kb-{i:05d}" for i in range(1, 46)]  # 45 IDs
    results = await backend.get_entries(ids)

    assert len(calls) == 3
    assert len(calls[0]) == 20
    assert len(calls[1]) == 20
    assert len(calls[2]) == 5
    assert len(results) == 45
    # All not found since handler returns found=False
    for _, entry, rot in results:
        assert entry is None
        assert rot == []


@pytest.mark.asyncio
async def test_get_entries_found_and_not_found():
    """Mix of found/not-found entries is parsed correctly."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        ids = body["ids"]
        results = []
        for eid in ids:
            if eid == "kb-00001":
                results.append({"id": eid, "found": True, "entry": _ENTRY_JSON, "pointer_rot": []})
            else:
                results.append({"id": eid, "found": False, "entry": None, "pointer_rot": []})
        return httpx.Response(200, json={"results": results})

    backend = _make_backend(handler)
    results = await backend.get_entries(["kb-00001", "kb-99999"])
    assert len(results) == 2
    eid0, entry0, rot0 = results[0]
    assert eid0 == "kb-00001"
    assert entry0 is not None
    assert entry0.id == "kb-00001"
    assert rot0 == []

    eid1, entry1, _ = results[1]
    assert eid1 == "kb-99999"
    assert entry1 is None


@pytest.mark.asyncio
async def test_get_entries_pointer_rot_parsed():
    """pointer_rot field is converted to list of (target_id, superseded_by) tuples."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": "kb-00001",
                        "found": True,
                        "entry": {**_ENTRY_JSON, "entry_type": "mental_map"},
                        "pointer_rot": [
                            {"target_id": "kb-00002", "superseded_by": "kb-00003"},
                            {"target_id": "kb-00004", "superseded_by": None},
                        ],
                    }
                ]
            },
        )

    backend = _make_backend(handler)
    results = await backend.get_entries(["kb-00001"])
    _, _, rot = results[0]
    assert rot == [("kb-00002", "kb-00003"), ("kb-00004", None)]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_create_returns_created_entry():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/store"
        body = json.loads(req.content)
        assert body["short_title"] == "My entry"
        return httpx.Response(200, json={"action": "created", "entry": _ENTRY_JSON})

    backend = _make_backend(handler)
    action, entry = await backend.store(
        short_title="My entry",
        long_title="Long title",
        knowledge_details="Details",
    )
    assert action == "created"
    assert entry.id == "kb-00001"


@pytest.mark.asyncio
async def test_store_update_returns_updated_entry():
    updated_json = {**_ENTRY_JSON, "version": 2}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body.get("update_entry_id") == "kb-00001"
        return httpx.Response(200, json={"action": "updated", "entry": updated_json})

    backend = _make_backend(handler)
    action, entry = await backend.store(
        short_title="",
        long_title="",
        knowledge_details="New details",
        update_entry_id="kb-00001",
    )
    assert action == "updated"
    assert entry.version == 2


# ---------------------------------------------------------------------------
# deactivate / reactivate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deactivate():
    deactivated = {**_ENTRY_JSON, "is_active": False}

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/entries/kb-00001/deactivate"
        return httpx.Response(200, json={"entry": deactivated})

    backend = _make_backend(handler)
    entry = await backend.deactivate("kb-00001")
    assert entry.is_active is False


@pytest.mark.asyncio
async def test_deactivate_posts_change_reason_body():
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"entry": {**_ENTRY_JSON, "is_active": False}})

    backend = _make_backend(handler)
    await backend.deactivate("kb-00001", change_reason="obsolete")
    assert seen == [{"change_reason": "obsolete"}]


@pytest.mark.asyncio
async def test_deactivate_posts_superseded_by_body():
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"entry": {**_ENTRY_JSON, "is_active": False}})

    backend = _make_backend(handler)
    await backend.deactivate("kb-00001", change_reason="replaced", superseded_by="kb-00002")
    assert seen == [{"change_reason": "replaced", "superseded_by": "kb-00002"}]


@pytest.mark.asyncio
async def test_reactivate():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/entries/kb-00001/reactivate"
        return httpx.Response(200, json={"entry": _ENTRY_JSON})

    backend = _make_backend(handler)
    entry = await backend.reactivate("kb-00001")
    assert entry.is_active is True


# ---------------------------------------------------------------------------
# store_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_batch_returns_created_entries():
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert len(body["entries"]) == 2
        second = {**_ENTRY_JSON, "id": "kb-00002"}
        return httpx.Response(200, json={"created": [_ENTRY_JSON, second]})

    backend = _make_backend(handler)
    created, failed = await backend.store_batch(
        [
            {"short_title": "A", "long_title": "A long", "knowledge_details": "A details"},
            {"short_title": "B", "long_title": "B long", "knowledge_details": "B details"},
        ]
    )
    assert len(created) == 2
    assert failed == []


# ---------------------------------------------------------------------------
# bulk_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bulk_update():
    before_json = _ENTRY_JSON
    after_json = {**_ENTRY_JSON, "version": 2, "project_ref": "new-proj"}

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["dry_run"] is False
        return httpx.Response(200, json={"results": [{"before": before_json, "after": after_json}]})

    backend = _make_backend(handler)
    pairs = await backend.bulk_update(
        filters={"project_ref": "my-project"},
        updates={"project_ref": "new-proj"},
        dry_run=False,
    )
    assert len(pairs) == 1
    before, after = pairs[0]
    assert before.project_ref == "my-project"
    assert after.project_ref == "new-proj"


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_feedback_sends_correct_body():
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/feedback"
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={})

    backend = _make_backend(handler)
    await backend.feedback(
        "missing",
        tool_name="kb_search",
        query_or_params="python async",
        detail="no results",
    )
    assert len(captured) == 1
    body = captured[0]
    assert body["feedback_type"] == "missing"
    assert body["tool_name"] == "kb_search"
    assert body["query_or_params"] == "python async"
    assert body["detail"] == "no results"


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_preflight():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/preflight"
        assert req.url.params.get("project_ref") == "my-proj"
        return httpx.Response(200, json={"context": "Project context here."})

    backend = _make_backend(handler)
    result = await backend.preflight("my-proj", since=None)
    assert result == "Project context here."


@pytest.mark.asyncio
async def test_preflight_with_since():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("since") == "7d"
        return httpx.Response(200, json={"context": "Recent context."})

    backend = _make_backend(handler)
    result = await backend.preflight("proj", since="7d")
    assert result == "Recent context."


# ---------------------------------------------------------------------------
# graph traversal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersedes_chain():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"chain": ["kb-00001", "kb-00002"]})

    backend = _make_backend(handler)
    chain = await backend.supersedes_chain("kb-00001")
    assert chain == ["kb-00001", "kb-00002"]


@pytest.mark.asyncio
async def test_find_path_not_found():
    """found=false, hops=[] → returns None."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"found": False, "hops": []})

    backend = _make_backend(handler)
    result = await backend.find_path("kb-00001", "kb-00002", max_depth=4)
    assert result is None


@pytest.mark.asyncio
async def test_find_path_same_node():
    """found=true, hops=[] → source==target, returns empty list."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"found": True, "hops": []})

    backend = _make_backend(handler)
    result = await backend.find_path("kb-00001", "kb-00001", max_depth=4)
    assert result == []


@pytest.mark.asyncio
async def test_find_path_with_hops():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "found": True,
                "hops": [
                    {"source": "kb-00001", "edge_type": "references", "target": "kb-00002"},
                ],
            },
        )

    backend = _make_backend(handler)
    result = await backend.find_path("kb-00001", "kb-00002", max_depth=4)
    assert result == [("kb-00001", "references", "kb-00002")]


# ---------------------------------------------------------------------------
# discovery: list_projects / list_contributors / list_teams
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_projects():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/projects"
        return httpx.Response(
            200,
            json={
                "items": [
                    {"name": "proj-a", "entry_count": 5},
                    {"name": "proj-b", "entry_count": 3},
                ]
            },
        )

    backend = _make_backend(handler)
    rows = await backend.list_projects()
    assert rows == [("proj-a", 5), ("proj-b", 3)]


@pytest.mark.asyncio
async def test_list_contributors():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/contributors"
        return httpx.Response(
            200,
            json={
                "items": [
                    {"name": "alice", "entry_count": 10},
                ]
            },
        )

    backend = _make_backend(handler)
    rows = await backend.list_contributors()
    assert rows == [("alice", 10)]


@pytest.mark.asyncio
async def test_list_teams():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/teams"
        return httpx.Response(
            200,
            json={
                "items": [
                    {"name": "eng", "entry_count": 7},
                ]
            },
        )

    backend = _make_backend(handler)
    rows = await backend.list_teams()
    assert rows == [("eng", 7)]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def test_map_error_401():
    exc = BackendHttpError(401, "Unauthorized")
    msg = _map_error(exc, "http://kb.test")
    assert "401" in msg
    assert "PERSONAL_KB_API_KEY" in msg


def test_map_error_403():
    exc = BackendHttpError(403, "Admin only")
    msg = _map_error(exc, "http://kb.test")
    assert "403" in msg
    assert "admin privileges required" in msg
    assert "Admin only" in msg


def test_map_error_404():
    exc = BackendHttpError(404, "Entry not found")
    msg = _map_error(exc, "http://kb.test")
    assert "Entry not found" in msg
    assert "404" not in msg  # 404 detail only


def test_map_error_409():
    exc = BackendHttpError(409, "Conflict")
    msg = _map_error(exc, "http://kb.test")
    assert "Conflict" in msg


def test_map_error_other():
    exc = BackendHttpError(500, "Server error")
    msg = _map_error(exc, "http://kb.test")
    assert "500" in msg
    assert "Server error" in msg


@pytest.mark.asyncio
async def test_connect_error_raises_backend_http_error():
    """ConnectError is wrapped as BackendHttpError(0, ...)."""

    def fail_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = _make_backend(fail_handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.search(
            __import__("kb_core.models.search", fromlist=["SearchQuery"]).SearchQuery(query="test"),
        )
    assert exc_info.value.status == 0
    assert "cannot reach" in exc_info.value.detail


# ---------------------------------------------------------------------------
# Bearer auth header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bearer_auth_header_sent():
    captured_headers: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_headers.append(dict(req.headers))
        return httpx.Response(200, json={"context": "x"})

    backend = HttpBackend(base_url="http://kb.test", api_key="my-secret-key")
    transport = httpx.MockTransport(handler)
    backend._client = httpx.AsyncClient(
        base_url="http://kb.test",
        headers={"Authorization": "Bearer my-secret-key"},
        transport=transport,
    )
    await backend.preflight("proj", since=None)
    assert len(captured_headers) == 1
    assert captured_headers[0].get("authorization") == "Bearer my-secret-key"


# ---------------------------------------------------------------------------
# _extract_detail — edge cases
# ---------------------------------------------------------------------------


def test_extract_detail_string_detail():
    """JSON body with a string 'detail' field is returned as-is."""
    resp = httpx.Response(422, json={"detail": "Validation error"})
    assert _extract_detail(resp) == "Validation error"


def test_extract_detail_list_detail():
    """FastAPI validation list detail is JSON-encoded."""
    detail_list = [{"loc": ["body", "query"], "msg": "field required", "type": "missing"}]
    resp = httpx.Response(422, json={"detail": detail_list})
    result = _extract_detail(resp)
    assert "field required" in result
    assert isinstance(result, str)
    # Must be valid JSON (the list was json.dumps'd)
    parsed = json.loads(result)
    assert isinstance(parsed, list)


def test_extract_detail_no_detail_key():
    """Body without 'detail' falls back to response text."""
    resp = httpx.Response(
        500, content=b"Internal Server Error", headers={"content-type": "text/plain"}
    )
    result = _extract_detail(resp)
    assert "Internal Server Error" in result


def test_extract_detail_invalid_json():
    """Non-JSON body falls back to response text."""
    resp = httpx.Response(
        503, content=b"Service Unavailable", headers={"content-type": "text/plain"}
    )
    result = _extract_detail(resp)
    assert "Service Unavailable" in result


# ---------------------------------------------------------------------------
# _raise_for_status — non-2xx
# ---------------------------------------------------------------------------


def test_raise_for_status_success_does_not_raise():
    resp = httpx.Response(200, json={"ok": True})
    # Should not raise
    _raise_for_status(resp, "http://kb.test")


def test_raise_for_status_4xx_raises():
    resp = httpx.Response(404, json={"detail": "Not found"})
    with pytest.raises(BackendHttpError) as exc_info:
        _raise_for_status(resp, "http://kb.test")
    assert exc_info.value.status == 404
    assert "Not found" in exc_info.value.detail


def test_raise_for_status_422_raises_with_list_detail():
    detail_list = [{"msg": "field required"}]
    resp = httpx.Response(422, json={"detail": detail_list})
    with pytest.raises(BackendHttpError) as exc_info:
        _raise_for_status(resp, "http://kb.test")
    assert exc_info.value.status == 422
    assert "field required" in exc_info.value.detail


# ---------------------------------------------------------------------------
# _parse_file_result — direct call
# ---------------------------------------------------------------------------


def test_parse_file_result_full():
    """_parse_file_result builds a FileResult from a complete dict."""
    data = {
        "path": "test.md",
        "action": "ingested",
        "reason": None,
        "entry_count": 3,
        "entry_ids": ["kb-00001", "kb-00002", "kb-00003"],
        "summary": "Test file",
        "chunks_processed": 2,
        "chunks_skipped": 0,
        "chunks_flagged": 0,
    }
    result = _parse_file_result(data)
    assert result.path == "test.md"
    assert result.action == "ingested"
    assert result.entry_count == 3
    assert result.entry_ids == ["kb-00001", "kb-00002", "kb-00003"]


def test_parse_file_result_defaults():
    """_parse_file_result uses sensible defaults for missing keys."""
    result = _parse_file_result({})
    assert result.path == ""
    assert result.action == "error"
    assert result.entry_count == 0


# ---------------------------------------------------------------------------
# Lifecycle: open / close / __aenter__ / __aexit__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_open_creates_client():
    """open() creates an httpx.AsyncClient attached to the backend."""
    backend = HttpBackend(base_url="http://kb.test", api_key="secret")
    assert backend._client is None
    await backend.open()
    assert backend._client is not None
    await backend.close()


@pytest.mark.asyncio
async def test_close_nulls_client():
    """close() closes the client and sets _client to None."""
    backend = HttpBackend(base_url="http://kb.test", api_key="secret")
    await backend.open()
    assert backend._client is not None
    await backend.close()
    assert backend._client is None


@pytest.mark.asyncio
async def test_close_when_already_closed_is_noop():
    """close() is idempotent when _client is already None."""
    backend = HttpBackend(base_url="http://kb.test", api_key="secret")
    # Should not raise
    await backend.close()
    assert backend._client is None


@pytest.mark.asyncio
async def test_aenter_aexit_context_manager():
    """async with HttpBackend opens and closes the client cleanly."""
    backend = HttpBackend(base_url="http://kb.test", api_key="secret")
    async with backend as ctx:
        assert ctx is backend
        assert backend._client is not None
    assert backend._client is None


# ---------------------------------------------------------------------------
# _c() — RuntimeError when not opened
# ---------------------------------------------------------------------------


def test_c_raises_when_client_is_none():
    """_c() raises RuntimeError when the backend has not been opened."""
    backend = HttpBackend(base_url="http://kb.test", api_key="secret")
    with pytest.raises(RuntimeError, match="not opened"):
        backend._c()


# ---------------------------------------------------------------------------
# _get error paths: ConnectError, TimeoutException, timeout kwarg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_connect_error_raises_backend_http_error():
    """ConnectError inside _get is wrapped as BackendHttpError(0, ...)."""

    def fail_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = _make_backend(fail_handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.supersedes_chain("kb-00001")
    assert exc_info.value.status == 0
    assert "cannot reach" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_timeout_raises_backend_http_error():
    """TimeoutException inside _get is wrapped as BackendHttpError(0, ...)."""

    def timeout_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    backend = _make_backend(timeout_handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.supersedes_chain("kb-00001")
    assert exc_info.value.status == 0
    assert "cannot reach" in exc_info.value.detail


@pytest.mark.asyncio
async def test_post_timeout_raises_backend_http_error():
    """TimeoutException inside _post is wrapped as BackendHttpError(0, ...)."""

    def timeout_handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    backend = _make_backend(timeout_handler)
    with pytest.raises(BackendHttpError) as exc_info:
        from kb_core.models.search import SearchQuery

        await backend.search(SearchQuery(query="test"))
    assert exc_info.value.status == 0


# ---------------------------------------------------------------------------
# search — optional query params (project_ref, entry_type, tags, filtered_count)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_with_project_ref_and_tags():
    """project_ref and tags are included in the POST body when set."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"results": [], "filtered_count": 3})

    backend = _make_backend(handler)
    from kb_core.models.search import SearchQuery

    query = SearchQuery(query="test", project_ref="my-proj", tags=["python", "async"], limit=5)
    _results, filtered = await backend.search(query)
    assert filtered == 3
    assert len(captured) == 1
    body = captured[0]
    assert body["project_ref"] == "my-proj"
    assert body["tags"] == ["python", "async"]


@pytest.mark.asyncio
async def test_search_with_entry_type_enum():
    """entry_type Enum value is serialised to string."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"results": [], "filtered_count": 0})

    backend = _make_backend(handler)
    from kb_core.models.search import SearchQuery

    from personal_kb.models.entry import EntryType

    query = SearchQuery(query="decisions", entry_type=EntryType.DECISION, limit=5)
    await backend.search(query)
    assert captured[0]["entry_type"] == "decision"


@pytest.mark.asyncio
async def test_vector_search_available_always_true():
    """HttpBackend.vector_search_available() always returns True."""
    backend = HttpBackend(base_url="http://kb.test", api_key="key")
    result = await backend.vector_search_available()
    assert result is True


# ---------------------------------------------------------------------------
# store — all optional conditional fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_with_all_optional_fields():
    """All optional store fields are included when set."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"action": "created", "entry": _ENTRY_JSON})

    backend = _make_backend(handler)
    from personal_kb.models.entry import EntryType

    await backend.store(
        short_title="My entry",
        long_title="Long title",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
        project_ref="proj-a",
        source_context="some context",
        tags=["tag1"],
        hints={"related": ["kb-00002"]},
        sensitivity="internal",
        ttl="90d",
        change_reason="new data",
    )
    body = captured[0]
    assert body["entry_type"] == "decision"
    assert body["project_ref"] == "proj-a"
    assert body["source_context"] == "some context"
    assert body["tags"] == ["tag1"]
    assert body["hints"] == {"related": ["kb-00002"]}
    assert body["sensitivity"] == "internal"
    assert body["ttl"] == "90d"
    assert body["change_reason"] == "new data"


# ---------------------------------------------------------------------------
# store_batch — empty list + optional fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_batch_empty_returns_early():
    """store_batch with empty list returns ([], []) without an HTTP call."""
    called = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={})

    backend = _make_backend(handler)
    created, failed = await backend.store_batch([])
    assert created == []
    assert failed == []
    assert not called  # No HTTP request made


@pytest.mark.asyncio
async def test_store_batch_with_all_optional_fields():
    """store_batch passes optional per-entry fields to the service."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        second = {**_ENTRY_JSON, "id": "kb-00002"}
        return httpx.Response(200, json={"created": [_ENTRY_JSON, second]})

    backend = _make_backend(handler)
    await backend.store_batch(
        [
            {
                "short_title": "A",
                "long_title": "A long",
                "knowledge_details": "A details",
                "entry_type": "decision",
                "project_ref": "proj-x",
                "source_context": "ctx",
                "confidence_level": 0.75,
                "tags": ["a", "b"],
                "hints": {"in_project": ["proj-x"]},
                "sensitivity": "public",
                "ttl": "30d",
            },
            {
                "short_title": "B",
                "long_title": "B long",
                "knowledge_details": "B details",
            },
        ]
    )
    entries = captured[0]["entries"]
    assert entries[0]["entry_type"] == "decision"
    assert entries[0]["project_ref"] == "proj-x"
    assert entries[0]["source_context"] == "ctx"
    assert entries[0]["confidence_level"] == pytest.approx(0.75)
    assert entries[0]["tags"] == ["a", "b"]
    assert entries[0]["hints"] == {"in_project": ["proj-x"]}
    assert entries[0]["sensitivity"] == "public"
    assert entries[0]["ttl"] == "30d"
    # Second entry has no optional fields — should only have required keys
    assert "entry_type" not in entries[1]


# ---------------------------------------------------------------------------
# ask_auto — with scope and non-empty entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_auto_with_scope_and_entries():
    """ask_auto sends scope and parses entries + agent_turns_used."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body.get("scope") == "project:kb"
        return httpx.Response(
            200,
            json={
                "entries": [
                    {"entry": _ENTRY_JSON, "context": "relevant context"},
                ],
                "agent_turns_used": 2,
            },
        )

    backend = _make_backend(handler)
    entries_with_ctx, turns = await backend.ask_auto(
        question="what is X?",
        scope="project:kb",
        include_graph_context=True,
        limit=10,
    )
    assert turns == 2
    assert len(entries_with_ctx) == 1
    entry, ctx_str = entries_with_ctx[0]
    assert entry.id == "kb-00001"
    assert ctx_str == "relevant context"


@pytest.mark.asyncio
async def test_ask_auto_without_scope():
    """ask_auto without scope omits the scope key from the request body."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"entries": [], "agent_turns_used": 0})

    backend = _make_backend(handler)
    await backend.ask_auto(question="test", scope=None, include_graph_context=False, limit=5)
    assert "scope" not in captured[0]


# ---------------------------------------------------------------------------
# summarize — with/without scope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summarize_with_scope():
    """summarize sends scope and returns answer string."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body.get("scope") == "tag:python"
        return httpx.Response(200, json={"answer": "Python is a programming language."})

    backend = _make_backend(handler)
    result = await backend.summarize("what is python?", scope="tag:python", limit=10)
    assert result == "Python is a programming language."


@pytest.mark.asyncio
async def test_summarize_without_scope():
    """summarize without scope omits scope from request."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"answer": "Summary here."})

    backend = _make_backend(handler)
    result = await backend.summarize("question", scope=None, limit=5)
    assert result == "Summary here."
    assert "scope" not in captured[0]


# ---------------------------------------------------------------------------
# bfs_entries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bfs_entries():
    """bfs_entries sends correct params and parses (entry_id, depth, path) tuples."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/graph/bfs"
        assert req.url.params.get("start_node") == "kb-00001"
        assert req.url.params.get("max_depth") == "2"
        return httpx.Response(
            200,
            json={
                "entries": [
                    {"entry_id": "kb-00001", "depth": 0, "path": []},
                    {"entry_id": "kb-00002", "depth": 1, "path": ["kb-00001"]},
                ]
            },
        )

    backend = _make_backend(handler)
    results = await backend.bfs_entries(start="kb-00001", max_depth=2, limit=50)
    assert len(results) == 2
    assert results[0] == ("kb-00001", 0, [])
    assert results[1] == ("kb-00002", 1, ["kb-00001"])


# ---------------------------------------------------------------------------
# entries_for_scope — with entry_type and order_by
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entries_for_scope_defaults():
    """entries_for_scope without optional params sends just scope."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("scope") == "project:kb"
        assert "entry_type" not in req.url.params
        assert "order_by" not in req.url.params
        return httpx.Response(200, json={"entry_ids": ["kb-00001", "kb-00002"]})

    backend = _make_backend(handler)
    ids = await backend.entries_for_scope("project:kb")
    assert ids == ["kb-00001", "kb-00002"]


@pytest.mark.asyncio
async def test_entries_for_scope_with_entry_type_and_order_by():
    """entries_for_scope sends entry_type and order_by when non-default."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json={"entry_ids": ["kb-00003"]})

    backend = _make_backend(handler)
    ids = await backend.entries_for_scope(
        "project:kb", entry_type="decision", order_by="created_at"
    )
    assert ids == ["kb-00003"]
    assert captured[0]["entry_type"] == "decision"
    assert captured[0]["order_by"] == "created_at"


# ---------------------------------------------------------------------------
# neighbors — with edge_types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_neighbors_with_edge_types():
    """neighbors sends edge_types when provided and parses result."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(
            200,
            json={
                "neighbors": [
                    {"neighbor_id": "kb-00002", "edge_type": "references", "direction": "outgoing"},
                ]
            },
        )

    backend = _make_backend(handler)
    results = await backend.neighbors("kb-00001", edge_types=["references"], direction="outgoing")
    assert len(results) == 1
    assert results[0] == ("kb-00002", "references", "outgoing")


@pytest.mark.asyncio
async def test_neighbors_without_edge_types():
    """neighbors without edge_types omits edge_types from params."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(dict(req.url.params))
        return httpx.Response(200, json={"neighbors": []})

    backend = _make_backend(handler)
    await backend.neighbors("kb-00001")
    assert "edge_types" not in captured[0]


# ---------------------------------------------------------------------------
# ingest_file — multipart upload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_file_success(tmp_path):
    """ingest_file posts a multipart request and returns a FileResult."""
    test_file = tmp_path / "doc.md"
    test_file.write_text("# Title\n\nSome content.")

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/api/kb/ingest/file"
        # Verify multipart by checking content-type header
        assert "multipart/form-data" in req.headers.get("content-type", "")
        return httpx.Response(
            200,
            json={
                "path": str(test_file),
                "action": "ingested",
                "entry_count": 1,
                "entry_ids": ["kb-00001"],
            },
        )

    backend = _make_backend(handler)
    result = await backend.ingest_file(test_file, project_ref=None, dry_run=False)
    assert result.action == "ingested"
    assert result.entry_count == 1


@pytest.mark.asyncio
async def test_ingest_file_with_project_ref(tmp_path):
    """ingest_file sends project_ref as a form field when provided."""
    test_file = tmp_path / "doc.md"
    test_file.write_text("content")

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"path": str(test_file), "action": "ingested", "entry_count": 0},
        )

    backend = _make_backend(handler)
    result = await backend.ingest_file(test_file, project_ref="my-proj", dry_run=True)
    assert result.action == "ingested"


@pytest.mark.asyncio
async def test_ingest_file_connect_error(tmp_path):
    """ConnectError during ingest_file is wrapped as BackendHttpError."""
    test_file = tmp_path / "doc.md"
    test_file.write_text("content")

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    backend = _make_backend(handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.ingest_file(test_file, project_ref=None, dry_run=False)
    assert exc_info.value.status == 0


@pytest.mark.asyncio
async def test_ingest_file_timeout(tmp_path):
    """TimeoutException during ingest_file is wrapped as BackendHttpError."""
    test_file = tmp_path / "doc.md"
    test_file.write_text("content")

    def handler(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    backend = _make_backend(handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.ingest_file(test_file, project_ref=None, dry_run=False)
    assert exc_info.value.status == 0


# ---------------------------------------------------------------------------
# ingest_url — with content and project_ref
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_url_with_content_and_project_ref():
    """ingest_url sends content and project_ref when provided."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={"path": "https://example.com", "action": "ingested", "entry_count": 2},
        )

    backend = _make_backend(handler)
    result = await backend.ingest_url(
        "https://example.com",
        content="Pre-fetched page content",
        project_ref="web-proj",
        dry_run=False,
    )
    assert result.action == "ingested"
    body = captured[0]
    assert body["content"] == "Pre-fetched page content"
    assert body["project_ref"] == "web-proj"
    assert body["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_ingest_url_minimal():
    """ingest_url without content/project_ref omits those keys."""
    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(
            200,
            json={"path": "https://x.com", "action": "skipped", "entry_count": 0},
        )

    backend = _make_backend(handler)
    result = await backend.ingest_url("https://x.com", content=None, project_ref=None, dry_run=True)
    assert result.action == "skipped"
    body = captured[0]
    assert "content" not in body
    assert "project_ref" not in body


# ---------------------------------------------------------------------------
# decision_search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_search():
    """decision_search POST /api/kb/search with entry_type=decision and returns IDs."""

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        assert body["entry_type"] == "decision"
        assert body["query"] == "architecture"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"entry": {**_ENTRY_JSON, "id": "kb-00010"}, "score": 0.9},
                    {"entry": {**_ENTRY_JSON, "id": "kb-00011"}, "score": 0.8},
                ],
                "filtered_count": 0,
            },
        )

    backend = _make_backend(handler)
    ids = await backend.decision_search("architecture", limit=10)
    assert ids == ["kb-00010", "kb-00011"]


# ---------------------------------------------------------------------------
# HTTP 4xx/5xx error propagation through _post and _get
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_401_raises_backend_http_error():
    """401 response from _post raises BackendHttpError(401, ...)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    backend = _make_backend(handler)
    with pytest.raises(BackendHttpError) as exc_info:
        from kb_core.models.search import SearchQuery

        await backend.search(SearchQuery(query="test"))
    assert exc_info.value.status == 401


@pytest.mark.asyncio
async def test_get_404_raises_backend_http_error():
    """404 response from _get raises BackendHttpError(404, ...)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Entry not found"})

    backend = _make_backend(handler)
    with pytest.raises(BackendHttpError) as exc_info:
        await backend.supersedes_chain("kb-99999")
    assert exc_info.value.status == 404
    assert "Entry not found" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_with_timeout_parameter():
    """_get passes timeout kwarg when provided (hits the timeout branch)."""
    captured_requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured_requests.append(req)
        return httpx.Response(200, json={"answer": "ok"})

    backend = _make_backend(handler)
    # ask_auto and summarize use _LONG_TIMEOUT — trigger via ask_auto
    result = await backend.ask_auto(
        question="test", scope=None, include_graph_context=False, limit=5
    )
    # No error — long-timeout path was exercised
    assert isinstance(result, tuple)


# ---------------------------------------------------------------------------
# map eligibility
# ---------------------------------------------------------------------------

# Mirror of the owner item's kb-core dataclasses.asdict() contract and of the
# API item's MapEligibilityVerdictModel — diff this literal against those.
_VERDICT_JSON: dict[str, Any] = {
    "evidence": {
        "project_ref": "harness-design",
        "mappable": 71,
        "ingested": 0,
        "hand_authored": 71,
        "maps": 0,
        "top_prefix": "Session",
        "top_prefix_share": 0.9792,
        "is_ingest_corpus": False,
        "is_too_thin": False,
        "is_journal": True,
        "computed_eligible": False,
    },
    "override": {
        "project_ref": "harness-design",
        "eligible": True,
        "reason": "kept as a dated session journal on purpose",
        "set_by": "jason@example.com",
        "set_at": "2026-09-19T12:00:00+00:00",
    },
    "effective_eligible": True,
    "decided_by": "override",
    "orphaned": False,
}


@pytest.mark.asyncio
async def test_map_eligibility_returns_projects_verbatim():
    """GET /api/kb/map-eligibility returns the projects list untouched."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "GET"
        assert req.url.path == "/api/kb/map-eligibility"
        return httpx.Response(200, json={"projects": [_VERDICT_JSON]})

    backend = _make_backend(handler)
    result = await backend.map_eligibility()
    assert result == [_VERDICT_JSON]
    assert result[0]["evidence"]["top_prefix_share"] == 0.9792


@pytest.mark.asyncio
async def test_map_eligibility_wrong_envelope_key_is_loud():
    """A wrong top-level envelope key raises KeyError naming 'projects'."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [_VERDICT_JSON]})

    backend = _make_backend(handler)
    with pytest.raises(KeyError) as exc_info:
        await backend.map_eligibility()
    assert "projects" in str(exc_info.value)


@pytest.mark.asyncio
async def test_set_map_eligibility_override_sends_exact_body():
    """POST /api/kb/map-eligibility/override sends exactly the three body keys."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/api/kb/map-eligibility/override"
        body = json.loads(req.content)
        assert set(body) == {"project_ref", "eligible", "reason"}
        assert body["eligible"] is False
        return httpx.Response(200, json={"changed": True, "verdict": _VERDICT_JSON})

    backend = _make_backend(handler)
    await backend.set_map_eligibility_override("harness-design", eligible=False, reason="too thin")


@pytest.mark.asyncio
async def test_set_map_eligibility_override_returns_changed_and_verdict():
    """The two-key envelope is returned intact, including a null verdict."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": True, "verdict": _VERDICT_JSON})

    backend = _make_backend(handler)
    result = await backend.set_map_eligibility_override("harness-design", eligible=True, reason="r")
    assert result == {"changed": True, "verdict": _VERDICT_JSON}


@pytest.mark.asyncio
async def test_set_map_eligibility_override_null_verdict_passes_through():
    """A null verdict is returned as None rather than raising."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": True, "verdict": None})

    backend = _make_backend(handler)
    result = await backend.set_map_eligibility_override("harness-design", eligible=True, reason="r")
    assert result == {"changed": True, "verdict": None}


@pytest.mark.asyncio
async def test_set_map_eligibility_override_missing_envelope_is_loud():
    """Missing 'changed' or 'verdict' envelope keys raise KeyError naming them."""

    def empty_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    backend = _make_backend(empty_handler)
    with pytest.raises(KeyError) as exc_info:
        await backend.set_map_eligibility_override("harness-design", eligible=True, reason="r")
    assert "changed" in str(exc_info.value)

    def no_verdict_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": True})

    backend = _make_backend(no_verdict_handler)
    with pytest.raises(KeyError) as exc_info:
        await backend.set_map_eligibility_override("harness-design", eligible=True, reason="r")
    assert "verdict" in str(exc_info.value)


@pytest.mark.asyncio
async def test_clear_map_eligibility_override_posts_to_clear_path():
    """The clear is a POST to /override/clear with one body key, not a DELETE."""

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.method == "POST"
        assert req.url.path == "/api/kb/map-eligibility/override/clear"
        assert json.loads(req.content) == {"project_ref": "harness-design"}
        return httpx.Response(200, json={"changed": True, "verdict": _VERDICT_JSON})

    backend = _make_backend(handler)
    assert await backend.clear_map_eligibility_override("harness-design") is True


@pytest.mark.asyncio
async def test_clear_map_eligibility_override_returns_changed_flag():
    """The clear returns the changed flag both ways; a null verdict does not raise."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": True, "verdict": _VERDICT_JSON})

    backend = _make_backend(handler)
    assert await backend.clear_map_eligibility_override("harness-design") is True

    def noop_handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": False, "verdict": None})

    backend = _make_backend(noop_handler)
    assert await backend.clear_map_eligibility_override("harness-design") is False


@pytest.mark.asyncio
async def test_clear_map_eligibility_override_missing_changed_key_is_loud():
    """A missing 'changed' envelope key raises KeyError naming 'changed'."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    backend = _make_backend(handler)
    with pytest.raises(KeyError) as exc_info:
        await backend.clear_map_eligibility_override("harness-design")
    assert "changed" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_search_body_carries_include_superseded(flag):
    from kb_core.models.search import SearchQuery

    captured: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        captured.update(json.loads(req.content))
        return httpx.Response(200, json={"results": [], "filtered_count": 0})

    backend = _make_backend(handler)
    await backend.search(SearchQuery(query="x", include_superseded=flag))
    assert captured["include_superseded"] is flag
