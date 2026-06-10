"""HTTP-mode (is_remote=True) coverage for MCP tool shims.

Each test injects an HttpBackend (with httpx.MockTransport) as the
``"backend"`` key of the lifespan context so ``backend_from_lifespan``
returns it directly.  This exercises every ``if backend.is_remote:``
branch in the tool files.

Pattern:
  1.  Register the tool on a mock MCP and capture the function.
  2.  Build an HttpBackend backed by MockTransport.
  3.  Put the backend under ``lifespan["backend"]``.
  4.  Call the captured tool and assert on the string result.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from personal_kb.backend.http import HttpBackend

# ---------------------------------------------------------------------------
# Shared helpers
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


def _make_http_lifespan(handler) -> dict[str, Any]:
    """Return a lifespan dict with an HttpBackend injected."""
    return {"backend": _make_http_backend(handler)}


def _make_ctx(handler) -> MagicMock:
    """Return a MagicMock Context with HTTP lifespan."""
    ctx = MagicMock()
    ctx.lifespan_context = _make_http_lifespan(handler)
    return ctx


def _register(register_fn, **kwargs) -> Any:
    """Register a tool on a mock MCP and return the captured function."""
    tools: dict[str, Any] = {}

    def capture(**_kw):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    register_fn(mcp, **kwargs)
    # Return the first registered callable
    return next(iter(tools.values()))


# ---------------------------------------------------------------------------
# kb_ask — HTTP-mode branches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_ask_http_auto_strategy_returns_entries():
    """HTTP mode with strategy='auto' calls backend.ask_auto and formats entries."""
    from personal_kb.tools.kb_ask import register_kb_ask

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "entries": [{"entry": _ENTRY_JSON, "context": "some context"}],
                "agent_turns_used": 1,
            },
        )

    kb_ask = _register(register_kb_ask)
    ctx = _make_ctx(handler)
    result = await kb_ask(question="test question", strategy="auto", ctx=ctx)
    assert "[Agent: 1 tool calls]" in result
    assert "Test Entry" in result


@pytest.mark.asyncio
async def test_kb_ask_http_auto_strategy_no_results():
    """HTTP mode with ask_auto returning empty entries shows 'No results'."""
    from personal_kb.tools.kb_ask import register_kb_ask

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"entries": [], "agent_turns_used": 0})

    kb_ask = _register(register_kb_ask)
    ctx = _make_ctx(handler)
    result = await kb_ask(question="obscure question", strategy="auto", ctx=ctx)
    assert "No results" in result
    assert "[Agent: 0 tool calls]" in result


@pytest.mark.asyncio
async def test_kb_ask_http_non_auto_strategy_error():
    """Non-'auto' strategies return an error in HTTP mode."""
    from personal_kb.tools.kb_ask import register_kb_ask

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})  # should not be called

    kb_ask = _register(register_kb_ask)
    ctx = _make_ctx(handler)
    result = await kb_ask(question="trace this", strategy="decision_trace", ctx=ctx)
    assert "Error" in result
    assert "decision_trace" in result
    assert "HTTP mode" in result


@pytest.mark.asyncio
async def test_kb_ask_http_backend_error_mapped():
    """BackendHttpError from ask_auto is mapped to an error string."""
    from personal_kb.tools.kb_ask import register_kb_ask

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    kb_ask = _register(register_kb_ask)
    ctx = _make_ctx(handler)
    result = await kb_ask(question="test", strategy="auto", ctx=ctx)
    assert "Error" in result
    assert "401" in result


# ---------------------------------------------------------------------------
# kb_list_projects / kb_list_contributors / kb_list_teams — HTTP-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_list_projects_http_mode():
    """kb_list_projects in HTTP mode calls backend.list_projects and formats output."""
    from personal_kb.tools.kb_list import register_kb_list_projects

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"name": "proj-a", "entry_count": 5}]})

    kb_list_projects = _register(register_kb_list_projects)
    ctx = _make_ctx(handler)
    result = await kb_list_projects(ctx=ctx)
    assert "proj-a" in result
    assert "5 entries" in result


@pytest.mark.asyncio
async def test_kb_list_projects_http_empty():
    """kb_list_projects in HTTP mode returns 'No projects' when list is empty."""
    from personal_kb.tools.kb_list import register_kb_list_projects

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    kb_list_projects = _register(register_kb_list_projects)
    ctx = _make_ctx(handler)
    result = await kb_list_projects(ctx=ctx)
    assert "No projects found" in result


@pytest.mark.asyncio
async def test_kb_list_projects_http_error():
    """BackendHttpError from list_projects is mapped to an error string."""
    from personal_kb.tools.kb_list import register_kb_list_projects

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    kb_list_projects = _register(register_kb_list_projects)
    ctx = _make_ctx(handler)
    result = await kb_list_projects(ctx=ctx)
    assert "Error" in result


@pytest.mark.asyncio
async def test_kb_list_contributors_http_mode():
    """kb_list_contributors in HTTP mode formats output correctly."""
    from personal_kb.tools.kb_list import register_kb_list_contributors

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"name": "alice", "entry_count": 10}]})

    fn = _register(register_kb_list_contributors)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "alice" in result
    assert "10 entries" in result


@pytest.mark.asyncio
async def test_kb_list_contributors_http_empty():
    from personal_kb.tools.kb_list import register_kb_list_contributors

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    fn = _register(register_kb_list_contributors)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "No contributors found" in result


@pytest.mark.asyncio
async def test_kb_list_contributors_http_error():
    from personal_kb.tools.kb_list import register_kb_list_contributors

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Admin only"})

    fn = _register(register_kb_list_contributors)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "Error" in result


@pytest.mark.asyncio
async def test_kb_list_teams_http_mode():
    from personal_kb.tools.kb_list import register_kb_list_teams

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"name": "eng", "entry_count": 3}]})

    fn = _register(register_kb_list_teams)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "eng" in result
    assert "3 entries" in result


@pytest.mark.asyncio
async def test_kb_list_teams_http_empty():
    from personal_kb.tools.kb_list import register_kb_list_teams

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": []})

    fn = _register(register_kb_list_teams)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "No teams found" in result


@pytest.mark.asyncio
async def test_kb_list_teams_http_error():
    from personal_kb.tools.kb_list import register_kb_list_teams

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Server error"})

    fn = _register(register_kb_list_teams)
    ctx = _make_ctx(handler)
    result = await fn(ctx=ctx)
    assert "Error" in result


# ---------------------------------------------------------------------------
# kb_preflight — HTTP-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_preflight_http_mode_success():
    """kb_preflight in HTTP mode returns the context string from the service."""
    from personal_kb.tools.kb_preflight import register_kb_preflight

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("project_ref") == "my-proj"
        return httpx.Response(200, json={"context": "Project context here."})

    kb_preflight = _register(register_kb_preflight)
    ctx = _make_ctx(handler)
    result = await kb_preflight(project_ref="my-proj", ctx=ctx)
    assert result == "Project context here."


@pytest.mark.asyncio
async def test_kb_preflight_http_mode_with_since():
    """kb_preflight in HTTP mode passes 'since' to the service."""
    from personal_kb.tools.kb_preflight import register_kb_preflight

    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get("since") == "7d"
        return httpx.Response(200, json={"context": "Recent context."})

    kb_preflight = _register(register_kb_preflight)
    ctx = _make_ctx(handler)
    result = await kb_preflight(project_ref="proj", since="7d", ctx=ctx)
    assert result == "Recent context."


@pytest.mark.asyncio
async def test_kb_preflight_http_mode_backend_error():
    """BackendHttpError from preflight is mapped to an error string."""
    from personal_kb.tools.kb_preflight import register_kb_preflight

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    kb_preflight = _register(register_kb_preflight)
    ctx = _make_ctx(handler)
    result = await kb_preflight(project_ref="proj", ctx=ctx)
    assert "Error" in result
    assert "401" in result


@pytest.mark.asyncio
async def test_kb_preflight_http_mode_invalid_since():
    """Invalid 'since' is caught before any HTTP call."""
    from personal_kb.tools.kb_preflight import register_kb_preflight

    called = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={"context": "x"})

    kb_preflight = _register(register_kb_preflight)
    ctx = _make_ctx(handler)
    result = await kb_preflight(project_ref="proj", since="notvalid", ctx=ctx)
    # parse_ttl raises on bad format — returned as Error
    assert "Error" in result
    assert not called  # No HTTP call was made


# ---------------------------------------------------------------------------
# kb_search — HTTP-mode: contributor/team filter error + search path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_search_http_rejects_contributor_filter():
    """kb_search in HTTP mode returns an error if contributor is passed."""
    from personal_kb.tools.kb_search import register_kb_search

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "filtered_count": 0})

    kb_search = _register(register_kb_search)
    ctx = _make_ctx(handler)
    result = await kb_search(query="test", contributor="alice", ctx=ctx)
    assert "Error" in result
    assert "contributor" in result.lower()


@pytest.mark.asyncio
async def test_kb_search_http_rejects_team_filter():
    """kb_search in HTTP mode returns an error if team is passed."""
    from personal_kb.tools.kb_search import register_kb_search

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "filtered_count": 0})

    kb_search = _register(register_kb_search)
    ctx = _make_ctx(handler)
    result = await kb_search(query="test", team="eng", ctx=ctx)
    assert "Error" in result


@pytest.mark.asyncio
async def test_kb_search_http_success():
    """kb_search in HTTP mode returns formatted results."""
    from personal_kb.tools.kb_search import register_kb_search

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/kb/search":
            body = json.loads(req.content)
            if body.get("query"):
                return httpx.Response(
                    200,
                    json={
                        "results": [
                            {
                                "entry": _ENTRY_JSON,
                                "score": 0.9,
                                "effective_confidence": 0.85,
                                "staleness_warning": None,
                                "match_source": "fts",
                            }
                        ],
                        "filtered_count": 0,
                    },
                )
        # neighbors call for graph hints (sparse check)
        return httpx.Response(200, json={"neighbors": []})

    kb_search = _register(register_kb_search)
    ctx = _make_ctx(handler)
    result = await kb_search(query="python testing", limit=5, ctx=ctx)
    assert "Test Entry" in result


# ---------------------------------------------------------------------------
# kb_summarize — HTTP-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_summarize_http_mode_success():
    """kb_summarize in HTTP mode calls backend.summarize and returns the answer."""
    from personal_kb.tools.kb_summarize import register_kb_summarize

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "Python is great for scripting."})

    kb_summarize = _register(register_kb_summarize)
    ctx = _make_ctx(handler)
    result = await kb_summarize(question="what is python?", ctx=ctx)
    assert result == "Python is great for scripting."


@pytest.mark.asyncio
async def test_kb_summarize_http_mode_with_scope():
    """kb_summarize in HTTP mode passes scope to the service."""
    from personal_kb.tools.kb_summarize import register_kb_summarize

    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={"answer": "Scoped answer."})

    kb_summarize = _register(register_kb_summarize)
    ctx = _make_ctx(handler)
    result = await kb_summarize(question="test", scope="project:kb", ctx=ctx)
    assert result == "Scoped answer."
    assert captured[0].get("scope") == "project:kb"


@pytest.mark.asyncio
async def test_kb_summarize_http_mode_backend_error():
    """BackendHttpError from summarize is mapped to an error string."""
    from personal_kb.tools.kb_summarize import register_kb_summarize

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "Unauthorized"})

    kb_summarize = _register(register_kb_summarize)
    ctx = _make_ctx(handler)
    result = await kb_summarize(question="test", ctx=ctx)
    assert "Error" in result
    assert "401" in result


# ---------------------------------------------------------------------------
# kb_feedback — registered tool function path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_feedback_registered_tool_valid():
    """The registered kb_feedback tool calls backend.feedback for a valid type."""
    from personal_kb.tools.kb_feedback import register_kb_feedback

    captured: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        captured.append(json.loads(req.content))
        return httpx.Response(200, json={})

    kb_feedback = _register(register_kb_feedback)
    ctx = _make_ctx(handler)
    result = await kb_feedback(
        feedback_type="missing",
        tool_name="kb_search",
        query_or_params="python async",
        detail="no results found",
        ctx=ctx,
    )
    assert "Feedback recorded" in result
    assert "missing" in result
    assert len(captured) == 1
    assert captured[0]["feedback_type"] == "missing"


@pytest.mark.asyncio
async def test_kb_feedback_registered_tool_invalid_type():
    """The registered kb_feedback tool rejects invalid feedback_type client-side."""
    from personal_kb.tools.kb_feedback import register_kb_feedback

    called = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={})

    kb_feedback = _register(register_kb_feedback)
    ctx = _make_ctx(handler)
    result = await kb_feedback(feedback_type="bad_type", ctx=ctx)
    assert "Invalid feedback_type" in result
    assert not called  # No HTTP call made


# ---------------------------------------------------------------------------
# kb_get — registered tool function body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_get_registered_tool_found():
    """The registered kb_get tool formats found entries."""
    from personal_kb.tools.kb_get import register_kb_get

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"id": "kb-00001", "found": True, "entry": _ENTRY_JSON, "pointer_rot": []}
                ]
            },
        )

    kb_get = _register(register_kb_get)
    ctx = _make_ctx(handler)
    result = await kb_get(entry_id="kb-00001", ctx=ctx)
    assert "Test Entry" in result


@pytest.mark.asyncio
async def test_kb_get_registered_tool_not_found():
    """The registered kb_get tool shows 'not found' for missing entries."""
    from personal_kb.tools.kb_get import register_kb_get

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"id": "kb-99999", "found": False, "entry": None}]},
        )

    kb_get = _register(register_kb_get)
    ctx = _make_ctx(handler)
    result = await kb_get(entry_id="kb-99999", ctx=ctx)
    assert "not found" in result


@pytest.mark.asyncio
async def test_kb_get_registered_tool_too_many_ids():
    """The registered kb_get tool rejects > 20 IDs."""
    from personal_kb.tools.kb_get import register_kb_get

    kb_get = _register(register_kb_get)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    ctx = _make_ctx(handler)
    too_many = [f"kb-{i:05d}" for i in range(21)]
    result = await kb_get(entry_id=too_many, ctx=ctx)
    assert "Error" in result
    assert "Maximum 20" in result


# ---------------------------------------------------------------------------
# kb_ingest_url — HTTP-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_ingest_url_http_mode_success():
    """kb_ingest_url in HTTP mode calls backend.ingest_url and formats the result."""
    from personal_kb.tools.kb_ingest_url import register_kb_ingest_url

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "path": "https://example.com",
                "action": "ingested",
                "reason": None,
                "entry_count": 2,
                "entry_ids": ["kb-00001", "kb-00002"],
                "summary": "Example page.",
                "chunks_processed": 1,
                "chunks_skipped": 0,
                "chunks_flagged": 0,
            },
        )

    kb_ingest_url = _register(register_kb_ingest_url)
    ctx = _make_ctx(handler)
    result = await kb_ingest_url(url="https://example.com", ctx=ctx)
    assert "ingested" in result
    assert "2 entries" in result


@pytest.mark.asyncio
async def test_kb_ingest_url_http_mode_empty_url():
    """kb_ingest_url returns error for empty URL before any backend call."""
    from personal_kb.tools.kb_ingest_url import register_kb_ingest_url

    called = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={})

    kb_ingest_url = _register(register_kb_ingest_url)
    ctx = _make_ctx(handler)
    result = await kb_ingest_url(url="", ctx=ctx)
    assert "Error" in result
    assert not called


@pytest.mark.asyncio
async def test_kb_ingest_url_http_mode_backend_error():
    """BackendHttpError from ingest_url is mapped to an error string."""
    from personal_kb.tools.kb_ingest_url import register_kb_ingest_url

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Invalid URL"})

    kb_ingest_url = _register(register_kb_ingest_url)
    ctx = _make_ctx(handler)
    result = await kb_ingest_url(url="https://example.com", ctx=ctx)
    assert "Error" in result


# ---------------------------------------------------------------------------
# kb_bulk_update — HTTP-mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_bulk_update_http_mode_success():
    """kb_bulk_update in HTTP mode calls backend.bulk_update and formats the result."""
    from personal_kb.tools.kb_bulk_update import register_kb_bulk_update

    before = _ENTRY_JSON
    after = {**_ENTRY_JSON, "project_ref": "new-proj", "version": 2}

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"before": before, "after": after}]})

    kb_bulk_update = _register(register_kb_bulk_update)
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"project_ref": "my-project"},
        updates={"project_ref": "new-proj"},
        dry_run=False,
        ctx=ctx,
    )
    assert result  # non-empty output
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_kb_bulk_update_http_mode_no_filters():
    """kb_bulk_update returns an error when no filters are provided."""
    from personal_kb.tools.kb_bulk_update import register_kb_bulk_update

    called = []

    def handler(req: httpx.Request) -> httpx.Response:
        called.append(True)
        return httpx.Response(200, json={})

    kb_bulk_update = _register(register_kb_bulk_update)
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(filters={}, updates={"project_ref": "x"}, ctx=ctx)
    assert "Error" in result
    assert not called


@pytest.mark.asyncio
async def test_kb_bulk_update_http_mode_backend_error():
    """BackendHttpError from bulk_update is mapped to an error string."""
    from personal_kb.tools.kb_bulk_update import register_kb_bulk_update

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Admin only"})

    kb_bulk_update = _register(register_kb_bulk_update)
    ctx = _make_ctx(handler)
    result = await kb_bulk_update(
        filters={"project_ref": "x"},
        updates={"project_ref": "y"},
        dry_run=False,
        ctx=ctx,
    )
    assert "Error" in result
    assert "admin" in result.lower()
