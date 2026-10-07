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


# ---------------------------------------------------------------------------
# kb_map_eligibility / kb_map_eligibility_override — HTTP-mode
# ---------------------------------------------------------------------------

# Two-project fixture with BOTH rows fully pinned.  The full-render test below
# asserts exact full-line equality on splitlines(), so any drift in the render
# format or in these literals fails loudly.

_ROW_A: dict[str, Any] = {
    "evidence": {
        "project_ref": "cleanr",
        "mappable": 10,
        "ingested": 5,
        "hand_authored": 5,
        "maps": 0,
        "top_prefix": "cleanr",
        "top_prefix_share": 0.1,
        "is_ingest_corpus": False,
        "is_too_thin": False,
        "is_journal": False,
        "computed_eligible": True,
    },
    "override": None,
    "effective_eligible": True,
    "decided_by": "computed",
    "orphaned": False,
}

_ROW_B: dict[str, Any] = {
    "evidence": {
        "project_ref": "dispatch-performance-log",
        "mappable": 624,
        "ingested": 0,
        "hand_authored": 624,
        "maps": 0,
        "top_prefix": "Run",
        "top_prefix_share": 0.9792,
        "is_ingest_corpus": False,
        "is_too_thin": False,
        "is_journal": True,
        "computed_eligible": False,
    },
    "override": None,
    "effective_eligible": False,
    "decided_by": "computed",
    "orphaned": False,
}


@pytest.mark.asyncio
async def test_map_eligibility_full_render_exact_lines():
    """The full render is asserted as FULL-LINE equality on splitlines()."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"projects": [_ROW_A, _ROW_B]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    lines = out.splitlines()
    assert len(lines) == 6
    assert lines[0] == (
        "Map eligibility — 2 projects | eligible 1 (computed 1, override 0) "
        "| ineligible 1 | eligible & unmapped 1 | evidence flags: too_thin 0, "
        "ingest_corpus 0, journal 1 | override rows 0"
    )
    assert lines[1] == ""
    assert lines[2] == (
        "cleanr | ELIGIBLE (computed) | mappable 10 = hand 5 + ingested 5 "
        "| maps 0 | top 'cleanr' 10.0% | flags: none"
    )
    assert lines[3] == (
        "dispatch-performance-log | INELIGIBLE (computed) "
        "| mappable 624 = hand 624 + ingested 0 | maps 0 | top 'Run' 97.9% "
        "| flags: journal"
    )
    assert lines[4] == ""
    assert lines[5] == (
        "Thresholds: too_thin when hand_authored < 5; journal when mappable >= 20 "
        "and top prefix share >= 60%. To change a verdict: "
        'kb_map_eligibility_override(project_ref=..., eligible=..., reason="...").'
    )


@pytest.mark.asyncio
async def test_map_eligibility_team_prefix_cross_reference():
    """The team-prefixed footer cross-references team_kb_map_eligibility_override."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"projects": [_ROW_A]})

    fn = _register(register_kb_map_eligibility, prefix="team_kb_")
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "team_kb_map_eligibility_override(" in out


@pytest.mark.asyncio
async def test_map_eligibility_project_ref_filter():
    """project_ref filters client-side to exactly one row."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"projects": [_ROW_A, _ROW_B]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="dispatch-performance-log", ctx=ctx)
    assert "cleanr" not in out
    assert "dispatch-performance-log" in out


@pytest.mark.asyncio
async def test_map_eligibility_unknown_project_ref():
    """An unknown project_ref points at kb_list_projects."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"projects": [_ROW_A, _ROW_B]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="nope", ctx=ctx)
    assert "No map-eligibility row" in out
    assert "kb_list_projects" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_row_rendered():
    """An override row renders its verdict, reason and set_by."""

    def handler(req: httpx.Request) -> httpx.Response:
        row = {
            **_ROW_A,
            "override": {
                "project_ref": "cleanr",
                "eligible": False,
                "reason": "dated session journal",
                "set_by": "jason@example.com",
                "set_at": "2026-09-19T12:00:00+00:00",
            },
            "effective_eligible": False,
            "decided_by": "override",
        }
        return httpx.Response(200, json={"projects": [row]})

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "override: ineligible" in out
    assert "computed said eligible" in out
    assert "set_by jason@example.com" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_missing_set_by_is_loud(caplog):
    """A null set_by renders the MISSING wording and logs a WARNING."""
    import logging

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        row = {
            **_ROW_A,
            "override": {
                "project_ref": "cleanr",
                "eligible": False,
                "reason": "dated session journal",
                "set_by": None,
                "set_at": "2026-09-19T12:00:00+00:00",
            },
            "effective_eligible": False,
            "decided_by": "override",
        }
        return httpx.Response(200, json={"projects": [row]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    with caplog.at_level(logging.WARNING):
        out = await fn(ctx=ctx)
    assert "MISSING — service recorded no identity" in out
    assert "map-eligibility" in caplog.text


@pytest.mark.asyncio
async def test_map_eligibility_orphaned_row_rendered():
    """An orphaned row appends the ORPHANED warning."""

    def handler(req: httpx.Request) -> httpx.Response:
        row = {
            **_ROW_A,
            "evidence": {**_ROW_A["evidence"], "mappable": 0},
            "effective_eligible": False,
            "orphaned": True,
        }
        return httpx.Response(200, json={"projects": [row]})

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "ORPHANED" in out


@pytest.mark.asyncio
async def test_map_eligibility_empty_projects():
    """An empty projects list renders the empty-corpus message."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"projects": []})

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "No project_refs returned" in out


@pytest.mark.asyncio
async def test_map_eligibility_row_missing_maps_field_is_loud(caplog):
    """A verdict row missing an evidence field renders the payload error."""
    import logging

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        row = {**_ROW_A, "evidence": {k: v for k, v in _ROW_A["evidence"].items() if k != "maps"}}
        return httpx.Response(200, json={"projects": [row]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    with caplog.at_level(logging.WARNING):
        out = await fn(ctx=ctx)
    assert "unexpected map-eligibility payload" in out
    assert "'maps'" in out
    assert "map-eligibility" in caplog.text


@pytest.mark.asyncio
async def test_map_eligibility_row_missing_evidence_key_is_loud():
    """A row with no evidence key at all trips the filter's strict subscript."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        row = {k: v for k, v in _ROW_A.items() if k != "evidence"}
        return httpx.Response(200, json={"projects": [row]})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="x", ctx=ctx)
    assert "unexpected map-eligibility payload" in out
    assert "'evidence'" in out


@pytest.mark.asyncio
async def test_map_eligibility_missing_projects_key_is_loud():
    """A 200 with no projects key renders the payload error naming 'projects'."""
    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "unexpected map-eligibility payload" in out
    assert "'projects'" in out


@pytest.mark.asyncio
async def test_map_eligibility_404_version_skew(caplog):
    """A 404 renders the version-skew message and logs a WARNING."""
    import logging

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Not Found"})

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    with caplog.at_level(logging.WARNING):
        out = await fn(ctx=ctx)
    assert "has no map-eligibility endpoint (404)" in out
    assert "map-eligibility" in caplog.text


@pytest.mark.asyncio
async def test_map_eligibility_403_admin():
    """A 403 renders the admin-privileges error."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "Admin only"})

    from personal_kb.tools.kb_map_eligibility import register_kb_map_eligibility

    fn = _register(register_kb_map_eligibility)
    ctx = _make_ctx(handler)
    out = await fn(ctx=ctx)
    assert "admin privileges required (403)" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_set_success():
    """A set POST sends exactly three keys, once, and renders the stored verdict."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        verdict = {
            **_ROW_B,
            "override": {
                "project_ref": "dispatch-performance-log",
                "eligible": False,
                "reason": "dated session journal",
                "set_by": "jason@example.com",
                "set_at": "2026-09-19T12:00:00+00:00",
            },
            "effective_eligible": False,
            "decided_by": "override",
        }
        return httpx.Response(200, json={"changed": True, "verdict": verdict})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(
        project_ref="dispatch-performance-log",
        eligible=False,
        reason="dated session journal",
        ctx=ctx,
    )
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/kb/map-eligibility/override"
    assert set(json.loads(requests[0].content)) == {"project_ref", "eligible", "reason"}
    assert "Override set" in out
    assert "INELIGIBLE" in out
    assert "(override)" in out
    assert "dated session journal" in out
    assert "has no mappable entries" not in out
    assert "changed" not in out


@pytest.mark.asyncio
async def test_map_eligibility_override_set_orphaned():
    """An orphaned write appends the spelling warning without a follow-up GET."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        verdict = {
            **_ROW_B,
            "orphaned": True,
            "effective_eligible": False,
        }
        return httpx.Response(200, json={"changed": True, "verdict": verdict})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="agent-gtd-dev", eligible=False, reason="r", ctx=ctx)
    assert len(requests) == 1
    assert "has no mappable entries" in out
    assert "list_projects" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_set_missing_set_by(caplog):
    """A set response with a null set_by warns about audit-trail attribution."""
    import logging

    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        verdict = {
            **_ROW_B,
            "override": {
                "project_ref": "dispatch-performance-log",
                "eligible": False,
                "reason": "dated session journal",
                "set_by": None,
                "set_at": "2026-09-19T12:00:00+00:00",
            },
            "effective_eligible": False,
            "decided_by": "override",
        }
        return httpx.Response(200, json={"changed": True, "verdict": verdict})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    with caplog.at_level(logging.WARNING):
        out = await fn(project_ref="dispatch-performance-log", eligible=False, reason="r", ctx=ctx)
    assert "no set_by" in out
    assert "map-eligibility" in caplog.text


@pytest.mark.asyncio
async def test_map_eligibility_override_clear_success():
    """clear=True POSTs to /override/clear and renders the revert message."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"changed": True, "verdict": None})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="harness-design", clear=True, ctx=ctx)
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/kb/map-eligibility/override/clear"
    assert json.loads(requests[0].content) == {"project_ref": "harness-design"}
    assert "reverts to the computed verdict" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_clear_noop():
    """A no-op clear (changed=false) renders 'nothing to clear'."""

    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"changed": False, "verdict": None})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="harness-design", clear=True, ctx=ctx)
    assert "nothing to clear" in out


@pytest.mark.asyncio
async def test_map_eligibility_override_validation_errors_no_request():
    """The five client-side validation errors return exact strings and send nothing."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"changed": True, "verdict": None})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)

    bad_ref = (
        "Error: project_ref must be non-empty and contain only letters, digits, '_', '.' or '-'."
    )
    assert await fn(project_ref="", eligible=True, reason="r", ctx=ctx) == bad_ref
    assert await fn(project_ref="a/b c", eligible=True, reason="r", ctx=ctx) == bad_ref

    assert await fn(project_ref="p", clear=True, eligible=True, ctx=ctx) == (
        "Error: clear=True takes only project_ref — omit eligible and reason."
    )
    assert await fn(project_ref="p", ctx=ctx) == (
        "Error: eligible is required (True or False) unless clear=True."
    )
    assert await fn(project_ref="p", eligible=True, reason="   ", ctx=ctx) == (
        "Error: reason is required — the override is a human verdict and the reason "
        "is its audit trail."
    )
    assert requests == []


@pytest.mark.asyncio
async def test_map_eligibility_override_reason_too_long_no_request():
    """A reason over the service's 2000-char cap is rejected client-side."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"changed": True, "verdict": None})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="p", eligible=True, reason="x" * 2001, ctx=ctx)
    assert "Error" in out
    assert "2000" in out
    assert requests == []


def test_map_eligibility_description_content():
    """The read tool description carries every contracted field and predicate."""
    from personal_kb.tools.kb_map_eligibility import _eligibility_description

    d = _eligibility_description("kb_")
    for name in (
        "project_ref",
        "mappable",
        "ingested",
        "hand_authored",
        "maps",
        "top_prefix",
        "top_prefix_share",
        "is_ingest_corpus",
        "is_too_thin",
        "is_journal",
        "computed_eligible",
    ):
        assert name in d
    assert "is_too_thin when hand_authored < 5" in d
    assert "mappable >= 20" in d
    assert "60%" in d
    assert "kb_map_eligibility_override" in d
    assert "team_kb_map_eligibility_override" in _eligibility_description("team_kb_")


def test_map_eligibility_override_description_content():
    """The write tool description carries the worked examples and the loop warning."""
    from personal_kb.tools.kb_map_eligibility import _override_description

    d = _override_description("kb_")
    assert "harness-design" in d
    assert "threat-intel" in d
    assert "clear=True" in d
    assert "never writes it" in d
    assert "team_kb_map_eligibility" in _override_description("team_kb_")


@pytest.mark.asyncio
async def test_map_eligibility_override_null_verdict_invariant_breach(caplog):
    """A set response with verdict=null renders the invariant-breach message."""
    import logging

    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json={"changed": True, "verdict": None})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    with caplog.at_level(logging.WARNING):
        out = await fn(project_ref="p", eligible=True, reason="r", ctx=ctx)
    assert "no resolved verdict" in out
    assert "invariant breach" in out
    assert "map-eligibility" in caplog.text
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_map_eligibility_override_verdict_missing_orphaned_is_loud():
    """A verdict dict missing 'orphaned' trips the write tool's KeyError chain."""
    from personal_kb.tools.kb_map_eligibility import (
        register_kb_map_eligibility_override,
    )

    def handler(req: httpx.Request) -> httpx.Response:
        verdict = {k: v for k, v in _ROW_B.items() if k != "orphaned"}
        return httpx.Response(200, json={"changed": True, "verdict": verdict})

    fn = _register(register_kb_map_eligibility_override)
    ctx = _make_ctx(handler)
    out = await fn(project_ref="p", eligible=True, reason="r", ctx=ctx)
    assert "unexpected map-eligibility payload" in out
    assert "'orphaned'" in out


@pytest.mark.asyncio
@pytest.mark.parametrize("passed,expected", [(True, True), (None, False)])
async def test_kb_search_forwards_include_superseded(passed, expected):
    """kb_search posts include_superseded (default False) to the service."""
    from personal_kb.tools.kb_search import register_kb_search

    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/kb/search":
            bodies.append(json.loads(req.content))
            return httpx.Response(200, json={"results": [], "filtered_count": 0})
        return httpx.Response(200, json={"neighbors": []})

    kb_search = _register(register_kb_search)
    ctx = _make_ctx(handler)
    kwargs = {} if passed is None else {"include_superseded": passed}
    await kb_search(query="python testing", limit=5, ctx=ctx, **kwargs)
    assert bodies[-1]["include_superseded"] is expected
