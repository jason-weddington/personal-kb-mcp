"""Parity between the stdio MCP server and kb-service's /mcp (B15/B16).

During the stdio overlap every MCP tool exists twice: ``src/personal_kb/tools/``
and ``packages/kb-service/src/kb_service/mcp_server/tools/``. These tests pin
the two together — schemas and instructions (B15) and rendered outputs for
the same call sequence (B16) — so a change to one twin fails until the other
follows.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import httpx
import pytest

if TYPE_CHECKING:
    from pathlib import Path

KB_INGEST_BULLET = (
    "- kb_ingest: Intelligent extraction from local files. An LLM reads the source "
    "and creates multiple properly structured KB entries (decisions, patterns, facts). "
    "Deduplicates against existing entries — safe to ingest overlapping files. "
    "Accepts file paths, directories, glob patterns (e.g. *.md, docs/**/*.txt).\n"
)

_SCHEMA_ENV_VARS = ("KB_INSTANCE_ROLE", "KB_MANAGER", "KB_CONTRIBUTOR", "KB_AUTH_MODE")

ENV_SETS: list[dict[str, str]] = [
    {},
    {"KB_INSTANCE_ROLE": "team", "KB_MANAGER": "TRUE", "KB_CONTRIBUTOR": "alice"},
]


async def _tool_table(server: Any) -> dict[str, tuple[Any, ...]]:
    return {
        t.name: (
            t.description,
            t.parameters,
            t.output_schema,
            t.annotations,
            t.title,
            t.tags,
            t.meta,
        )
        for t in await server.list_tools()
    }


@pytest.mark.parametrize("env", ENV_SETS, ids=["default", "team-manager"])
async def test_schema_and_instructions_parity(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    for var in _SCHEMA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    from kb_service.mcp_server.server import create_mcp_server

    from personal_kb.server import _get_tool_prefix, create_server

    prefix = _get_tool_prefix()
    stdio = create_server()
    http = create_mcp_server()

    stdio_tools = await _tool_table(stdio)
    http_tools = await _tool_table(http)
    assert f"{prefix}ingest" in stdio_tools
    stdio_tools.pop(f"{prefix}ingest")
    assert sorted(http_tools) == sorted(stdio_tools)
    for name, row in stdio_tools.items():
        assert http_tools[name] == row, name

    bullet = KB_INGEST_BULLET.replace("- kb_ingest:", f"- {prefix}ingest:")
    assert stdio.instructions is not None
    assert bullet in stdio.instructions
    assert http.instructions == stdio.instructions.replace(bullet, "")
    assert f"- {prefix}ingest:" not in http.instructions


# ═════════════════════════════════════════════════════════════════════════════
# B16 — output parity over one ordered call sequence
# ═════════════════════════════════════════════════════════════════════════════

_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?")

_PARITY_DELETE = (
    "ANTHROPIC_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "KB_DATABASE_URL",
    "KB_SERVICE_DATABASE_URL",
    "KB_SURPRISE_CAPTURE",
    "KB_INSTANCE_ROLE",
    "KB_SERVICE_PUBLIC_URL",
    "KB_SKIP_SAFETY",
)
_PARITY_SET = {
    "KB_AUTH_MODE": "none",
    "KB_OLLAMA_URL": "http://127.0.0.1:9",
    "KB_EXTRACTION_PROVIDER": "ollama",
    "KB_QUERY_PROVIDER": "ollama",
    "KB_MANAGER": "TRUE",
    "KB_CONTRIBUTOR": "parity",
    "PERSONAL_KB_URL": "http://testserver",
}

F: dict[str, Any] = {
    "long_title": "Parity long title",
    "knowledge_details": "Parity details for the HTTP MCP parity test.",
    "entry_type": "factual_reference",
    "project_ref": "parity",
    "tags": ["parity"],
}

CALLS: list[tuple[str, dict[str, Any]]] = [
    ("kb_store", {"short_title": "Parity entry", **F, "supersedes": "none"}),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "knowledge_details": "Parity updated.",
            "supersedes": "none",
        },
    ),
    ("kb_store", {"short_title": "Parity two", **F, "supersedes": ["kb-1"]}),
    ("kb_get", {"entry_id": ["kb-00001"]}),
    ("kb_get", {"entry_id": ["kb-99999"]}),
    ("kb_search", {"query": "Parity"}),
    ("kb_search", {"query": "Parity", "contributor": "x"}),
    ("kb_preflight", {"project_ref": "parity"}),
    ("kb_list_projects", {}),
    ("kb_list_contributors", {}),
    ("kb_list_teams", {}),
    ("kb_ask", {"question": "parity", "strategy": "timeline"}),
    ("kb_ask", {"question": "parity"}),
    ("kb_summarize", {"question": "parity"}),
    ("kb_explore", {}),
    ("kb_feedback", {"feedback_type": "friction", "detail": "parity"}),
    (
        "kb_store_batch",
        {
            "entries": [
                {"short_title": "Batch one", **F, "supersedes": "none"},
                {"short_title": "Batch two", **F, "supersedes": "none"},
            ]
        },
    ),
    ("kb_map_eligibility", {}),
    (
        "kb_map_eligibility_override",
        {"project_ref": "parity", "eligible": True, "reason": "parity test"},
    ),
    ("kb_map_eligibility_override", {"project_ref": "parity", "clear": True}),
    (
        "kb_bulk_update",
        {
            "filters": {"project_ref": "parity"},
            "updates": {"tags": ["parity", "bulk"]},
            "dry_run": True,
        },
    ),
    ("kb_maintain", {"action": "reconcile_supersession"}),
    (
        "kb_store",
        {
            "deactivate_entry_id": "kb-00001",
            "change_reason": "parity",
            "supersedes": "none",
        },
    ),
    ("kb_maintain", {"action": "reactivate", "entry_id": "kb-00001"}),
    (
        "kb_ingest_url",
        {
            "url": "file:///parity.md",
            "content": "# Parity\n\nParity ingest content.",
            "dry_run": True,
        },
    ),
]


def _norm(text: str) -> str:
    return _TS_RE.sub("<TS>", text)


def _parity_env(monkeypatch: pytest.MonkeyPatch, db_dir: Path | None) -> None:
    for var in _PARITY_DELETE:
        monkeypatch.delenv(var, raising=False)
    for key, value in _PARITY_SET.items():
        monkeypatch.setenv(key, value)
    if db_dir is not None:
        db_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("KB_DB_PATH", str(db_dir / "knowledge.db"))


def _stdio_tools() -> dict[str, Any]:
    """Capture every stdio tool function except kb_ingest, keyed by name."""
    from personal_kb.tools import (
        kb_ask,
        kb_bulk_update,
        kb_explore,
        kb_feedback,
        kb_get,
        kb_ingest_url,
        kb_list,
        kb_maintain,
        kb_map_eligibility,
        kb_preflight,
        kb_search,
        kb_store,
        kb_store_batch,
        kb_summarize,
    )

    tools: dict[str, Any] = {}

    def capture(**_kw: Any) -> Any:
        def decorator(fn: Any) -> Any:
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = capture
    for register in (
        kb_store.register_kb_store,
        kb_store_batch.register_kb_store_batch,
        kb_search.register_kb_search,
        kb_get.register_kb_get,
        kb_ask.register_kb_ask,
        kb_summarize.register_kb_summarize,
        kb_ingest_url.register_kb_ingest_url,
        kb_feedback.register_kb_feedback,
        kb_preflight.register_kb_preflight,
        kb_explore.register_kb_explore,
        kb_map_eligibility.register_kb_map_eligibility,
        kb_map_eligibility.register_kb_map_eligibility_override,
        kb_maintain.register_kb_maintain,
        kb_bulk_update.register_kb_bulk_update,
        kb_list.register_kb_list_projects,
        kb_list.register_kb_list_contributors,
        kb_list.register_kb_list_teams,
    ):
        register(mcp)
    return tools


def _stdio_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    from kb_core.models.entry import EntryType

    out = dict(kwargs)
    if isinstance(out.get("entry_type"), str):
        out["entry_type"] = EntryType(out["entry_type"])
    return out


async def _channel_a(
    app: Any, calls: list[tuple[str, dict[str, Any]]], transport: httpx.AsyncBaseTransport
) -> list[str]:
    from personal_kb.backend.http import HttpBackend

    tools = _stdio_tools()
    backend = HttpBackend(base_url="http://testserver", api_key="local-no-auth")
    backend._client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": "Bearer local-no-auth"},
    )
    ctx = MagicMock()
    ctx.lifespan_context = {"backend": backend}
    out: list[str] = []
    try:
        for name, kwargs in calls:
            out.append(await tools[name](**_stdio_kwargs(kwargs), ctx=ctx))
    finally:
        await backend.close()
    return out


async def _channel_b(app: Any, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    out: list[str] = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        for i, (name, kwargs) in enumerate(calls):
            resp = await client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": kwargs},
                },
                headers={"Accept": "application/json, text/event-stream"},
            )
            assert resp.status_code == 200, resp.text
            result = resp.json()["result"]
            assert not result.get("isError"), (name, result)
            out.append(result["content"][0]["text"])
    return out


async def test_output_parity_over_call_sequence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from kb_service.main import app

    _parity_env(monkeypatch, tmp_path / "a")
    async with app.router.lifespan_context(app):
        a = await _channel_a(
            app,
            CALLS,
            httpx.ASGITransport(app=app, raise_app_exceptions=False),
        )

    _parity_env(monkeypatch, tmp_path / "b")
    async with app.router.lifespan_context(app):
        b = await _channel_b(app, CALLS)

    assert len(a) == len(b) == len(CALLS) == 25
    for i, (name, _kwargs) in enumerate(CALLS):
        assert _norm(b[i]) == _norm(a[i]), f"call {i + 1} {name}"
    # Sanity: the sequence exercised real writes, not just errors.
    assert a[0].startswith("Created kb-00001 (v1)")
    assert b[16].startswith("Batch: 2 entries created")


WRITE_POLICY_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("kb_store", {"short_title": "Queued one", **F, "supersedes": "none"}),
    (
        "kb_store_batch",
        {
            "entries": [
                {"short_title": "Queued two", **F, "supersedes": "none"},
                {"short_title": "Queued three", **F, "supersedes": "none"},
            ]
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "knowledge_details": "x",
            "change_reason": "parity",
            "supersedes": "none",
        },
    ),
    (
        "kb_store",
        {
            "deactivate_entry_id": "kb-00001",
            "change_reason": "parity",
            "supersedes": "none",
        },
    ),
    ("kb_store", {"short_title": "Queued four", **F, "supersedes": ["kb-00001"]}),
]

WRITE_POLICY_EXPECTED = [
    "Queued as candidate 1 (write policy: headless surface). Not in the KB: capture"
    " mode is shadow, so it is recorded for audit only.",
    "Batch: 2 entries queued as candidates 2, 3 (write policy: headless surface)."
    " Not in the KB: capture mode is shadow, so they are recorded for audit only.",
    "Error: write policy: updating an entry requires an interactive surface; this"
    " request is headless. Store a new entry instead and it is queued for review.",
    "Error: write policy: deactivating an entry requires an interactive surface;"
    " this request is headless.",
    "Error: write policy: superseding entries requires an interactive surface; this"
    ' request is headless. Store the entry with supersedes "none" and it is queued'
    " for review.",
]


async def test_write_policy_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from kb_service.main import app

    def _env(db_dir: Path) -> None:
        _parity_env(monkeypatch, db_dir)
        monkeypatch.setenv("KB_WRITE_POLICY_DEFAULT_SURFACE", "headless")
        monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")

    _env(tmp_path / "a")
    async with app.router.lifespan_context(app):
        a = await _channel_a(
            app,
            WRITE_POLICY_CALLS,
            httpx.ASGITransport(app=app, raise_app_exceptions=False),
        )

    _env(tmp_path / "b")
    async with app.router.lifespan_context(app):
        b = await _channel_b(app, WRITE_POLICY_CALLS)

    assert a == b == WRITE_POLICY_EXPECTED


# ─── client-side rejection table ─────────────────────────────────────────────
#
# One row per client-side rejection branch reachable without any backend call.
# Stdio source lines (src/personal_kb/tools/, at the time of writing):
#   kb_store.py           create: missing_field 487, supersedes empty/bad 494,
#                         hints conflict 497, superseded_by 501,
#                         distinct_from 507, sensitivity 511, secrets 516,
#                         orphan map 528, ttl 533; update: change_reason 400,
#                         supersedes 403, hints conflict 406, superseded_by 410,
#                         distinct_from 417, sensitivity 423, secrets 428,
#                         ttl 434; deactivate: change_reason 352,
#                         supersedes 359, superseded_by 369, distinct_from 374
#   kb_store_batch.py     max batch 35, empty 38, missing fields 44,
#                         supersedes/hints/distinct_from 67, sensitivity 76,
#                         secrets 86, all-TTL-invalid (no request) 106
#   kb_maintain.py (HTTP) unknown action 124, deactivate entry_id 133,
#                         change_reason 135, superseded_by 138,
#                         reactivate entry_id 150, unsupported action 169
#   kb_bulk_update.py     no filters 99, no updates 102
#   kb_map_eligibility.py override: project_ref 295, clear+args 307,
#                         eligible 317, reason 320, reason length 325
#   (kb_map_eligibility has no pre-request rejection: its project_ref filter
#   runs on the fetched rows.)

_LEAKY_DETAILS = 'password = "hunter2"'

REJECTIONS: list[tuple[str, dict[str, Any]]] = [
    # kb_store — create
    ("kb_store", {"short_title": "", **F, "supersedes": "none"}),
    ("kb_store", {"short_title": "s", **F, "supersedes": []}),
    ("kb_store", {"short_title": "s", **F, "supersedes": ["kb-1"]}),
    (
        "kb_store",
        {"short_title": "s", **F, "supersedes": "none", "hints": {"supersedes": "kb-00002"}},
    ),
    ("kb_store", {"short_title": "s", **F, "supersedes": "none", "superseded_by": "kb-00002"}),
    ("kb_store", {"short_title": "s", **F, "supersedes": "none", "distinct_from": ["x"]}),
    ("kb_store", {"short_title": "s", **F, "supersedes": "none", "sensitivity": "top"}),
    (
        "kb_store",
        {"short_title": "s", **F, "knowledge_details": _LEAKY_DETAILS, "supersedes": "none"},
    ),
    (
        "kb_store",
        {
            "short_title": "s",
            **F,
            "entry_type": "mental_map",
            "knowledge_details": "no pointers",
            "supersedes": "none",
        },
    ),
    ("kb_store", {"short_title": "s", **F, "supersedes": "none", "ttl": "forever"}),
    # kb_store — update
    ("kb_store", {"update_entry_id": "kb-00001", "supersedes": "none"}),
    ("kb_store", {"update_entry_id": "kb-00001", "change_reason": "r", "supersedes": []}),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "hints": {"supersedes": ["kb-00002"]},
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "superseded_by": "kb-00002",
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "distinct_from": ["kb-00002"],
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "sensitivity": "nope",
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "knowledge_details": _LEAKY_DETAILS,
        },
    ),
    (
        "kb_store",
        {
            "update_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "ttl": "later",
        },
    ),
    # kb_store — deactivate
    ("kb_store", {"deactivate_entry_id": "kb-00001", "supersedes": "none"}),
    (
        "kb_store",
        {"deactivate_entry_id": "kb-00001", "change_reason": "r", "supersedes": ["kb-00002"]},
    ),
    (
        "kb_store",
        {
            "deactivate_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "superseded_by": "bogus",
        },
    ),
    (
        "kb_store",
        {
            "deactivate_entry_id": "kb-00001",
            "change_reason": "r",
            "supersedes": "none",
            "distinct_from": ["kb-00002"],
        },
    ),
    # kb_store_batch
    (
        "kb_store_batch",
        {"entries": [{"short_title": "s", **F, "supersedes": "none"}] * 11},
    ),
    ("kb_store_batch", {"entries": []}),
    ("kb_store_batch", {"entries": [{"short_title": "s"}]}),
    ("kb_store_batch", {"entries": [{"short_title": "s", **F, "supersedes": []}]}),
    (
        "kb_store_batch",
        {
            "entries": [
                {
                    "short_title": "s",
                    **F,
                    "supersedes": "none",
                    "hints": {"supersedes": ["kb-00002"]},
                }
            ]
        },
    ),
    (
        "kb_store_batch",
        {"entries": [{"short_title": "s", **F, "supersedes": "none", "sensitivity": "x"}]},
    ),
    (
        "kb_store_batch",
        {
            "entries": [
                {"short_title": "s", **F, "knowledge_details": _LEAKY_DETAILS, "supersedes": "none"}
            ]
        },
    ),
    (
        "kb_store_batch",
        {"entries": [{"short_title": "s", **F, "supersedes": "none", "ttl": "bad"}]},
    ),
    # kb_maintain (HTTP branch)
    ("kb_maintain", {"action": "explode"}),
    ("kb_maintain", {"action": "deactivate"}),
    ("kb_maintain", {"action": "deactivate", "entry_id": "kb-00001"}),
    (
        "kb_maintain",
        {
            "action": "deactivate",
            "entry_id": "kb-00001",
            "change_reason": "r",
            "superseded_by": "kb-1",
        },
    ),
    ("kb_maintain", {"action": "reactivate"}),
    ("kb_maintain", {"action": "stats"}),
    # kb_bulk_update
    ("kb_bulk_update", {"filters": {}, "updates": {"team": "t"}}),
    ("kb_bulk_update", {"filters": {"team": "t"}, "updates": {}}),
    # kb_map_eligibility_override
    (
        "kb_map_eligibility_override",
        {"project_ref": "bad ref", "eligible": True, "reason": "r"},
    ),
    (
        "kb_map_eligibility_override",
        {"project_ref": "p", "clear": True, "reason": "r"},
    ),
    ("kb_map_eligibility_override", {"project_ref": "p", "reason": "r"}),
    ("kb_map_eligibility_override", {"project_ref": "p", "eligible": False}),
    (
        "kb_map_eligibility_override",
        {"project_ref": "p", "eligible": True, "reason": "x" * 2001},
    ),
]


def _refuse(request: httpx.Request) -> httpx.Response:
    pytest.fail(f"client-side rejection made a request: {request.method} {request.url}")


async def test_rejection_table_has_enough_rows() -> None:
    assert len(REJECTIONS) >= 25


@pytest.mark.parametrize(
    ("name", "kwargs"),
    REJECTIONS,
    ids=[f"{i:02d}-{n}" for i, (n, _k) in enumerate(REJECTIONS)],
)
async def test_client_side_rejection_parity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, kwargs: dict[str, Any]
) -> None:
    from kb_service.main import app

    _parity_env(monkeypatch, tmp_path / "rej")
    async with app.router.lifespan_context(app):
        [a] = await _channel_a(app, [(name, kwargs)], httpx.MockTransport(_refuse))
        [b] = await _channel_b(app, [(name, kwargs)])
    assert b == a
    assert a.startswith(("Error", "Unknown action", "Batch failed"))
    json.dumps(kwargs)  # rows stay JSON-serialisable for the /mcp channel
