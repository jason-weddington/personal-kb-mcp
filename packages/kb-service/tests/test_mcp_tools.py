"""Every /mcp tool: success, error and admin paths over POST /mcp (B14c).

The second half ports the HTTP-mode branch tests of the stdio tools
(``tests/tools/test_http_mode_tools.py`` and the HTTP-mode cases of the
kb_store / kb_store_batch / kb_maintain / kb_bulk_update suites): tool
functions are called directly with ``context.backend_for_request`` patched to
return a stub implementing the InProcessBackend method signatures.
"""

import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from kb_core.models.entry import EntryType, KnowledgeEntry

from kb_service.mcp_server import context
from kb_service.mcp_server.backend import QueuedBatch, QueuedStore
from kb_service.mcp_server.errors import BackendHttpError
from kb_service.mcp_server.tools import (
    kb_ask,
    kb_bulk_update,
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
from kb_service.routes import (
    ingest_routes,
    kb_read_routes,
    kb_routes,
    kb_write_routes,
    map_eligibility_routes,
    query_routes,
)
from tests.conftest import FakeKnowledgeBase, make_entry, make_search_result
from tests.test_mcp_endpoint import ADMIN, USER, rpc

# ─── over-the-wire helpers ───────────────────────────────────────────────────


def call(
    client: TestClient,
    name: str,
    arguments: dict[str, Any] | None = None,
    headers: dict[str, str] = ADMIN,
) -> dict[str, Any]:
    resp = client.post(
        "/mcp",
        json=rpc("tools/call", {"name": name, "arguments": arguments or {}}),
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    result: dict[str, Any] = resp.json()["result"]
    return result


def text(result: dict[str, Any]) -> str:
    out: str = result["content"][0]["text"]
    return out


STORE_ARGS = {
    "short_title": "s",
    "long_title": "l",
    "knowledge_details": "d",
    "supersedes": "none",
}

# ─── success paths ───────────────────────────────────────────────────────────


def test_store_create_update_deactivate(
    mcp_client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    fake_kb.entries["kb-00001"] = make_entry()
    created = text(call(mcp_client, "kb_store", STORE_ARGS))
    assert created.startswith("Created kb-00001 (v1)\n[kb-00001] factual_reference")
    updated = text(
        call(
            mcp_client,
            "kb_store",
            {
                "update_entry_id": "kb-00001",
                "knowledge_details": "d2",
                "change_reason": "r",
                "supersedes": "none",
            },
        )
    )
    assert updated.startswith("Updated kb-00001")
    gone = text(
        call(
            mcp_client,
            "kb_store",
            {
                "deactivate_entry_id": "kb-00001",
                "change_reason": "r",
                "supersedes": "none",
            },
        )
    )
    assert gone == "Deactivated entry kb-00001: Fake entry (r)"


def test_store_batch_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_store_batch", {"entries": [STORE_ARGS]}))
    assert out.startswith("Batch: 1 entries created")
    assert "Created kb-00001 (v1)" in out


@pytest.mark.parametrize("fake_kb", ["one_result"], indirect=True)
def test_search_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_search", {"query": "x"}))
    assert "Example entry" in out


def test_get_success_and_not_found(
    mcp_client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    fake_kb.entries["kb-00001"] = make_entry()
    out = text(call(mcp_client, "kb_get", {"entry_id": ["kb-00001", "kb-00009"]}))
    assert "Fake details." in out
    assert "[kb-00009] not found" in out


def test_ask_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_ask", {"question": "q"}))
    assert out.startswith("[Agent: 3 tool calls]")
    assert "fake ask context" in out


def test_summarize_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_summarize", {"question": "q"}))
    assert out == "fake synthesized answer"


def test_ingest_url_success(mcp_client: TestClient) -> None:
    out = text(
        call(
            mcp_client,
            "kb_ingest_url",
            {"url": "https://e.com", "content": "hello", "dry_run": True},
        )
    )
    assert out == "[DRY RUN]   ingested: doc.md (1 entries) [kb-00002]"


def test_feedback_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_feedback", {"feedback_type": "friction"}))
    assert out == (
        "Feedback recorded (friction). Thank you — this helps improve the KB."
    )


def test_preflight_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_preflight", {"project_ref": "p"}))
    assert out == "preflight context"


def test_explore_uses_public_url(
    mcp_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert text(call(mcp_client, "kb_explore")) == (
        "KB explorer is hosted at http://testserver — open it in a browser."
    )
    monkeypatch.setenv("KB_SERVICE_PUBLIC_URL", "https://kb.example.com/")
    assert text(call(mcp_client, "kb_explore")) == (
        "KB explorer is hosted at https://kb.example.com — open it in a browser."
    )


def test_map_eligibility_success(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_map_eligibility"))
    assert out.startswith("No project_refs returned")


def test_map_eligibility_override_set_and_clear(mcp_client: TestClient) -> None:
    out = text(
        call(
            mcp_client,
            "kb_map_eligibility_override",
            {"project_ref": "p", "eligible": True, "reason": "r"},
        )
    )
    assert out.startswith("Override stored for p")
    out = text(
        call(
            mcp_client,
            "kb_map_eligibility_override",
            {"project_ref": "p", "clear": True},
        )
    )
    assert out.startswith("Override cleared — p")


def test_maintain_http_actions(
    mcp_client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    fake_kb.entries["kb-00001"] = make_entry()
    assert text(
        call(
            mcp_client, "kb_maintain", {"action": "reactivate", "entry_id": "kb-00001"}
        )
    ) == ("Reactivated entry kb-00001: Fake entry")
    assert text(
        call(mcp_client, "kb_maintain", {"action": "reconcile_supersession"})
    ) == ("Supersession reconcile: edges_added=0 set=0 cleared=0")
    assert text(
        call(
            mcp_client,
            "kb_maintain",
            {"action": "deactivate", "entry_id": "kb-00001", "change_reason": "r"},
        )
    ) == ("Deactivated entry kb-00001: Fake entry")
    assert text(call(mcp_client, "kb_maintain", {"action": "vacuum"})) == (
        "Error: action vacuum is not supported in HTTP mode"
        " — run it on the KB service host."
    )
    assert text(call(mcp_client, "kb_maintain", {"action": "nope"})).startswith(
        "Unknown action 'nope'."
    )


def test_bulk_update_success(
    mcp_client: TestClient, fake_kb: FakeKnowledgeBase
) -> None:
    out = text(
        call(
            mcp_client,
            "kb_bulk_update",
            {
                "filters": {"project_ref": "p"},
                "updates": {"tags_add": ["x"]},
                "dry_run": True,
            },
        )
    )
    assert out.startswith("DRY RUN — 1 entries would be updated:")


def test_list_tools_success(mcp_client: TestClient) -> None:
    assert text(call(mcp_client, "kb_list_projects")) == "No projects found."
    assert text(call(mcp_client, "kb_list_contributors")) == "No contributors found."
    assert text(call(mcp_client, "kb_list_teams")) == "No teams found."


# ─── error paths: handler raises HTTPException(503) ──────────────────────────

# (tool, arguments, module, handler attr)
CAUGHT: list[tuple[str, dict[str, Any], ModuleType, str]] = [
    ("kb_store", STORE_ARGS, kb_write_routes, "store"),
    ("kb_store_batch", {"entries": [STORE_ARGS]}, kb_write_routes, "store_batch"),
    ("kb_ask", {"question": "q"}, query_routes, "ask"),
    ("kb_summarize", {"question": "q"}, query_routes, "summarize"),
    ("kb_ingest_url", {"url": "https://e.com"}, ingest_routes, "ingest_url"),
    ("kb_preflight", {"project_ref": "p"}, kb_read_routes, "preflight"),
    (
        "kb_maintain",
        {"action": "deactivate", "entry_id": "kb-00001", "change_reason": "r"},
        kb_write_routes,
        "deactivate",
    ),
    (
        "kb_bulk_update",
        {"filters": {"project_ref": "p"}, "updates": {"tags": ["x"]}},
        kb_write_routes,
        "bulk_update",
    ),
    ("kb_map_eligibility", {}, map_eligibility_routes, "map_eligibility"),
    (
        "kb_map_eligibility_override",
        {"project_ref": "p", "eligible": True, "reason": "r"},
        map_eligibility_routes,
        "set_override",
    ),
    ("kb_list_projects", {}, kb_read_routes, "list_projects"),
    ("kb_list_contributors", {}, kb_read_routes, "list_contributors"),
    ("kb_list_teams", {}, kb_read_routes, "list_teams"),
]

UNCAUGHT: list[tuple[str, dict[str, Any], ModuleType, str]] = [
    ("kb_get", {"entry_id": "kb-00001"}, kb_read_routes, "get_entries"),
    ("kb_search", {"query": "x"}, kb_routes, "search"),
    ("kb_feedback", {"feedback_type": "friction"}, kb_write_routes, "feedback"),
]


@pytest.mark.parametrize(("tool", "args", "module", "attr"), CAUGHT)
def test_caught_backend_error_renders_error_text(
    mcp_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    args: dict[str, Any],
    module: ModuleType,
    attr: str,
) -> None:
    monkeypatch.setattr(module, attr, AsyncMock(side_effect=HTTPException(503, "boom")))
    result = call(mcp_client, tool, args)
    assert text(result) == "Error: KB service returned 503: boom"
    assert not result.get("isError")


@pytest.mark.parametrize(("tool", "args", "module", "attr"), UNCAUGHT)
def test_uncaught_backend_error_is_tool_error(
    mcp_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tool: str,
    args: dict[str, Any],
    module: ModuleType,
    attr: str,
) -> None:
    monkeypatch.setattr(module, attr, AsyncMock(side_effect=HTTPException(503, "boom")))
    result = call(mcp_client, tool, args)
    assert result["isError"] is True
    assert text(result) == (
        f"Error calling tool '{tool}': KB service returned 503: boom"
    )


# ─── admin-only tools as a non-admin key ─────────────────────────────────────

ADMIN_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("kb_maintain", {"action": "reactivate", "entry_id": "kb-00001"}),
    ("kb_maintain", {"action": "reconcile_supersession"}),
    (
        "kb_bulk_update",
        {"filters": {"project_ref": "p"}, "updates": {"tags": ["x"]}, "dry_run": True},
    ),
    ("kb_map_eligibility", {}),
    (
        "kb_map_eligibility_override",
        {"project_ref": "p", "eligible": True, "reason": "r"},
    ),
    ("kb_map_eligibility_override", {"project_ref": "p", "clear": True}),
]


@pytest.mark.parametrize(("tool", "args"), ADMIN_CALLS)
def test_admin_tools_refuse_non_admin_key(
    mcp_client: TestClient, tool: str, args: dict[str, Any]
) -> None:
    result = call(mcp_client, tool, args, headers=USER)
    assert text(result) == "Error: admin privileges required (403): Admin only"


def test_non_admin_store_succeeds(mcp_client: TestClient) -> None:
    out = text(call(mcp_client, "kb_store", STORE_ARGS, headers=USER))
    assert out.startswith("Created kb-00001")


# ═════════════════════════════════════════════════════════════════════════════
# Ported HTTP-mode branch tests (direct calls with a stub backend)
# ═════════════════════════════════════════════════════════════════════════════


def capture(register: Callable[..., None], prefix: str = "kb_") -> Any:
    """Register on a mock MCP and return the single captured tool function."""
    tools: dict[str, Any] = {}

    def _tool(**_kw: Any) -> Callable[[Any], Any]:
        def decorator(fn: Any) -> Any:
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp = MagicMock()
    mcp.tool = _tool
    register(mcp, prefix)
    return next(iter(tools.values()))


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """A stub backend returned by ``context.backend_for_request``."""
    backend = MagicMock()
    monkeypatch.setattr(context, "backend_for_request", lambda: backend)
    return backend


def entry(**kw: Any) -> KnowledgeEntry:
    base = make_entry()
    return base.model_copy(update=kw)


ERR_503 = BackendHttpError(503, "down")

# ─── kb_store: deactivate path ───────────────────────────────────────────────


@pytest.fixture
def store_fn() -> Any:
    return capture(kb_store.register_kb_store)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"change_reason": None},
            "Error: change_reason is required when updating or deactivating an "
            "entry: say what changed and why.",
        ),
        (
            {"change_reason": "r", "supersedes": ["kb-00002"]},
            "Error: supersedes does not apply to deactivate_entry_id; "
            'pass "none" and name the replacing entry with superseded_by.',
        ),
        (
            {"change_reason": "r", "superseded_by": "bogus"},
            "Error: superseded_by must be a kb-XXXXX id (got 'bogus').",
        ),
        (
            {"change_reason": "r", "distinct_from": ["kb-00002"]},
            "Error: distinct_from applies to create only.",
        ),
    ],
)
async def test_store_deactivate_rejections(
    store_fn: Any, stub: MagicMock, kwargs: dict[str, Any], expected: str
) -> None:
    args = {"supersedes": "none", "deactivate_entry_id": "kb-00001", **kwargs}
    assert await store_fn(**args) == expected
    stub.deactivate.assert_not_called()


async def test_store_deactivate_with_superseded_by(
    store_fn: Any, stub: MagicMock
) -> None:
    stub.deactivate = AsyncMock(return_value=entry())
    out = await store_fn(
        supersedes="none",
        deactivate_entry_id="kb-00001",
        change_reason="old",
        superseded_by="kb-00002",
    )
    assert out == "Deactivated entry kb-00001: Fake entry (old); superseded by kb-00002"
    stub.deactivate.assert_awaited_once_with(
        "kb-00001", change_reason="old", superseded_by="kb-00002"
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ERR_503, "Error: KB service returned 503: down"),
        (
            BackendHttpError(404, "Entry kb-00001 not found"),
            "Error: Entry kb-00001 not found",
        ),
        (RuntimeError("weird"), "Error: weird"),
    ],
)
async def test_store_deactivate_errors(
    store_fn: Any, stub: MagicMock, exc: Exception, expected: str
) -> None:
    stub.deactivate = AsyncMock(side_effect=exc)
    out = await store_fn(
        supersedes="none", deactivate_entry_id="kb-00001", change_reason="r"
    )
    assert out == expected


# ─── kb_store: update path ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected_prefix"),
    [
        ({"change_reason": " "}, "Error: change_reason is required"),
        ({"supersedes": []}, "Error: supersedes=[] is ambiguous"),
        ({"supersedes": ["nope"]}, "Error: supersedes must be a list"),
        (
            {"hints": {"supersedes": ["kb-00003"]}},
            'Error: supersedes="none" conflicts with hints.supersedes',
        ),
        ({"superseded_by": "kb-00002"}, "Error: superseded_by applies to"),
        ({"distinct_from": ["kb-00002"]}, "Error: distinct_from applies to create"),
        ({"sensitivity": "secret"}, 'Error: Invalid sensitivity "secret"'),
        ({"knowledge_details": 'password = "hunter2"'}, "Error: Potential secrets"),
        ({"ttl": "forever"}, "Error: "),
    ],
)
async def test_store_update_rejections(
    store_fn: Any, stub: MagicMock, kwargs: dict[str, Any], expected_prefix: str
) -> None:
    args = {
        "supersedes": "none",
        "update_entry_id": "kb-00001",
        "change_reason": "r",
        **kwargs,
    }
    out = await store_fn(**args)
    assert out.startswith(expected_prefix), out
    stub.store.assert_not_called()


async def test_store_update_success_with_supersedes(
    store_fn: Any, stub: MagicMock
) -> None:
    stub.store = AsyncMock(return_value=("updated", entry(version=2), ["kb-00002"]))
    out = await store_fn(
        supersedes=["kb-00002"],
        update_entry_id="kb-00001",
        change_reason="r",
        tags=["a"],
    )
    assert out.startswith("Updated kb-00001 (v2)")
    assert out.endswith("\nSupersedes: kb-00002")
    kwargs = stub.store.await_args.kwargs
    assert kwargs["update_entry_id"] == "kb-00001"
    assert kwargs["supersedes"] == ["kb-00002"]


async def test_store_update_mismatch_logs_warning(
    store_fn: Any, stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    stub.store = AsyncMock(return_value=("updated", entry(), None))
    caplog.set_level(logging.WARNING, logger="kb_service")
    await store_fn(
        supersedes=["kb-00002"], update_entry_id="kb-00001", change_reason="r"
    )
    assert "supersession-client mismatch op=update" in caplog.text


async def test_store_update_mental_map_advisories(
    store_fn: Any, stub: MagicMock
) -> None:
    stub.store = AsyncMock(
        return_value=("updated", entry(entry_type=EntryType.MENTAL_MAP), None)
    )
    out = await store_fn(
        supersedes="none",
        update_entry_id="kb-00001",
        change_reason="r",
        knowledge_details="see kb-00002 on port 8080",
    )
    assert out.startswith("Map lint (advisory): ")
    assert "\n\nUpdated kb-00001" in out


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ERR_503, "Error: KB service returned 503: down"),
        (RuntimeError("weird"), "Error: weird"),
    ],
)
async def test_store_update_errors(
    store_fn: Any, stub: MagicMock, exc: Exception, expected: str
) -> None:
    stub.store = AsyncMock(side_effect=exc)
    out = await store_fn(supersedes="none", update_entry_id="kb-1", change_reason="r")
    assert out == expected


# ─── kb_store: create path ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected_prefix"),
    [
        ({"short_title": ""}, "Error: short_title, long_title, and knowledge_details"),
        ({"supersedes": []}, "Error: supersedes=[] is ambiguous"),
        ({"supersedes": "kb-00002"}, "Error: supersedes must be a list"),
        (
            {"hints": {"supersedes": "kb-00003"}},
            'Error: supersedes="none" conflicts',
        ),
        ({"superseded_by": "kb-00002"}, "Error: superseded_by applies to"),
        ({"distinct_from": ["x"]}, "Error: distinct_from must be a list"),
        ({"sensitivity": "top"}, 'Error: Invalid sensitivity "top"'),
        ({"knowledge_details": 'password = "hunter2"'}, "Error: Potential secrets"),
        (
            {"entry_type": EntryType.MENTAL_MAP, "knowledge_details": "no pointers"},
            "Error: A mental_map entry requires at least one outbound pointer",
        ),
        ({"ttl": "soon"}, "Error: "),
    ],
)
async def test_store_create_rejections(
    store_fn: Any, stub: MagicMock, kwargs: dict[str, Any], expected_prefix: str
) -> None:
    args = {**STORE_ARGS, **kwargs}
    out = await store_fn(**args)
    assert out.startswith(expected_prefix), out
    stub.store.assert_not_called()


async def test_store_create_skip_safety(
    store_fn: Any, stub: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SKIP_SAFETY", "TRUE")
    stub.store = AsyncMock(return_value=("created", entry(), []))
    out = await store_fn(**{**STORE_ARGS, "knowledge_details": 'password = "x1"'})
    assert out.startswith("Created kb-00001")


async def test_store_create_success_with_supersedes_and_hints(
    store_fn: Any, stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    stub.store = AsyncMock(return_value=("created", entry(), ["kb-00002", "kb-00003"]))
    caplog.set_level(logging.WARNING, logger="kb_service")
    out = await store_fn(
        **{
            **STORE_ARGS,
            "supersedes": ["kb-00002"],
            "hints": {"supersedes": ["kb-00003"]},
            "distinct_from": ["kb-00009"],
            "ttl": "7d",
        }
    )
    assert out.endswith("\nSupersedes: kb-00002, kb-00003")
    assert "mismatch" not in caplog.text
    kwargs = stub.store.await_args.kwargs
    assert kwargs["entry_type"] == EntryType.FACTUAL_REFERENCE
    assert kwargs["distinct_from"] == ["kb-00009"]
    assert kwargs["ttl"] == "7d"


async def test_store_create_mismatch_logs_warning(
    store_fn: Any, stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    stub.store = AsyncMock(return_value=("created", entry(), ["kb-00002"]))
    caplog.set_level(logging.WARNING, logger="kb_service")
    await store_fn(
        **{**STORE_ARGS, "supersedes": ["kb-00002"], "hints": {"supersedes": "kb-3"}}
    )
    assert "supersession-client mismatch op=create" in caplog.text


async def test_store_create_mental_map_with_related_entities_hint(
    store_fn: Any, stub: MagicMock
) -> None:
    stub.store = AsyncMock(
        return_value=(
            "created",
            entry(entry_type=EntryType.MENTAL_MAP, has_embedding=True),
            [],
        )
    )
    out = await store_fn(
        **{
            **STORE_ARGS,
            "entry_type": EntryType.MENTAL_MAP,
            "knowledge_details": "orientation on port 8080",
            "hints": {"related_entities": [{"id": "kb-00002"}, "kb-00003"]},
        }
    )
    assert out.startswith("Map lint (advisory): ")
    assert "Note: Entry will be embedded" not in out


@pytest.mark.parametrize(
    "hints",
    [
        {"related_entities": ["kb-00003"]},
        {"related_entities": {"target": "kb-00003"}},
    ],
)
def test_mental_map_pointer_detection(hints: dict[str, object]) -> None:
    assert kb_store._mental_map_has_pointer("no refs", hints) is True
    assert kb_store._mental_map_has_pointer("no refs", {"tool": "x"}) is False
    assert kb_store._mental_map_has_pointer("no refs", {"related_entities": [""]}) is (
        False
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (BackendHttpError(409, "near duplicate"), "Error: near duplicate"),
        (
            BackendHttpError(401, "x"),
            "Error: KB service authentication failed (401). Check PERSONAL_KB_API_KEY.",
        ),
        (RuntimeError("weird"), "Error: weird"),
    ],
)
async def test_store_create_errors(
    store_fn: Any, stub: MagicMock, exc: Exception, expected: str
) -> None:
    stub.store = AsyncMock(side_effect=exc)
    assert await store_fn(**STORE_ARGS) == expected


# ─── kb_store_batch ──────────────────────────────────────────────────────────


@pytest.fixture
def batch_fn() -> Any:
    return capture(kb_store_batch.register_kb_store_batch)


def _batch_entry(**kw: Any) -> dict[str, Any]:
    return {**STORE_ARGS, **kw}


@pytest.mark.parametrize(
    ("entries", "expected_prefix"),
    [
        ([_batch_entry()] * 11, "Error: Maximum 10 entries per batch (got 11)."),
        ([], "Error: entries list is empty."),
        (
            [{"short_title": "s"}],
            "Error: entry 0 missing required fields: knowledge_details, long_title, "
            "supersedes",
        ),
        ([_batch_entry(supersedes=[])], "Error: entry 0: supersedes=[] is ambiguous"),
        (
            [_batch_entry(), _batch_entry(supersedes="x")],
            "Error: entry 1: supersedes must be a list",
        ),
        (
            [_batch_entry(hints={"supersedes": ["kb-00002"]})],
            'Error: entry 0: supersedes="none" conflicts',
        ),
        (
            [_batch_entry(distinct_from="kb-00002")],
            "Error: entry 0: distinct_from must be a list",
        ),
        (
            [_batch_entry(sensitivity="nope")],
            'Error: entry 0 has invalid sensitivity "nope".',
        ),
        (
            [_batch_entry(knowledge_details='password = "hunter2"')],
            "Error: Potential secrets detected in entry 0",
        ),
    ],
)
async def test_store_batch_rejections(
    batch_fn: Any, stub: MagicMock, entries: list[dict[str, Any]], expected_prefix: str
) -> None:
    out = await batch_fn(entries=entries)
    assert out.startswith(expected_prefix), out
    stub.store_batch.assert_not_called()


async def test_store_batch_ttl_failures_and_supersedes(
    batch_fn: Any, stub: MagicMock
) -> None:
    created = [
        entry(id="kb-00005"),
        entry(
            id="kb-00006",
            entry_type=EntryType.MENTAL_MAP,
            knowledge_details="see kb-00005 port 8080",
        ),
    ]
    stub.store_batch = AsyncMock(return_value=(created, [], [["kb-00001"], []]))
    out = await batch_fn(
        entries=[
            _batch_entry(short_title="bad", ttl="never"),
            _batch_entry(supersedes=["kb-00001"]),
            _batch_entry(ttl="7d"),
            _batch_entry(short_title="lost"),
        ]
    )
    sent = stub.store_batch.await_args.args[0]
    assert len(sent) == 3
    assert out.startswith("Batch: 2 entries created, 2 failed")
    assert "Created kb-00005 (v1)" in out
    assert "\nSupersedes: kb-00001" in out
    assert "Map lint (advisory):" in out
    assert "Failed entries (retry these):\n  Entry 0 (bad): " in out


async def test_store_batch_all_failed(batch_fn: Any, stub: MagicMock) -> None:
    stub.store_batch = AsyncMock(return_value=([], [], []))
    out = await batch_fn(entries=[_batch_entry(short_title="t", ttl="bogus")])
    assert out.startswith("Batch failed: all 1 entries failed.\n  Entry 0 (t): ")


async def test_store_batch_backend_error_remaps_entry_index(
    batch_fn: Any, stub: MagicMock
) -> None:
    stub.store_batch = AsyncMock(
        side_effect=BackendHttpError(409, "entry 0 is a near duplicate; entry 7 too")
    )
    out = await batch_fn(
        entries=[_batch_entry(ttl="bad"), _batch_entry(short_title="second")]
    )
    assert out == "Error: entry 1 is a near duplicate; entry 7 too"


async def test_store_batch_backend_failure_indices_restored(
    batch_fn: Any, stub: MagicMock
) -> None:
    stub.store_batch = AsyncMock(
        return_value=([entry(id="kb-00007")], [(0, "x", "db error"), (9, "y", "e")], [])
    )
    out = await batch_fn(entries=[_batch_entry(ttl="bad"), _batch_entry()])
    assert "Entry 1 (x): db error" in out
    assert "Entry 9 (y): e" in out


# ─── kb_search ───────────────────────────────────────────────────────────────


@pytest.fixture
def search_fn() -> Any:
    return capture(kb_search.register_kb_search)


@pytest.mark.parametrize("kwargs", [{"contributor": "a"}, {"team": "t"}])
async def test_search_rejects_contributor_team(
    search_fn: Any, stub: MagicMock, kwargs: dict[str, str]
) -> None:
    out = await search_fn(query="x", **kwargs)
    assert out == "Error: contributor/team filters are not supported in HTTP mode."
    stub.search.assert_not_called()


@pytest.mark.parametrize("passed", [True, False])
async def test_search_forwards_query_and_superseded_marker(
    search_fn: Any, stub: MagicMock, passed: bool
) -> None:
    result = make_search_result()
    result.entry.superseded_by = "kb-00009"
    stub.search = AsyncMock(return_value=([result] * 3, 0))
    stub.vector_search_available = AsyncMock(return_value=False)
    out = await search_fn(
        query="x",
        project_ref="p",
        entry_type=EntryType.DECISION,
        tags=["t"],
        include_superseded=passed,
    )
    query = stub.search.await_args.args[0]
    assert query.include_superseded is passed
    assert query.entry_type == EntryType.DECISION
    assert stub.search.await_args.kwargs == {"contributor": None}
    assert "[SUPERSEDED by kb-00009]" in out
    assert "Vector search unavailable" in out


async def test_search_sparse_results_collect_graph_hints(
    search_fn: Any, stub: MagicMock
) -> None:
    result = make_search_result()
    first_id = result.entry.id
    stub.search = AsyncMock(return_value=([result], 0))
    stub.vector_search_available = AsyncMock(return_value=True)

    async def neighbors(node_id: str, limit: int = 10) -> list[tuple[str, str, str]]:
        if node_id == first_id:
            return [
                ("kb-00020", "depends_on", "outgoing"),
                (first_id, "self", "both"),
                ("tag:python", "has_tag", "outgoing"),
            ]
        return [
            (first_id, "has_tag", "incoming"),
            ("concept:x", "rel", "both"),
            ("kb-00021", "has_tag", "incoming"),
            ("kb-00022", "has_tag", "incoming"),
            ("kb-00023", "has_tag", "incoming"),
        ]

    async def get_entries(
        ids: list[str],
    ) -> list[tuple[str, KnowledgeEntry | None, list[Any]]]:
        eid = ids[0]
        if eid == "kb-00022":
            return [(eid, entry(id=eid, superseded_by="kb-00099"), [])]
        return [(eid, entry(id=eid, short_title=f"T{eid}"), [])]

    stub.neighbors = neighbors
    stub.get_entries = get_entries
    out = await search_fn(query="x")
    assert "Tkb-00020" in out
    assert "Tkb-00021" in out
    assert "Tkb-00023" in out
    assert "kb-00022" not in out


async def test_graph_hints_stop_at_max_direct_neighbours(stub: MagicMock) -> None:
    result = make_search_result()
    stub.neighbors = AsyncMock(
        return_value=[(f"kb-0003{i}", "rel", "both") for i in range(5)]
    )
    stub.get_entries = AsyncMock(
        side_effect=lambda ids: [(ids[0], entry(id=ids[0]), [])]
    )
    hints = await kb_search.collect_graph_hints(stub, [result])
    assert len(hints) == 3


# ─── kb_get ──────────────────────────────────────────────────────────────────


@pytest.fixture
def get_fn() -> Any:
    return capture(kb_get.register_kb_get)


async def test_get_too_many_ids(get_fn: Any, stub: MagicMock) -> None:
    out = await get_fn(entry_id=[f"kb-{i:05d}" for i in range(21)])
    assert out == "Error: Maximum 20 IDs per request (got 21)."


async def test_get_superseded_banner_and_pointer_rot(
    get_fn: Any, stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[list[str]] = []

    async def get_entries(ids: list[str]) -> list[Any]:
        calls.append(ids)
        if ids == ["kb-00008", "kb-00009"]:
            return [
                ("kb-00008", entry(id="kb-00008", short_title="Newer"), []),
                ("kb-00009", None, []),
            ]
        return [
            (
                "kb-00001",
                entry(superseded_by="kb-00008"),
                [("kb-00004", "kb-00005"), ("kb-00006", None)],
            ),
            ("kb-00002", entry(id="kb-00002", superseded_by="kb-00009"), []),
        ]

    stub.get_entries = get_entries
    caplog.set_level(logging.WARNING, logger="kb_service")
    out = await get_fn(entry_id=["kb-00001", "kb-00002"])
    assert "SUPERSEDED by kb-00008 — Newer" in out
    assert "SUPERSEDED by kb-00009\n" in out
    assert "  Pointer-rot:\n    [kb-00004] superseded by [kb-00005]" in out
    assert "    [kb-00006] deactivated" in out
    assert "supersession-read invariant_breach" in caplog.text


# ─── kb_ask / kb_summarize ───────────────────────────────────────────────────


async def test_ask_non_auto_strategy_errors(stub: MagicMock) -> None:
    fn = capture(kb_ask.register_kb_ask)
    out = await fn(question="q", strategy="timeline")
    assert out == (
        "Error: strategy 'timeline' requires a local KB"
        " — only 'auto' is supported in HTTP mode."
    )


async def test_ask_no_results_and_errors(stub: MagicMock) -> None:
    fn = capture(kb_ask.register_kb_ask)
    stub.ask_auto = AsyncMock(return_value=([], 2))
    assert await fn(question="q") == "[Agent: 2 tool calls] No results found."
    stub.ask_auto = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(question="q") == "Error: x"
    stub.ask_auto = AsyncMock(side_effect=BackendHttpError(403, "Admin only"))
    assert await fn(question="q") == (
        "Error: admin privileges required (403): Admin only"
    )


async def test_summarize_forwards_scope_and_maps_errors(stub: MagicMock) -> None:
    fn = capture(kb_summarize.register_kb_summarize)
    stub.summarize = AsyncMock(return_value="answer")
    assert await fn(question="q", scope="project:p", limit=7) == "answer"
    stub.summarize.assert_awaited_once_with("q", "project:p", 7)
    stub.summarize = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(question="q") == "Error: x"


# ─── kb_ingest_url / kb_preflight / kb_feedback / kb_list ────────────────────


async def test_ingest_url_branches(stub: MagicMock) -> None:
    from kb_core.ingest.ingester import FileResult

    fn = capture(kb_ingest_url.register_kb_ingest_url)
    assert await fn(url="") == "Error: url is required."
    stub.ingest_url = AsyncMock(
        return_value=FileResult(
            path="https://e.com",
            action="ingested",
            reason="chunked",
            entry_count=3,
            entry_ids=["kb-00001"],
            summary="A summary",
            chunks_processed=3,
            chunks_skipped=1,
            chunks_flagged=1,
        )
    )
    out = await fn(url="https://e.com", project_ref="p")
    assert out == (
        "  ingested: https://e.com — chunked (3 entries)"
        " [3 chunks, 1 deduped, 1 redacted (secrets)]\n  Summary: A summary"
    )
    stub.ingest_url.assert_awaited_once_with("https://e.com", None, "p", False)
    stub.ingest_url = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(url="https://e.com") == "Error: x"


async def test_preflight_branches(stub: MagicMock) -> None:
    fn = capture(kb_preflight.register_kb_preflight)
    out = await fn(project_ref="p", since="bogus")
    assert out.startswith("Error: ")
    stub.preflight.assert_not_called()
    stub.preflight = AsyncMock(return_value="ctx")
    assert await fn(project_ref="p", since="7d") == "ctx"
    stub.preflight.assert_awaited_once_with("p", "7d")
    stub.preflight = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(project_ref="p") == "Error: x"


async def test_feedback_invalid_type(stub: MagicMock) -> None:
    fn = capture(kb_feedback.register_kb_feedback)
    out = await fn(feedback_type="bogus")
    assert out == (
        "Invalid feedback_type 'bogus'. Must be one of: friction, missing, unhelpful"
    )
    stub.feedback.assert_not_called()


@pytest.mark.parametrize(
    ("register", "method", "empty"),
    [
        (kb_list.register_kb_list_projects, "list_projects", "No projects found."),
        (
            kb_list.register_kb_list_contributors,
            "list_contributors",
            "No contributors found.",
        ),
        (kb_list.register_kb_list_teams, "list_teams", "No teams found."),
    ],
)
async def test_list_tools_render(
    stub: MagicMock, register: Callable[..., None], method: str, empty: str
) -> None:
    fn = capture(register)
    setattr(stub, method, AsyncMock(return_value=[("a", 3), ("b", 1)]))
    assert await fn() == "a (3 entries)\nb (1 entries)"
    setattr(stub, method, AsyncMock(return_value=[]))
    assert await fn() == empty
    setattr(stub, method, AsyncMock(side_effect=RuntimeError("x")))
    assert await fn() == "Error: x"


# ─── kb_maintain ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"action": "deactivate"},
            "Error: entry_id is required for deactivate action.",
        ),
        (
            {"action": "deactivate", "entry_id": "kb-00001"},
            "Error: change_reason is required for deactivate action.",
        ),
        (
            {
                "action": "deactivate",
                "entry_id": "kb-00001",
                "change_reason": "r",
                "superseded_by": "kb-1",
            },
            "Error: superseded_by 'kb-1' is not a valid entry ID (expected kb-NNNNN).",
        ),
        (
            {"action": "reactivate"},
            "Error: entry_id is required for reactivate action.",
        ),
    ],
)
async def test_maintain_rejections(
    stub: MagicMock, kwargs: dict[str, Any], expected: str
) -> None:
    fn = capture(kb_maintain.register_kb_maintain)
    assert await fn(**kwargs) == expected


async def test_maintain_deactivate_forwards_reason_and_superseded_by(
    stub: MagicMock,
) -> None:
    fn = capture(kb_maintain.register_kb_maintain)
    stub.deactivate = AsyncMock(return_value=entry())
    out = await fn(
        action="deactivate",
        entry_id="kb-00001",
        change_reason="r",
        superseded_by="kb-00002",
    )
    assert out == "Deactivated entry kb-00001: Fake entry"
    stub.deactivate.assert_awaited_once_with(
        "kb-00001", change_reason="r", superseded_by="kb-00002"
    )


async def test_maintain_reactivate_error_and_reconcile_render(stub: MagicMock) -> None:
    fn = capture(kb_maintain.register_kb_maintain)
    stub.reactivate = AsyncMock(side_effect=BackendHttpError(404, "gone"))
    assert await fn(action="reactivate", entry_id="kb-00001") == "Error: gone"
    stub.reconcile_supersession = AsyncMock(
        return_value={
            "edges_added": 1,
            "set_count": 2,
            "cleared_count": 0,
            "changed": [["kb-00001", None, "kb-00002"]],
        }
    )
    assert await fn(action="reconcile_supersession") == (
        "Supersession reconcile: edges_added=1 set=2 cleared=0\n"
        "  drift kb-00001: None -> 'kb-00002'"
    )
    stub.reconcile_supersession = AsyncMock(side_effect=ERR_503)
    assert await fn(action="reconcile_supersession") == (
        "Error: KB service returned 503: down"
    )


# ─── kb_bulk_update ──────────────────────────────────────────────────────────


async def test_bulk_update_branches(stub: MagicMock) -> None:
    fn = capture(kb_bulk_update.register_kb_bulk_update)
    assert await fn(filters={}, updates={"team": "t"}) == (
        "Error: At least one filter is required to prevent accidental mass updates."
    )
    assert await fn(filters={"team": "t"}, updates={}) == "Error: No updates specified."
    before = entry(tags=["a"])
    after = entry(tags=["a", "b"], version=2, team="t")
    stub.bulk_update = AsyncMock(return_value=[(before, after)])
    out = await fn(filters={"project_ref": "p"}, updates={"tags_add": ["b"]})
    assert out.splitlines() == [
        "DRY RUN — 1 entries would be updated:",
        "",
        "  kb-00001 (v1 → v2)",
        "  tags: ['a'] → ['a', 'b']",
        "  team: None → 't'",
    ]
    out = await fn(filters={"project_ref": "p"}, updates={"team": "t"}, dry_run=False)
    assert out.startswith("1 entries updated:")
    stub.bulk_update = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(filters={"a": 1}, updates={"b": 2}) == "Error: x"


# ─── kb_map_eligibility ──────────────────────────────────────────────────────

ROW_A: dict[str, Any] = {
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

ROW_B: dict[str, Any] = {
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


async def test_map_eligibility_full_render_exact_lines(stub: MagicMock) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility)
    stub.map_eligibility = AsyncMock(return_value=[ROW_A, ROW_B])
    lines = (await fn()).splitlines()
    assert lines == [
        "Map eligibility — 2 projects | eligible 1 (computed 1, override 0) "
        "| ineligible 1 | eligible & unmapped 1 | evidence flags: too_thin 0, "
        "ingest_corpus 0, journal 1 | override rows 0",
        "",
        "cleanr | ELIGIBLE (computed) | mappable 10 = hand 5 + ingested 5 "
        "| maps 0 | top 'cleanr' 10.0% | flags: none",
        "dispatch-performance-log | INELIGIBLE (computed) "
        "| mappable 624 = hand 624 + ingested 0 | maps 0 | top 'Run' 97.9% "
        "| flags: journal",
        "",
        "Thresholds: too_thin when hand_authored < 5; journal when mappable >= 20 "
        "and top prefix share >= 60%. To change a verdict: "
        'kb_map_eligibility_override(project_ref=..., eligible=..., reason="...").',
    ]


async def test_map_eligibility_filters_overrides_and_orphans(
    stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility, prefix="team_kb_")
    row = {
        **ROW_A,
        "evidence": {
            **ROW_A["evidence"],
            "is_too_thin": True,
            "is_ingest_corpus": True,
        },
        "override": {
            "eligible": False,
            "reason": "journal",
            "set_by": None,
            "set_at": "2026-01-01",
        },
        "decided_by": "override",
        "orphaned": True,
    }
    stub.map_eligibility = AsyncMock(return_value=[row, ROW_B])
    caplog.set_level(logging.WARNING, logger="kb_service")
    out = await fn(project_ref="cleanr")
    assert "dispatch-performance-log" not in out
    assert "flags: too_thin, ingest_corpus" in out
    assert "ORPHANED (0 mappable entries" in out
    assert (
        "  override: ineligible — journal | set_by MISSING — service recorded no "
        "identity | set_at 2026-01-01 | computed said eligible"
    ) in out
    assert "team_kb_map_eligibility_override(" in out
    assert "have no set_by" in caplog.text
    unknown = await fn(project_ref="nope")
    assert unknown == (
        "No map-eligibility row for project_ref 'nope'. "
        "Use team_kb_list_projects to see valid project_refs."
    )


async def test_map_eligibility_payload_drift_and_404(stub: MagicMock) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility)
    broken = {
        **ROW_A,
        "evidence": {k: v for k, v in ROW_A["evidence"].items() if k != "maps"},
    }
    stub.map_eligibility = AsyncMock(return_value=[broken])
    assert (await fn()).startswith(
        "Error: KB service returned an unexpected map-eligibility payload — "
        "missing field 'maps'."
    )
    stub.map_eligibility = AsyncMock(return_value=[{"override": None}])
    assert "missing field 'evidence'" in await fn(project_ref="x")
    stub.map_eligibility = AsyncMock(side_effect=KeyError("projects"))
    assert "missing field 'projects'" in await fn()
    stub.map_eligibility = AsyncMock(side_effect=BackendHttpError(404, "Not Found"))
    assert (await fn()).startswith(
        "Error: this KB service has no map-eligibility endpoint (404)"
    )
    stub.map_eligibility = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn() == "Error: x"


@pytest.mark.parametrize(
    ("kwargs", "expected_prefix"),
    [
        (
            {"project_ref": "bad ref", "eligible": True, "reason": "r"},
            "Error: project_ref must",
        ),
        (
            {"project_ref": "p", "clear": True, "eligible": True},
            "Error: clear=True takes",
        ),
        ({"project_ref": "p"}, "Error: eligible is required"),
        (
            {"project_ref": "p", "eligible": True, "reason": " "},
            "Error: reason is required",
        ),
        (
            {"project_ref": "p", "eligible": True, "reason": "x" * 2001},
            "Error: reason exceeds the service's 2000-character cap",
        ),
    ],
)
async def test_override_validation_makes_no_request(
    stub: MagicMock, kwargs: dict[str, Any], expected_prefix: str
) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility_override)
    assert (await fn(**kwargs)).startswith(expected_prefix)
    stub.set_map_eligibility_override.assert_not_called()
    stub.clear_map_eligibility_override.assert_not_called()


async def test_override_set_renders_stored_verdict(
    stub: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility_override)
    verdict = {
        "effective_eligible": False,
        "decided_by": "override",
        "orphaned": False,
        "override": {"set_by": "alice", "set_at": "2026-01-02", "reason": "stored"},
    }
    stub.set_map_eligibility_override = AsyncMock(
        return_value={"changed": True, "verdict": verdict}
    )
    out = await fn(project_ref="p", eligible=False, reason=" stored ")
    assert out == (
        "Override set — p is now INELIGIBLE (override), beating the computed "
        "verdict until cleared. reason: stored | set_by alice | set_at 2026-01-02"
    )
    stub.set_map_eligibility_override.assert_awaited_once_with(
        "p", eligible=False, reason=" stored "
    )
    orphan = {**verdict, "orphaned": True, "override": {}}
    stub.set_map_eligibility_override = AsyncMock(
        return_value={"changed": True, "verdict": orphan}
    )
    caplog.set_level(logging.WARNING, logger="kb_service")
    out = await fn(project_ref="p", eligible=True, reason="why")
    assert "reason: why | set_by MISSING | set_at unknown" in out
    assert "WARNING: the service recorded no set_by" in out
    assert "WARNING: p has no mappable entries" in out
    assert "returned no set_by" in caplog.text


async def test_override_clear_noop_and_errors(stub: MagicMock) -> None:
    fn = capture(kb_map_eligibility.register_kb_map_eligibility_override)
    stub.clear_map_eligibility_override = AsyncMock(return_value=False)
    assert await fn(project_ref="p", clear=True) == (
        "No override existed for p — nothing to clear."
    )
    stub.set_map_eligibility_override = AsyncMock(return_value={"verdict": {}})
    assert "missing field 'effective_eligible'" in await fn(
        project_ref="p", eligible=True, reason="r"
    )
    stub.set_map_eligibility_override = AsyncMock(
        side_effect=BackendHttpError(404, "Not Found")
    )
    assert (await fn(project_ref="p", eligible=True, reason="r")).startswith(
        "Error: this KB service has no map-eligibility endpoint (404)"
    )
    stub.set_map_eligibility_override = AsyncMock(side_effect=RuntimeError("x"))
    assert await fn(project_ref="p", eligible=True, reason="r") == "Error: x"


# ─── write policy: queued results (twins of the stdio tools) ─────────────────

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
QUEUED_BATCH_TEXT = {
    "on": (
        "Batch: 2 entries queued as candidates 7, 8 for review (write policy:"
        " headless surface). Not in the KB yet: the candidate pipeline's distiller"
        " and critic decide whether each is written."
    ),
    "shadow": (
        "Batch: 2 entries queued as candidates 7, 8 (write policy: headless"
        " surface). Not in the KB: capture mode is shadow, so they are recorded for"
        " audit only."
    ),
    "off": (
        "Batch: 2 entries queued as candidates 7, 8 (write policy: headless"
        " surface). Not in the KB: capture mode is off, so they are recorded for"
        " audit only."
    ),
}


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
async def test_store_queued_renders(store_fn: Any, stub: MagicMock, mode: str) -> None:
    stub.store = AsyncMock(return_value=QueuedStore(7, "headless", mode))
    assert await store_fn(**STORE_ARGS) == QUEUED_STORE_TEXT[mode]
    assert (
        kb_store.format_queued_store(QueuedStore(7, "headless", mode))
        == (QUEUED_STORE_TEXT[mode])
    )


async def test_store_update_queued_is_error(store_fn: Any, stub: MagicMock) -> None:
    stub.store = AsyncMock(return_value=QueuedStore(7, "headless", "on"))
    out = await store_fn(
        update_entry_id="kb-00001",
        knowledge_details="d",
        change_reason="r",
        supersedes="none",
    )
    assert out == (
        "Error: the KB service queued an update as candidate 7; updates are never"
        " queued."
    )


@pytest.mark.parametrize("mode", ["on", "shadow", "off"])
async def test_store_batch_queued_renders(
    batch_fn: Any, stub: MagicMock, mode: str
) -> None:
    stub.store_batch = AsyncMock(return_value=QueuedBatch((7, 8), "headless", mode))
    out = await batch_fn(entries=[_batch_entry(), _batch_entry(short_title="t")])
    assert out == QUEUED_BATCH_TEXT[mode]


async def test_store_batch_queued_appends_ttl_failures(
    batch_fn: Any, stub: MagicMock
) -> None:
    stub.store_batch = AsyncMock(return_value=QueuedBatch((7, 8), "headless", "shadow"))
    out = await batch_fn(
        entries=[
            _batch_entry(),
            _batch_entry(short_title="bad", ttl="never"),
            _batch_entry(short_title="t"),
        ]
    )
    lines = out.split("\n")
    assert lines[0] == QUEUED_BATCH_TEXT["shadow"]
    assert lines[1] == "Failed entries (retry these):"
    assert lines[2].startswith("  Entry 1 (bad): ")
    assert len(lines) == 3
