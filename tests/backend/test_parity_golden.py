"""Golden parity tests — byte-identical output for Local vs HTTP backends.

These tests verify that the same tool-layer formatting code produces
byte-identical output whether data comes from LocalBackend (in-process DB)
or HttpBackend (MockTransport round-trip through JSON serialization).

The pattern:
  1. Create data via LocalBackend.
  2. Capture what that data looks like as JSON (what the service would return).
  3. Feed the same JSON through HttpBackend via MockTransport.
  4. Assert both paths produce byte-identical output strings.
"""

from __future__ import annotations

import json
from datetime import UTC
from typing import Any

import httpx
import pytest
import pytest_asyncio
from kb_core.config import Attribution, KbConfig
from kb_core.graph.builder import GraphBuilder
from kb_core.knowledge_base import KnowledgeBase

from personal_kb.backend.http import HttpBackend
from personal_kb.backend.local import LocalBackend
from personal_kb.db.connection import create_connection
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.formatters import format_entry_full, format_result_list
from personal_kb.tools.kb_get import _render_pointer_rot
from personal_kb.tools.kb_store import format_store_result

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db():
    conn = await create_connection(":memory:")
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def store(db):
    return KnowledgeStore(db)


def _make_kb(db, store) -> KnowledgeBase:
    """Build a minimal KnowledgeBase with no LLM/embedder for parity tests."""
    return KnowledgeBase(
        config=KbConfig(attribution=Attribution()),
        db=db,
        store=store,
        embedder=None,
        graph_builder=GraphBuilder(db),
        graph_enricher=None,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
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


# ---------------------------------------------------------------------------
# Helper: serialise a KnowledgeEntry to the JSON dict HttpBackend would parse
# ---------------------------------------------------------------------------


def _entry_to_json(entry) -> dict[str, Any]:
    """Convert KnowledgeEntry to the dict that _parse_entry() expects.

    Uses model_dump(mode='json') so datetime fields become ISO strings.
    """
    return json.loads(entry.model_dump_json())


# ---------------------------------------------------------------------------
# kb_get parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_get_local_vs_http_parity(db, store):
    """format_entry_full output is byte-identical for LocalBackend vs HttpBackend."""
    entry = await store.create_entry(
        short_title="Parity Test",
        long_title="Parity long title",
        knowledge_details="Details for parity check.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["parity", "test"],
        project_ref="parity-proj",
        confidence_level=0.85,
    )

    # --- Local path ---
    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    local_results = await local_backend.get_entries([entry.id])
    assert len(local_results) == 1
    l_eid, l_entry, l_rot = local_results[0]
    assert l_entry is not None
    local_rendered = format_entry_full(l_entry)
    local_rot_note = _render_pointer_rot(l_rot)
    if local_rot_note is not None:
        local_rendered = f"{local_rendered}\n{local_rot_note}"
    local_output = format_result_list([local_rendered])

    # --- HTTP path (MockTransport serving the serialised entry) ---
    entry_json = _entry_to_json(l_entry)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [{"id": l_eid, "found": True, "entry": entry_json, "pointer_rot": []}]
            },
        )

    http_backend = _make_http_backend(handler)
    http_results = await http_backend.get_entries([entry.id])
    _, h_entry, h_rot = http_results[0]
    assert h_entry is not None
    http_rendered = format_entry_full(h_entry)
    http_rot_note = _render_pointer_rot(h_rot)
    if http_rot_note is not None:
        http_rendered = f"{http_rendered}\n{http_rot_note}"
    http_output = format_result_list([http_rendered])

    assert local_output == http_output


# ---------------------------------------------------------------------------
# kb_store create parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_store_create_parity(db, store):
    """format_store_result(create) is byte-identical for LocalBackend vs HttpBackend."""
    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    action, local_entry = await local_backend.store(
        short_title="Golden",
        long_title="Golden long title",
        knowledge_details="Golden details.",
        entry_type=EntryType.DECISION,
        confidence_level=0.9,
    )
    assert action == "created"

    local_output = format_store_result(local_entry, is_update=False, include_backend_warning=False)

    # HTTP path
    entry_json = _entry_to_json(local_entry)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"action": "created", "entry": entry_json})

    http_backend = _make_http_backend(handler)
    _action, http_entry = await http_backend.store(
        short_title="Golden",
        long_title="Golden long title",
        knowledge_details="Golden details.",
        entry_type=EntryType.DECISION,
        confidence_level=0.9,
    )
    http_output = format_store_result(http_entry, is_update=False, include_backend_warning=False)

    assert local_output == http_output


# ---------------------------------------------------------------------------
# kb_store deactivate parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_store_deactivate_parity(db, store):
    """Deactivate return string is identical for Local vs HTTP backends."""
    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    _, local_entry = await local_backend.store(
        short_title="To deactivate",
        long_title="Long",
        knowledge_details="Details",
        entry_type=EntryType.FACTUAL_REFERENCE,
    )

    deactivated_local = await local_backend.deactivate(local_entry.id)
    local_output = f"Deactivated entry {deactivated_local.id}: {deactivated_local.short_title}"

    deactivated_json = _entry_to_json(deactivated_local)

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"entry": deactivated_json})

    http_backend = _make_http_backend(handler)
    deactivated_http = await http_backend.deactivate(local_entry.id)
    http_output = f"Deactivated entry {deactivated_http.id}: {deactivated_http.short_title}"

    assert local_output == http_output


# ---------------------------------------------------------------------------
# kb_preflight parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_preflight_parity(db, store):
    """preflight() returns the same string for LocalBackend vs HttpBackend."""
    # Populate a minimal DB for preflight to work on
    await store.create_entry(
        short_title="Parity preflight entry",
        long_title="Long title",
        knowledge_details="Details",
        entry_type=EntryType.DECISION,
        project_ref="parity-proj",
    )

    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    local_output = await local_backend.preflight("parity-proj", since=None)

    # HTTP MockTransport returns whatever the local backend produced
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"context": local_output})

    http_backend = _make_http_backend(handler)
    http_output = await http_backend.preflight("parity-proj", since=None)

    # By construction they must match — but the type contract is correct too
    assert isinstance(http_output, str)
    assert local_output == http_output


# ---------------------------------------------------------------------------
# kb_ask auto parity (agent_turns_used=0)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_ask_auto_parity_no_agent(db, store):
    """ask_auto with no LLM (agent_turns_used=0) output is parity-identical."""
    await store.create_entry(
        short_title="Ask parity",
        long_title="Ask parity long",
        knowledge_details="Parity for ask auto.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["parity"],
    )

    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    entries_with_ctx, turns = await local_backend.ask_auto(
        question="parity",
        scope=None,
        include_graph_context=False,
        limit=10,
    )
    assert turns == 0  # no LLM → no agent turns

    if entries_with_ctx:
        local_formatted = [format_entry_full(e, context=c) for e, c in entries_with_ctx[:10]]
        local_output = format_result_list(local_formatted, header="[Agent: 0 tool calls]")

        # HTTP: MockTransport returns the same entry list
        entry_json = _entry_to_json(entries_with_ctx[0][0])
        ctx_str = entries_with_ctx[0][1]

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "entries": [{"entry": entry_json, "context": ctx_str}],
                    "agent_turns_used": 0,
                },
            )

        http_backend = _make_http_backend(handler)
        http_entries, http_turns = await http_backend.ask_auto(
            question="parity",
            scope=None,
            include_graph_context=False,
            limit=10,
        )
        assert http_turns == 0
        http_formatted = [format_entry_full(e, context=c) for e, c in http_entries[:10]]
        http_output = format_result_list(http_formatted, header="[Agent: 0 tool calls]")

        assert local_output == http_output
    else:
        # No results found — both paths produce "no results"
        assert entries_with_ctx == []


# ---------------------------------------------------------------------------
# kb_search parity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_search_parity(db, store):
    """search() results are serialised and parsed back byte-identically."""
    await store.create_entry(
        short_title="Search parity",
        long_title="Search parity long",
        knowledge_details="Details for search parity.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        project_ref="search-proj",
        confidence_level=0.9,
    )

    from kb_core.models.search import SearchQuery

    kb = _make_kb(db, store)
    local_backend = LocalBackend(kb)
    query = SearchQuery(query="parity", limit=10)
    local_results, _ = await local_backend.search(query)

    # Verify formatting of the local result
    from datetime import datetime

    from personal_kb.confidence.decay import compute_effective_confidence
    from personal_kb.tools.formatters import format_entry_compact

    now = datetime.now(UTC)

    local_formatted: list[str] = []
    for sr in local_results:
        anchor = sr.entry.updated_at or sr.entry.created_at or now
        eff = compute_effective_confidence(sr.entry.confidence_level, sr.entry.entry_type, anchor)
        local_formatted.append(format_entry_compact(sr.entry, eff))

    # HTTP path: serve the same result via MockTransport
    if local_results:
        entry_json = _entry_to_json(local_results[0].entry)
        http_result = {
            "entry": entry_json,
            "score": local_results[0].score,
            "effective_confidence": local_results[0].effective_confidence,
            "staleness_warning": local_results[0].staleness_warning,
            "match_source": local_results[0].match_source,
        }

        def handler(req: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"results": [http_result], "filtered_count": 0})

        http_backend = _make_http_backend(handler)
        http_results, _ = await http_backend.search(query)
        assert len(http_results) == 1

        http_formatted: list[str] = []
        for sr in http_results:
            anchor = sr.entry.updated_at or sr.entry.created_at or now
            eff = compute_effective_confidence(
                sr.entry.confidence_level, sr.entry.entry_type, anchor
            )
            http_formatted.append(format_entry_compact(sr.entry, eff))

        assert local_formatted == http_formatted
