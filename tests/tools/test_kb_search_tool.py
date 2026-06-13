"""Tests for the kb_search MCP tool formatting and graph hints.

Graph-hint tests run against an :class:`HttpBackend` backed by
``httpx.MockTransport`` — the in-process local backend has been removed, so
``collect_graph_hints`` is exercised against canned HTTP responses matching
the service's ``/api/kb/graph/neighbors`` and ``/api/kb/get`` route shapes.
"""

import json
from datetime import UTC, datetime

import httpx
import pytest

from personal_kb.backend.http import HttpBackend
from personal_kb.models.entry import EntryType, KnowledgeEntry
from personal_kb.models.search import SearchResult
from personal_kb.tools.formatters import format_graph_hint
from personal_kb.tools.kb_search import collect_graph_hints, format_search_results


def _make_entry(
    entry_id: str = "kb-00001",
    short_title: str = "Test",
    long_title: str = "Test entry",
    knowledge_details: str = "Some details here",
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


def _make_result(
    entry: KnowledgeEntry,
    score: float = 0.025,
    confidence: float = 0.85,
    staleness: str | None = None,
) -> SearchResult:
    return SearchResult(
        entry=entry,
        score=score,
        effective_confidence=confidence,
        staleness_warning=staleness,
        match_source="hybrid",
    )


# --- format_search_results ---


def test_format_empty_results():
    result = format_search_results([])
    assert result == "No results found."


def test_format_results_with_note():
    result = format_search_results([], match_source_note="FTS only")
    assert "No results found." in result


def test_format_results():
    entry = _make_entry(project_ref="test-proj", tags=["a", "b"])
    results = [_make_result(entry)]
    output = format_search_results(results)
    assert "kb-00001" in output
    assert "Test" in output
    assert "test-proj" in output
    assert "85%" in output
    assert "#a #b" in output
    # Compact format should NOT include knowledge_details
    assert "Some details here" not in output


def test_format_results_with_staleness():
    entry = _make_entry(
        entry_id="kb-00002",
        short_title="Old fact",
        long_title="An old factual reference",
        knowledge_details="Outdated info",
    )
    results = [_make_result(entry, score=0.01, confidence=0.3, staleness="Stale")]
    output = format_search_results(results)
    assert "[STALE]" in output


def test_format_results_with_graph_hints():
    """Graph hints should appear in formatted output."""
    entry = _make_entry()
    results = [_make_result(entry)]
    hints = ["See also: [kb-00042] Related entry (via tag:python)"]
    output = format_search_results(results, graph_hints=hints)
    assert "Related entries via graph:" in output
    assert "kb-00042" in output
    assert "via tag:python" in output


def test_format_results_no_hints_when_none():
    """No graph hints section when hints is None."""
    entry = _make_entry()
    results = [_make_result(entry)]
    output = format_search_results(results, graph_hints=None)
    assert "Related entries via graph:" not in output


def test_format_results_no_hints_when_empty():
    """No graph hints section when hints list is empty."""
    entry = _make_entry()
    results = [_make_result(entry)]
    output = format_search_results(results, graph_hints=[])
    assert "Related entries via graph:" not in output


# --- format_graph_hint ---


def test_format_graph_hint():
    entry = _make_entry(entry_id="kb-00042", short_title="Chose aiosqlite")
    hint = format_graph_hint(entry, "concept:async-io")
    assert hint == "See also: [kb-00042] Chose aiosqlite (via concept:async-io)"


# --- collect_graph_hints ---
# HTTP-contract helpers: build an HttpBackend over a MockTransport and serve
# canned neighbors / get responses (matching the service route shapes).


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


def _graph_backend(
    neighbors_map: dict[str, list[tuple[str, str, str]]],
    entries_map: dict[str, KnowledgeEntry | None],
) -> HttpBackend:
    """Build an HttpBackend whose neighbors/get routes are driven by maps.

    ``neighbors_map`` maps a node id → list of ``(neighbor_id, edge_type,
    direction)``. ``entries_map`` maps an entry id → ``KnowledgeEntry`` (or
    ``None`` to model a missing/inactive entry).
    """

    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path == "/api/kb/graph/neighbors":
            node_id = req.url.params.get("node_id")
            nbrs = neighbors_map.get(node_id, [])
            return httpx.Response(
                200,
                json={
                    "neighbors": [
                        {"neighbor_id": n, "edge_type": e, "direction": d} for n, e, d in nbrs
                    ]
                },
            )
        if path == "/api/kb/get":
            body = json.loads(req.content)
            results = []
            for eid in body.get("ids", []):
                entry = entries_map.get(eid)
                if entry is None:
                    results.append({"id": eid, "found": False, "entry": None})
                else:
                    results.append(
                        {
                            "id": eid,
                            "found": True,
                            "entry": entry.model_dump(mode="json"),
                            "pointer_rot": [],
                        }
                    )
            return httpx.Response(200, json={"results": results})
        return httpx.Response(404, json={"detail": f"unexpected path {path}"})

    return _make_http_backend(handler)


@pytest.mark.asyncio
async def test_collect_graph_hints_empty_results():
    """No hints when there are no search results."""
    backend = _graph_backend({}, {})
    hints = await collect_graph_hints(backend, [])
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_no_graph_edges():
    """No hints when entries have no graph connections."""
    entry = _make_entry()
    backend = _graph_backend({"kb-00001": []}, {"kb-00001": entry})
    results = [_make_result(entry)]
    hints = await collect_graph_hints(backend, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_via_shared_tag():
    """Should find related entries via shared tag nodes (two-hop)."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", tags=["python"])
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", tags=["python"])

    backend = _graph_backend(
        {
            "kb-00001": [("tag:python", "tagged", "both")],
            "tag:python": [("kb-00002", "tagged", "both")],
        },
        {"kb-00002": entry2},
    )

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(backend, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "tag:python" in hints[0]


@pytest.mark.asyncio
async def test_collect_graph_hints_skips_result_entries():
    """Should not hint at entries already in the result set."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", tags=["python"])
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", tags=["python"])

    backend = _graph_backend(
        {
            "kb-00001": [("tag:python", "tagged", "both")],
            "kb-00002": [("tag:python", "tagged", "both")],
            "tag:python": [("kb-00001", "tagged", "both"), ("kb-00002", "tagged", "both")],
        },
        {"kb-00001": entry1, "kb-00002": entry2},
    )

    # Both entries in results — no hints
    results = [_make_result(entry1), _make_result(entry2)]
    hints = await collect_graph_hints(backend, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_max_limit():
    """Should cap hints at max_hints."""
    related = [
        _make_entry(entry_id=f"kb-{i:05d}", short_title=f"Entry {i}", tags=["shared"])
        for i in range(2, 7)
    ]
    entry1 = _make_entry(entry_id="kb-00001", short_title="Entry 1", tags=["shared"])

    backend = _graph_backend(
        {
            "kb-00001": [("tag:shared", "tagged", "both")],
            "tag:shared": [(e.id, "tagged", "both") for e in related],
        },
        {e.id: e for e in related},
    )

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(backend, results, max_hints=3)
    assert len(hints) == 3


@pytest.mark.asyncio
async def test_collect_graph_hints_direct_entry_edge():
    """Should find hints via direct entry-to-entry edges (e.g. supersedes)."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="Original decision")
    entry2 = _make_entry(
        entry_id="kb-00002",
        short_title="Updated decision",
        entry_type=EntryType.DECISION,
    )

    backend = _graph_backend(
        {"kb-00001": [("kb-00002", "supersedes", "incoming")]},
        {"kb-00002": entry2},
    )

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(backend, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "supersedes" in hints[0]


@pytest.mark.asyncio
async def test_collect_graph_hints_skips_inactive():
    """Should not include hints for inactive/missing entries."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="Active", tags=["python"])

    # entry2 resolves to None via /api/kb/get (inactive → not found)
    backend = _graph_backend(
        {
            "kb-00001": [("tag:python", "tagged", "both")],
            "tag:python": [("kb-00002", "tagged", "both")],
        },
        {"kb-00002": None},
    )

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(backend, results)
    assert hints == []


@pytest.mark.asyncio
async def test_collect_graph_hints_via_project():
    """Should find related entries via shared project node."""
    entry1 = _make_entry(entry_id="kb-00001", short_title="First", project_ref="my-proj")
    entry2 = _make_entry(entry_id="kb-00002", short_title="Second", project_ref="my-proj")

    backend = _graph_backend(
        {
            "kb-00001": [("project:my-proj", "in_project", "both")],
            "project:my-proj": [("kb-00002", "in_project", "both")],
        },
        {"kb-00002": entry2},
    )

    results = [_make_result(entry1)]
    hints = await collect_graph_hints(backend, results)
    assert len(hints) == 1
    assert "kb-00002" in hints[0]
    assert "project:my-proj" in hints[0]
