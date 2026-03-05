"""Tests for the kb_explore MCP tool."""

import asyncio
import contextlib
import json

import pytest

from personal_kb.explorer.graph_data import extract_graph_data
from personal_kb.explorer.renderer import render_explorer_html
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType
from personal_kb.tools import kb_explore
from personal_kb.tools.kb_explore import explore_logic


@pytest.fixture(autouse=True)
def _reset_web_server():
    """Cancel and reset the module-level web server task between tests."""
    yield
    task = kb_explore._web_server_task
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            asyncio.get_event_loop().run_until_complete(task)
    kb_explore._web_server_task = None


@pytest.mark.asyncio
async def test_explore_logic_returns_html_and_summary(db, store, monkeypatch):
    """explore_logic returns valid HTML and a summary with stats."""
    builder = GraphBuilder(db)
    entry = await store.create_entry(
        short_title="Test",
        long_title="Test entry",
        knowledge_details="Just a test.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["python"],
    )
    await builder.build_for_entry(entry)
    await db.commit()

    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened_urls.append)

    html, summary = await explore_logic(db)

    assert "<!DOCTYPE html>" in html
    assert "GRAPH_DATA" in html
    assert "force-graph" in html
    node_count = summary.split("nodes")[0].strip().split()[-1]
    assert int(node_count) >= 2  # entry + tag (+ possible project)
    assert "Explorer opened" in summary
    assert len(opened_urls) == 1
    assert opened_urls[0].startswith("file://") or opened_urls[0].startswith("http://localhost:")


@pytest.mark.asyncio
async def test_html_contains_correct_graph_data(db, store):
    """The GRAPH_DATA JSON blob in HTML has correct node/edge counts."""
    builder = GraphBuilder(db)
    entry = await store.create_entry(
        short_title="GraphTest",
        long_title="Graph test entry",
        knowledge_details="For testing.",
        entry_type=EntryType.DECISION,
        tags=["sqlite", "testing"],
        project_ref="test-proj",
    )
    await builder.build_for_entry(entry)
    await db.commit()

    data = await extract_graph_data(db)
    html = render_explorer_html(data)

    # Extract GRAPH_DATA from HTML
    marker = "const GRAPH_DATA = "
    start = html.index(marker) + len(marker)
    end = html.index(";\nconst NODE_COLORS")
    embedded_json = json.loads(html[start:end])

    assert embedded_json["stats"]["node_count"] == data["stats"]["node_count"]
    assert embedded_json["stats"]["edge_count"] == data["stats"]["edge_count"]
    assert len(embedded_json["nodes"]) == len(data["nodes"])
    assert len(embedded_json["edges"]) == len(data["edges"])


@pytest.mark.asyncio
async def test_explore_empty_db(db, monkeypatch):
    """explore_logic works with an empty DB."""
    opened_urls: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened_urls.append)

    _html, summary = await explore_logic(db)

    assert "0 nodes" in summary
    assert "0 edges" in summary
    assert len(opened_urls) == 1


@pytest.mark.asyncio
async def test_html_has_node_colors(db):
    """The rendered HTML includes the NODE_COLORS mapping."""
    data = await extract_graph_data(db)
    html = render_explorer_html(data)

    assert "NODE_COLORS" in html
    assert "#00bcd4" in html  # tag color
    assert "#ff9800" in html  # project color
