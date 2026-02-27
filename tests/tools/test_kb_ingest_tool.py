"""Tests for the kb_ingest MCP tool."""

import json
from unittest.mock import MagicMock

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.kb_ingest import register_kb_ingest
from tests.conftest import FakeEmbedder, FakeLLM


class SequenceLLM(FakeLLM):
    """FakeLLM that returns a sequence of responses."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)
        self._call_index = 0

    async def generate(self, prompt, *, system=None):
        self.last_prompt = prompt
        self.last_system = system
        self.generate_count += 1
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return None


@pytest_asyncio.fixture
async def tool_context():
    """Create a mock MCP context with all lifespan dependencies."""
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = FakeEmbedder(db)
    graph_builder = GraphBuilder(db)
    fake_llm = FakeLLM()
    enricher = GraphEnricher(db, fake_llm)

    lifespan = {
        "db": db,
        "store": store,
        "embedder": embedder,
        "graph_builder": graph_builder,
        "graph_enricher": enricher,
        "query_llm": None,  # Will be overridden per test
    }

    ctx = MagicMock()
    ctx.lifespan_context = lifespan

    yield ctx, lifespan

    await db.close()


def _register_and_capture(mcp_mock):
    """Register kb_ingest on a mock MCP and return the captured tools dict."""
    tools = {}

    def capture_tool():
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp_mock.tool = capture_tool
    register_kb_ingest(mcp_mock)
    return tools


class TestKbIngestTool:
    async def test_error_when_no_llm(self, tool_context, tmp_path):
        ctx, lifespan = tool_context
        lifespan["query_llm"] = None

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path=str(tmp_path / "test.md"),
            ctx=ctx,
        )
        assert "No LLM available" in result

    async def test_error_on_nonexistent_path(self, tool_context):
        ctx, lifespan = tool_context
        lifespan["query_llm"] = FakeLLM()

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path="/nonexistent/path/file.md",
            ctx=ctx,
        )
        assert "does not exist" in result

    async def test_ingest_single_file(self, tool_context, tmp_path):
        ctx, lifespan = tool_context

        f = tmp_path / "notes.md"
        f.write_text("# My Notes\n\nSome knowledge here.")

        entries_json = [
            {
                "short_title": "test entry",
                "long_title": "A test knowledge entry",
                "knowledge_details": "Some knowledge.",
                "entry_type": "factual_reference",
                "tags": ["test"],
            }
        ]
        llm = SequenceLLM(["Summary of notes.", json.dumps(entries_json)])
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path=str(f),
            project_ref="test-project",
            ctx=ctx,
        )
        assert "ingested" in result
        assert "1 entries" in result

    async def test_dry_run_single_file(self, tool_context, tmp_path):
        ctx, lifespan = tool_context

        f = tmp_path / "notes.md"
        f.write_text("# Dry run test")

        entries_json = [
            {
                "short_title": "dry entry",
                "long_title": "Dry run entry",
                "knowledge_details": "Details.",
                "entry_type": "decision",
                "tags": [],
            }
        ]
        llm = SequenceLLM(["Dry run summary.", json.dumps(entries_json)])
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path=str(f),
            dry_run=True,
            ctx=ctx,
        )
        assert "DRY RUN" in result
        assert "Summary: Dry run summary." in result

    async def test_ingest_directory(self, tool_context, tmp_path):
        ctx, lifespan = tool_context

        (tmp_path / "a.md").write_text("# File A")
        (tmp_path / "b.md").write_text("# File B")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm = SequenceLLM(
            [
                "Summary A.",
                json.dumps(entries_json),
                "Summary B.",
                json.dumps(entries_json),
            ]
        )
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path=str(tmp_path),
            ctx=ctx,
        )
        assert "Ingestion complete" in result
        assert "2 ingested" in result

    async def test_glob_pattern_matches_files(self, tool_context, tmp_path, monkeypatch):
        ctx, lifespan = tool_context
        monkeypatch.chdir(tmp_path)

        # Create .md and .txt files — glob *.md should only match .md
        (tmp_path / "notes.md").write_text("# Notes")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "data.txt").write_text("plain text")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm = SequenceLLM(
            [
                "Summary 1.",
                json.dumps(entries_json),
                "Summary 2.",
                json.dumps(entries_json),
            ]
        )
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path="*.md",
            ctx=ctx,
        )
        assert "Ingestion complete" in result
        assert "2 ingested" in result

    async def test_glob_no_matches(self, tool_context, tmp_path, monkeypatch):
        ctx, lifespan = tool_context
        monkeypatch.chdir(tmp_path)
        lifespan["query_llm"] = FakeLLM()

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path="*.nonexistent",
            ctx=ctx,
        )
        assert "No files matched pattern" in result

    async def test_glob_recursive_pattern(self, tool_context, tmp_path, monkeypatch):
        ctx, lifespan = tool_context
        monkeypatch.chdir(tmp_path)

        # Create nested structure
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("# Top")
        (sub / "nested.md").write_text("# Nested")

        entries_json = [
            {
                "short_title": "entry",
                "long_title": "An entry",
                "knowledge_details": "Details",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm = SequenceLLM(
            [
                "Summary 1.",
                json.dumps(entries_json),
                "Summary 2.",
                json.dumps(entries_json),
            ]
        )
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest"](
            path="**/*.md",
            ctx=ctx,
        )
        assert "Ingestion complete" in result
        assert "2 ingested" in result
