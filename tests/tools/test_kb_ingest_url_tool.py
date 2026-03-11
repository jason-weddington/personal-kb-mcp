"""Tests for the kb_ingest_url MCP tool."""

import json
from unittest.mock import MagicMock

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.store.knowledge_store import KnowledgeStore
from personal_kb.tools.kb_ingest_url import register_kb_ingest_url
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
        "query_llm": None,
    }

    ctx = MagicMock()
    ctx.lifespan_context = lifespan

    yield ctx, lifespan

    await db.close()


def _register_and_capture(mcp_mock):
    """Register kb_ingest_url on a mock MCP and return the captured tools dict."""
    tools = {}

    def capture_tool(**_kwargs):
        def decorator(func):
            tools[func.__name__] = func
            return func

        return decorator

    mcp_mock.tool = capture_tool
    register_kb_ingest_url(mcp_mock)
    return tools


class TestKbIngestUrlTool:
    async def test_prefetched_content_skips_fetch(self, tool_context):
        """When content is provided, should ingest directly without fetching."""
        ctx, lifespan = tool_context

        entries_json = [
            {
                "short_title": "intranet doc",
                "long_title": "Internal deployment guide",
                "knowledge_details": "Deploy with kubectl apply.",
                "entry_type": "factual_reference",
                "tags": ["deployment"],
            }
        ]
        llm = SequenceLLM(["Summary of deployment guide.", json.dumps(entries_json)])
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest_url"](
            url="https://intranet.example.com/deploy-guide",
            content="# Deployment Guide\n\nDeploy with kubectl apply.",
            project_ref="infra",
            ctx=ctx,
        )
        assert "ingested" in result
        assert "1 entries" in result

    async def test_prefetched_content_dry_run(self, tool_context):
        """Dry run with pre-fetched content should preview without storing."""
        ctx, lifespan = tool_context

        entries_json = [
            {
                "short_title": "wiki page",
                "long_title": "Team wiki page",
                "knowledge_details": "Some wiki content.",
                "entry_type": "factual_reference",
                "tags": [],
            }
        ]
        llm = SequenceLLM(["Wiki page summary.", json.dumps(entries_json)])
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest_url"](
            url="https://wiki.example.com/page",
            content="Some wiki content.",
            dry_run=True,
            ctx=ctx,
        )
        assert "DRY RUN" in result

    async def test_no_content_calls_ingest_url(self, tool_context, monkeypatch):
        """When content is None, should call ingest_url (the fetch path)."""
        ctx, lifespan = tool_context
        llm = SequenceLLM(["Summary.", "[]"])
        lifespan["query_llm"] = llm

        tools = _register_and_capture(MagicMock())

        # Patch ingest_url on the FileIngester class to verify it's called
        from personal_kb.ingest.ingester import FileIngester, FileResult

        called_with = {}

        async def mock_ingest_url(self, url, *, project_ref=None, dry_run=False):
            called_with["url"] = url
            called_with["project_ref"] = project_ref
            return FileResult(path=url, action="ingested", entry_ids=[], summary="Test")

        monkeypatch.setattr(FileIngester, "ingest_url", mock_ingest_url)

        await tools["kb_ingest_url"](
            url="https://example.com/article",
            ctx=ctx,
        )
        assert called_with["url"] == "https://example.com/article"

    async def test_error_when_no_llm(self, tool_context):
        ctx, lifespan = tool_context
        lifespan["query_llm"] = None

        tools = _register_and_capture(MagicMock())

        result = await tools["kb_ingest_url"](
            url="https://example.com",
            content="Some content",
            ctx=ctx,
        )
        assert "No LLM available" in result
