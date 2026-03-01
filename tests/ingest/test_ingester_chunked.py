"""Tests for chunked ingestion pipeline."""

import json

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.graph.enricher import GraphEnricher
from personal_kb.ingest.ingester import FileIngester
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder, FakeLLM


@pytest_asyncio.fixture
async def ingester_deps():
    """Create all dependencies for FileIngester with in-memory DB."""
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = FakeEmbedder(db)
    graph_builder = GraphBuilder(db)
    llm = FakeLLM()
    enricher = GraphEnricher(db, llm)
    yield {
        "db": db,
        "store": store,
        "embedder": embedder,
        "graph_builder": graph_builder,
        "enricher": enricher,
        "llm": llm,
    }
    await db.close()


def _make_entry_json(title: str) -> dict:
    return {
        "short_title": title,
        "long_title": f"Entry about {title}",
        "knowledge_details": f"Details about {title}.",
        "entry_type": "factual_reference",
        "tags": ["test"],
    }


class _MultiChunkLLM(FakeLLM):
    """LLM that returns summary first, then entries for each extract call."""

    def __init__(self, summary: str, entries_per_call: list[list[dict]]):
        super().__init__()
        self._responses: list[str] = [summary]
        for entries in entries_per_call:
            self._responses.append(json.dumps(entries))
        self._call_index = 0

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        self.last_prompt = prompt
        self.last_system = system
        self.generate_count += 1
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return json.dumps([])  # Fallback: empty array


class TestChunkedIngestion:
    async def test_large_file_produces_multiple_chunks(self, ingester_deps, tmp_path, monkeypatch):
        """A file larger than chunk_size produces entries from multiple chunks."""
        monkeypatch.setenv("KB_INGEST_CHUNK_SIZE", "100")
        monkeypatch.setenv("KB_INGEST_CHUNK_OVERLAP", "0")

        # Two sections, each >100 chars
        content = "# Section One\n\n" + "a" * 100 + "\n\n# Section Two\n\n" + "b" * 100
        f = tmp_path / "big.md"
        f.write_text(content)

        deps = ingester_deps
        llm = _MultiChunkLLM(
            "Summary of big file.",
            [
                [_make_entry_json("entry from section one")],
                [_make_entry_json("entry from section two")],
            ],
        )
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "ingested"
        assert result.entry_count == 2
        assert result.chunks_processed >= 2

    async def test_small_file_single_chunk(self, ingester_deps, tmp_path):
        """A small file goes through as a single chunk — backward compatible."""
        f = tmp_path / "small.md"
        f.write_text("# Small\n\nTiny content.")

        deps = ingester_deps
        llm = _MultiChunkLLM(
            "Summary.",
            [[_make_entry_json("small entry")]],
        )
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "ingested"
        assert result.entry_count == 1
        assert result.chunks_processed == 1
        assert result.chunks_skipped == 0

    async def test_running_context_passed_to_subsequent_chunks(
        self, ingester_deps, tmp_path, monkeypatch
    ):
        """Entries from chunk 1 appear in running context for chunk 2."""
        monkeypatch.setenv("KB_INGEST_CHUNK_SIZE", "100")
        monkeypatch.setenv("KB_INGEST_CHUNK_OVERLAP", "0")

        content = "# Part A\n\n" + "x" * 100 + "\n\n# Part B\n\n" + "y" * 100
        f = tmp_path / "doc.md"
        f.write_text(content)

        class ContextCaptureLLM(FakeLLM):
            """Captures system prompts for each call to verify running context."""

            def __init__(self):
                super().__init__()
                self._call_index = 0
                self.system_prompts: list[str | None] = []
                self._responses = [
                    "Summary.",
                    json.dumps([_make_entry_json("alpha concept")]),
                    json.dumps([_make_entry_json("beta concept")]),
                ]

            async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
                self.generate_count += 1
                self.system_prompts.append(system)
                if self._call_index < len(self._responses):
                    resp = self._responses[self._call_index]
                    self._call_index += 1
                    return resp
                return json.dumps([])

        deps = ingester_deps
        llm = ContextCaptureLLM()
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path)
        assert result.action == "ingested"
        assert result.entry_count == 2

        # First extract call (system_prompts[1]) should NOT have running context
        assert "already been extracted" not in (llm.system_prompts[1] or "")

        # Second extract call (system_prompts[2]) should include "alpha concept"
        assert "already been extracted" in (llm.system_prompts[2] or "")
        assert "alpha concept" in (llm.system_prompts[2] or "")

    async def test_dry_run_chunks_file(self, ingester_deps, tmp_path, monkeypatch):
        """Dry run also chunks and reports chunk count."""
        monkeypatch.setenv("KB_INGEST_CHUNK_SIZE", "100")
        monkeypatch.setenv("KB_INGEST_CHUNK_OVERLAP", "0")

        content = "# A\n\n" + "a" * 100 + "\n\n# B\n\n" + "b" * 100
        f = tmp_path / "preview.md"
        f.write_text(content)

        deps = ingester_deps
        llm = _MultiChunkLLM(
            "Preview summary.",
            [
                [_make_entry_json("entry a")],
                [_make_entry_json("entry b")],
            ],
        )
        ingester = FileIngester(
            deps["db"],
            deps["store"],
            deps["embedder"],
            deps["graph_builder"],
            GraphEnricher(deps["db"], FakeLLM()),
            llm,
        )

        result = await ingester.ingest_file(f, base_dir=tmp_path, dry_run=True)
        assert result.action == "dry_run"
        assert result.entry_count == 2
        assert result.chunks_processed >= 2
