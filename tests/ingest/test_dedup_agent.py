"""Tests for KB-aware dedup agent."""

import json

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.ingest.chunker import Chunk
from personal_kb.ingest.dedup_agent import DedupAgent, _parse_dedup_response
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder, FakeLLM


@pytest_asyncio.fixture
async def dedup_deps():
    """Create dependencies for DedupAgent tests with in-memory DB."""
    db = await create_connection(":memory:")
    store = KnowledgeStore(db)
    embedder = FakeEmbedder(db)
    yield {"db": db, "store": store, "embedder": embedder}
    await db.close()


def _make_chunk(text: str = "# Test\n\nSome content.", heading: str | None = "Test") -> Chunk:
    return Chunk(text=text, index=0, start_char=0, heading=heading)


async def _seed_entry(store: KnowledgeStore, embedder: FakeEmbedder, title: str) -> str:
    """Create a KB entry and embed it for search."""
    entry = await store.create_entry(
        short_title=title,
        long_title=f"Entry about {title}",
        knowledge_details=f"Detailed knowledge about {title}.",
        entry_type=EntryType.FACTUAL_REFERENCE,
        tags=["test"],
    )
    embedding = await embedder.embed(entry.embedding_text)
    if embedding:
        await embedder.store_embedding(entry.id, embedding)
        await store.mark_embedding(entry.id, True)
    return entry.id


class TestDedupAgentNoOverlap:
    async def test_empty_kb_returns_extract(self, dedup_deps):
        """When KB has no entries, dedup should return extract."""
        deps = dedup_deps
        llm = FakeLLM()
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.06)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"
        assert llm.generate_count == 0  # No LLM call needed

    async def test_low_score_returns_extract(self, dedup_deps):
        """When search scores are below threshold, returns extract without LLM."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "unrelated topic")

        llm = FakeLLM()
        # Very high threshold so score will be below
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=999.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"
        assert llm.generate_count == 0


class TestDedupAgentWithOverlap:
    async def test_skip_verdict(self, dedup_deps):
        """LLM says skip → DedupResult(action='skip')."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "Test heading content")

        llm = FakeLLM(
            response=json.dumps(
                {"verdict": "skip", "reason": "fully covered", "existing_titles": []}
            )
        )
        # Very low threshold so we always pass to LLM
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "skip"
        assert llm.generate_count == 1

    async def test_partial_verdict_with_titles(self, dedup_deps):
        """LLM says partial → DedupResult(action='partial', existing_titles=[...])."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "overlapping concept")

        llm = FakeLLM(
            response=json.dumps(
                {
                    "verdict": "partial",
                    "reason": "some overlap",
                    "existing_titles": ["overlapping concept"],
                }
            )
        )
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "partial"
        assert "overlapping concept" in result.existing_titles

    async def test_extract_verdict(self, dedup_deps):
        """LLM says extract → DedupResult(action='extract')."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "something")

        llm = FakeLLM(
            response=json.dumps(
                {"verdict": "extract", "reason": "mostly new", "existing_titles": []}
            )
        )
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"


class TestDedupAgentGracefulDegradation:
    async def test_llm_unavailable_returns_extract(self, dedup_deps):
        """When LLM is unavailable but search hits threshold, returns extract."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "topic")

        llm = FakeLLM(available=False)
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"
        assert "unavailable" in result.reason

    async def test_llm_returns_none(self, dedup_deps):
        """When LLM returns None, returns extract."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "topic")

        llm = FakeLLM(response=None, available=True)
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"

    async def test_empty_chunk_returns_extract(self, dedup_deps):
        """Empty chunk text → extract."""
        deps = dedup_deps
        llm = FakeLLM()
        agent = DedupAgent(deps["db"], deps["embedder"], llm, threshold=0.0)

        empty_chunk = Chunk(text="", index=0, start_char=0, heading=None)
        result = await agent.check_chunk(empty_chunk, [])
        assert result.action == "extract"


class TestParseDedupResponse:
    def test_valid_skip(self):
        raw = json.dumps({"verdict": "skip", "reason": "covered", "existing_titles": []})
        result = _parse_dedup_response(raw)
        assert result.action == "skip"
        assert result.reason == "covered"

    def test_valid_partial_with_titles(self):
        raw = json.dumps(
            {"verdict": "partial", "reason": "some overlap", "existing_titles": ["a", "b"]}
        )
        result = _parse_dedup_response(raw)
        assert result.action == "partial"
        assert result.existing_titles == ["a", "b"]

    def test_valid_extract(self):
        raw = json.dumps({"verdict": "extract", "reason": "new", "existing_titles": []})
        result = _parse_dedup_response(raw)
        assert result.action == "extract"

    def test_strips_markdown_fences(self):
        inner = json.dumps({"verdict": "skip", "reason": "test", "existing_titles": []})
        raw = f"```json\n{inner}\n```"
        result = _parse_dedup_response(raw)
        assert result.action == "skip"

    def test_malformed_json_returns_extract(self):
        result = _parse_dedup_response("not json at all")
        assert result.action == "extract"

    def test_no_json_object_returns_extract(self):
        result = _parse_dedup_response("here is some text without json")
        assert result.action == "extract"

    def test_invalid_verdict_defaults_to_extract(self):
        raw = json.dumps({"verdict": "invalid", "reason": "test", "existing_titles": []})
        result = _parse_dedup_response(raw)
        assert result.action == "extract"

    def test_missing_verdict_defaults_to_extract(self):
        raw = json.dumps({"reason": "test"})
        result = _parse_dedup_response(raw)
        assert result.action == "extract"

    def test_non_list_existing_titles_ignored(self):
        raw = json.dumps({"verdict": "partial", "reason": "test", "existing_titles": "not a list"})
        result = _parse_dedup_response(raw)
        assert result.existing_titles == []

    def test_surrounding_text(self):
        inner = json.dumps({"verdict": "skip", "reason": "done", "existing_titles": []})
        raw = f"Here is my response:\n{inner}\nDone."
        result = _parse_dedup_response(raw)
        assert result.action == "skip"
