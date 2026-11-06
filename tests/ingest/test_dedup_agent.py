"""Tests for KB-aware dedup agent."""

import json
import logging

import pytest_asyncio
from kb_core.models.search import SearchQuery
from kb_core.search.hybrid import RRF_K, hybrid_search

from personal_kb.db.connection import create_connection
from personal_kb.ingest.chunker import Chunk
from personal_kb.ingest.dedup_agent import DedupAgent, _parse_dedup_response
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore
from tests.conftest import FakeEmbedder, FakeLLM, ScriptedEmbedder

# Log record source for the gate assertions below.
_DEDUP_LOGGER = "kb_core.ingest.dedup_agent"

_SKIP_RESPONSE = json.dumps({"verdict": "skip", "reason": "covered", "existing_titles": []})


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


def _flat_chunk() -> Chunk:
    """Chunk that FTS-matches entry A ("photosynthesis chlorophyll") verbatim."""
    return Chunk(text="photosynthesis chlorophyll", index=0, start_char=0, heading=None)


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


async def _seed_pair(deps) -> dict[str, str]:
    """Seed entry A (FTS-matchable) and entry B (never FTS-matched by the chunk).

    Seeding goes through the fixture ``FakeEmbedder`` so the entries have real
    stored vectors; ``ScriptedEmbedder`` is only ever handed to ``DedupAgent``,
    which controls the vector leg the gate reads.
    """
    return {
        "a": await _seed_entry(deps["store"], deps["embedder"], "photosynthesis chlorophyll"),
        "b": await _seed_entry(deps["store"], deps["embedder"], "quokka xylophone"),
    }


class TestDedupAgentNoOverlap:
    async def test_empty_kb_returns_extract(self, dedup_deps):
        """When KB has no entries, dedup should return extract."""
        deps = dedup_deps
        llm = FakeLLM()
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.85)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"
        assert llm.generate_count == 0  # No LLM call needed

    async def test_low_similarity_returns_extract(self, dedup_deps):
        """When the best similarity is below the floor, returns extract without LLM."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "unrelated topic")

        llm = FakeLLM()
        # Floor no cosine similarity can reach, so nothing clears the gate
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=999.0)

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
        # Floor of zero so every similarity clears the gate
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

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
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

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
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"


class TestDedupAgentGracefulDegradation:
    async def test_llm_unavailable_returns_extract(self, dedup_deps):
        """When LLM is unavailable but the similarity clears the floor, returns extract."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "topic")

        llm = FakeLLM(available=False)
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"
        assert "unavailable" in result.reason

    async def test_llm_returns_none(self, dedup_deps):
        """When LLM returns None, returns extract."""
        deps = dedup_deps
        await _seed_entry(deps["store"], deps["embedder"], "topic")

        llm = FakeLLM(response=None, available=True)
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

        result = await agent.check_chunk(_make_chunk(), [])
        assert result.action == "extract"

    async def test_empty_chunk_returns_extract(self, dedup_deps):
        """Empty chunk text → extract."""
        deps = dedup_deps
        llm = FakeLLM()
        agent = DedupAgent(deps["db"], deps["embedder"], llm, similarity_floor=0.0)

        empty_chunk = Chunk(text="", index=0, start_char=0, heading=None)
        result = await agent.check_chunk(empty_chunk, [])
        assert result.action == "extract"


class TestDedupSimilarityGate:
    """The gate compares the best cosine similarity against the floor.

    The vector leg is scripted, so each case pins one branch of the gate:
    below the floor, at or above it (skip / partial / extract verdicts),
    the maximum-over-results read, and the no-similarity degradation path.
    """

    async def test_vector_only_below_floor_extracts(self, dedup_deps, caplog):
        """A below-floor similarity extracts with no LLM call, and is logged."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["b"], 0.50)]), llm)

        with caplog.at_level(logging.INFO, logger=_DEDUP_LOGGER):
            result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "extract"
        assert llm.generate_count == 0
        assert "below floor" in result.reason
        gate = [r.getMessage() for r in caplog.records if "op=gate" in r.getMessage()]
        assert gate, "the below-floor branch must be logged"
        assert "top_similarity=0.5000" in gate[0]
        assert "above_floor=False" in gate[0]

    async def test_both_legs_high_similarity_asks_llm(self, dedup_deps):
        """Similarity 0.93 clears the floor although the fused RRF score is 0.0328."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["a"], 0.07)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "skip"
        assert llm.generate_count == 1

    async def test_skip_verdict_vector_only(self, dedup_deps):
        """A vector-only hit at 0.93 clears the floor and reaches the LLM."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["b"], 0.07)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "skip"
        assert llm.generate_count == 1

    async def test_partial_verdict_carries_titles(self, dedup_deps):
        """Similarity 0.88 (distance 0.12) clears the floor and partial keeps titles."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(
            response=json.dumps(
                {
                    "verdict": "partial",
                    "reason": "some overlap",
                    "existing_titles": ["photosynthesis chlorophyll"],
                }
            )
        )
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["a"], 0.12)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "partial"
        assert result.existing_titles == ["photosynthesis chlorophyll"]
        assert llm.generate_count == 1

    async def test_extract_verdict_above_floor(self, dedup_deps):
        """The LLM may still say extract about a chunk that cleared the floor."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(
            response=json.dumps({"verdict": "extract", "reason": "new", "existing_titles": []})
        )
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["a"], 0.07)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "extract"
        assert llm.generate_count == 1

    async def test_gate_reads_max_similarity_over_results(self, dedup_deps):
        """The gate takes the best similarity, not the top-ranked result's."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        scripted = ScriptedEmbedder([(ids["b"], 0.10)])

        # Fixture shape: the RRF top result is the FTS-only entry A (no similarity),
        # and the 0.90 similarity sits on the lower-ranked vector-only entry B.
        results, _ = await hybrid_search(
            deps["db"],
            scripted,
            SearchQuery(query="photosynthesis chlorophyll", limit=5, include_superseded=True),
        )
        assert results[0].vector_similarity is None
        assert results[1].vector_similarity is not None
        assert abs(results[1].vector_similarity - 0.90) < 1e-9

        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], scripted, llm)
        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "skip"
        assert llm.generate_count == 1

    async def test_vector_leg_unavailable_extracts(self, dedup_deps, caplog):
        """No result carries a similarity → extract, logged as a vector-leg tripwire."""
        deps = dedup_deps
        await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder(available=False), llm)

        with caplog.at_level(logging.INFO, logger=_DEDUP_LOGGER):
            result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "extract"
        assert result.reason == "vector leg unavailable"
        assert result.reason != "no existing entries found"
        assert llm.generate_count == 0
        assert [r for r in caplog.records if "tripwire=vector_leg_unavailable" in r.getMessage()]

    async def test_constructor_fallback_floor_below(self, dedup_deps):
        """No explicit floor → the kb_core default 0.85 applies; 0.84 extracts."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["a"], 0.16)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "extract"
        assert llm.generate_count == 0
        assert "below floor 0.85" in result.reason

    async def test_constructor_fallback_floor_above(self, dedup_deps):
        """The default 0.85 floor is reached at similarity 0.86."""
        deps = dedup_deps
        ids = await _seed_pair(deps)
        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], ScriptedEmbedder([(ids["a"], 0.14)]), llm)

        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "skip"
        assert llm.generate_count == 1


class TestDedupFloorTripwire:
    async def test_unreachable_floor_warns(self, dedup_deps, caplog):
        """A floor above 1.0 can never fire and is reported at construction."""
        deps = dedup_deps
        with caplog.at_level(logging.INFO, logger=_DEDUP_LOGGER):
            DedupAgent(deps["db"], deps["embedder"], FakeLLM(), similarity_floor=1.2)

        assert [
            r for r in caplog.records if "tripwire=floor_unreachable floor=1.2000" in r.getMessage()
        ]

    async def test_default_floor_does_not_warn(self, dedup_deps, caplog):
        """The 0.85 default is reachable, so no tripwire is emitted."""
        deps = dedup_deps
        with caplog.at_level(logging.INFO, logger=_DEDUP_LOGGER):
            DedupAgent(deps["db"], deps["embedder"], FakeLLM())

        assert not [r for r in caplog.records if "tripwire=floor_unreachable" in r.getMessage()]


class TestDedupOldGateRegression:
    async def test_regression_old_rrf_gate_could_never_fire(self, dedup_deps):
        """The old 0.06-vs-RRF gate was unreachable; the similarity gate is not.

        The two numeric assertions pin the unchanged RRF arithmetic (the
        documentation of the old bug): one leg contributes 1/61 = 0.0164 and
        the reachable maximum over both legs, 2/61 = 0.0328, is below 0.06.
        They pass before and after the fix. The behavioural pair is the guard:
        against the old code this input returned ``extract`` with
        ``generate_count == 0``; now it returns the LLM's verdict.
        """
        deps = dedup_deps
        a_id = await _seed_entry(deps["store"], deps["embedder"], "photosynthesis chlorophyll")
        scripted = ScriptedEmbedder([(a_id, 0.07)])

        assert 2 * (1.0 / (RRF_K + 1)) < 0.06  # 0.03279

        results, _ = await hybrid_search(
            deps["db"],
            scripted,
            SearchQuery(query="zebra xylophone quokka", limit=5, include_superseded=True),
        )
        assert len(results) == 1
        assert abs(results[0].score - 1.0 / (RRF_K + 1)) < 1e-12  # 0.01639, below 0.06
        assert results[0].vector_similarity > 0.9

        llm = FakeLLM(response=_SKIP_RESPONSE)
        agent = DedupAgent(deps["db"], scripted, llm)
        result = await agent.check_chunk(_flat_chunk(), [])

        assert result.action == "skip"
        assert llm.generate_count == 1


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


class TestDedupSupersededStillMatches:
    async def test_superseded_entry_still_dedupes(self, dedup_deps):
        """Superseded content must still be found so re-ingestion dedupes."""
        deps = dedup_deps
        db = deps["db"]
        old_id = await _seed_entry(deps["store"], deps["embedder"], "Test heading content")
        new_id = await _seed_entry(deps["store"], deps["embedder"], "unrelated zzz replacement")
        await db.execute(
            "UPDATE knowledge_entries SET superseded_by = ? WHERE id = ?", (new_id, old_id)
        )
        await db.commit()

        llm = FakeLLM(
            response=json.dumps({"verdict": "skip", "reason": "covered", "existing_titles": []})
        )
        agent = DedupAgent(db, deps["embedder"], llm, similarity_floor=0.0)
        result = await agent.check_chunk(_make_chunk(), [])
        assert result.reason != "no existing entries found"
        assert llm.generate_count == 1
