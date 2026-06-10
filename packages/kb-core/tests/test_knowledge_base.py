"""End-to-end tests for the :class:`KnowledgeBase` facade.

These tests exercise the public ``kb_core`` API the way library and
microservice consumers will use it — they construct a facade against a
brand-new SQLite database via :func:`kb_core.create_sqlite`, run the
whole CRUD + search + ingest + query pipeline through it, and assert
the engine still produces today's behavior.

Two correctness anchors:

* The facade's :meth:`KnowledgeBase.search` is a thin pass-through over
  :func:`kb_core.search.hybrid.hybrid_search`. The parity test below
  calls both with the same query and asserts byte-identical results
  (entry IDs, scores, match source). If a future change drifts those
  apart, this test catches it.
* The facade explicitly supports ``embedder=None`` as a first-class
  FTS-only configuration. The FTS-only test asserts that
  :meth:`search` still returns results when no embedder is configured
  — the engine must NOT fall back silently to "no results" when the
  vector leg is disabled.

Deterministic embeddings come from a small controlled :class:`FakeEmbedder`
that hashes its input — the same pattern used elsewhere in this repo, but
inlined here so the kb-core test package stays import-independent of the
main repo's ``tests/conftest.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from kb_core import (
    Attribution,
    EmbeddingConfig,
    IngestConfig,
    KnowledgeBase,
    SqliteConfig,
    create_sqlite,
)
from kb_core.config import KbConfig
from kb_core.models.entry import EntryType
from kb_core.models.search import SearchQuery
from kb_core.search.hybrid import hybrid_search

if TYPE_CHECKING:
    from kb_core.db.backend import Database


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic in-process embedder for tests.

    Implements both :class:`kb_core.search.embedder_protocol.Embedder` and
    :class:`~kb_core.search.embedder_protocol.BatchEmbedder` so the
    facade's full store/ingest pipelines can exercise it. Vectors come
    from a simple hash, so identical text always produces identical
    vectors — no Ollama needed.
    """

    def __init__(self, db: Database, dim: int = 1024) -> None:
        """Bind to the test database; store a fixed embedding dimensionality."""
        self.db = db
        self.dim = dim

    async def is_available(self) -> bool:
        """Always available for tests."""
        return True

    async def embed(self, text: str) -> list[float] | None:
        """Hash ``text`` to a deterministic normalized vector."""
        h = hash(text) & 0xFFFFFFFF
        vec: list[float] = []
        for i in range(self.dim):
            val = ((h * (i + 1) * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
            vec.append(val * 2 - 1)
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Batch wrapper over :meth:`embed` — single-call semantics for tests."""
        out: list[list[float]] = []
        for text in texts:
            vec = await self.embed(text)
            if vec is None:
                return None
            out.append(vec)
        return out

    async def store_embeddings(self, entries: list[tuple[str, list[float]]]) -> None:
        """Persist each ``(entry_id, vector)`` pair to the vector store."""
        for entry_id, embedding in entries:
            await self.db.vector_store(entry_id, embedding)
        await self.db.commit()

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        """Single-entry persist helper used by the store CRUD path."""
        await self.db.vector_store(entry_id, embedding)
        await self.db.commit()

    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 20,
        *,
        project_ref: str | None = None,
        entry_type: str | None = None,
        tags: list[str] | None = None,
        contributor: str | None = None,
        team: str | None = None,
    ) -> list[tuple[str, float]]:
        """Forward to the backend's KNN query with metadata filters."""
        return await self.db.vector_search(
            query_embedding,
            limit=limit,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
            contributor=contributor,
            team=team,
        )

    async def close(self) -> None:
        """No-op (no HTTP client to clean up)."""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def kb_with_embedder(tmp_path: Any) -> Any:
    """Open a :class:`KnowledgeBase` over a temp SQLite file with the fake embedder.

    Yields the open facade and tears it down — using the ``async with``
    contract the public API documents.
    """
    db_path = tmp_path / "kb.db"
    kb = await create_sqlite(db_path)
    # Inject a deterministic embedder after open.
    fake = FakeEmbedder(kb.db)
    kb._embedder = fake  # type: ignore[attr-defined]
    try:
        yield kb, fake
    finally:
        await kb.close()


@pytest.fixture
async def kb_fts_only(tmp_path: Any) -> Any:
    """Open a facade with explicit ``embedder=None`` (FTS-only)."""
    db_path = tmp_path / "kb_fts.db"
    kb = await create_sqlite(db_path)  # default: no embedding
    try:
        yield kb
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


async def test_create_sqlite_applies_schema_and_opens_clean_db(tmp_path: Any) -> None:
    """A brand-new SQLite file gets its schema applied automatically.

    Library consumers hand the facade a path that may not exist yet; the
    facade has to apply the schema on open so search_events, knowledge_*,
    graph_*, etc. all exist. We verify by reading the tables list.
    """
    kb = await create_sqlite(tmp_path / "fresh.db")
    try:
        cursor = await kb.db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        names = {row[0] for row in rows}
        # Required tables for the engine — search_events is the W2-flagged
        # precondition that the facade MUST apply on open.
        assert "knowledge_entries" in names
        assert "knowledge_fts" in names
        assert "graph_nodes" in names
        assert "graph_edges" in names
        assert "search_events" in names
        assert "agent_feedback" in names
        assert "audit_events" in names
        assert "ingested_files" in names
    finally:
        await kb.close()


async def test_create_sqlite_default_is_fts_only(tmp_path: Any) -> None:
    """The factory default builds NO embedder — FTS-only is a first-class choice."""
    kb = await create_sqlite(tmp_path / "fts.db")
    try:
        assert kb.embedder is None
    finally:
        await kb.close()


async def test_create_sqlite_with_embedding_config_builds_embedder(tmp_path: Any) -> None:
    """Explicit :class:`EmbeddingConfig` opts the caller into Ollama embeddings."""
    kb = await create_sqlite(
        tmp_path / "embed.db",
        embedding=EmbeddingConfig(),
    )
    try:
        assert kb.embedder is not None
    finally:
        await kb.close()


async def test_context_manager_closes_db(tmp_path: Any) -> None:
    """``async with`` releases the DB connection on exit."""
    db_path = tmp_path / "ctx.db"
    async with await create_sqlite(db_path) as kb:
        # Use the DB.
        cursor = await kb.db.execute("SELECT 1")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 1


async def test_knowledge_base_create_from_kb_config(tmp_path: Any) -> None:
    """The low-level :meth:`KnowledgeBase.create` works the same as the factory."""
    config = KbConfig(
        database=SqliteConfig(path=tmp_path / "cfg.db"),
        embedding=None,
        attribution=Attribution(contributor="alice", team="alphas"),
    )
    kb = await KnowledgeBase.create(config)
    try:
        assert kb.config.attribution.contributor == "alice"
        assert kb.embedder is None
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Store CRUD
# ---------------------------------------------------------------------------


async def test_store_creates_versioned_entry_with_graph_edges(
    kb_with_embedder: Any,
) -> None:
    """The store path runs create + embed + deterministic graph builder."""
    kb, _embedder = kb_with_embedder
    entry = await kb.store(
        short_title="async sqlite",
        long_title="Using aiosqlite for async DB access",
        knowledge_details="aiosqlite wraps sqlite3 in async APIs.",
        entry_type=EntryType.PATTERN_CONVENTION,
        project_ref="personal-kb",
        tags=["python", "sqlite"],
        enrich=False,  # No LLM in this test path.
    )
    assert entry.id.startswith("kb-")
    assert entry.version == 1
    assert entry.has_embedding is False  # Set asynchronously after embed

    # Re-fetch — embedding flag is updated by _embed_one().
    refreshed = await kb.get(entry.id)
    assert refreshed is not None
    assert refreshed.has_embedding is True

    # Deterministic graph edges: project + tag edges should exist.
    neighbors = await kb.graph.neighbors(entry.id)
    edge_types = {edge_type for _nid, edge_type, _dir in neighbors}
    assert "has_tag" in edge_types
    assert "in_project" in edge_types


async def test_store_batch_creates_multiple_entries(kb_with_embedder: Any) -> None:
    """:meth:`store_batch` creates entries in input order; embedder + graph fire."""
    kb, _ = kb_with_embedder
    created = await kb.store_batch(
        [
            {
                "short_title": "alpha",
                "long_title": "Alpha entry",
                "knowledge_details": "Alpha details.",
                "tags": ["t1"],
            },
            {
                "short_title": "beta",
                "long_title": "Beta entry",
                "knowledge_details": "Beta details.",
                "tags": ["t2"],
            },
        ],
        enrich=False,
    )
    assert len(created) == 2
    assert [e.short_title for e in created] == ["alpha", "beta"]


async def test_update_creates_new_version(kb_with_embedder: Any) -> None:
    """:meth:`update` writes a new entry_versions row and bumps ``version``."""
    kb, _ = kb_with_embedder
    entry = await kb.store(
        short_title="orig",
        long_title="Original",
        knowledge_details="Original content.",
        enrich=False,
    )
    updated = await kb.update(
        entry.id,
        knowledge_details="New content.",
        change_reason="Reworded",
        enrich=False,
    )
    assert updated.version == 2
    assert updated.knowledge_details == "New content."


async def test_deactivate_then_reactivate_roundtrip(kb_with_embedder: Any) -> None:
    """:meth:`deactivate` and :meth:`reactivate` flip the ``is_active`` flag."""
    kb, _ = kb_with_embedder
    entry = await kb.store(
        short_title="x",
        long_title="X",
        knowledge_details="x",
        enrich=False,
    )
    deactivated = await kb.deactivate(entry.id)
    assert deactivated.is_active is False

    reactivated = await kb.reactivate(entry.id)
    assert reactivated.is_active is True


async def test_get_returns_none_for_missing_entry(kb_with_embedder: Any) -> None:
    """Missing IDs return ``None`` rather than raising."""
    kb, _ = kb_with_embedder
    assert await kb.get("kb-99999") is None


async def test_bulk_update_applies_metadata_to_matching_entries(
    kb_with_embedder: Any,
) -> None:
    """:meth:`bulk_update` writes new versions for every matched entry."""
    kb, _ = kb_with_embedder
    for i in range(3):
        await kb.store(
            short_title=f"e{i}",
            long_title=f"Entry {i}",
            knowledge_details="details",
            project_ref="alpha",
            tags=["k"],
            enrich=False,
        )
    results = await kb.bulk_update(
        filters={"project_ref": "alpha"},
        updates={"tags_add": ["bulk"]},
    )
    assert len(results) == 3
    for _before, after in results:
        assert "bulk" in after.tags


# ---------------------------------------------------------------------------
# Search — parity with hybrid_search + FTS-only path
# ---------------------------------------------------------------------------


async def test_search_parity_with_direct_hybrid_search(kb_with_embedder: Any) -> None:
    """:meth:`KnowledgeBase.search` returns IDENTICAL results to direct ``hybrid_search``.

    The facade must be a thin pass-through over the engine. Calling
    both with the same query against the same DB must produce the same
    entry IDs, the same scores, and the same match source.
    """
    kb, embedder = kb_with_embedder

    # Seed a small corpus.
    for short_title, details, tags in [
        ("async sqlite", "Using aiosqlite for async DB access.", ["python", "sqlite"]),
        ("rrf ranking", "Reciprocal rank fusion combines FTS + vector.", ["search"]),
        ("graph builder", "Deterministic edges from tags + projects.", ["graph"]),
    ]:
        await kb.store(
            short_title=short_title,
            long_title=short_title,
            knowledge_details=details,
            tags=tags,
            enrich=False,
        )

    query = SearchQuery(query="async sqlite python", limit=5)

    facade_results, facade_filtered = await kb.search(query)
    direct_results, direct_filtered = await hybrid_search(kb.db, embedder, query)

    assert facade_filtered == direct_filtered
    assert [r.entry.id for r in facade_results] == [r.entry.id for r in direct_results]
    assert [round(r.score, 8) for r in facade_results] == [
        round(r.score, 8) for r in direct_results
    ]
    assert [r.match_source for r in facade_results] == [r.match_source for r in direct_results]


async def test_search_fts_only_returns_results_without_embedder(
    kb_fts_only: Any,
) -> None:
    """With ``embedder=None``, search still runs (FTS-only path).

    Asserts the "first-class FTS-only" promise: no silent "no results"
    when the vector leg is disabled.
    """
    kb = kb_fts_only
    await kb.store(
        short_title="sqlite vector search",
        long_title="sqlite-vec extension provides KNN over BLOBs",
        knowledge_details="The sqlite-vec extension implements approximate nearest neighbour.",
        tags=["sqlite", "vector"],
        enrich=False,
    )
    # Facade-side search.
    results, _ = await kb.search(SearchQuery(query="sqlite vector", limit=5))
    assert len(results) >= 1
    # Match source should be ``"fts"`` because no embedder was configured.
    assert results[0].match_source == "fts"


# ---------------------------------------------------------------------------
# Embed accessors
# ---------------------------------------------------------------------------


async def test_embed_returns_none_when_no_embedder(kb_fts_only: Any) -> None:
    """:meth:`embed` and :meth:`embed_batch` return ``None`` in FTS-only mode."""
    kb = kb_fts_only
    assert await kb.embed("hello") is None
    assert await kb.embed_batch(["a", "b"]) is None


async def test_embed_returns_vector_with_fake_embedder(kb_with_embedder: Any) -> None:
    """With an embedder configured, :meth:`embed` returns a vector."""
    kb, _ = kb_with_embedder
    vec = await kb.embed("hello world")
    assert vec is not None
    assert len(vec) == 1024


# ---------------------------------------------------------------------------
# Project context + maps
# ---------------------------------------------------------------------------


async def test_preflight_returns_project_context(kb_with_embedder: Any) -> None:
    """:meth:`preflight` returns a non-empty ToC for a project with entries."""
    kb, _ = kb_with_embedder
    await kb.store(
        short_title="convention",
        long_title="Our convention",
        knowledge_details="Always use UTC.",
        entry_type=EntryType.PATTERN_CONVENTION,
        project_ref="alpha",
        enrich=False,
    )
    summary = await kb.preflight("alpha")
    assert "alpha" in summary
    assert "convention" in summary


async def test_maps_for_project_returns_active_mental_maps(
    kb_with_embedder: Any,
) -> None:
    """:meth:`maps_for_project` returns only active ``mental_map`` entries."""
    kb, _ = kb_with_embedder

    # Seed two maps + one non-map.
    map1 = await kb.store(
        short_title="map A",
        long_title="Map A long title",
        knowledge_details="Map A details.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="alpha",
        enrich=False,
    )
    await kb.store(
        short_title="map B",
        long_title="Map B long title",
        knowledge_details="Map B details.",
        entry_type=EntryType.MENTAL_MAP,
        project_ref="alpha",
        enrich=False,
    )
    await kb.store(
        short_title="decision X",
        long_title="Decision X",
        knowledge_details="Some decision.",
        entry_type=EntryType.DECISION,
        project_ref="alpha",
        enrich=False,
    )

    maps = await kb.maps_for_project("alpha")
    assert len(maps) == 2
    map_ids = {m["id"] for m in maps}
    assert map1.id in map_ids
    # Schema: just id + titles.
    assert set(maps[0].keys()) == {"id", "short_title", "long_title"}

    # Deactivating one removes it from the maps list.
    await kb.deactivate(map1.id)
    maps_after = await kb.maps_for_project("alpha")
    assert len(maps_after) == 1
    assert map1.id not in {m["id"] for m in maps_after}


# ---------------------------------------------------------------------------
# Graph accessor
# ---------------------------------------------------------------------------


async def test_graph_accessor_exposes_traversals(kb_with_embedder: Any) -> None:
    """The :attr:`graph` accessor's traversal helpers all work end-to-end."""
    kb, _ = kb_with_embedder

    # Seed two entries sharing a tag so they're connected in the graph.
    e1 = await kb.store(
        short_title="alpha note",
        long_title="Alpha",
        knowledge_details="A note about alpha.",
        tags=["shared"],
        project_ref="proj-1",
        enrich=False,
    )
    e2 = await kb.store(
        short_title="beta note",
        long_title="Beta",
        knowledge_details="A note about beta.",
        tags=["shared"],
        project_ref="proj-1",
        enrich=False,
    )

    # Neighbors of e1: should include the shared tag node.
    neighbors = await kb.graph.neighbors(e1.id)
    assert any(nid == "tag:shared" for nid, _et, _dir in neighbors)

    # entries_for_scope: both entries are in proj-1.
    in_project = await kb.graph.entries_for_scope("project:proj-1")
    assert e1.id in in_project
    assert e2.id in in_project

    # Vocabulary: should expose the "tag" type.
    vocab = await kb.graph.vocabulary()
    assert "tag" in vocab
    assert "shared" in vocab["tag"]

    # BFS from e1 reaches e2 via the shared tag.
    bfs = await kb.graph.bfs_entries(e1.id, max_depth=2)
    reached_ids = {nid for nid, _depth, _path in bfs}
    assert e2.id in reached_ids

    # find_path between e1 and e2 returns a non-empty path.
    path = await kb.graph.find_path(e1.id, e2.id, max_depth=4)
    assert path is not None
    assert len(path) > 0

    # supersedes_chain: only contains the entry itself when nothing supersedes.
    chain = await kb.graph.supersedes_chain(e1.id)
    assert chain == [e1.id]


# ---------------------------------------------------------------------------
# Ask / summarize (no LLM configured) — exercise the no-LLM fallback paths
# ---------------------------------------------------------------------------


async def test_ask_without_llm_uses_single_shot_path(kb_with_embedder: Any) -> None:
    """:meth:`ask` falls back to the single-shot planner path when ``query_llm`` is None.

    No LLM is configured, so the planner path skips refinement and goes
    straight to :func:`_auto_search_entries`. The returned tuple has
    ``agent_turns == 0`` (no LLM turns happened).
    """
    kb, _ = kb_with_embedder
    await kb.store(
        short_title="ask target",
        long_title="Ask target",
        knowledge_details="Some content the ask path can match.",
        tags=["t"],
        enrich=False,
    )
    entries, agent_turns = await kb.ask("ask target", agentic=False, max_tool_calls=0)
    assert agent_turns == 0
    assert any(e.short_title == "ask target" for e, _ctx in entries)


async def test_summarize_without_llm_returns_formatted_fallback(
    kb_with_embedder: Any,
) -> None:
    """:meth:`summarize` falls back to formatted raw entries when no LLM is configured."""
    kb, _ = kb_with_embedder
    await kb.store(
        short_title="summarize target",
        long_title="Summarize target",
        knowledge_details="Something to summarize.",
        enrich=False,
    )
    out = await kb.summarize("summarize target", agentic=False, max_tool_calls=0)
    # The fallback path prefixes a "LLM unavailable" notice.
    assert "LLM unavailable" in out or "summarize target" in out


# ---------------------------------------------------------------------------
# embedder=None must not be silent fallback
# ---------------------------------------------------------------------------


async def test_embedder_none_is_explicit_choice_not_fallback(tmp_path: Any) -> None:
    """``embedder=None`` is an explicit FTS-only choice — the facade doesn't paper over it.

    A consumer who explicitly passes ``embedding=None`` to the factory
    MUST end up with ``kb.embedder is None`` and search must use the
    FTS-only path (match_source == "fts"). No silent fallback to
    "I'll just retry Ollama at search time" — that's the W5 contract.
    """
    kb = await create_sqlite(tmp_path / "explicit.db", embedding=None)
    try:
        assert kb.embedder is None
        await kb.store(
            short_title="explicit fts",
            long_title="Explicit FTS-only mode",
            knowledge_details="No vector leg should run.",
            enrich=False,
        )
        results, _ = await kb.search(SearchQuery(query="explicit fts", limit=5))
        # Match source must be "fts" since there's no embedder.
        assert all(r.match_source == "fts" for r in results)
    finally:
        await kb.close()


# ---------------------------------------------------------------------------
# Per-request attribution kwargs + dry_run threading on ingest_text
# ---------------------------------------------------------------------------


class _SequenceLLM:
    """LLM test double that returns pre-scripted responses in order.

    Cycles back to the last response once the list is exhausted, so
    a two-element list works for any number of chunks.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    async def is_available(self) -> bool:
        return True

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        resp = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        return resp

    async def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        system: str | None = None,
    ) -> str | None:
        return await self.generate(
            next((m["content"] for m in reversed(messages) if m["role"] == "user"), ""),
            system=system,
        )

    async def close(self) -> None:
        pass


_INGEST_ENTRY_JSON = (
    '[{"short_title": "test entry", "long_title": "A test knowledge entry",'
    ' "knowledge_details": "This is a test entry for attribution testing.",'
    ' "entry_type": "lesson_learned", "tags": ["test"]}]'
)


async def _make_ingest_kb(
    tmp_path: Any,
    *,
    contributor: str | None = None,
    team: str | None = None,
) -> KnowledgeBase:
    """Build a KB wired for ingest tests: fake LLM + fake embedder + no dedup."""
    llm = _SequenceLLM(["Test summary.", _INGEST_ENTRY_JSON])
    kb = await create_sqlite(
        tmp_path / "ingest.db",
        attribution=Attribution(contributor=contributor, team=team),
        ingest=IngestConfig(agentic_ingest=False),
        extraction_llm=llm,
    )
    # Inject a deterministic batch embedder (both extraction and vector store need it).
    fake = FakeEmbedder(kb.db)
    kb._embedder = fake  # type: ignore[attr-defined]
    return kb


async def test_ingest_text_per_call_attribution_overrides_ctor(tmp_path: Any) -> None:
    """``contributor``/``team`` kwargs on :meth:`ingest_text` override ctor attribution.

    The KB is constructed with ``contributor="ctor_user"``/``team="ctor_team"``.
    When :meth:`ingest_text` is called with ``contributor="alice@x"``/``team="t1"``,
    the created entries must carry the per-call attribution — not the ctor values.
    """
    kb = await _make_ingest_kb(tmp_path, contributor="ctor_user", team="ctor_team")
    try:
        result = await kb.ingest_text(
            "A note about a useful pattern.",
            "notes.md",
            contributor="alice@x",
            team="t1",
        )
        assert result.action == "ingested", f"Unexpected action: {result.action}"
        assert len(result.entry_ids) >= 1, "Expected at least one entry to be created"
        entry = await kb.get(result.entry_ids[0])
        assert entry is not None
        assert entry.contributor == "alice@x"
        assert entry.team == "t1"
    finally:
        await kb.close()


async def test_ingest_text_default_attribution_uses_ctor(tmp_path: Any) -> None:
    """Without per-call kwargs, :meth:`ingest_text` uses the ctor attribution (regression guard).

    The KB is constructed with ``contributor="ctor_user"``/``team="ctor_team"``.
    Calling :meth:`ingest_text` without attribution kwargs must produce entries
    stamped with the ctor values, preserving today's behavior byte-for-byte.
    """
    kb = await _make_ingest_kb(tmp_path, contributor="ctor_user", team="ctor_team")
    try:
        result = await kb.ingest_text(
            "A note about a useful pattern.",
            "notes.md",
        )
        assert result.action == "ingested", f"Unexpected action: {result.action}"
        assert len(result.entry_ids) >= 1, "Expected at least one entry to be created"
        entry = await kb.get(result.entry_ids[0])
        assert entry is not None
        assert entry.contributor == "ctor_user"
        assert entry.team == "ctor_team"
    finally:
        await kb.close()


async def test_ingest_text_dry_run_creates_no_entries(tmp_path: Any) -> None:
    """``dry_run=True`` on :meth:`ingest_text` runs the pipeline without storing entries.

    The returned :class:`FileResult` must have ``action="dry_run"`` and the
    database must remain empty (zero ``knowledge_entries`` rows).
    """
    kb = await _make_ingest_kb(tmp_path, contributor="ctor_user", team="ctor_team")
    try:
        result = await kb.ingest_text(
            "A note about a useful pattern.",
            "notes.md",
            dry_run=True,
        )
        assert result.action == "dry_run", f"Unexpected action: {result.action}"
        # entry_ids must be empty — nothing was persisted.
        assert result.entry_ids == []
        # Verify no rows in the knowledge_entries table.
        cursor = await kb.db.execute("SELECT COUNT(*) FROM knowledge_entries WHERE is_active = 1")
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] == 0, f"Expected 0 active entries, got {row[0]}"
    finally:
        await kb.close()
