"""Eval framework fixtures — ControlledEmbedder, corpus loader, eval_kb."""

import json
import math
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.graph.builder import GraphBuilder
from personal_kb.models.entry import EntryType
from personal_kb.store.knowledge_store import KnowledgeStore

_EVAL_DIR = Path(__file__).parent


class ControlledEmbedder:
    """Embedder with pre-assigned vectors for deterministic ranking tests.

    Registered texts get exact vectors; unregistered texts fall back to
    a hash-based vector (same as FakeEmbedder but at dim=64).
    """

    def __init__(self, db, dim: int = 64):
        self.db = db
        self.dim = dim
        self._vectors: dict[str, list[float]] = {}
        self._available = True

    def register(self, name: str, vector: list[float]) -> None:
        """Pre-assign a unit vector for a given text key."""
        self._vectors[name] = vector

    async def is_available(self) -> bool:
        return self._available

    async def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        # Check registered vectors (match on substring for embedding_text)
        for name, vec in self._vectors.items():
            if name in text:
                return vec
        # Hash-based fallback for unregistered text
        return _hash_vector(text, self.dim)

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        await self.db.execute("DELETE FROM knowledge_vec WHERE entry_id = ?", (entry_id,))
        await self.db.execute(
            "INSERT INTO knowledge_vec (entry_id, embedding) VALUES (?, ?)",
            (entry_id, blob),
        )
        await self.db.commit()

    async def search_similar(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        cursor = await self.db.execute(
            """SELECT entry_id, distance
            FROM knowledge_vec
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?""",
            (blob, limit),
        )
        rows = await cursor.fetchall()
        return [(row[0], row[1]) for row in rows]

    async def close(self):
        pass


def _hash_vector(text: str, dim: int) -> list[float]:
    """Deterministic hash-based vector (same as FakeEmbedder)."""
    h = hash(text) & 0xFFFFFFFF
    vec = []
    for i in range(dim):
        val = ((h * (i + 1) * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
        vec.append(val * 2 - 1)
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec]


def build_cluster_vectors(dim: int, axis: int, names: list[str]) -> dict[str, list[float]]:
    """Build unit vectors for a cluster sharing a primary axis.

    Each name gets a vector with a strong primary component on `axis`
    and decreasing alignment — first name is closest to the cluster
    centroid (pure axis), later names drift slightly.

    Args:
        dim: Vector dimensionality.
        axis: Primary axis index for this cluster.
        names: Ordered by desired proximity to cluster centroid.

    Returns:
        Mapping of name → normalized vector.
    """
    result: dict[str, list[float]] = {}
    for i, name in enumerate(names):
        vec = [0.0] * dim
        # Strong primary component
        vec[axis] = 1.0
        # Increasing noise on secondary axes to spread items apart
        if i > 0:
            # Use a different secondary axis per item for spread
            secondary = (axis + i) % dim
            if secondary == axis:
                secondary = (secondary + 1) % dim
            vec[secondary] = 0.1 * i
        # Normalize to unit vector
        norm = math.sqrt(sum(v * v for v in vec))
        result[name] = [v / norm for v in vec]
    return result


async def load_corpus(
    db,
    store: KnowledgeStore,
    graph_builder: GraphBuilder,
    embedder: ControlledEmbedder,
) -> dict[str, str]:
    """Load corpus.json into the DB, returning short_title → entry_id mapping."""
    with open(_EVAL_DIR / "corpus.json") as f:
        entries = json.load(f)

    title_to_id: dict[str, str] = {}
    deactivate_ids: list[str] = []
    now = datetime.now(UTC)

    for item in entries:
        entry = await store.create_entry(
            short_title=item["short_title"],
            long_title=item["long_title"],
            knowledge_details=item["knowledge_details"],
            entry_type=EntryType(item["entry_type"]),
            project_ref=item.get("project_ref"),
            tags=item.get("tags"),
            hints=item.get("hints"),
            confidence_level=item.get("confidence_level", 0.9),
        )

        title_to_id[item["short_title"]] = entry.id

        # Backdate if days_old is set
        days_old = item.get("days_old")
        if days_old:
            past = now - timedelta(days=days_old)
            past_str = past.isoformat()
            await db.execute(
                "UPDATE knowledge_entries SET created_at = ?, updated_at = ? WHERE id = ?",
                (past_str, past_str, entry.id),
            )
            await db.commit()

        # Build deterministic graph edges (tags, hints, project)
        # Re-fetch entry to get correct timestamps
        fresh_entry = await store.get_entry(entry.id)
        await graph_builder.build_for_entry(fresh_entry)

        # Embed and store vector
        embedding = await embedder.embed(fresh_entry.embedding_text)
        if embedding:
            await embedder.store_embedding(entry.id, embedding)
            await store.mark_embedding(entry.id)

        # Track entries to deactivate
        if item.get("deactivate"):
            deactivate_ids.append(entry.id)

    # Deactivate entries after all are created (so supersedes chains work)
    for eid in deactivate_ids:
        await store.deactivate_entry(eid)

    return title_to_id


@pytest_asyncio.fixture
async def eval_kb():
    """Full eval environment: DB, embedder, corpus, and queries.

    Yields (db, embedder, title_to_id, queries) where:
    - db: in-memory aiosqlite connection with full schema
    - embedder: ControlledEmbedder with cluster vectors registered
    - title_to_id: mapping of short_title → entry_id
    - queries: list of query dicts from queries.json
    """
    db = await create_connection(":memory:", embedding_dim=64)

    embedder = ControlledEmbedder(db, dim=64)

    # Load similarity map and register cluster vectors
    with open(_EVAL_DIR / "similarity_map.json") as f:
        sim_map = json.load(f)

    with open(_EVAL_DIR / "queries.json") as f:
        queries = json.load(f)

    # Build query ID → actual query text mapping
    query_text_map = {q["id"]: q["query"] for q in queries}

    for cluster in sim_map["clusters"]:
        all_names = cluster["queries"] + cluster["entries"]
        vectors = build_cluster_vectors(sim_map["dim"], cluster["axis"], all_names)
        for name, vec in vectors.items():
            embedder.register(name, vec)
            # Also register the actual query text with the same vector
            if name in query_text_map:
                embedder.register(query_text_map[name], vec)

    store = KnowledgeStore(db)
    graph_builder = GraphBuilder(db)

    title_to_id = await load_corpus(db, store, graph_builder, embedder)

    yield db, embedder, title_to_id, queries

    await db.close()
