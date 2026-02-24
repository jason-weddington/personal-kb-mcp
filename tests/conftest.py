"""Shared test fixtures."""

import struct

import pytest_asyncio

from personal_kb.db.connection import create_connection
from personal_kb.store.knowledge_store import KnowledgeStore


@pytest_asyncio.fixture
async def db():
    """In-memory database with full schema and sqlite-vec."""
    conn = await create_connection(":memory:")
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def store(db):
    """Knowledge store backed by in-memory DB."""
    return KnowledgeStore(db)


class FakeEmbedder:
    """Deterministic fake embedder for testing.

    Produces embeddings based on a simple hash of the text,
    so identical texts always get identical vectors.
    """

    def __init__(self, db, dim: int = 1024):
        self.db = db
        self.dim = dim
        self._available = True

    async def is_available(self) -> bool:
        return self._available

    async def embed(self, text: str) -> list[float] | None:
        if not self._available:
            return None
        # Simple deterministic embedding from text hash
        h = hash(text) & 0xFFFFFFFF
        vec = []
        for i in range(self.dim):
            # Use a simple PRNG seeded with hash + position
            val = ((h * (i + 1) * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
            vec.append(val * 2 - 1)  # Range [-1, 1]
        # Normalize
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec]

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


@pytest_asyncio.fixture
async def fake_embedder(db):
    """Fake embedding client for tests."""
    return FakeEmbedder(db)
