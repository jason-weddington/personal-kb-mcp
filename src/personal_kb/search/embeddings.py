"""Ollama embedding client with graceful degradation."""

import logging
import struct

import httpx

from personal_kb.config import get_embedding_model, get_ollama_timeout, get_ollama_url
from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates embeddings via Ollama and stores them in sqlite-vec."""

    def __init__(self, db: Database, http_client: httpx.AsyncClient | None = None):
        """Initialize with a database connection and optional HTTP client."""
        self.db = db
        self._http = http_client
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Ollama is reachable. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            resp = await client.get(f"{get_ollama_url()}/api/tags", timeout=get_ollama_timeout())
            resp.raise_for_status()
            self._available = True
        except Exception:
            logger.warning("Ollama not available — embeddings disabled")
            self._available = None  # Will retry next call
        return self._available is True

    async def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for the given text. Returns None if unavailable."""
        if not await self.is_available():
            return None
        try:
            client = self._get_client()
            resp = await client.post(
                f"{get_ollama_url()}/api/embed",
                json={"model": get_embedding_model(), "input": text},
                timeout=get_ollama_timeout(),
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/embed returns {"embeddings": [[...]]}
            result: list[float] = data["embeddings"][0]
            return result
        except Exception:
            logger.warning("Embedding generation failed", exc_info=True)
            self._available = None  # Will retry next call
            return None

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        """Store an embedding in the vec0 table."""
        blob = _serialize_f32(embedding)
        # Upsert: delete then insert (vec0 doesn't support ON CONFLICT)
        await self.db.execute("DELETE FROM knowledge_vec WHERE entry_id = ?", (entry_id,))
        await self.db.execute(
            "INSERT INTO knowledge_vec (entry_id, embedding) VALUES (?, ?)",
            (entry_id, blob),
        )
        await self.db.commit()

    async def search_similar(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        """Find similar entries by vector distance. Returns (entry_id, distance) pairs."""
        blob = _serialize_f32(query_embedding)
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

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    async def close(self) -> None:
        """Close the HTTP client if open."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None


def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a list of floats to a compact binary format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)
