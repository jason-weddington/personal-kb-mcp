"""Ollama embedding client with graceful degradation."""

import logging

import httpx

from personal_kb.config import get_embedding_model, get_ollama_timeout, get_ollama_url
from personal_kb.db.backend import Database

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates embeddings via Ollama and stores them in the database."""

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
        """Store an embedding via the database backend."""
        await self.db.vector_store(entry_id, embedding)
        await self.db.commit()

    async def search_similar(
        self, query_embedding: list[float], limit: int = 20
    ) -> list[tuple[str, float]]:
        """Find similar entries by vector distance. Returns (entry_id, distance) pairs."""
        return await self.db.vector_search(query_embedding, limit=limit)

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    async def close(self) -> None:
        """Close the HTTP client if open."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
