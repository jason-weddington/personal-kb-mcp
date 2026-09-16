"""Ollama embedding client with graceful degradation.

Env-free version of the embedding client: instead of reading
``KB_OLLAMA_URL`` / ``KB_EMBEDDING_MODEL`` / ``KB_OLLAMA_TIMEOUT`` from
``os.environ`` at each call site, the client takes an
:class:`kb_core.config.EmbeddingConfig` at construction time and reads
the snapshotted values off of it. The channel-side adapter
(``personal_kb.config.build_embedding_config``) is what reads env.

This client structurally satisfies the
:class:`kb_core.search.embedder_protocol.Embedder` and
:class:`~kb_core.search.embedder_protocol.BatchEmbedder` Protocols used
by hybrid and vector search.
"""

import logging

import httpx

from kb_core.config import EmbeddingConfig
from kb_core.db.backend import Database

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Generates embeddings via Ollama and stores them in the database."""

    def __init__(
        self,
        db: Database,
        config: EmbeddingConfig | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize with a database connection, embedding config, and optional HTTP client.

        ``config`` carries the env-free embedding tunables (Ollama URL,
        model, timeout). Construction sites build it from
        ``personal_kb.config.build_embedding_config()`` and pass it in.
        ``None`` falls back to defaults that mirror today's behavior.
        """
        self.db = db
        self._config = config if config is not None else EmbeddingConfig()
        self._http = http_client
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Ollama is reachable. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            resp = await client.get(
                f"{self._config.ollama_url}/api/tags",
                timeout=self._config.timeout,
            )
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
                f"{self._config.ollama_url}/api/embed",
                json={
                    "model": self._config.model,
                    "input": text,
                    "keep_alive": self._config.keep_alive,
                },
                timeout=self._config.timeout,
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

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Generate embeddings for multiple texts in a single Ollama call.

        Returns None if Ollama is unavailable or the call fails.
        """
        if not texts:
            return []
        if not await self.is_available():
            return None
        try:
            client = self._get_client()
            resp = await client.post(
                f"{self._config.ollama_url}/api/embed",
                json={
                    "model": self._config.model,
                    "input": texts,
                    "keep_alive": self._config.keep_alive,
                },
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings: list[list[float]] = data["embeddings"]
            if len(embeddings) != len(texts):
                logger.warning(
                    "Embedding count mismatch: got %d, expected %d",
                    len(embeddings),
                    len(texts),
                )
                return None
            return embeddings
        except Exception:
            logger.warning("Batch embedding generation failed", exc_info=True)
            self._available = None
            return None

    async def store_embeddings(self, entries: list[tuple[str, list[float]]]) -> None:
        """Store multiple embeddings. Each item is (entry_id, embedding)."""
        for entry_id, embedding in entries:
            await self.db.vector_store(entry_id, embedding)
        await self.db.commit()

    async def store_embedding(self, entry_id: str, embedding: list[float]) -> None:
        """Store an embedding via the database backend."""
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
        """Find similar entries by vector distance. Returns (entry_id, distance) pairs.

        Optional metadata filters are passed through to the backend so the
        KNN result set is restricted to matching entries at the SQL level.
        """
        return await self.db.vector_search(
            query_embedding,
            limit=limit,
            project_ref=project_ref,
            entry_type=entry_type,
            tags=tags,
            contributor=contributor,
            team=team,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    async def close(self) -> None:
        """Close the HTTP client if open."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
