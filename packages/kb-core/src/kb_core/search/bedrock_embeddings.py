"""Bedrock embedding client (Amazon Titan Text Embeddings V2 by default).

Same surface as the Ollama :class:`~kb_core.search.embeddings.EmbeddingClient`
(it subclasses it for the vector-store and KNN methods) and the same graceful
degradation: every failure logs and returns ``None``, so search falls back to
full-text and the write path queues the entry for the embedding-retry worker.

Credentials and region come from the :class:`~kb_core.config.BedrockProviderConfig`
on :attr:`EmbeddingConfig.bedrock`, resolved by the same
:func:`~kb_core.llm.bedrock.build_runtime_client` the Bedrock LLM provider uses,
so embeddings and LLM calls sign with the same AWS account.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from kb_core.config import TITAN_V2_DIMENSIONS, BedrockProviderConfig, EmbeddingConfig
from kb_core.search.embeddings import EmbeddingClient

if TYPE_CHECKING:
    from kb_core.db.backend import Database

logger = logging.getLogger(__name__)

TITAN_MAX_INPUT_CHARS = 50_000
"""Titan V2 rejects ``inputText`` longer than this; longer text is truncated before the call."""

BEDROCK_EMBED_CONCURRENCY = 8
"""Parallel ``InvokeModel`` calls per batch; Titan embeds one text per request."""


class BedrockEmbeddingClient(EmbeddingClient):
    """Generates embeddings via the Bedrock ``InvokeModel`` API."""

    def __init__(self, db: Database, config: EmbeddingConfig) -> None:
        """Initialize with a database connection and a bedrock embedding config."""
        super().__init__(db, config)
        self._bedrock = config.bedrock if config.bedrock is not None else BedrockProviderConfig()
        self._client: Any = None
        self._auth_method: str | None = None

    def _get_runtime(self) -> Any:
        if self._client is None:
            from kb_core.llm.bedrock import build_runtime_client

            client, self._auth_method = build_runtime_client(self._bedrock)
            if client is None:
                logger.warning(
                    "aws-sdk-bedrock-runtime package not installed — Bedrock embeddings disabled"
                )
                return None
            self._client = client
        return self._client

    async def is_available(self) -> bool:
        """True when the SDK is installed, credentials resolve and the dimension is valid."""
        if self._available is True:
            return True
        if self._config.model.startswith("amazon.titan-embed-text-v2") and (
            self._config.dim not in TITAN_V2_DIMENSIONS
        ):
            logger.warning(
                "Bedrock embeddings disabled: %s accepts dimensions %s, KB_EMBEDDING_DIM is %d",
                self._config.model,
                TITAN_V2_DIMENSIONS,
                self._config.dim,
            )
            return False
        try:
            client = self._get_runtime()
        except Exception:
            logger.warning("Bedrock embedding client setup failed", exc_info=True)
            self._client = None
            return False
        if client is None:
            return False
        if self._auth_method is None:
            logger.warning("No AWS credentials found — Bedrock embeddings disabled")
            self._client = None
            return False
        self._available = True
        logger.info(
            "Bedrock embeddings enabled (auth=%s, model=%s, dim=%d)",
            self._auth_method,
            self._config.model,
            self._config.dim,
        )
        return True

    async def _invoke(self, text: str) -> list[float]:
        from aws_sdk_bedrock_runtime.models import InvokeModelInput

        if len(text) > TITAN_MAX_INPUT_CHARS:
            logger.info(
                "Bedrock embedding input truncated from %d to %d chars",
                len(text),
                TITAN_MAX_INPUT_CHARS,
            )
        body = {
            "inputText": text[:TITAN_MAX_INPUT_CHARS],
            "dimensions": self._config.dim,
            "normalize": True,
        }
        response = await asyncio.wait_for(
            self._get_runtime().invoke_model(
                InvokeModelInput(
                    model_id=self._config.model,
                    body=json.dumps(body).encode(),
                    # No content_type: aws-sdk-bedrock-runtime 0.4.0 then sends
                    # Content-Type twice (ours plus its octet-stream default) and
                    # Bedrock rejects the request with InvalidSignatureException.
                    accept="application/json",
                )
            ),
            timeout=self._config.timeout,
        )
        vector: list[float] = json.loads(response.body)["embedding"]
        if len(vector) != self._config.dim:
            msg = f"Bedrock returned {len(vector)} dimensions, expected {self._config.dim}"
            raise ValueError(msg)
        return vector

    async def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for the given text. Returns None if unavailable."""
        if not await self.is_available():
            return None
        try:
            return await self._invoke(text)
        except Exception:
            logger.warning("Bedrock embedding generation failed", exc_info=True)
            self._available = None
            return None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed each text with its own call, a few at a time. None if any call fails."""
        if not texts:
            return []
        if not await self.is_available():
            return None
        sem = asyncio.Semaphore(BEDROCK_EMBED_CONCURRENCY)

        async def one(text: str) -> list[float]:
            async with sem:
                return await self._invoke(text)

        try:
            return list(await asyncio.gather(*(one(t) for t in texts)))
        except Exception:
            logger.warning("Bedrock batch embedding failed", exc_info=True)
            self._available = None
            return None

    async def close(self) -> None:
        """No-op — the SDK client needs no cleanup."""
