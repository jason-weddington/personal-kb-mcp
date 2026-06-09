"""Ollama LLM client with graceful degradation.

Channel-agnostic: takes an :class:`~kb_core.config.OllamaProviderConfig`
at construction time. No environment reads — every tunable is on the
config.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from kb_core.config import OllamaProviderConfig
    from kb_core.llm.provider import Message

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """Generates text via Ollama's /api/generate endpoint."""

    def __init__(
        self,
        config: OllamaProviderConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize with an explicit config and optional HTTP client."""
        self._config = config
        self._http = http_client
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Ollama is reachable. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            resp = await client.get(f"{self._config.url}/api/tags", timeout=self._config.timeout)
            resp.raise_for_status()
            self._available = True
        except Exception:
            logger.warning("Ollama not available — LLM disabled")
            self._available = None
        return self._available is True

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        if not await self.is_available():
            return None
        try:
            client = self._get_client()
            payload: dict[str, object] = {
                "model": self._config.model,
                "prompt": prompt,
                "stream": False,
            }
            if system is not None:
                payload["system"] = system
            resp = await client.post(
                f"{self._config.url}/api/generate",
                json=payload,
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result: str = data["response"]
            return result
        except Exception:
            logger.warning("LLM generation failed", exc_info=True)
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history via /api/chat."""
        if not await self.is_available():
            return None
        try:
            client = self._get_client()
            api_messages: list[dict[str, str]] = []
            if system is not None:
                api_messages.append({"role": "system", "content": system})
            for m in messages:
                api_messages.append({"role": m["role"], "content": m["content"]})
            resp = await client.post(
                f"{self._config.url}/api/chat",
                json={
                    "model": self._config.model,
                    "messages": api_messages,
                    "stream": False,
                },
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            result: str = data["message"]["content"]
            return result
        except Exception:
            logger.warning("Ollama chat generation failed", exc_info=True)
            self._available = None
            return None

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    async def close(self) -> None:
        """Close the HTTP client if open."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None
