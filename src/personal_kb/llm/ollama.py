"""Ollama LLM client with graceful degradation."""

import logging

import httpx

from personal_kb.config import get_llm_model, get_llm_timeout, get_ollama_url

logger = logging.getLogger(__name__)


class OllamaLLMClient:
    """Generates text via Ollama's /api/generate endpoint."""

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        """Initialize with an optional HTTP client."""
        self._http = http_client
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check if Ollama is reachable. Only caches success — retries on failure."""
        if self._available is True:
            return True
        try:
            client = self._get_client()
            resp = await client.get(f"{get_ollama_url()}/api/tags", timeout=get_llm_timeout())
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
                "model": get_llm_model(),
                "prompt": prompt,
                "stream": False,
            }
            if system is not None:
                payload["system"] = system
            resp = await client.post(
                f"{get_ollama_url()}/api/generate",
                json=payload,
                timeout=get_llm_timeout(),
            )
            resp.raise_for_status()
            data = resp.json()
            result: str = data["response"]
            return result
        except Exception:
            logger.warning("LLM generation failed", exc_info=True)
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
