"""Anthropic LLM client with graceful degradation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from personal_kb.config import get_anthropic_model, get_anthropic_timeout

if TYPE_CHECKING:
    from personal_kb.llm.provider import Message

logger = logging.getLogger(__name__)


_SONNET_MODEL = "claude-sonnet-4-6"


class AnthropicLLMClient:
    """Generates text via the Anthropic Messages API."""

    def __init__(self, *, model_override: str | None = None) -> None:
        """Initialize with lazy client creation."""
        self._client: Any = None
        self._available: bool | None = None
        self._model_override = model_override

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure."""
        if self._available is True:
            return True
        # Verify the SDK is importable and a key is configured
        try:
            client = self._get_client()
            if client is None:
                return False
            import os

            if not os.environ.get("ANTHROPIC_API_KEY"):
                logger.warning("ANTHROPIC_API_KEY not set — Anthropic LLM disabled")
                return False
            # Key is set — assume available until proven otherwise
            # (first generate() will confirm and cache _available = True)
            return True
        except Exception:
            return False

    def _model(self) -> str:
        return self._model_override or get_anthropic_model()

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        try:
            client = self._get_client()
            if client is None:
                return None

            kwargs: dict[str, Any] = {
                "model": self._model(),
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system is not None:
                kwargs["system"] = system

            response = await client.messages.create(
                **kwargs,
                timeout=get_anthropic_timeout(),
            )
            result: str = response.content[0].text
            self._available = True
            return result
        except Exception:
            logger.warning("Anthropic generation failed", exc_info=True)
            self._available = None
            return None

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history."""
        try:
            client = self._get_client()
            if client is None:
                return None

            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            kwargs: dict[str, Any] = {
                "model": self._model(),
                "max_tokens": 4096,
                "messages": api_messages,
            }
            if system is not None:
                kwargs["system"] = system

            response = await client.messages.create(
                **kwargs,
                timeout=get_anthropic_timeout(),
            )
            result: str = response.content[0].text
            self._available = True
            return result
        except Exception:
            logger.warning("Anthropic chat generation failed", exc_info=True)
            self._available = None
            return None

    def _get_client(self) -> Any:
        """Lazily create the AsyncAnthropic client. Returns None if SDK missing."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic()
            except ImportError:
                logger.warning("anthropic package not installed — Anthropic LLM disabled")
                return None
        return self._client

    async def close(self) -> None:
        """Close the Anthropic client if open."""
        if self._client is not None:
            await self._client.close()
            self._client = None
