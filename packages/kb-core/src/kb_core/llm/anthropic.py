"""Anthropic LLM client with graceful degradation.

Channel-agnostic: takes an :class:`~kb_core.config.AnthropicProviderConfig`
at construction time. No environment reads — every tunable is on the
config. If ``config.api_key`` is ``None`` the underlying SDK is allowed
to resolve credentials on its own (env, keychain, etc.). The kb_core
class itself never touches ``os.environ``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kb_core.config import AnthropicProviderConfig
    from kb_core.llm.provider import Message

logger = logging.getLogger(__name__)


_SONNET_MODEL = "claude-sonnet-4-6"
"""Default model identifier for the human-facing synthesis role."""


class AnthropicLLMClient:
    """Generates text via the Anthropic Messages API."""

    def __init__(self, config: AnthropicProviderConfig) -> None:
        """Initialize with an explicit provider config."""
        self._config = config
        self._client: Any = None
        self._available: bool | None = None

    async def is_available(self) -> bool:
        """Check availability. Only caches success — retries on failure.

        Returns True if the SDK is importable. We no longer pre-check for
        an explicit API key — when ``config.api_key`` is ``None`` we
        defer credential resolution to the SDK. A failed ``generate()``
        will clear the cached availability so the next call retries.
        """
        if self._available is True:
            return True
        try:
            client = self._get_client()
            return client is not None
        except Exception:
            return False

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        try:
            client = self._get_client()
            if client is None:
                return None

            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system is not None:
                kwargs["system"] = system

            response = await client.messages.create(
                **kwargs,
                timeout=self._config.timeout,
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
        """Generate text from a conversation history.

        Prompt caching is enabled via the top-level ``cache_control`` kwarg
        (SDK ≥ 0.40). This places the cache breakpoint on the last cacheable
        block of the request — the correct multi-turn pattern — so each turn
        reuses the prefix built by the previous turn (5-min TTL, refreshed on
        read). Requests below the model's minimum cacheable token count
        silently no-op (``cache_creation_input_tokens=0``), which is harmless.
        """
        try:
            client = self._get_client()
            if client is None:
                return None

            api_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            kwargs: dict[str, Any] = {
                "model": self._config.model,
                "max_tokens": 4096,
                "messages": api_messages,
                "cache_control": {"type": "ephemeral"},
            }
            if system is not None:
                kwargs["system"] = system

            response = await client.messages.create(
                **kwargs,
                timeout=self._config.timeout,
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

                kwargs: dict[str, Any] = {}
                if self._config.api_key is not None:
                    kwargs["api_key"] = self._config.api_key
                self._client = AsyncAnthropic(**kwargs)
            except ImportError:
                logger.warning("anthropic package not installed — Anthropic LLM disabled")
                return None
        return self._client

    async def close(self) -> None:
        """Close the Anthropic client if open."""
        if self._client is not None:
            await self._client.close()
            self._client = None
