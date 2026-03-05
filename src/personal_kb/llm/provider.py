"""LLM provider protocol for pluggable language model backends."""

from typing import Protocol, runtime_checkable

# Message dict: {"role": "user"|"assistant", "content": "..."}
Message = dict[str, str]


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model providers with graceful degradation."""

    async def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        ...

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        ...

    async def generate_chat(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
    ) -> str | None:
        """Generate text from a conversation history. Returns None if unavailable."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
