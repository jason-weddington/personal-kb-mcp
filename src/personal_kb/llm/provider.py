"""LLM provider protocol for pluggable language model backends."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for language model providers with graceful degradation."""

    async def is_available(self) -> bool:
        """Check if the LLM backend is reachable."""
        ...

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Generate text from a prompt. Returns None if unavailable."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
