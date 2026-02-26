"""LLM provider module."""

from personal_kb.llm.ollama import OllamaLLMClient
from personal_kb.llm.provider import LLMProvider

try:
    from personal_kb.llm.anthropic import AnthropicLLMClient
except ImportError:
    AnthropicLLMClient = None  # type: ignore[assignment,misc]

try:
    from personal_kb.llm.bedrock import BedrockLLMClient
except ImportError:
    BedrockLLMClient = None  # type: ignore[assignment,misc]

__all__ = ["AnthropicLLMClient", "BedrockLLMClient", "LLMProvider", "OllamaLLMClient"]
