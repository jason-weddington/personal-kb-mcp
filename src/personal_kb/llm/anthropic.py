"""Re-export shim — real code moved to ``kb_core.llm.anthropic``.

Shim: real code moved to kb_core (kb-core extraction wave 3a).
Channel-rewiring wave removes this.
"""

from kb_core.llm.anthropic import _SONNET_MODEL, AnthropicLLMClient

__all__ = ["_SONNET_MODEL", "AnthropicLLMClient"]
