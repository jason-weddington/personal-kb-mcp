"""Re-export shim — real code moved to ``kb_core.llm.provider``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.llm.provider import LLMProvider, Message

__all__ = ["LLMProvider", "Message"]
