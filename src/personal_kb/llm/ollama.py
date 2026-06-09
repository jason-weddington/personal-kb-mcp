"""Re-export shim — real code moved to ``kb_core.llm.ollama``.

Shim: real code moved to kb_core (kb-core extraction wave 3a).
Channel-rewiring wave removes this.
"""

from kb_core.llm.ollama import OllamaLLMClient

__all__ = ["OllamaLLMClient"]
