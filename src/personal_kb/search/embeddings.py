"""Re-export shim — real code moved to ``kb_core.search.embeddings``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.search.embeddings import EmbeddingClient

__all__ = ["EmbeddingClient"]
