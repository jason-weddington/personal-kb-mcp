"""Re-export shim — real code moved to ``kb_core.search.vector``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.search.vector import vector_search

__all__ = ["vector_search"]
