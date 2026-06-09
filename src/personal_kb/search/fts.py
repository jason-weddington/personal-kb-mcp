"""Re-export shim — real code moved to ``kb_core.search.fts``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.search.fts import fts_search

__all__ = ["fts_search"]
