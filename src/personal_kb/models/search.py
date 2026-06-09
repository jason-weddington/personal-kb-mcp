"""Re-export shim — real code moved to ``kb_core.models.search``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.models.search import SearchQuery, SearchResult

__all__ = ["SearchQuery", "SearchResult"]
