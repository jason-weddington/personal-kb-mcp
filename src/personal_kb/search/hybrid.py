"""Re-export shim — real code moved to ``kb_core.search.hybrid``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.search.hybrid import (
    RRF_K,
    _filter_only_search,
    _has_filters,
    _record_search_event,
    hybrid_search,
)

__all__ = [
    "RRF_K",
    "_filter_only_search",
    "_has_filters",
    "_record_search_event",
    "hybrid_search",
]
