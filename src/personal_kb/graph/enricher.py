"""Re-export shim — real code moved to ``kb_core.graph.enricher``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.graph.enricher import (
    _DEDUP_SIMILARITY_THRESHOLD,
    _MAX_BATCH_CONTENT,
    _MAX_RELATIONSHIPS,
    _VALID_ENTITY_TYPES,
    GraphEnricher,
    _PrefixIndex,
)

__all__ = [
    "_DEDUP_SIMILARITY_THRESHOLD",
    "_MAX_BATCH_CONTENT",
    "_MAX_RELATIONSHIPS",
    "_VALID_ENTITY_TYPES",
    "GraphEnricher",
    "_PrefixIndex",
]
