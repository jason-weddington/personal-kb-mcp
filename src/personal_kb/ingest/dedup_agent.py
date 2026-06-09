"""Re-export shim — real code moved to ``kb_core.ingest.dedup_agent``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.dedup_agent import (
    DedupAgent,
    DedupResult,
    _parse_dedup_response,
)

__all__ = [
    "DedupAgent",
    "DedupResult",
    "_parse_dedup_response",
]
