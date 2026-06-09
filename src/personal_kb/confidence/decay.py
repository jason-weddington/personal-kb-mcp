"""Re-export shim — real code moved to ``kb_core.confidence.decay``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.confidence.decay import (
    HALF_LIVES,
    STALENESS_THRESHOLD,
    compute_effective_confidence,
    staleness_warning,
)

__all__ = [
    "HALF_LIVES",
    "STALENESS_THRESHOLD",
    "compute_effective_confidence",
    "staleness_warning",
]
