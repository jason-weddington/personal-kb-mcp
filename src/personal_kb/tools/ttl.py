"""Re-export shim — real code moved to ``kb_core.ttl``.

Shim: real code moved to kb_core (kb-core extraction wave 3).
Channel-rewiring wave removes this.
"""

from kb_core.ttl import compute_expires_at, parse_ttl

__all__ = ["compute_expires_at", "parse_ttl"]
