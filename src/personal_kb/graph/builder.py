"""Re-export shim — real code moved to ``kb_core.graph.builder``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.graph.builder import GraphBuilder, _as_list

__all__ = ["GraphBuilder", "_as_list"]
