"""Re-export shim — real code moved to ``kb_core.graph.planner``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.graph.planner import QueryPlan, QueryPlanner

__all__ = ["QueryPlan", "QueryPlanner"]
