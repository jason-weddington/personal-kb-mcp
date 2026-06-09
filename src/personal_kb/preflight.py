"""Re-export shim — real code moved to ``kb_core.preflight``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.preflight import (
    _conventions_sql,
    _expiring_sql,
    _format_duration,
    _format_expiry_badge,
    _format_toc_line,
    _graph_related,
    _maps_sql,
    _recent_sql,
    build_project_context,
)

__all__ = [
    "_conventions_sql",
    "_expiring_sql",
    "_format_duration",
    "_format_expiry_badge",
    "_format_toc_line",
    "_graph_related",
    "_maps_sql",
    "_recent_sql",
    "build_project_context",
]
