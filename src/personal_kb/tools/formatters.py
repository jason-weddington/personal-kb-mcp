"""Re-export shim — real code moved to ``kb_core.formatting``.

Shim: real code moved to kb_core (kb-core extraction wave 3).
Channel-rewiring wave removes this.
"""

from kb_core.formatting import (
    format_entry_compact,
    format_entry_full,
    format_entry_header,
    format_entry_meta,
    format_graph_hint,
    format_result_list,
)

__all__ = [
    "format_entry_compact",
    "format_entry_full",
    "format_entry_header",
    "format_entry_meta",
    "format_graph_hint",
    "format_result_list",
]
