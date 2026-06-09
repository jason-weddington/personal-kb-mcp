"""Re-export shim — real code moved to ``kb_core.graph.queries``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.graph.queries import (
    _ENTRY_TYPES,
    _KB_ID_RE,
    _parse_scope,
    _safe_order,
    _sort_entries,
    bfs_entries,
    entries_for_scope,
    find_path,
    get_graph_vocabulary,
    get_neighbors,
    supersedes_chain,
)

__all__ = [
    "_ENTRY_TYPES",
    "_KB_ID_RE",
    "_parse_scope",
    "_safe_order",
    "_sort_entries",
    "bfs_entries",
    "entries_for_scope",
    "find_path",
    "get_graph_vocabulary",
    "get_neighbors",
    "supersedes_chain",
]
