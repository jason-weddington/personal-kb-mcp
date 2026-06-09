"""Re-export shim — real code moved to ``kb_core.db.queries``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.queries import (
    deactivate_entry_db,
    delete_entry_cascade,
    get_all_active_entry_ids,
    get_db_stats,
    get_entry,
    insert_entry,
    insert_version,
    next_entry_id,
    reactivate_entry_db,
    row_to_entry,
    touch_accessed,
    update_entry,
)

__all__ = [
    "deactivate_entry_db",
    "delete_entry_cascade",
    "get_all_active_entry_ids",
    "get_db_stats",
    "get_entry",
    "insert_entry",
    "insert_version",
    "next_entry_id",
    "reactivate_entry_db",
    "row_to_entry",
    "touch_accessed",
    "update_entry",
]
