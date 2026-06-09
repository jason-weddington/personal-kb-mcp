"""Re-export shim — real code moved to ``kb_core.db.sqlite_backend``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.sqlite_backend import (
    SQLiteBackend,
    SQLiteCursor,
    _escape_fts_query,
    _serialize_f32,
)

__all__ = ["SQLiteBackend", "SQLiteCursor", "_escape_fts_query", "_serialize_f32"]
