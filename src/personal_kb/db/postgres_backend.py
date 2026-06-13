"""Re-export shim — real code moved to ``kb_core.db.postgres_backend``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.postgres_backend import (
    PostgresBackend,
    PostgresCursor,
    PostgresRow,
    _translate_placeholders,
    _txn_conn,
)

__all__ = [
    "PostgresBackend",
    "PostgresCursor",
    "PostgresRow",
    "_translate_placeholders",
    "_txn_conn",
]
