"""Re-export shim — real code moved to ``kb_core.db.postgres_backend``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.postgres_backend import (
    _LISTENER_RECONNECT_DELAY,
    NOTIFY_CHANNEL,
    PostgresBackend,
    PostgresCursor,
    PostgresRow,
    _translate_placeholders,
    _txn_conn,
)

__all__ = [
    "NOTIFY_CHANNEL",
    "_LISTENER_RECONNECT_DELAY",
    "PostgresBackend",
    "PostgresCursor",
    "PostgresRow",
    "_translate_placeholders",
    "_txn_conn",
]
