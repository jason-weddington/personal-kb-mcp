"""Re-export shim — real code moved to ``kb_core.db.backend``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.backend import Cursor, Database, Row

__all__ = ["Cursor", "Database", "Row"]
