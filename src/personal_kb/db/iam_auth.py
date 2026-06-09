"""Re-export shim — real code moved to ``kb_core.db.iam_auth``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.iam_auth import (
    DSNComponents,
    make_ssl_context,
    make_token_factory,
    parse_dsn,
)

__all__ = ["DSNComponents", "make_ssl_context", "make_token_factory", "parse_dsn"]
