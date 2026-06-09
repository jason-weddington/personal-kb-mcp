"""Re-export shim — real code moved to ``kb_core.ingest.ingester``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.ingester import (
    _ALLOWED_EXTENSIONS,
    _ALLOWED_NAMES,
    FileIngester,
    FileResult,
    IngestResult,
    ProgressCallback,
    _is_allowed_file,
    _read_pdf,
)

__all__ = [
    "_ALLOWED_EXTENSIONS",
    "_ALLOWED_NAMES",
    "FileIngester",
    "FileResult",
    "IngestResult",
    "ProgressCallback",
    "_is_allowed_file",
    "_read_pdf",
]
