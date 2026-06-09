"""Re-export shim — real code moved to ``kb_core.ingest.extractor``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.extractor import (
    ExtractedEntry,
    _is_code_file,
    _is_prose_file,
    _parse_entries,
    extract_entries,
    summarize_file,
)

__all__ = [
    "ExtractedEntry",
    "_is_code_file",
    "_is_prose_file",
    "_parse_entries",
    "extract_entries",
    "summarize_file",
]
