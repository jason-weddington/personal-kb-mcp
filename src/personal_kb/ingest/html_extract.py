"""Re-export shim — real code moved to ``kb_core.ingest.html_extract``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.html_extract import extract_content

__all__ = ["extract_content"]
