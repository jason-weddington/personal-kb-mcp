"""Re-export shim — real code moved to ``kb_core.ingest.chunker``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.chunker import (
    Chunk,
    chunk_content,
)

__all__ = [
    "Chunk",
    "chunk_content",
]
