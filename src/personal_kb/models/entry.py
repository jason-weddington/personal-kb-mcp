"""Re-export shim — real code moved to ``kb_core.models.entry``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.models.entry import EntryType, KnowledgeEntry

__all__ = ["EntryType", "KnowledgeEntry"]
