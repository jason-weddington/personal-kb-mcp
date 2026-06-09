"""Re-export shim — real code moved to ``kb_core.models.version``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.models.version import EntryVersion

__all__ = ["EntryVersion"]
