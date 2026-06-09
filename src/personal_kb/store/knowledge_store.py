"""Re-export shim — real code moved to ``kb_core.store.knowledge_store``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.store.knowledge_store import KnowledgeStore, _record_audit_event

__all__ = ["KnowledgeStore", "_record_audit_event"]
