"""Re-export shim — real code moved to ``kb_core.db.schema``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.db.schema import (
    AUDIT_EVENTS_SCHEMA_SQL,
    DEPLOYMENT_CONFIG_SCHEMA_SQL,
    FEEDBACK_SCHEMA_SQL,
    GRAPH_SCHEMA_SQL,
    INGEST_SCHEMA_SQL,
    INIT_SEQ_SQL,
    SCHEMA_SQL,
    SCHEMA_VERSION,
    SEARCH_EVENTS_SCHEMA_SQL,
    _migrate_add_expires_at,
    _migrate_add_last_accessed,
    _migrate_v2_multi_user,
    apply_audit_events_schema,
    apply_deployment_config_schema,
    apply_feedback_schema,
    apply_graph_schema,
    apply_ingest_schema,
    apply_schema,
    apply_search_events_schema,
    apply_vec_schema,
)

__all__ = [
    "AUDIT_EVENTS_SCHEMA_SQL",
    "DEPLOYMENT_CONFIG_SCHEMA_SQL",
    "FEEDBACK_SCHEMA_SQL",
    "GRAPH_SCHEMA_SQL",
    "INGEST_SCHEMA_SQL",
    "INIT_SEQ_SQL",
    "SCHEMA_SQL",
    "SCHEMA_VERSION",
    "SEARCH_EVENTS_SCHEMA_SQL",
    "_migrate_add_expires_at",
    "_migrate_add_last_accessed",
    "_migrate_v2_multi_user",
    "apply_audit_events_schema",
    "apply_deployment_config_schema",
    "apply_feedback_schema",
    "apply_graph_schema",
    "apply_ingest_schema",
    "apply_schema",
    "apply_search_events_schema",
    "apply_vec_schema",
]
