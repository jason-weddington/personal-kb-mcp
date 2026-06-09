"""Re-export shim — real code moved to ``kb_core.ingest.safety``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.ingest.safety import (
    SafetyResult,
    check_deny_list,
    detect_secrets_in_content,
    redact_pii,
    run_content_safety,
    run_safety_pipeline,
)

__all__ = [
    "SafetyResult",
    "check_deny_list",
    "detect_secrets_in_content",
    "redact_pii",
    "run_content_safety",
    "run_safety_pipeline",
]
