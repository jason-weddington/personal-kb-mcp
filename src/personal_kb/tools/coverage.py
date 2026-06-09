"""Re-export shim — real code moved to ``kb_core.coverage``.

Shim: real code moved to kb_core (kb-core extraction wave 3).
Channel-rewiring wave removes this.
"""

from kb_core.coverage import CoverageResult, _parse_coverage_response, assess_coverage

__all__ = ["CoverageResult", "_parse_coverage_response", "assess_coverage"]
