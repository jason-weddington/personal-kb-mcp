"""Re-export shim — real code moved to ``kb_core.llm.json_parser``.

Shim: real code moved to kb_core (kb-core extraction).
Channel-rewiring wave removes this.
"""

from kb_core.llm.json_parser import _strip_fences, parse_json_array, parse_json_object

__all__ = ["_strip_fences", "parse_json_array", "parse_json_object"]
