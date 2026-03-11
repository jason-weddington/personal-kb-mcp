"""Shared JSON parser for LLM responses.

LLMs wrap JSON in markdown fences, mix it with commentary, or return
bare structures.  Every caller previously copy-pasted the same
fence-strip → regex-find → json.loads pipeline.  This module
consolidates that into two helpers:

- ``parse_json_object(raw)`` — extract the first JSON **object**
- ``parse_json_array(raw)``  — extract the first JSON **array**

Both return ``None`` on failure so callers can apply their own
fallback logic.
"""

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _strip_fences(raw: str) -> str:
    """Remove markdown code fences if present."""
    m = _FENCE_RE.search(raw)
    return m.group(1) if m else raw


def parse_json_object(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from an LLM response.

    Strips markdown fences, then uses ``json.JSONDecoder`` to find the
    first valid ``{...}`` structure — handles nested braces correctly
    (unlike the old greedy regex).

    Returns the parsed dict, or ``None`` if no valid object is found.
    """
    text = _strip_fences(raw)
    idx = text.find("{")
    if idx == -1:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text, idx)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(obj, dict):
        return obj
    return None


def parse_json_array(raw: str) -> list[Any] | None:
    """Extract the first JSON array from an LLM response.

    Strips markdown fences, then uses ``json.JSONDecoder`` to find the
    first valid ``[...]`` structure.

    Returns the parsed list, or ``None`` if no valid array is found.
    """
    text = _strip_fences(raw)
    idx = text.find("[")
    if idx == -1:
        return None
    try:
        arr, _ = json.JSONDecoder().raw_decode(text, idx)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(arr, list):
        return arr
    return None
