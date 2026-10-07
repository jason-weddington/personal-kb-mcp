"""Validate and stamp ``hints.resolution`` on KB writes.

A *resolution* is a corrected belief stored in an entry's ``hints`` under the
key ``"resolution"``. The format is owned by ``kb_service/prevention.py``
(GTD 32605015 AC1), which reads it; this module is the producer-side validator.
``provenance.event_id`` is an additive key the owner's parser ignores; it is
reserved for ``grounding=observed``.

Format::

    {"resolution": {
        "corrected_fact": str (required, non-blank, <= 1000 chars),
        "wrong_belief": str, "evidence": str,
        "cue": {"tool": str, "target_class": str, "args_prefix": str},
        "provenance": {"capture": "deliberate"|"autonomous",
                       "grounding": "observed"|"asserted", "event_id": str},
        "observed_sessions": int >= 1,
        "scope": "project"|"global"}}

Pure functions; the only side effect is logging.
"""

from __future__ import annotations

import logging
from typing import Any

from kb_core.cues import target_class

RESOLUTION_KEYS = frozenset(
    {
        "corrected_fact",
        "wrong_belief",
        "evidence",
        "cue",
        "provenance",
        "observed_sessions",
        "scope",
    }
)
CUE_KEYS = frozenset({"tool", "target_class", "args_prefix"})
PROVENANCE_KEYS = frozenset({"capture", "grounding", "event_id"})
CAPTURE_VALUES = ("deliberate", "autonomous")
GROUNDING_VALUES = ("observed", "asserted")
SCOPE_VALUES = ("project", "global")
CORRECTED_FACT_MAX = 1000

logger = logging.getLogger(__name__)


class ResolutionHintError(ValueError):
    """A ``hints.resolution`` value failed validation."""

    def __init__(self, message: str, reason: str) -> None:
        """Store *reason* (a short machine-readable code) beside *message*."""
        super().__init__(message)
        self.reason = reason


def _nonempty_str(value: object) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _validate_shape(res: Any, entry_type: str) -> None:
    if not isinstance(res, dict):
        raise ResolutionHintError("hints.resolution must be an object", "invalid_shape")
    unknown = set(res) - RESOLUTION_KEYS
    if unknown:
        raise ResolutionHintError(
            f"hints.resolution: unknown keys {sorted(unknown)!r}", "unknown_keys"
        )
    cue = res.get("cue")
    if isinstance(cue, dict):
        unknown = set(cue) - CUE_KEYS
        if unknown:
            raise ResolutionHintError(
                f"hints.resolution.cue: unknown keys {sorted(unknown)!r}",
                "unknown_keys",
            )
    prov = res.get("provenance")
    if isinstance(prov, dict):
        unknown = set(prov) - PROVENANCE_KEYS
        if unknown:
            raise ResolutionHintError(
                f"hints.resolution.provenance: unknown keys {sorted(unknown)!r}",
                "unknown_keys",
            )

    fact = res.get("corrected_fact")
    if not isinstance(fact, str) or fact.strip() == "":
        raise ResolutionHintError(
            "hints.resolution.corrected_fact is required and must be a "
            "non-empty string",
            "corrected_fact",
        )
    if len(fact) > CORRECTED_FACT_MAX:
        raise ResolutionHintError(
            f"hints.resolution.corrected_fact exceeds {CORRECTED_FACT_MAX} characters",
            "corrected_fact",
        )

    for key in ("wrong_belief", "evidence"):
        if key in res and not isinstance(res[key], str):
            raise ResolutionHintError(
                f"hints.resolution.{key} must be a string", "type_error"
            )

    if "cue" in res:
        if not (
            isinstance(cue, dict)
            and _nonempty_str(cue.get("tool"))
            and _nonempty_str(cue.get("target_class"))
        ):
            raise ResolutionHintError(
                "hints.resolution.cue: tool and target_class are required "
                "non-empty strings",
                "type_error",
            )
        if "args_prefix" in cue and not _nonempty_str(cue["args_prefix"]):
            raise ResolutionHintError(
                "hints.resolution.cue.args_prefix must be a non-empty string",
                "type_error",
            )

    if "provenance" in res:
        if not isinstance(prov, dict):
            raise ResolutionHintError(
                "hints.resolution.provenance must be an object", "type_error"
            )
        if "capture" in prov and prov["capture"] not in CAPTURE_VALUES:
            raise ResolutionHintError(
                "hints.resolution.provenance.capture must be one of "
                f"{CAPTURE_VALUES!r}",
                "type_error",
            )
        if "grounding" in prov and prov["grounding"] not in GROUNDING_VALUES:
            raise ResolutionHintError(
                "hints.resolution.provenance.grounding must be one of "
                f"{GROUNDING_VALUES!r}",
                "type_error",
            )
        if "event_id" in prov and not _nonempty_str(prov["event_id"]):
            raise ResolutionHintError(
                "hints.resolution.provenance.event_id must be a non-empty string",
                "type_error",
            )

    if "observed_sessions" in res:
        obs = res["observed_sessions"]
        if isinstance(obs, bool) or not isinstance(obs, int) or obs < 1:
            raise ResolutionHintError(
                "hints.resolution.observed_sessions must be an integer >= 1",
                "type_error",
            )

    if "scope" in res and res["scope"] not in SCOPE_VALUES:
        raise ResolutionHintError(
            f"hints.resolution.scope must be one of {SCOPE_VALUES!r}", "type_error"
        )

    if entry_type == "mental_map":
        raise ResolutionHintError(
            "hints.resolution is not allowed on a mental_map entry", "mental_map"
        )


def _check_bash_cue(res: dict[str, Any]) -> None:
    cue = res.get("cue")
    if not isinstance(cue, dict) or cue.get("tool") != "Bash":
        return
    tc = cue["target_class"]
    normalized = target_class("Bash", tc)
    if normalized != tc:
        raise ResolutionHintError(
            f"hints.resolution.cue.target_class {tc!r} is not a normalized class; "
            f"kb_core.cues.target_class gives {normalized!r}",
            "not_normalized",
        )


def _existing_capture(existing_hints: dict[str, Any] | None) -> str | None:
    """Capture of the stored resolution, or None when there is none."""
    if existing_hints is None:
        return None
    existing = existing_hints.get("resolution")
    if not isinstance(existing, dict):
        return None
    prov = existing.get("provenance")
    if isinstance(prov, dict) and isinstance(prov.get("capture"), str):
        return str(prov["capture"])
    return "deliberate"  # fail closed


def validate_and_stamp_resolution(
    hints: dict[str, Any] | None,
    *,
    is_machine: bool,
    entry_type: str,
    existing_hints: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Validate ``hints['resolution']`` and stamp its provenance.

    Returns *hints* itself when there is no resolution; otherwise a new dict
    with a validated, stamped copy of the resolution. Inputs are not mutated.

    Raises:
        ResolutionHintError: the resolution is malformed or not allowed.
    """
    if hints is None or "resolution" not in hints:
        return hints
    res = hints["resolution"]
    _validate_shape(res, str(getattr(entry_type, "value", entry_type)))
    _check_bash_cue(res)

    prov = dict(res.get("provenance") or {})
    if is_machine:
        if prov.get("capture") == "deliberate":
            logger.info("resolution_capture_forced", extra={"supplied": "deliberate"})
        prov["capture"] = "autonomous"
    else:
        prov["capture"] = prov.get("capture") or "deliberate"
    prov["grounding"] = prov.get("grounding") or "asserted"
    if prov["grounding"] == "observed" and "event_id" not in prov:
        raise ResolutionHintError(
            'hints.resolution.provenance: grounding "observed" requires a '
            "non-empty event_id",
            "observed_needs_event_id",
        )

    existing_capture = _existing_capture(existing_hints)
    if (
        existing_capture == "deliberate"
        and prov["capture"] == "autonomous"
        and prov["grounding"] != "observed"
    ):
        raise ResolutionHintError(
            "hints.resolution: an autonomous resolution may replace a deliberate "
            'one only with grounding "observed" and an event_id',
            "deliberate_protected",
        )

    stamped = dict(res)
    stamped["provenance"] = prov
    return {**hints, "resolution": stamped}
