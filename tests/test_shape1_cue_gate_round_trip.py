"""Round trip: a precise shape-1 cue reaches the gate index and the hook matches it.

The surprise distiller (kb-service) builds the cue with ``kb_core.cues``; the
standalone hook matches it with its vendored ``cues_lite``. Both packages are
importable only here, in the root test suite.
"""

from __future__ import annotations

import json

from kb_service.prevention import build_gate_index, parse_resolution
from kb_service.surprise_distill import shape1_cue
from personal_kb_hook import cues_lite
from personal_kb_hook.prevention import _matches


def _hook_matches(entry: dict[str, object], command: str) -> bool:
    return any(
        _matches(entry, "Bash", seg_class, seg_args)
        for seg_class, seg_args in cues_lite.bash_segments(command)
    )


def test_shape1_cue_round_trips_through_gate_index_and_hook() -> None:
    cue = shape1_cue("git push origin main", "git push origin HEAD:refs/for/main")
    assert cue == {
        "tool": "Bash",
        "target_class": "git push",
        "args_prefix": "origin main",
    }
    hints = {
        "resolution": {
            "corrected_fact": "Push for review with git push origin HEAD:refs/for/main",
            "wrong_belief": "git push origin main",
            "cue": cue,
            "provenance": {
                "capture": "autonomous",
                "grounding": "observed",
                "event_id": "s1:0",
            },
            "observed_sessions": 1,
        },
        "surprise_capture": {"shape": 1, "mode": "interactive"},
    }
    res = parse_resolution("kb-00001", "2026-10-10", json.dumps(hints))
    index, truncated = build_gate_index([res])
    assert truncated == 0
    (entry,) = [c.model_dump() for c in index]
    assert entry["args_prefix"] == "origin main"
    assert _hook_matches(entry, "git push origin main 2>&1")
    assert not _hook_matches(entry, "git push origin HEAD:refs/for/main")
