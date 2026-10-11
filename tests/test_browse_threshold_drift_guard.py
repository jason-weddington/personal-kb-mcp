"""Drift guard: the web UI's browse thresholds match ``kb_core.confidence.decay``.

``packages/kb-service/frontend/src/browse/format.ts`` hard-codes
``STALE_FILTER_THRESHOLD`` and ``STALENESS_THRESHOLD`` to colour confidences
and write the Stale captions. If either side changes, this test fails until
both agree again.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from kb_core.confidence import decay

FORMAT_TS = (
    Path(__file__).resolve().parent.parent
    / "packages"
    / "kb-service"
    / "frontend"
    / "src"
    / "browse"
    / "format.ts"
)

_CONST_RE = re.compile(r"export const (STALE_FILTER_THRESHOLD|STALENESS_THRESHOLD) = ([0-9.]+)")


def _frontend_constants() -> dict[str, float]:
    return {m.group(1): float(m.group(2)) for m in _CONST_RE.finditer(FORMAT_TS.read_text())}


@pytest.mark.parametrize("name", ["STALE_FILTER_THRESHOLD", "STALENESS_THRESHOLD"])
def test_frontend_threshold_matches_kb_core(name: str) -> None:
    """Each format.ts threshold equals the kb-core constant of the same name."""
    constants = _frontend_constants()
    assert name in constants, f"{name} not found in {FORMAT_TS}"
    assert constants[name] == getattr(decay, name)
