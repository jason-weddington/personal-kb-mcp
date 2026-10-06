"""Hermetic tests for ``scripts/listener_decision_report.py``.

The report is the reader half of the listener telemetry columns added in
GTD 268e2af3 (candidate_ids, whispered_ids, n_retrieved, n_after_a,
n_after_b, retrieval_path) — without it the two-week measurement window
produces data and no verdict. These tests drive ``render_report`` directly
with fabricated row dicts (the shape ``_fetch_rows`` returns from asyncpg
records), no live Postgres involved.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# scripts/ is not a package (no __init__.py, not installed) -- add it to
# sys.path directly, mirroring how the script is invoked standalone.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from listener_decision_report import _parse_since, _pct, render_report  # noqa: E402

_SINCE = datetime(2026, 9, 1, tzinfo=UTC)


def _row(
    *,
    reason: str,
    decision: str,
    candidates_considered: int = 0,
    candidate_ids: list[str] | None = None,
    whispered_ids: list[str] | None = None,
    n_retrieved: int = 0,
    n_after_a: int = 0,
    n_after_b: int = 0,
    retrieval_path: str = "primary",
    candidate_signal: str = "detail",
) -> dict[str, Any]:
    return {
        "reason": reason,
        "decision": decision,
        "candidates_considered": candidates_considered,
        "candidate_ids": json.dumps(candidate_ids or []),
        "whispered_ids": json.dumps(whispered_ids or []),
        "n_retrieved": n_retrieved,
        "n_after_a": n_after_a,
        "n_after_b": n_after_b,
        "retrieval_path": retrieval_path,
        "candidate_signal": candidate_signal,
    }


def test_empty_window_reports_zero_decisions() -> None:
    out = render_report([], _SINCE)
    assert "0 decision(s)" in out


def test_reason_counts_and_whisper_rate() -> None:
    rows = [
        _row(reason="kill-switch", decision="declined"),
        _row(reason="kill-switch", decision="declined"),
        _row(reason="whispered", decision="whisper", candidate_ids=["kb-1"]),
    ]
    out = render_report(rows, _SINCE)
    assert "kill-switch" in out
    assert "Whisper rate: 1/3" in out


def test_jury_reaching_and_vote_none_share() -> None:
    rows = [
        _row(reason="rule-a", decision="declined"),  # never reaches the jury
        _row(reason="vote-none", decision="declined"),
        _row(reason="vote-split", decision="declined"),
        _row(reason="whispered", decision="whisper"),
    ]
    out = render_report(rows, _SINCE)
    # 3 of 4 reach the jury (vote-none, vote-split, whispered).
    assert "Jury-reaching rate" in out
    assert "3/4" in out
    # vote-none is 1 of the 3 jury-reaching decisions.
    assert "vote-none share of jury-reaching: 1/3" in out


def test_per_stage_attrition_totals() -> None:
    rows = [
        _row(
            reason="rule-a",
            decision="declined",
            n_retrieved=5,
            n_after_a=2,
            n_after_b=2,
        ),
        _row(
            reason="rule-b",
            decision="declined",
            n_retrieved=3,
            n_after_a=3,
            n_after_b=0,
        ),
    ]
    out = render_report(rows, _SINCE)
    assert "retrieved:          8" in out
    # dropped by rule-A: (5-2) + (3-3) = 3
    assert "dropped by rule-A:  3" in out
    # dropped by rule-B: (2-2) + (3-0) = 3
    assert "dropped by rule-B:  3" in out


def test_fallback_rate_counts_fallback_direct_retrieval_path() -> None:
    rows = [
        _row(reason="rule-a", decision="declined", retrieval_path="fallback-direct"),
        _row(reason="rule-a", decision="declined", retrieval_path="primary"),
    ]
    out = render_report(rows, _SINCE)
    assert "Fallback retrieval path used: 1/2" in out


def test_map_id_considered_vs_whispered_is_the_load_bearing_line() -> None:
    """The load-bearing measurement: candidate_ids vs whispered_ids per map id."""
    rows = [
        _row(
            reason="rule-a",
            decision="declined",
            candidate_ids=["kb-1", "kb-2"],
        ),
        _row(
            reason="whispered",
            decision="whisper",
            candidate_ids=["kb-1"],
            whispered_ids=["kb-1"],
        ),
    ]
    out = render_report(rows, _SINCE)
    # kb-1 was considered twice, whispered once; kb-2 considered once, never.
    assert "kb-1           considered=   2  whispered=   1" in out
    assert "kb-2           considered=   1  whispered=   0" in out


def test_no_candidate_ids_in_window_prints_placeholder() -> None:
    rows = [_row(reason="kill-switch", decision="declined")]
    out = render_report(rows, _SINCE)
    assert "no candidate_ids recorded in this window" in out


def test_pct_helper_handles_zero_denominator() -> None:
    assert _pct(0, 0) == "n/a"
    assert _pct(1, 2) == "50.0%"


def test_parse_since_accepts_hours_days_weeks() -> None:
    now_ish = datetime.now(UTC)
    for spec in ("24h", "14d", "2w"):
        cutoff = _parse_since(spec)
        assert cutoff < now_ish


def test_parse_since_rejects_bad_unit() -> None:
    try:
        _parse_since("14x")
    except SystemExit as exc:
        assert "must end in h/d/w" in str(exc)
    else:
        raise AssertionError("expected SystemExit for an invalid unit")


def test_whispers_are_broken_down_by_candidate_signal() -> None:
    """GTD be964e94 added candidate_signal; this reader is its only consumer.

    Without this line the "lexical vs detail vs fallback" question is only
    answerable by hand-typed SQL — the seam neither item's spec owned.
    """
    rows = [
        _row(reason="whispered", decision="whisper", candidate_signal="lexical"),
        _row(reason="whispered", decision="whisper", candidate_signal="lexical"),
        _row(reason="whispered", decision="whisper", candidate_signal="detail"),
        # Declines carry no surfaced pointer, so they must not be attributed.
        _row(reason="rule-b", decision="declined", candidate_signal="detail"),
    ]
    out = render_report(rows, _SINCE)
    assert "Whispers by candidate signal (3 whispers)" in out
    assert "lexical      2 (66.7%)" in out
    assert "detail       1 (33.3%)" in out


def test_missing_candidate_signal_column_degrades_to_unset() -> None:
    """A rolling deploy can leave one instance on a pre-candidate_signal schema;
    the report must degrade rather than crash on the nice-to-have column."""
    row = _row(reason="whispered", decision="whisper")
    del row["candidate_signal"]
    out = render_report([row], _SINCE)
    assert "(unset)" in out
