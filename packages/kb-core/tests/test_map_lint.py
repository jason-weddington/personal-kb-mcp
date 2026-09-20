"""Unit tests for the structured mental_map lint (kb_core.map_lint).

This module is the single source of truth for what a valid mental_map body
is; the MCP channel's advisory lint and the web service's machine-principal
write gate both consume it. These tests pin:

* every purity rule fires exactly its own machine-readable code,
* the moved constants stay at their corpus-validated values (900 / 175 —
  "move, do not retune"),
* the human messages stay byte-identical to the advisory strings the MCP
  channel has always rendered (minus the channel prefix), so the extraction
  wave changed nothing observable,
* the budget boundary: a body at exactly its budget passes, one char over
  fails.

Hermetic: pure functions only — no DB, no LLM, no network.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import pairwise

import pytest

from kb_core.map_lint import (
    MAP_BODY_BASE_CHARS,
    MAP_BODY_PER_POINTER_CHARS,
    MapLintCode,
    MapLintFinding,
    count_map_pointers,
    lint_map_body,
    map_body_budget,
)


def _calibration_body(total_chars: int, pointer_count: int) -> str:
    """Synthetic body with the exact requested length and distinct kb- count.

    Every non-id position is filler, so no purity heuristic can fire and the
    fixture isolates the length rule from the purity rules.
    """
    ids_text = " ".join(f"kb-{1000 + i:05d}" for i in range(pointer_count))
    body = ids_text + "x" * (total_chars - len(ids_text))
    assert len(body) == total_chars
    assert count_map_pointers(body) == pointer_count
    return body


# --- constants: moved verbatim, do not retune ------------------------------


def test_budget_constants_are_verbatim():
    assert MAP_BODY_BASE_CHARS == 900
    assert MAP_BODY_PER_POINTER_CHARS == 175
    assert map_body_budget(0) == MAP_BODY_BASE_CHARS
    assert map_body_budget(1) == 1075


def test_budget_is_strictly_monotonic_in_pointer_count():
    budgets = [map_body_budget(p) for p in range(0, 30)]
    assert all(earlier < later for earlier, later in pairwise(budgets))


# --- each purity rule fires its code (exact lists, cross-trips included) ----
#
# The regexes are deliberately naive and a single token can trip more than
# one rule (a URL is also an absolute path and carries a dotted host; a
# dotted path segment trips the dotted-identifier rule). These exact lists
# pin that behavior as MOVED, including its rough edges — none of it may
# change in the extraction wave.

RULE_CASES = [
    # body -> exact ordered code list
    # (a bare URL is covered by test_path_rule_fires — it trips three rules)
    ("set KB_DB_PATH to override", [MapLintCode.ENV_VAR]),
    ("call store.create_entry(short_title)", [MapLintCode.DOTTED_IDENTIFIER]),
    ('the flag is "--no-push"', [MapLintCode.QUOTED_LITERAL]),
    ("the flag is `--no-push`", [MapLintCode.QUOTED_LITERAL]),
    ("the explorer runs on port 8767", [MapLintCode.CONFIG_NUMERAL]),
    ("WireGuard listens on 51820", [MapLintCode.CONFIG_NUMERAL]),
    ("dedup threshold is 0.06", [MapLintCode.CONFIG_NUMERAL]),
    ("timeout:30", [MapLintCode.CONFIG_NUMERAL]),
]


@pytest.mark.parametrize(("body", "codes"), RULE_CASES)
def test_rule_fires_exactly_its_code(body: str, codes: list[MapLintCode]):
    assert [f.code for f in lint_map_body(body)] == codes


@pytest.mark.parametrize(
    ("body", "codes"),
    [
        (
            "see https://example.com/x",
            [MapLintCode.URL, MapLintCode.PATH, MapLintCode.DOTTED_IDENTIFIER],
        ),
        (
            "config at ~/.config/kb/settings.toml",
            [MapLintCode.PATH, MapLintCode.DOTTED_IDENTIFIER],
        ),
        ("lives in /srv/kb/main.py", [MapLintCode.PATH, MapLintCode.DOTTED_IDENTIFIER]),
        ("/etc/kb-service/env holds the config", [MapLintCode.PATH]),
    ],
)
def test_path_rule_fires(body: str, codes: list[MapLintCode]):
    assert [f.code for f in lint_map_body(body)] == codes


def test_multiple_findings_are_all_reported_in_rule_order():
    # One body tripping several rules reports one finding per rule, in the
    # module's fixed rule order — the order the MCP channel has always
    # rendered advisory strings in. Every rule fires here.
    body = 'port 8767, see https://example.com/x, KB_DB_PATH, store.f(), "x"'
    codes = [f.code for f in lint_map_body(body)]
    assert codes == [
        MapLintCode.URL,
        MapLintCode.PATH,
        MapLintCode.ENV_VAR,
        MapLintCode.DOTTED_IDENTIFIER,
        MapLintCode.QUOTED_LITERAL,
        MapLintCode.CONFIG_NUMERAL,
    ]


# --- pointers-in-disguise and kb-refs are never flagged --------------------


@pytest.mark.parametrize(
    "body",
    [
        "the pipeline has three stages and 4 components",
        "there are 12 stages",
        "there are 100 nodes",
        "kb-00001 kb-99999",
        "",
        "orients the auth subsystem; follow the edges to sources",
    ],
)
def test_pointer_bodies_are_clean(body: str):
    assert lint_map_body(body) == []


# --- budget boundary --------------------------------------------------------


def test_body_at_exactly_its_budget_passes():
    assert lint_map_body("x" * map_body_budget(0)) == []
    assert lint_map_body(_calibration_body(map_body_budget(1), 1)) == []


def test_body_one_char_over_budget_fails():
    findings = lint_map_body(_calibration_body(map_body_budget(2) + 1, 2))
    assert [f.code for f in findings] == [MapLintCode.OVER_BUDGET]


def test_over_budget_finding_states_numbers_and_codes():
    findings = lint_map_body(_calibration_body(1200, 1))
    assert [f.code for f in findings] == [MapLintCode.OVER_BUDGET]
    message = findings[0].message
    assert "1200" in message and "1075" in message and "1 pointer" in message
    assert "900 base + 175 per pointer" in message


# --- corpus calibration (kb-01724 / kb-03257, measured live 2026-09-20) ------


def test_kb_01724_calibration_passes():
    # 2319 chars / 11 pointers -> budget 2825 (82% used); one of the two
    # best maps in the corpus and one of only two that exceeded the old
    # flat ~1500 cap.
    body = _calibration_body(2319, 11)
    assert map_body_budget(11) == 2825
    assert len(body) / map_body_budget(11) == pytest.approx(0.82, abs=0.01)
    assert lint_map_body(body) == []


def test_kb_03257_calibration_passes():
    # 2540 chars / 20 pointers -> budget 4400 (58% used).
    body = _calibration_body(2540, 20)
    assert map_body_budget(20) == 4400
    assert len(body) / map_body_budget(20) == pytest.approx(0.58, abs=0.01)
    assert lint_map_body(body) == []


# --- message byte-stability ---------------------------------------------------
#
# The MCP channel renders "Map lint (advisory): " + finding.message; these
# exact strings predate the extraction wave. A rewording is a regression:
# pin them.


def test_messages_are_byte_identical_to_the_original_advisory_prose():
    body = 'port 8767, see https://example.com/x, KB_DB_PATH, store.f(), "x"' + "y" * 1600
    url, path, env, dotted, quoted, numeral, budget = lint_map_body(body)
    assert url.message == (
        "contains a URL — point to a kb entry instead of embedding a retrievable link."
    )
    assert path.message == (
        "contains a file path — that is a retrievable value, not an orientation pointer."
    )
    assert env.message == (
        "contains an ENV_VAR-style token — that is a retrievable config value, not a pointer."
    )
    assert dotted.message == (
        "contains a dotted code identifier/signature — that is a retrievable value, not a pointer."
    )
    assert quoted.message == "contains a quoted literal value."
    assert numeral.message == (
        "contains a config-like numeral — that is a retrievable value, not a count of parts."
    )
    # len(body) == 1664 with zero distinct kb- pointers -> budget 900.
    assert len(body) == 1664
    assert budget.message == (
        f"body is {len(body)} chars, exceeding the advisory budget of 900 chars for "
        "0 pointers (900 base + 175 per pointer) — cut orientation prose, "
        "not pointer glosses."
    )


def test_finding_is_a_frozen_structured_pair():
    finding = lint_map_body("see https://example.com/x")[0]
    assert isinstance(finding, MapLintFinding)
    assert finding.code == "url"
    assert isinstance(finding.code, MapLintCode)
    with pytest.raises(FrozenInstanceError):
        finding.code = MapLintCode.PATH  # type: ignore[misc]
