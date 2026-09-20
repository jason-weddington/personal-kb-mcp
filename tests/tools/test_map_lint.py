"""Unit tests for the pure-function mental_map lint (src/personal_kb/tools/map_lint.py)."""

from itertools import pairwise

import pytest

from personal_kb.tools.map_lint import (
    MAP_BODY_BASE_CHARS,
    MAP_BODY_PER_POINTER_CHARS,
    count_map_pointers,
    lint_map_body,
    map_body_budget,
)

# --- MUST-FLAG: §7.3 retrievable-value categories ---

MUST_FLAG = [
    "the explorer runs on port 8767",  # config numeral adjacent to config-noun
    "WireGuard listens on 51820",  # bare 5-digit numeral
    "dedup threshold is 0.06",  # decimal config value
    "see https://example.com/x",  # URL
    "config at ~/.config/kb/settings.toml",  # file path
    "set KB_DB_PATH to override",  # ENV_CAPS env-var token
    "call store.create_entry(short_title)",  # dotted code identifier/signature
    'the flag is "--no-push"',  # quoted literal value
]


@pytest.mark.parametrize("body", MUST_FLAG)
def test_must_flag(body):
    assert lint_map_body(body) != []


# --- MUST-NOT-FLAG: §7.3 pointers-in-disguise / structural counts / kb-refs ---

MUST_NOT_FLAG = [
    "the pipeline has three stages and 4 components",  # word + digit counts
    "the ingestion node has five edges",  # word-form count
    "there are 12 stages",  # digit count, magnitude > 9
    "this map orients kb-00421 and kb-01670",  # kb-refs are pointers
]


@pytest.mark.parametrize("body", MUST_NOT_FLAG)
def test_must_not_flag(body):
    assert lint_map_body(body) == []


# --- empty / clean ---


def test_empty_on_clean():
    assert lint_map_body("orients the auth subsystem; follow the edges to sources") == []


def test_empty_string_is_clean():
    assert lint_map_body("") == []


# --- compositional body budget (calibrated on the two best corpus maps) ---

# kb-01724: 11 distinct pointers, 2319-char body. Measured live 2026-09-20; one
# of the two best maps in the corpus (inline bulleted `- kb-XXXXX -- gloss`
# gloss lines) and one of only two maps that exceeded the old flat ~1500 cap.
KB_01724_CHARS = 2319
KB_01724_POINTERS = 11

# kb-03257: 20 distinct kb- ids, 2540-char body. Measured live 2026-09-20; the
# other of the two best maps, using a bare comma-separated id list plus two
# separate prose gloss paragraphs — an incompatible gloss layout to kb-01724,
# which is why pointers are counted format-agnostically.
KB_03257_CHARS = 2540
KB_03257_POINTERS = 20


def _calibration_body(total_chars: int, pointer_count: int) -> str:
    """Synthetic body with the exact measured length and distinct kb- id count.

    Every non-id position is a filler character, so no purity heuristic can
    fire and the fixture isolates the length rule. The real corpus bodies are
    deliberately NOT reproduced here: the budget must be measured on length
    and distinct-id count, not on any specific gloss layout.
    """
    ids_text = " ".join(f"kb-{1000 + i:05d}" for i in range(pointer_count))
    body = ids_text + "x" * (total_chars - len(ids_text))
    assert len(body) == total_chars
    assert count_map_pointers(body) == pointer_count
    return body


def test_kb_01724_calibration_passes():
    body = _calibration_body(KB_01724_CHARS, KB_01724_POINTERS)
    budget = map_body_budget(KB_01724_POINTERS)
    assert budget == 2825
    assert len(body) / budget == pytest.approx(0.82, abs=0.01)
    assert lint_map_body(body) == []


def test_kb_03257_calibration_passes():
    body = _calibration_body(KB_03257_CHARS, KB_03257_POINTERS)
    budget = map_body_budget(KB_03257_POINTERS)
    assert budget == 4400
    assert len(body) / budget == pytest.approx(0.58, abs=0.01)
    assert lint_map_body(body) == []


def test_thin_map_budget_is_smaller_than_the_old_flat_cap():
    # The old flat ~1500 cap was too loose for thin maps: a 1-pointer body of
    # 1200 chars passed before and must fail now (budget 900 + 175 = 1075).
    body = _calibration_body(1200, 1)
    assert map_body_budget(1) == 1075
    warnings = lint_map_body(body)
    assert warnings != []
    assert any("1200" in w and "1075" in w and "1 pointer" in w for w in warnings)


def test_budget_is_strictly_monotonic_in_pointer_count():
    budgets = [map_body_budget(p) for p in range(0, 30)]
    assert all(earlier < later for earlier, later in pairwise(budgets))


def test_zero_pointer_budget_is_exactly_the_base_allowance():
    # A zero-pointer map is separately rejected by the orphan check; this pins
    # that the arithmetic is well-defined at the boundary.
    assert map_body_budget(0) == MAP_BODY_BASE_CHARS
    assert MAP_BODY_BASE_CHARS == 900
    assert MAP_BODY_PER_POINTER_CHARS == 175


def test_warning_states_actual_budget_and_pointer_count():
    # 2 pointers -> budget 900 + 2*175 = 1250; a 1600-char body must say so.
    body = _calibration_body(1600, 2)
    warnings = lint_map_body(body)
    assert any("1600" in w and "1250" in w and "2 pointers" in w for w in warnings)


def test_body_at_exactly_its_budget_passes():
    assert lint_map_body("x" * map_body_budget(0)) == []
    assert lint_map_body(_calibration_body(map_body_budget(1), 1)) == []


def test_purity_warnings_identical_at_any_body_length():
    # The purity rules are unchanged by this item: the same body trips the
    # same purity warnings whether it is under or over its length budget.
    for snippet in MUST_FLAG:
        padded = snippet + " " + "x" * 1200  # over every relevant budget
        short = [w for w in lint_map_body(snippet) if "chars" not in w]
        long = [w for w in lint_map_body(padded) if "chars" not in w]
        assert short != []
        assert short == long


# --- contract guards ---


def test_all_warnings_have_advisory_prefix():
    body = 'port 8767, see https://example.com/x, KB_DB_PATH, store.f(), "x"'
    warnings = lint_map_body(body)
    assert warnings != []
    assert all(w.startswith("Map lint (advisory):") for w in warnings)


def test_no_warning_signals_rejection():
    body = 'port 8767, https://example.com/x, KB_DB_PATH, store.f(), "x"' + "y" * 1600
    warnings = lint_map_body(body)
    assert warnings != []
    assert not any(w.startswith("Error:") for w in warnings)


def test_count_of_parts_large_magnitude_not_flagged():
    assert lint_map_body("there are 100 nodes") == []


def test_kb_ref_never_flagged():
    assert lint_map_body("kb-00001 kb-99999") == []
