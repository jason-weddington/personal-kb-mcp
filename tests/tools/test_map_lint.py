"""Unit tests for the pure-function mental_map lint (src/personal_kb/tools/map_lint.py)."""

import pytest

from personal_kb.tools.map_lint import MAP_BODY_ADVISORY_CHARS, lint_map_body

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


# --- char-cap advisory ---


def test_char_cap_warning():
    body = "x" * 1600
    warnings = lint_map_body(body)
    assert warnings != []
    assert any(str(len(body)) in w and "1500" in w for w in warnings)


def test_char_cap_not_triggered_at_or_below_limit():
    body = "a" * MAP_BODY_ADVISORY_CHARS  # exactly 1500, not > 1500
    assert all("1500" not in w for w in lint_map_body(body))


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
