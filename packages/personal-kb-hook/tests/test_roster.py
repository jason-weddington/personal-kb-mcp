"""Tests for personal_kb_hook.roster.load_roster().

Covers the 14 unit cases enumerated in the P0 acceptance criteria plus
the load-bearing identity tests (legacy-fallback, listener-gate
unchanged, config-path-literal). Uses ``tmp_path`` + ``monkeypatch``
matching the style of tests/test_resolver.py and tests/test_http_index.py.
The tests.* mypy override (pyproject.toml lines 54-56) relaxes
disallow_untyped_defs, so test functions are intentionally lightly
annotated.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli, http_index, listener, listener_worker
from personal_kb_hook.roster import KbEntry, load_roster

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def roster_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate XDG_CONFIG_HOME and clear all legacy env vars.

    Points HOME at an empty subdir too, so a stray ~/.config/personal_kb/kbs.json
    on the build machine cannot influence the test.
    """
    xdg = tmp_path / "xdg"
    xdg.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)
    return {
        "xdg": xdg,
        "home": home,
        "config_dir": xdg / "personal_kb",
        "config_file": xdg / "personal_kb" / "kbs.json",
    }


def _write_roster(config_file: Path, entries: list[dict[str, Any]]) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(entries), encoding="utf-8")


def _write_key_file(tmp_path: Path, name: str, contents: str) -> Path:
    key_path = tmp_path / name
    key_path.write_text(contents, encoding="utf-8")
    return key_path


# ---------------------------------------------------------------------------
# Case 1: two valid entries → both returned in file order
# ---------------------------------------------------------------------------


def test_case01_two_valid_entries_returned_in_file_order(
    roster_env: dict[str, Path], tmp_path: Path
) -> None:
    key_a = _write_key_file(tmp_path, "key_a", "alpha-secret\n")
    key_b = _write_key_file(tmp_path, "key_b", "beta-secret\n")
    _write_roster(
        roster_env["config_file"],
        [
            {"label": "alpha", "url": "https://alpha.kb/", "key_file": str(key_a)},
            {"label": "beta", "url": "http://beta.kb/", "key_file": str(key_b)},
        ],
    )
    result = load_roster()
    assert result == [
        KbEntry(label="alpha", url="https://alpha.kb/", key="alpha-secret"),
        KbEntry(label="beta", url="http://beta.kb/", key="beta-secret"),
    ]


# ---------------------------------------------------------------------------
# Case 2: non-http(s) url dropped
# ---------------------------------------------------------------------------


def test_case02_non_http_url_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key_ok = _write_key_file(tmp_path, "k_ok", "good\n")
    key_bad = _write_key_file(tmp_path, "k_bad", "should-be-dropped\n")
    _write_roster(
        roster_env["config_file"],
        [
            {"label": "ok", "url": "https://ok.kb/", "key_file": str(key_ok)},
            {"label": "ftp", "url": "ftp://bad.kb/", "key_file": str(key_bad)},
            {"label": "file", "url": "file:///etc/passwd", "key_file": str(key_bad)},
        ],
    )
    result = load_roster()
    assert result == [KbEntry(label="ok", url="https://ok.kb/", key="good")]


# ---------------------------------------------------------------------------
# Case 3: missing/nonexistent key_file dropped
# ---------------------------------------------------------------------------


def test_case03_nonexistent_key_file_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    _write_roster(
        roster_env["config_file"],
        [
            {
                "label": "ghost",
                "url": "https://ghost.kb/",
                "key_file": str(tmp_path / "does-not-exist"),
            }
        ],
    )
    assert load_roster() == []


# ---------------------------------------------------------------------------
# Case 4: blank-contents key_file dropped
# ---------------------------------------------------------------------------


def test_case04_blank_contents_key_file_dropped(
    roster_env: dict[str, Path], tmp_path: Path
) -> None:
    blank = _write_key_file(tmp_path, "blank_key", "   \n\t\n")
    _write_roster(
        roster_env["config_file"],
        [{"label": "blank", "url": "https://blank.kb/", "key_file": str(blank)}],
    )
    assert load_roster() == []


# ---------------------------------------------------------------------------
# Case 5: non-dict element dropped
# ---------------------------------------------------------------------------


def test_case05_non_dict_element_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key_ok = _write_key_file(tmp_path, "k_ok", "good\n")
    # Manually serialize mixed types — _write_roster accepts list[dict] only.
    roster_env["config_file"].parent.mkdir(parents=True, exist_ok=True)
    roster_env["config_file"].write_text(
        json.dumps(
            [
                "a string is not a dict",
                42,
                ["a list", "is not a dict"],
                None,
                {"label": "ok", "url": "https://ok.kb/", "key_file": str(key_ok)},
            ]
        ),
        encoding="utf-8",
    )
    assert load_roster() == [KbEntry(label="ok", url="https://ok.kb/", key="good")]


# ---------------------------------------------------------------------------
# Case 6: malformed/non-JSON file → legacy fallback
# ---------------------------------------------------------------------------


def test_case06_malformed_json_falls_back_to_legacy(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    roster_env["config_file"].parent.mkdir(parents=True, exist_ok=True)
    roster_env["config_file"].write_text("{this is not (valid json", encoding="utf-8")
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == [KbEntry(label="personal", url="https://legacy.kb/", key="legacy-key")]


# ---------------------------------------------------------------------------
# Case 7: absent file + both legacy env vars → single 'personal' entry
# (the load-bearing IDENTITY regression test)
# ---------------------------------------------------------------------------


def test_case07_legacy_fallback_identity(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MANDATORY identity guard. Absent kbs.json + both legacy env vars set →
    exactly ``[KbEntry(label='personal', url=<URL>, key=<KEY>)]``.
    """
    # No kbs.json written to roster_env["xdg"]/personal_kb/.
    assert not roster_env["config_file"].exists()
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    result = load_roster()
    assert result == [KbEntry(label="personal", url="https://legacy.kb/", key="legacy-key")]
    # Defense-in-depth: exactly one element, exact label, exact url/key.
    assert len(result) == 1
    assert result[0].label == "personal"
    assert result[0].url == "https://legacy.kb/"
    assert result[0].key == "legacy-key"


# ---------------------------------------------------------------------------
# Case 8: absent file + both legacy env vars absent → []
# ---------------------------------------------------------------------------


def test_case08_absent_file_no_legacy_env_returns_empty(
    roster_env: dict[str, Path],
) -> None:
    assert not roster_env["config_file"].exists()
    # No config and no env: local-mode defaults synthesize the personal entry.
    assert load_roster() == [
        KbEntry(label="personal", url="http://127.0.0.1:8765", key="local-no-auth")
    ]


# ---------------------------------------------------------------------------
# Case 9: XDG_CONFIG_HOME honored / ~/.config used when unset or empty
# ---------------------------------------------------------------------------


def test_case09a_xdg_config_home_honored_when_set(
    roster_env: dict[str, Path], tmp_path: Path
) -> None:
    key_ok = _write_key_file(tmp_path, "k_ok", "via-xdg\n")
    _write_roster(
        roster_env["config_file"],
        [{"label": "team", "url": "https://team.kb/", "key_file": str(key_ok)}],
    )
    assert load_roster() == [KbEntry(label="team", url="https://team.kb/", key="via-xdg")]


def test_case09b_dotconfig_used_when_xdg_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    key_ok = _write_key_file(tmp_path, "k_ok", "via-dotconfig\n")
    config_file = home / ".config" / "personal_kb" / "kbs.json"
    _write_roster(
        config_file,
        [{"label": "dc", "url": "https://dc.kb/", "key_file": str(key_ok)}],
    )
    assert load_roster() == [KbEntry(label="dc", url="https://dc.kb/", key="via-dotconfig")]


def test_case09c_dotconfig_used_when_xdg_empty_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", "")  # empty string → treat as unset
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)

    key_ok = _write_key_file(tmp_path, "k_ok", "via-dotconfig-empty\n")
    config_file = home / ".config" / "personal_kb" / "kbs.json"
    _write_roster(
        config_file,
        [{"label": "dc", "url": "https://dc.kb/", "key_file": str(key_ok)}],
    )
    assert load_roster() == [KbEntry(label="dc", url="https://dc.kb/", key="via-dotconfig-empty")]


# ---------------------------------------------------------------------------
# Case 10: key_file contents are .strip()ed (trailing newline removed)
# ---------------------------------------------------------------------------


def test_case10_key_file_contents_are_stripped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key = _write_key_file(tmp_path, "k", "  padded-key\n\n")
    _write_roster(
        roster_env["config_file"],
        [{"label": "s", "url": "https://s.kb/", "key_file": str(key)}],
    )
    result = load_roster()
    assert result == [KbEntry(label="s", url="https://s.kb/", key="padded-key")]
    # Defense-in-depth: no whitespace survived.
    assert result[0].key == "padded-key"


# ---------------------------------------------------------------------------
# Case 11: explicit [] roster + both legacy env vars set → [] (NOT legacy)
# ---------------------------------------------------------------------------


def test_case11_empty_list_does_not_fall_back_to_legacy(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_roster(roster_env["config_file"], [])
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == []


# ---------------------------------------------------------------------------
# Case 12: all-entries-dropped + both legacy env vars set → [] (NOT legacy)
# ---------------------------------------------------------------------------


def test_case12_all_dropped_does_not_fall_back_to_legacy(
    roster_env: dict[str, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_bad = _write_key_file(tmp_path, "k_bad", "ignored\n")
    _write_roster(
        roster_env["config_file"],
        [{"label": "ftp", "url": "ftp://bad.kb/", "key_file": str(key_bad)}],
    )
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == []


# ---------------------------------------------------------------------------
# Case 13: URL set, API_KEY unset, no config file → []
# ---------------------------------------------------------------------------


def test_case13_url_set_key_unset_no_config_returns_empty(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not roster_env["config_file"].exists()
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    # PERSONAL_KB_API_KEY is unset by the fixture and we don't set it here.
    assert load_roster() == []


# ---------------------------------------------------------------------------
# Case 14: API_KEY set, URL = empty string, no config file → []
# ---------------------------------------------------------------------------


def test_case14_key_set_url_empty_string_no_config_returns_empty(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not roster_env["config_file"].exists()
    monkeypatch.setenv("PERSONAL_KB_URL", "")  # empty string → treated as unset
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == [
        KbEntry(label="personal", url="http://127.0.0.1:8765", key="legacy-key")
    ]


# ---------------------------------------------------------------------------
# Config-path-literal test (AC10): underscore dir + kbs.json filename are
# pinned. A hyphen-dir or wrong-filename file MUST NOT be read.
# ---------------------------------------------------------------------------


def test_config_path_literal_underscore_dir_and_filename(
    roster_env: dict[str, Path], tmp_path: Path
) -> None:
    """The roster file MUST live at <config>/personal_kb/kbs.json (underscore)."""
    key_ok = _write_key_file(tmp_path, "k_ok", "correct-path\n")
    correct = roster_env["xdg"] / "personal_kb" / "kbs.json"
    _write_roster(
        correct,
        [{"label": "ok", "url": "https://ok.kb/", "key_file": str(key_ok)}],
    )
    assert load_roster() == [KbEntry(label="ok", url="https://ok.kb/", key="correct-path")]


def test_config_path_hyphen_dir_not_read(roster_env: dict[str, Path], tmp_path: Path) -> None:
    """A hyphen-dir variant (``personal-kb``) MUST NOT be discovered."""
    key_ok = _write_key_file(tmp_path, "k_ok", "hyphen-path\n")
    hyphen = roster_env["xdg"] / "personal-kb" / "kbs.json"
    _write_roster(
        hyphen,
        [{"label": "ok", "url": "https://ok.kb/", "key_file": str(key_ok)}],
    )
    # Correct path (underscore) is absent → legacy fallback path taken;
    # no legacy env vars set → local-mode default entry.
    assert load_roster() == [
        KbEntry(label="personal", url="http://127.0.0.1:8765", key="local-no-auth")
    ]


def test_config_path_wrong_filename_not_read(roster_env: dict[str, Path], tmp_path: Path) -> None:
    """A wrong filename (``kb.json``) MUST NOT be discovered."""
    key_ok = _write_key_file(tmp_path, "k_ok", "wrong-name\n")
    wrong = roster_env["xdg"] / "personal_kb" / "kb.json"
    _write_roster(
        wrong,
        [{"label": "ok", "url": "https://ok.kb/", "key_file": str(key_ok)}],
    )
    assert load_roster() == [
        KbEntry(label="personal", url="http://127.0.0.1:8765", key="local-no-auth")
    ]


# ---------------------------------------------------------------------------
# Listener-gate identity test (AC9): is_listener_enabled MUST remain the
# single global env gate, untouched by this item.
# ---------------------------------------------------------------------------


def test_listener_gate_unchanged_flag_unset(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://x.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "k")
    monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)
    assert listener.is_listener_enabled() is False


def test_listener_gate_unchanged_all_set(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "https://x.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "k")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "true")
    assert listener.is_listener_enabled() is True


def test_listener_gate_unchanged_url_empty(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PERSONAL_KB_URL", "")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "k")
    monkeypatch.setenv("PERSONAL_KB_LISTENER", "1")
    # Empty URL means the local default; the opt-in flag still gates.
    assert listener.is_listener_enabled() is True


# ---------------------------------------------------------------------------
# Extra: missing-label / blank-label / non-str url / wholly-empty file
# ---------------------------------------------------------------------------


def test_missing_label_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key = _write_key_file(tmp_path, "k", "good\n")
    _write_roster(
        roster_env["config_file"],
        [{"url": "https://ok.kb/", "key_file": str(key)}],
    )
    assert load_roster() == []


def test_blank_label_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key = _write_key_file(tmp_path, "k", "good\n")
    _write_roster(
        roster_env["config_file"],
        [{"label": "", "url": "https://ok.kb/", "key_file": str(key)}],
    )
    assert load_roster() == []


def test_non_str_url_dropped(roster_env: dict[str, Path], tmp_path: Path) -> None:
    key = _write_key_file(tmp_path, "k", "good\n")
    roster_env["config_file"].parent.mkdir(parents=True, exist_ok=True)
    roster_env["config_file"].write_text(
        json.dumps([{"label": "ok", "url": 12345, "key_file": str(key)}]),
        encoding="utf-8",
    )
    assert load_roster() == []


def test_empty_file_triggers_legacy_fallback(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A zero-byte / whitespace-only file is treated as 'unparseable'."""
    roster_env["config_file"].parent.mkdir(parents=True, exist_ok=True)
    roster_env["config_file"].write_text("   \n\t\n", encoding="utf-8")
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == [KbEntry(label="personal", url="https://legacy.kb/", key="legacy-key")]


def test_non_list_top_level_triggers_legacy_fallback(
    roster_env: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON object (not a list) is treated as 'unparseable as a roster'."""
    roster_env["config_file"].parent.mkdir(parents=True, exist_ok=True)
    roster_env["config_file"].write_text(
        json.dumps({"label": "oops", "url": "https://x.kb/"}), encoding="utf-8"
    )
    monkeypatch.setenv("PERSONAL_KB_URL", "https://legacy.kb/")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "legacy-key")
    assert load_roster() == [KbEntry(label="personal", url="https://legacy.kb/", key="legacy-key")]


def test_tilde_in_key_file_path_is_expanded(roster_env: dict[str, Path], tmp_path: Path) -> None:
    """``~`` in ``key_file`` resolves via expanduser() against $HOME."""
    home = roster_env["home"]
    secret_path = home / ".my_kb_key"
    secret_path.write_text("tilde-expanded\n", encoding="utf-8")
    _write_roster(
        roster_env["config_file"],
        [{"label": "t", "url": "https://t.kb/", "key_file": "~/.my_kb_key"}],
    )
    assert load_roster() == [KbEntry(label="t", url="https://t.kb/", key="tilde-expanded")]


# ---------------------------------------------------------------------------
# Type-safety regression: KbEntry is structurally distinct from MapEntry.
# ---------------------------------------------------------------------------


def test_kb_entry_shape() -> None:
    """KbEntry has exactly (label, url, key) in that order, all str."""
    entry = KbEntry(label="x", url="https://x/", key="k")
    assert entry.label == "x"
    assert entry.url == "https://x/"
    assert entry.key == "k"
    assert entry._fields == ("label", "url", "key")
    # Tuple positional unpacking matches declared order.
    label, url, key = entry
    assert (label, url, key) == ("x", "https://x/", "k")


# ---------------------------------------------------------------------------
# Wiring guard (post-P2): load_roster is wired into cli.py (the directory
# pipeline) AND listener_worker (the detached Stop-event whisper fan-out —
# AC-2 of P2 hands the roster lookup to the worker, NOT the hook).
# http_index continues to take the roster as an explicit parameter from
# cli.py and does not import load_roster; listener still does not call
# load_roster itself (the cache helpers are roster-agnostic).
# ---------------------------------------------------------------------------


def test_load_roster_wired_into_cli_and_worker_only() -> None:
    """Post-P2 wiring: ``load_roster`` is wired in cli.py AND listener_worker.

    The Stop event's hook path still does not pass the roster to the worker
    (spawn_worker keeps its 2-positional-arg signature); the worker reads
    the roster itself when it wakes up. http_index continues to receive
    the roster as an explicit parameter from cli.py — it does NOT import
    load_roster. listener (the cache helpers + transcript tail + env gate)
    is roster-agnostic and still does not reference load_roster.
    """
    cli_src = inspect.getsource(cli)
    assert "load_roster" in cli_src, (
        "Post-P2 wiring guard: 'load_roster' must be wired into cli.py for the "
        "SessionStart/UserPromptSubmit directory pipeline + whisper rendering."
    )
    worker_src = inspect.getsource(listener_worker)
    assert "load_roster" in worker_src, (
        "Post-P2 wiring guard: 'load_roster' must be wired into listener_worker "
        "for the Stop-event whisper fan-out (AC-2)."
    )
    for module in (http_index, listener):
        src = inspect.getsource(module)
        assert "load_roster" not in src, (
            f"Post-P2 scope guard: 'load_roster' must NOT appear in "
            f"{module.__name__} — only cli.py and listener_worker wire it in P2."
        )
