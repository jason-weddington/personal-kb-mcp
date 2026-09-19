"""Tests for the new-maps delta FYI (GTD 64f71a8a).

Covers:
* ``render.render_new_maps`` — one-line delta rendering, always-labelled,
  the ``NEW_MAPS_DISPLAY_CAP`` display cap + ``(+N more)`` suffix, stable
  ordering, empty input.
* ``suppression.get_surfaced_map_ids`` — the read-only accessor the delta
  path uses to compute what is genuinely new.
* CLI integration: orientation reasons (first-emission, scope-change,
  compact) still emit the byte-identical FULL roster; new-maps emits ONLY
  the delta; announce-once; the display cap does not re-announce omitted
  maps; the (should-be-unreachable) empty-delta guard.
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli
from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.paths import get_whisper_log_path
from personal_kb_hook.render import (
    BANNED_TOKENS,
    NEW_MAPS_DISPLAY_CAP,
    compose_directory,
    render_new_maps,
)
from personal_kb_hook.suppression import EmitReason, get_surfaced_map_ids, mark_emitted

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# render_new_maps unit tests
# ---------------------------------------------------------------------------


def _entry(id_: str, short: str, long: str = "") -> dict[str, str]:
    return {"id": id_, "short_title": short, "long_title": long}


def test_render_new_maps_empty_returns_none() -> None:
    assert render_new_maps([]) is None


def test_render_new_maps_single() -> None:
    pairs = [(MapKey(label="personal", id="kb-2"), _entry("kb-2", "ingest"))]
    assert render_new_maps(pairs) == "New map — personal/[kb-2] ingest"


def test_render_new_maps_several_uses_plural_header_and_comma_join() -> None:
    pairs = [
        (MapKey(label="personal", id="kb-1"), _entry("kb-1", "auth")),
        (MapKey(label="team", id="kb-9"), _entry("kb-9", "ops")),
    ]
    assert render_new_maps(pairs) == "New maps — personal/[kb-1] auth, team/[kb-9] ops"


def test_render_new_maps_always_includes_label_single_kb() -> None:
    """Unlike render_whisper, the label is present even with only one KB configured."""
    pairs = [(MapKey(label="personal", id="kb-1"), _entry("kb-1", "auth"))]
    out = render_new_maps(pairs)
    assert out is not None
    assert "personal/[kb-1]" in out


def test_render_new_maps_cap_constant_is_five() -> None:
    assert NEW_MAPS_DISPLAY_CAP == 5


def test_render_new_maps_caps_at_five_with_overflow_suffix() -> None:
    pairs = [
        (MapKey(label="personal", id=f"kb-{i}"), _entry(f"kb-{i}", f"title{i}"))
        for i in range(1, 8)  # 7 new maps
    ]
    out = render_new_maps(pairs)
    assert out is not None
    assert out == (
        "New maps — personal/[kb-1] title1, personal/[kb-2] title2, "
        "personal/[kb-3] title3, personal/[kb-4] title4, "
        "personal/[kb-5] title5 (+2 more)"
    )
    # Omitted entries (kb-6, kb-7) are not named anywhere in the text.
    assert "kb-6" not in out
    assert "kb-7" not in out


def test_render_new_maps_stable_order_independent_of_input_order() -> None:
    """Display order is MapKey's natural (label, id) order, not input order."""
    pairs_a = [
        (MapKey(label="personal", id="kb-2"), _entry("kb-2", "b")),
        (MapKey(label="personal", id="kb-1"), _entry("kb-1", "a")),
    ]
    pairs_b = list(reversed(pairs_a))
    assert render_new_maps(pairs_a) == render_new_maps(pairs_b)
    assert render_new_maps(pairs_a) == "New maps — personal/[kb-1] a, personal/[kb-2] b"


def test_render_new_maps_no_banned_tokens() -> None:
    pairs = [(MapKey(label="personal", id="kb-1"), _entry("kb-1", "auth"))]
    out = render_new_maps(pairs)
    assert out is not None
    lowered = out.lower()
    for tok in BANNED_TOKENS:
        assert tok not in lowered


# ---------------------------------------------------------------------------
# suppression.get_surfaced_map_ids unit tests
# ---------------------------------------------------------------------------


def test_get_surfaced_map_ids_empty_when_no_scratch(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch.json"
    assert get_surfaced_map_ids(session_id="s1", scratch_path=scratch) == frozenset()


def test_get_surfaced_map_ids_reflects_mark_emitted(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch.json"
    keys = [MapKey(label="personal", id="kb-1"), MapKey(label="team", id="kb-2")]
    mark_emitted(session_id="s1", scope="proj", map_ids=keys, scratch_path=scratch)
    assert get_surfaced_map_ids(session_id="s1", scratch_path=scratch) == frozenset(keys)


def test_get_surfaced_map_ids_is_read_only(tmp_path: Path) -> None:
    """Calling the accessor does not itself mutate the scratch file."""
    scratch = tmp_path / "scratch.json"
    assert not scratch.exists()
    get_surfaced_map_ids(session_id="s1", scratch_path=scratch)
    assert not scratch.exists()


# ---------------------------------------------------------------------------
# CLI integration helpers (mirrors tests/test_whisper_telemetry.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    cache_dir = tmp_path / ".cache" / "personal_kb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    return {"root": tmp_path, "cache_dir": cache_dir}


def _read_log(session_id: str) -> list[dict[str, Any]]:
    path = get_whisper_log_path(session_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if not s:
            continue
        rows.append(json.loads(s))
    return rows


def _run_cli(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    args: list[str] | None = None,
) -> tuple[int, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    stdout_buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout_buf)
    rc = 0
    try:
        cli.main(args or [])
    except SystemExit as exc:
        rc = int(exc.code or 0)
    return rc, stdout_buf.getvalue()


def _stub_http_index(
    monkeypatch: pytest.MonkeyPatch,
    projects: dict[str, list[tuple[str, dict[str, str]]]],
) -> None:
    monkeypatch.setattr(cli.http_index, "load_index", lambda roster_arg: projects)


def _prompt_payload(cwd: Path, session_id: str) -> dict[str, str]:
    """Build a minimal UserPromptSubmit payload for ``cwd``/``session_id``."""
    return {"hook_event_name": "UserPromptSubmit", "cwd": str(cwd), "session_id": session_id}


# ---------------------------------------------------------------------------
# Orientation reasons: byte-identical full roster
# ---------------------------------------------------------------------------


def test_first_emission_full_roster_byte_identical(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    index = {"personal-kb": [("personal", _entry("kb-1", "auth", "Auth"))]}
    _stub_http_index(monkeypatch, index)
    session_id = "sess-first"
    rc, out = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        ["--format=text"],
    )
    assert rc == 0
    expected = compose_directory("personal-kb", [_entry("kb-1", "auth", "Auth")], index)
    assert out == expected
    rows = _read_log(session_id)
    assert rows[-1]["trigger_context"]["emit_reason"] == "first-emission"


def test_scope_change_full_roster_byte_identical(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    proj_a = hook_env["root"] / "proj_a"
    proj_b = hook_env["root"] / "proj_b"
    proj_a.mkdir()
    proj_b.mkdir()
    (proj_a / ".kb_project").write_text("proj-a\n", encoding="utf-8")
    (proj_b / ".kb_project").write_text("proj-b\n", encoding="utf-8")
    index = {
        "proj-a": [("personal", _entry("kb-1", "a1", "A1"))],
        "proj-b": [("personal", _entry("kb-2", "b1", "B1"))],
    }
    _stub_http_index(monkeypatch, index)
    session_id = "sess-scope-change"
    rc1, _ = _run_cli(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(proj_a), "session_id": session_id},
        ["--format=text"],
    )
    assert rc1 == 0
    rc2, out2 = _run_cli(
        monkeypatch,
        {"hook_event_name": "UserPromptSubmit", "cwd": str(proj_b), "session_id": session_id},
        ["--format=text"],
    )
    assert rc2 == 0
    expected2 = compose_directory("proj-b", [_entry("kb-2", "b1", "B1")], index)
    assert out2 == expected2
    rows = _read_log(session_id)
    assert rows[-1]["trigger_context"]["emit_reason"] == "scope-change"


def test_compact_full_roster_byte_identical(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    index = {"personal-kb": [("personal", _entry("kb-1", "auth", "Auth"))]}
    _stub_http_index(monkeypatch, index)
    session_id = "sess-compact"
    rc1, out1 = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        ["--format=text"],
    )
    assert rc1 == 0
    rc2, out2 = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
            "source": "compact",
        },
        ["--format=text"],
    )
    assert rc2 == 0
    expected = compose_directory("personal-kb", [_entry("kb-1", "auth", "Auth")], index)
    assert out1 == expected
    assert out2 == expected
    rows = _read_log(session_id)
    assert rows[-1]["trigger_context"]["emit_reason"] == "compact"


# ---------------------------------------------------------------------------
# new-maps: delta only, announce-once, cap, empty-delta guard
# ---------------------------------------------------------------------------


def test_new_maps_emits_delta_not_full_roster(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    session_id = "sess-delta-basic"

    _stub_http_index(monkeypatch, {"personal-kb": [("personal", _entry("kb-1", "auth"))]})
    rc1, _ = _run_cli(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(hook_env["root"]), "session_id": session_id},
        ["--format=text"],
    )
    assert rc1 == 0

    _stub_http_index(
        monkeypatch,
        {
            "personal-kb": [
                ("personal", _entry("kb-1", "auth")),
                ("personal", _entry("kb-2", "ingest")),
            ]
        },
    )
    rc2, out2 = _run_cli(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(hook_env["root"]),
            "session_id": session_id,
        },
        ["--format=text"],
    )
    assert rc2 == 0
    assert out2 == "New map — personal/[kb-2] ingest"
    assert "Maps for" not in out2


def test_new_maps_announced_exactly_once(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """The same new map is announced once; an unchanged re-run is fully silent."""
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    session_id = "sess-once"

    _stub_http_index(monkeypatch, {"personal-kb": [("personal", _entry("kb-1", "auth"))]})
    rc1, _ = _run_cli(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(hook_env["root"]), "session_id": session_id},
        ["--format=text"],
    )
    assert rc1 == 0

    index_v2 = {
        "personal-kb": [
            ("personal", _entry("kb-1", "auth")),
            ("personal", _entry("kb-2", "ingest")),
        ]
    }
    _stub_http_index(monkeypatch, index_v2)
    rc2, out2 = _run_cli(
        monkeypatch,
        _prompt_payload(hook_env["root"], session_id),
        ["--format=text"],
    )
    assert rc2 == 0
    assert "kb-2" in out2

    # Third run: identical map set -> should_emit() returns None -> silent,
    # NOT an empty FYI line.
    rc3, out3 = _run_cli(
        monkeypatch,
        _prompt_payload(hook_env["root"], session_id),
        ["--format=text"],
    )
    assert rc3 == 0
    assert out3 == ""


def test_new_maps_cap_records_all_and_does_not_reannounce_overflow(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """>5 new maps in one emission: only 5 are named + '(+N more)', but ALL are
    marked surfaced and telemetered, so the omitted ones are never re-announced.
    """
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    session_id = "sess-cap"

    _stub_http_index(monkeypatch, {"personal-kb": [("personal", _entry("kb-0", "seed"))]})
    rc1, _ = _run_cli(
        monkeypatch,
        {"hook_event_name": "SessionStart", "cwd": str(hook_env["root"]), "session_id": session_id},
        ["--format=text"],
    )
    assert rc1 == 0

    new_entries = [("personal", _entry(f"kb-{i}", f"title{i}")) for i in range(1, 8)]  # 7 new
    index_v2 = {"personal-kb": [("personal", _entry("kb-0", "seed")), *new_entries]}
    _stub_http_index(monkeypatch, index_v2)
    rc2, out2 = _run_cli(
        monkeypatch,
        _prompt_payload(hook_env["root"], session_id),
        ["--format=text"],
    )
    assert rc2 == 0
    assert "(+2 more)" in out2
    assert "kb-6" not in out2
    assert "kb-7" not in out2

    # Telemetry: all 7 new maps get rows (not just the 5 displayed).
    rows = _read_log(session_id)
    second_batch_ids = {
        r["map_id"] for r in rows if r["trigger_context"]["emit_reason"] == "new-maps"
    }
    assert second_batch_ids == {f"kb-{i}" for i in range(1, 8)}

    # Fourth call, same full set (kb-0..kb-7): fully suppressed, including the
    # display-capped kb-6/kb-7, which must NOT be re-announced.
    rc3, out3 = _run_cli(
        monkeypatch,
        _prompt_payload(hook_env["root"], session_id),
        ["--format=text"],
    )
    assert rc3 == 0
    assert out3 == ""


def test_empty_delta_guard_emits_nothing_and_writes_no_telemetry(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """The should-be-unreachable state (reason=new-maps, delta empty) must
    emit NOTHING — never fall back to the full directory, never a header
    with no entries.
    """
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    monkeypatch.chdir(hook_env["root"])
    session_id = "sess-empty-delta"

    index = {"personal-kb": [("personal", _entry("kb-1", "auth"))]}
    _stub_http_index(monkeypatch, index)

    # Pre-seed the scratch so kb-1 is already surfaced, then force
    # should_emit() to (artificially) report NEW_MAPS anyway.
    mark_emitted(
        session_id=session_id,
        scope="personal-kb",
        map_ids=[MapKey(label="personal", id="kb-1")],
    )
    monkeypatch.setattr(cli, "should_emit", lambda **kwargs: EmitReason.NEW_MAPS)

    rc, out = _run_cli(
        monkeypatch,
        _prompt_payload(hook_env["root"], session_id),
        ["--format=text"],
    )
    assert rc == 0
    assert out == ""
    assert _read_log(session_id) == []
