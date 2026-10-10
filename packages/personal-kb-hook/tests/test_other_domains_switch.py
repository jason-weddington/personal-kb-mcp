"""KB_ROSTER_OTHER_DOMAINS: the roster's 'Maps in other domains' line is opt-in."""

from __future__ import annotations

import contextlib
import io
import json
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli
from personal_kb_hook.paths import get_whisper_log_path
from personal_kb_hook.render import other_domains_enabled

if TYPE_CHECKING:
    from pathlib import Path

OTHER = "Maps in other domains"


@pytest.mark.parametrize("value", ["", "0", "false", "off", "no", "garbage"])
def test_switch_off_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("KB_ROSTER_OTHER_DOMAINS", value)
    assert other_domains_enabled() is False


@pytest.mark.parametrize("value", ["1", "TRUE", " on ", "yes"])
def test_switch_on_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("KB_ROSTER_OTHER_DOMAINS", value)
    assert other_domains_enabled() is True


def test_switch_unset_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KB_ROSTER_OTHER_DOMAINS", raising=False)
    assert other_domains_enabled() is False


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("HEADLESS_BUILD_ENGINE", raising=False)
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "secret")
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    (tmp_path / "kb").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "kb" / "knowledge.db"))
    (tmp_path / ".kb_project").write_text("proj-a\n", encoding="utf-8")
    return tmp_path


def _entry(id_: str, short: str) -> dict[str, str]:
    return {"id": id_, "short_title": short, "long_title": ""}


def _stub(monkeypatch: pytest.MonkeyPatch, projects: dict[str, Any]) -> None:
    monkeypatch.setattr(cli.http_index, "load_index", lambda roster_arg: projects)


def _run(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> str:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", buf)
    with contextlib.suppress(SystemExit):
        cli.main(["--format=text"])
    return buf.getvalue()


def _rows(session_id: str) -> list[dict[str, Any]]:
    path = get_whisper_log_path(session_id)
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").split("\n") if x.strip()]


def _start(root: Path, sid: str) -> dict[str, str]:
    return {"hook_event_name": "SessionStart", "cwd": str(root), "session_id": sid}


def _prompt(root: Path, sid: str) -> dict[str, str]:
    return {"hook_event_name": "UserPromptSubmit", "cwd": str(root), "session_id": sid}


def test_off_own_line_only_and_telemetry_own_ids(
    monkeypatch: pytest.MonkeyPatch, hook_env: Path
) -> None:
    monkeypatch.delenv("KB_ROSTER_OTHER_DOMAINS", raising=False)
    _stub(
        monkeypatch,
        {
            "proj-a": [("personal", _entry("kb-1", "a-map"))],
            "proj-b": [("personal", _entry("kb-2", "b-map"))],
        },
    )
    out = _run(monkeypatch, _start(hook_env, "s1"))
    assert "Maps for proj-a" in out
    assert "kb-1" in out
    assert OTHER not in out
    assert "kb-2" not in out
    assert {r["map_id"] for r in _rows("s1")} == {"kb-1"}


def test_off_no_own_maps_emits_nothing(monkeypatch: pytest.MonkeyPatch, hook_env: Path) -> None:
    monkeypatch.delenv("KB_ROSTER_OTHER_DOMAINS", raising=False)
    _stub(monkeypatch, {"proj-b": [("personal", _entry("kb-2", "b-map"))]})
    assert _run(monkeypatch, _start(hook_env, "s2")) == ""
    assert _rows("s2") == []


@pytest.mark.parametrize("enabled", [False, True])
def test_delta_for_other_project_only_when_on(
    monkeypatch: pytest.MonkeyPatch, hook_env: Path, enabled: bool
) -> None:
    if enabled:
        monkeypatch.setenv("KB_ROSTER_OTHER_DOMAINS", "1")
    else:
        monkeypatch.delenv("KB_ROSTER_OTHER_DOMAINS", raising=False)
    base = {
        "proj-a": [("personal", _entry("kb-1", "a-map"))],
        "proj-b": [("personal", _entry("kb-2", "b-map"))],
    }
    _stub(monkeypatch, base)
    _run(monkeypatch, _start(hook_env, "s3"))
    grown = {**base, "proj-b": [*base["proj-b"], ("personal", _entry("kb-3", "b-new"))]}
    _stub(monkeypatch, grown)
    out = _run(monkeypatch, _prompt(hook_env, "s3"))
    if enabled:
        assert "kb-3" in out
    else:
        assert out == ""


def test_on_output_includes_other_domains(monkeypatch: pytest.MonkeyPatch, hook_env: Path) -> None:
    monkeypatch.setenv("KB_ROSTER_OTHER_DOMAINS", "1")
    _stub(
        monkeypatch,
        {
            "proj-a": [("personal", _entry("kb-1", "a-map"))],
            "proj-b": [("personal", _entry("kb-2", "b-map"))],
        },
    )
    out = _run(monkeypatch, _start(hook_env, "s4"))
    assert f"{OTHER} — proj-b: [kb-2] b-map" in out
