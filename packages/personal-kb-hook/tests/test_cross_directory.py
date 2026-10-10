"""Tests for cross-project directory rendering (Gate 0 + P1 multi-KB).

Covers:
* ``render_cross_directory`` unit tests (format, ordering, banned tokens)
* ``compose_directory`` unit tests (line-1-only, line-2-only, both, neither)
* Multi-KB label-prefixed Line-2 format (P1)
* CLI integration: two-line emission (legacy single-KB fallback via env)
* CLI integration: roster-only when the resolved project has no maps
* CLI integration: suppression union (new cross-project map re-triggers)
* BANNED_TOKENS guard over single- and multi-KB roster output
"""

from __future__ import annotations

import io
import json
import unittest.mock
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import cli
from personal_kb_hook.render import (
    BANNED_TOKENS,
    compose_directory,
    render_cross_directory,
)

if TYPE_CHECKING:
    from pathlib import Path

    from personal_kb_hook.index_reader import MapEntry


# ---------------------------------------------------------------------------
# Helpers for the new (label, MapEntry) tuple shape
# ---------------------------------------------------------------------------


def _wrap(
    label: str, mapping: dict[str, list[dict[str, str]]]
) -> dict[str, list[tuple[str, MapEntry]]]:
    """Wrap ``{proj: [entry,...]}`` as the post-P1 ``{proj: [(label, entry),...]}``.

    A common shorthand for tests that only exercise the single-label case
    (the byte-identical legacy form).
    """
    return {proj: [(label, dict(entry)) for entry in entries] for proj, entries in mapping.items()}


# ---------------------------------------------------------------------------
# render_cross_directory unit tests
# ---------------------------------------------------------------------------


def test_render_cross_directory_basic_format() -> None:
    """Two other projects (single-label) → byte-identical pre-P1 form."""
    index = _wrap(
        "personal",
        {
            "personal-kb": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}],
            "agent-gtd": [
                {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"},
                {"id": "gtd-2", "short_title": "inbox", "long_title": "Inbox flow"},
            ],
            "home-network": [{"id": "hn-1", "short_title": "dns", "long_title": "DNS config"}],
        },
    )
    result = render_cross_directory("personal-kb", index)
    assert result is not None
    assert result.startswith("Maps in other domains — ")
    # personal-kb excluded
    assert "personal-kb" not in result
    # Other projects present
    assert "agent-gtd:" in result
    assert "home-network:" in result
    # Short titles only — long titles must be absent
    assert "tasks" in result
    assert "inbox" in result
    assert "dns" in result
    assert "Task tracker" not in result
    assert "DNS config" not in result
    assert "Auth map" not in result
    # Single-label mode: no label-prefix anywhere.
    assert "personal/" not in result


def test_render_cross_directory_empty_index_returns_none() -> None:
    """Empty index → None."""
    assert render_cross_directory("personal-kb", {}) is None


def test_render_cross_directory_only_current_project_returns_none() -> None:
    """Only the current project present → None."""
    index = _wrap(
        "personal",
        {"personal-kb": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}]},
    )
    assert render_cross_directory("personal-kb", index) is None


def test_render_cross_directory_other_projects_all_empty_returns_none() -> None:
    """Other projects present but all have zero maps → None."""
    index = _wrap(
        "personal",
        {
            "personal-kb": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}],
            "agent-gtd": [],
        },
    )
    assert render_cross_directory("personal-kb", index) is None


def test_render_cross_directory_projects_sorted_alphabetically() -> None:
    """Other projects appear in alphabetical order (single-label)."""
    index = _wrap(
        "personal",
        {
            "zebra-proj": [{"id": "z-1", "short_title": "zeta", "long_title": ""}],
            "alpha-proj": [{"id": "a-1", "short_title": "alpha", "long_title": ""}],
            "middle-proj": [{"id": "m-1", "short_title": "mid", "long_title": ""}],
        },
    )
    result = render_cross_directory("current", index)
    assert result is not None
    idx_alpha = result.index("alpha-proj")
    idx_middle = result.index("middle-proj")
    idx_zebra = result.index("zebra-proj")
    assert idx_alpha < idx_middle < idx_zebra


def test_render_cross_directory_maps_in_index_order() -> None:
    """Maps within a project appear in index order (not sorted)."""
    index = _wrap(
        "personal",
        {
            "other-proj": [
                {"id": "m-1", "short_title": "first", "long_title": ""},
                {"id": "m-2", "short_title": "second", "long_title": ""},
                {"id": "m-3", "short_title": "third", "long_title": ""},
            ],
        },
    )
    result = render_cross_directory("current", index)
    assert result is not None
    assert result.index("first") < result.index("second") < result.index("third")


def test_render_cross_directory_comma_join_within_project() -> None:
    """Maps within a project are joined with ', ' (comma-space)."""
    index = _wrap(
        "personal",
        {
            "other": [
                {"id": "m-1", "short_title": "alpha", "long_title": ""},
                {"id": "m-2", "short_title": "beta", "long_title": ""},
            ],
        },
    )
    result = render_cross_directory("current", index)
    assert result is not None
    assert "[m-1] alpha, [m-2] beta" in result


def test_render_cross_directory_semicolon_join_between_projects() -> None:
    """Projects are joined with '; ' (semicolon-space)."""
    index = _wrap(
        "personal",
        {
            "aaa": [{"id": "a-1", "short_title": "alpha", "long_title": ""}],
            "bbb": [{"id": "b-1", "short_title": "beta", "long_title": ""}],
        },
    )
    result = render_cross_directory("current", index)
    assert result is not None
    # Both projects present, aaa before bbb, separated by '; '
    assert "aaa: [a-1] alpha; bbb: [b-1] beta" in result


def test_render_cross_directory_uses_em_dash() -> None:
    """The heading uses U+2014 EM DASH, not a hyphen."""
    index = _wrap("personal", {"other": [{"id": "m-1", "short_title": "foo", "long_title": ""}]})
    result = render_cross_directory("current", index)
    assert result is not None
    assert "—" in result  # U+2014 EM DASH
    assert " — " in result


def test_render_cross_directory_no_banned_tokens() -> None:
    """The roster output contains no banned imperative tokens (single-label)."""
    index = _wrap(
        "personal",
        {
            "agent-gtd": [
                {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"},
            ],
            "home-network": [
                {"id": "hn-1", "short_title": "topology", "long_title": "Network layout"},
            ],
        },
    )
    result = render_cross_directory("personal-kb", index)
    assert result is not None
    lowered = result.lower()
    for tok in BANNED_TOKENS:
        assert tok not in lowered, f"banned token {tok!r} found in: {result}"


# ---------------------------------------------------------------------------
# Multi-KB Line-2 format (P1)
# ---------------------------------------------------------------------------


def test_render_cross_directory_multi_label_prefixed_format() -> None:
    """More than one distinct label → groups prefixed with ``{label}/{proj}``."""
    index: dict[str, list[tuple[str, MapEntry]]] = {
        "alpha-proj": [
            ("personal", {"id": "kb-a-1", "short_title": "alpha", "long_title": ""}),
        ],
        "beta-proj": [
            ("team", {"id": "kb-b-1", "short_title": "beta", "long_title": ""}),
        ],
    }
    result = render_cross_directory("personal-kb", index)
    assert result is not None
    assert result.startswith("Maps in other domains — ")
    # Both groups prefixed with their label.
    assert "personal/alpha-proj:" in result
    assert "team/beta-proj:" in result
    # Sort by (label, project_ref): personal/alpha-proj < team/beta-proj
    assert result.index("personal/alpha-proj:") < result.index("team/beta-proj:")
    # Short titles preserved.
    assert "[kb-a-1] alpha" in result
    assert "[kb-b-1] beta" in result


def test_render_cross_directory_multi_label_shared_project_split_by_label() -> None:
    """A project_ref present under two labels yields two label-prefixed groups."""
    index: dict[str, list[tuple[str, MapEntry]]] = {
        "shared": [
            ("personal", {"id": "kb-p-1", "short_title": "p-share", "long_title": ""}),
            ("team", {"id": "kb-t-1", "short_title": "t-share", "long_title": ""}),
        ],
    }
    result = render_cross_directory("not-shared", index)
    assert result is not None
    assert "personal/shared:" in result
    assert "team/shared:" in result
    # Sort by (label, project_ref): personal/shared before team/shared.
    assert result.index("personal/shared:") < result.index("team/shared:")


def test_render_cross_directory_single_label_byte_identical_no_prefix() -> None:
    """Exactly one distinct label → byte-identical to pre-P1 form (no prefix)."""
    index = _wrap(
        "personal",
        {
            "agent-gtd": [{"id": "gtd-1", "short_title": "tasks", "long_title": ""}],
            "home-network": [{"id": "hn-1", "short_title": "dns", "long_title": ""}],
        },
    )
    result = render_cross_directory("personal-kb", index)
    assert result == "Maps in other domains — agent-gtd: [gtd-1] tasks; home-network: [hn-1] dns"
    # Make sure label prefix is absent for the single-KB case (byte-identical).
    assert "personal/" not in result


def test_render_cross_directory_multi_label_no_banned_tokens() -> None:
    """The label-prefixed roster output also passes BANNED_TOKENS."""
    index: dict[str, list[tuple[str, MapEntry]]] = {
        "agent-gtd": [
            ("personal", {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"}),
        ],
        "home-network": [
            ("team", {"id": "hn-1", "short_title": "topology", "long_title": "Network layout"}),
        ],
    }
    result = render_cross_directory("personal-kb", index)
    assert result is not None
    # Sanity: prefix IS present.
    assert "personal/agent-gtd:" in result
    assert "team/home-network:" in result
    lowered = result.lower()
    for tok in BANNED_TOKENS:
        assert tok not in lowered, f"banned token {tok!r} found in: {result}"


def test_render_cross_directory_separator_is_slash_not_banned() -> None:
    """The chosen '/' separator and label literals introduce no banned tokens."""
    # Sanity check at the unit level: '/' is not in BANNED_TOKENS, and
    # neither 'personal' nor 'team' (the canonical labels) is banned.
    assert "/" not in BANNED_TOKENS
    for label in ("personal", "team"):
        assert label.lower() not in BANNED_TOKENS


# ---------------------------------------------------------------------------
# compose_directory unit tests
# ---------------------------------------------------------------------------


def test_compose_directory_both_lines() -> None:
    """Own maps + cross-project maps → two lines joined by a single newline."""
    index = _wrap(
        "personal",
        {
            "personal-kb": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}],
            "agent-gtd": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"}],
        },
    )
    own_maps = [entry for (_label, entry) in index["personal-kb"]]
    result = compose_directory("personal-kb", own_maps, index)
    assert result is not None
    lines = result.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Maps for personal-kb — ")
    assert lines[1].startswith("Maps in other domains — ")


def test_compose_directory_line1_only_when_no_other_projects() -> None:
    """Own maps, no other projects → line 1 only (no newline)."""
    index = _wrap(
        "personal",
        {"personal-kb": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}]},
    )
    own_maps = [entry for (_label, entry) in index["personal-kb"]]
    result = compose_directory("personal-kb", own_maps, index)
    assert result is not None
    assert "\n" not in result
    assert result.startswith("Maps for personal-kb — ")


def test_compose_directory_line2_only_when_no_own_maps() -> None:
    """No own maps, other project has maps → line 2 only (no leading newline)."""
    index = _wrap(
        "personal",
        {"agent-gtd": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"}]},
    )
    result = compose_directory("personal-kb", [], index)
    assert result is not None
    assert "\n" not in result
    assert result.startswith("Maps in other domains — ")
    assert not result.startswith("\n")


def test_compose_directory_both_empty_returns_none() -> None:
    """No own maps and no other projects → None."""
    assert compose_directory("personal-kb", [], {}) is None


def test_compose_directory_line1_unchanged_format() -> None:
    """Line 1 produced by compose_directory is byte-identical to render_directory."""
    from personal_kb_hook.render import render_directory

    raw_maps: list[MapEntry] = [
        {"id": "kb-1", "short_title": "auth", "long_title": "Authentication map"},
        {"id": "kb-2", "short_title": "ingest", "long_title": "Ingestion flow"},
    ]
    index = _wrap("personal", {"personal-kb": [dict(m) for m in raw_maps]})
    composed = compose_directory("personal-kb", raw_maps, index)
    assert composed is not None
    # Only one line (no other projects)
    assert "\n" not in composed
    assert composed == render_directory("personal-kb", raw_maps)


# ---------------------------------------------------------------------------
# Shared helpers for CLI integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def hook_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Isolate the hook's filesystem touchpoints to ``tmp_path``."""
    db_path = tmp_path / "kb" / "knowledge.db"
    db_path.parent.mkdir(parents=True)
    monkeypatch.setenv("KB_DB_PATH", str(db_path))
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    # Point HOME / XDG_CONFIG_HOME at tmp_path so load_roster finds no
    # kbs.json → falls back to PERSONAL_KB_URL/KEY (set per test).
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / ".config"))
    (tmp_path / ".cache" / "personal_kb").mkdir(parents=True, exist_ok=True)
    return {
        "root": tmp_path,
        "db_path": db_path,
        # role unset → "default" → CLI globs maps_index.*.jsonl
        "maps_index": db_path.parent / "maps_index.default.jsonl",
    }


def _run(
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
    args: list[str] | None = None,
) -> tuple[int, str]:
    """Run cli.main() with ``payload`` on stdin and capture stdout."""
    stdin_text = json.dumps(payload) if not isinstance(payload, str) else payload
    monkeypatch.setattr("sys.stdin", io.StringIO(stdin_text))
    stdout_buf = io.StringIO()
    monkeypatch.setattr("sys.stdout", stdout_buf)
    rc = 0
    try:
        cli.main(args or [])
    except SystemExit as exc:
        rc = int(exc.code or 0)
    return rc, stdout_buf.getvalue()


# ---------------------------------------------------------------------------
# CLI: two-line emission (single-KB legacy fallback)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("other_domains_on")
def test_two_line_emission_local_jsonl(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """personal-kb + agent-gtd + home-network in index → two-line output."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "personal-kb",
                "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}],
            },
            {
                "project_ref": "agent-gtd",
                "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"}],
            },
            {
                "project_ref": "home-network",
                "maps": [{"id": "hn-1", "short_title": "topology", "long_title": "Network layout"}],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-two-line-local",
        },
        ["--format=text"],
    )
    assert rc == 0
    lines = out.split("\n")
    assert len(lines) == 2
    # Line 1: existing format unchanged
    assert lines[0].startswith("Maps for personal-kb — ")
    assert "[kb-1] auth: Auth map" in lines[0]
    # Line 2: cross-project roster — single-KB byte-identical form (no label prefix)
    assert lines[1].startswith("Maps in other domains — ")
    assert "agent-gtd:" in lines[1]
    assert "home-network:" in lines[1]
    # personal-kb excluded from line 2
    assert "personal-kb" not in lines[1]
    # Short titles only in line 2
    assert "tasks" in lines[1]
    assert "topology" in lines[1]
    assert "Task tracker" not in lines[1]
    assert "Network layout" not in lines[1]
    # Alphabetical: agent-gtd before home-network
    assert lines[1].index("agent-gtd") < lines[1].index("home-network")
    # No label prefix (single-KB legacy form)
    assert "personal/" not in lines[1]


@pytest.mark.usefixtures("other_domains_on")
def test_two_line_claude_json_envelope(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """--format=claude-json: additionalContext contains the two-line string."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "personal-kb",
                "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth"}],
            },
            {
                "project_ref": "agent-gtd",
                "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"}],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-json-two-line",
        },
        ["--format=claude-json"],
    )
    assert rc == 0
    obj = json.loads(out)
    context = obj["hookSpecificOutput"]["additionalContext"]
    lines = context.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Maps for personal-kb — ")
    assert lines[1].startswith("Maps in other domains — ")


# ---------------------------------------------------------------------------
# CLI: roster-only (resolved project has zero own maps)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("other_domains_on")
def test_roster_only_local_jsonl(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """personal-kb resolved but absent from index; agent-gtd has maps → line 2 only."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "agent-gtd",
                "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"}],
            },
            # personal-kb intentionally absent
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-roster-only-local",
        },
        ["--format=text"],
    )
    assert rc == 0
    assert out != ""
    # Only one line — no leading or trailing newline
    assert "\n" not in out
    assert out.startswith("Maps in other domains — ")
    assert "agent-gtd:" in out
    assert "personal-kb" not in out


# ---------------------------------------------------------------------------
# CLI: suppression union — new cross-project map re-triggers emission
# ---------------------------------------------------------------------------


def test_suppression_second_prompt_silent_with_cross_project(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Same session / same union of keys → second prompt is suppressed."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "personal-kb",
                "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth"}],
            },
            {
                "project_ref": "agent-gtd",
                "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"}],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(hook_env["root"]),
        "session_id": "s-suppress-cross-union",
    }
    rc1, out1 = _run(monkeypatch, payload)
    assert rc1 == 0
    assert out1 != ""

    rc2, out2 = _run(monkeypatch, payload)
    assert rc2 == 0
    assert out2 == ""


@pytest.mark.usefixtures("other_domains_on")
def test_suppression_new_cross_project_map_retriggers(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """New cross-project map id → re-triggers even though own maps unchanged."""
    import urllib.request

    # Use a mutable store so we can update the response between calls.
    projects_v1 = [
        {
            "project_ref": "personal-kb",
            "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth"}],
        },
        {
            "project_ref": "agent-gtd",
            "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"}],
        },
    ]
    response_store: list[list[dict[str, Any]]] = [projects_v1]

    def mutable_urlopen(req: object, timeout: float = 3.0) -> object:
        body = json.dumps({"projects": response_store[0]}).encode("utf-8")
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return mock_resp

    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", mutable_urlopen)
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "cwd": str(hook_env["root"]),
        "session_id": "s-suppress-new-cross",
    }
    # First emission
    rc1, out1 = _run(monkeypatch, payload)
    assert rc1 == 0
    assert out1 != ""

    # Second call → suppressed
    rc2, out2 = _run(monkeypatch, payload)
    assert rc2 == 0
    assert out2 == ""

    # Add a new cross-project map id by updating the response store.
    response_store[0] = [
        {
            "project_ref": "personal-kb",
            "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth"}],
        },
        {
            "project_ref": "agent-gtd",
            "maps": [
                {"id": "gtd-1", "short_title": "tasks", "long_title": "Tasks"},
                {"id": "gtd-2", "short_title": "inbox", "long_title": "Inbox"},  # NEW
            ],
        },
    ]
    # Third call → re-triggered because gtd-2 is a new id
    rc3, out3 = _run(monkeypatch, payload)
    assert rc3 == 0
    assert out3 != ""
    assert "gtd-2" in out3


# ---------------------------------------------------------------------------
# CLI: BANNED_TOKENS over cross-project roster output (single-KB)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("other_domains_on")
def test_banned_tokens_not_in_cross_directory_output(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """Cross-project roster in CLI output contains no banned imperative tokens."""
    import urllib.request

    # personal-kb has no maps → roster-only output
    service_response = {
        "projects": [
            {
                "project_ref": "agent-gtd",
                "maps": [{"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"}],
            },
            {
                "project_ref": "home-network",
                "maps": [{"id": "hn-1", "short_title": "topology", "long_title": "Network layout"}],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")
    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-banned-cross",
        },
        ["--format=text"],
    )
    assert rc == 0
    assert out != ""
    lowered = out.lower()
    for tok in BANNED_TOKENS:
        assert tok not in lowered, f"banned token {tok!r} found in output: {out}"


# ---------------------------------------------------------------------------
# CLI: two-line emission via HTTP mock
# ---------------------------------------------------------------------------


def _make_fake_urlopen(service_response: dict[str, Any]) -> Any:
    """Return a fake urlopen function that yields ``service_response`` as JSON."""

    def fake_urlopen(req: object, timeout: float = 3.0) -> Any:
        body = json.dumps(service_response).encode("utf-8")
        mock_resp = unittest.mock.MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = unittest.mock.MagicMock(return_value=False)
        return mock_resp

    return fake_urlopen


@pytest.mark.usefixtures("other_domains_on")
def test_two_line_emission_http_mock(
    monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]
) -> None:
    """HTTP index with personal-kb + agent-gtd → two lines emitted."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "personal-kb",
                "maps": [{"id": "kb-1", "short_title": "auth", "long_title": "Auth map"}],
            },
            {
                "project_ref": "agent-gtd",
                "maps": [
                    {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"},
                ],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-http-two-line",
        },
        ["--format=text"],
    )
    assert rc == 0
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Maps for personal-kb — ")
    assert "[kb-1] auth: Auth map" in lines[0]
    assert lines[1].startswith("Maps in other domains — ")
    assert "agent-gtd:" in lines[1]
    assert "tasks" in lines[1]
    assert "Task tracker" not in lines[1]
    assert "personal-kb" not in lines[1]


@pytest.mark.usefixtures("other_domains_on")
def test_roster_only_http_mock(monkeypatch: pytest.MonkeyPatch, hook_env: dict[str, Path]) -> None:
    """HTTP index: personal-kb absent; agent-gtd has maps → roster line only."""
    import urllib.request

    service_response = {
        "projects": [
            {
                "project_ref": "agent-gtd",
                "maps": [
                    {"id": "gtd-1", "short_title": "tasks", "long_title": "Task tracker"},
                ],
            },
        ]
    }
    monkeypatch.setenv("PERSONAL_KB_URL", "https://kb.example.com")
    monkeypatch.setenv("PERSONAL_KB_API_KEY", "test-api-key")
    monkeypatch.setattr(urllib.request, "urlopen", _make_fake_urlopen(service_response))
    (hook_env["root"] / ".kb_project").write_text("personal-kb\n", encoding="utf-8")

    rc, out = _run(
        monkeypatch,
        {
            "hook_event_name": "SessionStart",
            "cwd": str(hook_env["root"]),
            "session_id": "s-http-roster-only",
        },
        ["--format=text"],
    )
    assert rc == 0
    assert out != ""
    assert "\n" not in out
    assert out.startswith("Maps in other domains — ")
    assert "agent-gtd:" in out
    assert "personal-kb" not in out
