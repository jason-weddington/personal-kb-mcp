"""Tests for personal_kb_hook.listener and related helpers.

Covers:
- extract_manifest: tail bound, truncated-first-line discard, record-schema
  tolerance, last-assistant-text selection, head cap, operated regex,
  <200-char skip, missing/corrupt transcript.
- read_listener_cache / write_listener_cache: tolerance + dedup.
- render_whisper: format and BANNED_TOKENS scaffold disjointness.
- get_listener_cache_path: correct path formula.
- is_listener_enabled: three-var gate.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from personal_kb_hook import listener
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import BANNED_TOKENS, render_whisper

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_assistant_record(
    text_blocks: list[str] | None = None,
    tool_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build a minimal assistant record in the Claude Code JSONL schema."""
    content: list[dict[str, Any]] = []
    for text in text_blocks or []:
        content.append({"type": "text", "text": text})
    for name in tool_names or []:
        content.append({"type": "tool_use", "name": name})
    return {
        "type": "assistant",
        "message": {"content": content},
    }


def _write_jsonl(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# is_listener_enabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url,key,flag,expected",
    [
        ("https://kb.example.com", "mykey", "true", True),
        ("https://kb.example.com", "mykey", "1", True),
        ("https://kb.example.com", "mykey", "TRUE", True),  # case-insensitive
        ("https://kb.example.com", "mykey", "false", False),
        ("https://kb.example.com", "mykey", "", False),
        ("https://kb.example.com", "", "true", False),
        ("", "mykey", "true", False),
        ("", "", "", False),
    ],
    ids=[
        "all-set-true",
        "all-set-1",
        "flag-uppercase",
        "flag-false",
        "flag-empty",
        "key-missing",
        "url-missing",
        "all-missing",
    ],
)
def test_is_listener_enabled(
    url: str,
    key: str,
    flag: str,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each of the three env vars independently gates the listener."""
    if url:
        monkeypatch.setenv("PERSONAL_KB_URL", url)
    else:
        monkeypatch.delenv("PERSONAL_KB_URL", raising=False)
    if key:
        monkeypatch.setenv("PERSONAL_KB_API_KEY", key)
    else:
        monkeypatch.delenv("PERSONAL_KB_API_KEY", raising=False)
    if flag:
        monkeypatch.setenv("PERSONAL_KB_LISTENER", flag)
    else:
        monkeypatch.delenv("PERSONAL_KB_LISTENER", raising=False)
    assert listener.is_listener_enabled() == expected


# ---------------------------------------------------------------------------
# extract_manifest — tail bound and first-line discard
# ---------------------------------------------------------------------------


def test_extract_manifest_reads_tail_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the file is larger than 262144 bytes, only the tail is read."""
    transcript = tmp_path / "transcript.jsonl"
    # Prepend garbage that should NOT be parsed (it will be in the discarded
    # first line or outside the tail window).
    text_payload = "A" * 300  # long enough to pass the 200-char minimum
    record = _make_assistant_record(text_blocks=[text_payload])

    # Create padding that exceeds 262144 bytes so we seek into the middle.
    # The valid assistant record is at the very end.
    padding_line = "x" * 500
    # Build file: lots of padding lines, then the assistant record
    num_padding = (262144 // (len(padding_line) + 1)) + 10
    lines = [padding_line] * num_padding + [json.dumps(record)]
    transcript.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))

    result = listener.extract_manifest(str(transcript))
    assert result is not None
    text, _operated = result
    assert text.startswith("A" * 300)


def test_extract_manifest_truncated_first_line_discarded(tmp_path: Path) -> None:
    """When offset > 0, the first (possibly truncated) line is discarded."""
    transcript = tmp_path / "transcript.jsonl"
    # Write a file where the first bytes form a partial record (bad JSON),
    # followed by a valid assistant record.
    text_payload = "B" * 300
    record = _make_assistant_record(text_blocks=[text_payload])
    record_line = json.dumps(record)

    # Put enough padding to push offset > 0 (> 262144 bytes total),
    # making the first line after seeking garbage.
    padding = "P" * 262200
    # file = <padding>\n<record_line>\n
    content = padding + "\n" + record_line + "\n"
    transcript.write_bytes(content.encode("utf-8"))

    result = listener.extract_manifest(str(transcript))
    # Should succeed even though the first partial line is bad JSON
    assert result is not None
    assert result[0].startswith("B" * 300)


# ---------------------------------------------------------------------------
# extract_manifest — record-schema tolerance
# ---------------------------------------------------------------------------


def test_extract_manifest_skips_non_assistant_records(tmp_path: Path) -> None:
    """Records with type != 'assistant' are skipped."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "C" * 300
    _write_jsonl(
        transcript,
        [
            {"type": "user", "message": {"content": [{"type": "text", "text": text_payload}]}},
            _make_assistant_record(text_blocks=[text_payload]),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert result[0].startswith("C" * 300)


def test_extract_manifest_skips_non_dict_record(tmp_path: Path) -> None:
    """Non-dict top-level records are skipped."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "D" * 300
    _write_jsonl(
        transcript,
        [
            "not-a-dict",
            42,
            _make_assistant_record(text_blocks=[text_payload]),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None


def test_extract_manifest_skips_non_dict_message(tmp_path: Path) -> None:
    """Assistant records with non-dict message are skipped."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "E" * 300
    _write_jsonl(
        transcript,
        [
            {"type": "assistant", "message": "not-a-dict"},
            _make_assistant_record(text_blocks=[text_payload]),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None


def test_extract_manifest_skips_non_list_content(tmp_path: Path) -> None:
    """Assistant records with non-list content are skipped."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "F" * 300
    _write_jsonl(
        transcript,
        [
            {"type": "assistant", "message": {"content": "not-a-list"}},
            _make_assistant_record(text_blocks=[text_payload]),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None


def test_extract_manifest_skips_bad_json_lines(tmp_path: Path) -> None:
    """Unparseable lines are skipped; valid records still processed."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "G" * 300
    transcript.write_bytes(
        (
            "this is bad json {{{\n"
            + json.dumps(_make_assistant_record(text_blocks=[text_payload]))
            + "\n"
        ).encode("utf-8")
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None


def test_extract_manifest_skips_non_dict_content_blocks(tmp_path: Path) -> None:
    """Non-dict content blocks are skipped; surrounding text still collected."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "H" * 300
    record: dict[str, Any] = {
        "type": "assistant",
        "message": {
            "content": [
                "not-a-dict-block",
                42,
                {"type": "text", "text": text_payload},
            ]
        },
    }
    _write_jsonl(transcript, [record])
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert text_payload in result[0]


# ---------------------------------------------------------------------------
# extract_manifest — last-assistant-text selection
# ---------------------------------------------------------------------------


def test_extract_manifest_returns_last_assistant_text(tmp_path: Path) -> None:
    """text comes from the LAST assistant record that has text blocks."""
    transcript = tmp_path / "t.jsonl"
    first_text = "I" * 300
    last_text = "J" * 300
    _write_jsonl(
        transcript,
        [
            _make_assistant_record(text_blocks=[first_text]),
            _make_assistant_record(text_blocks=[last_text]),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert result[0].startswith("J" * 300)
    assert "I" * 300 not in result[0]


def test_extract_manifest_skips_assistant_with_empty_text(tmp_path: Path) -> None:
    """Assistant records with only empty text blocks don't count as having text."""
    transcript = tmp_path / "t.jsonl"
    real_text = "K" * 300
    empty_record: dict[str, Any] = {
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": ""}]},
    }
    _write_jsonl(
        transcript,
        [
            _make_assistant_record(text_blocks=[real_text]),
            empty_record,
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert result[0].startswith("K" * 300)


# ---------------------------------------------------------------------------
# extract_manifest — text[:4000] head cap
# ---------------------------------------------------------------------------


def test_extract_manifest_head_cap_4000(tmp_path: Path) -> None:
    """text is truncated to at most 4000 chars (head, not tail)."""
    transcript = tmp_path / "t.jsonl"
    long_text = "L" * 5000
    _write_jsonl(transcript, [_make_assistant_record(text_blocks=[long_text])])
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert len(result[0]) == 4000
    assert result[0] == "L" * 4000


# ---------------------------------------------------------------------------
# extract_manifest — operated regex across all assistant records
# ---------------------------------------------------------------------------


def test_extract_manifest_operated_across_all_records(tmp_path: Path) -> None:
    """operated is collected from ALL assistant records, not just the last."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "M" * 300
    _write_jsonl(
        transcript,
        [
            _make_assistant_record(
                text_blocks=["ignored text"],
                tool_names=["mcp__agent-gtd__add_item", "mcp__personal-kb__kb_search"],
            ),
            _make_assistant_record(
                text_blocks=[text_payload],
                tool_names=["mcp__agent-gtd__list_items"],
            ),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    _, operated = result
    assert "mcp:agent-gtd" in operated
    assert "mcp:personal-kb" in operated
    assert operated == sorted(set(operated))  # sorted + deduped


def test_extract_manifest_operated_deduped(tmp_path: Path) -> None:
    """Duplicate mcp server names appear only once in operated."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "N" * 300
    _write_jsonl(
        transcript,
        [
            _make_assistant_record(
                text_blocks=[text_payload],
                tool_names=[
                    "mcp__agent-gtd__add_item",
                    "mcp__agent-gtd__list_items",  # same server
                ],
            ),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    _, operated = result
    assert operated.count("mcp:agent-gtd") == 1


def test_extract_manifest_non_mcp_tool_names_ignored(tmp_path: Path) -> None:
    """Tool names not matching ^mcp__(.+?)__ are ignored."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "O" * 300
    _write_jsonl(
        transcript,
        [
            _make_assistant_record(
                text_blocks=[text_payload],
                tool_names=["Bash", "Read", "Write", "mcp__kb__search"],
            ),
        ],
    )
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    _, operated = result
    assert "mcp:kb" in operated
    assert "Bash" not in operated
    assert "Read" not in operated


def test_extract_manifest_non_str_tool_name_ignored(tmp_path: Path) -> None:
    """Non-str tool_use names are ignored (tolerant)."""
    transcript = tmp_path / "t.jsonl"
    text_payload = "P" * 300
    record: dict[str, Any] = {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": 42},
                {"type": "text", "text": text_payload},
            ]
        },
    }
    _write_jsonl(transcript, [record])
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert result[1] == []  # no operated


# ---------------------------------------------------------------------------
# extract_manifest — <200-char skip
# ---------------------------------------------------------------------------


def test_extract_manifest_short_text_returns_none(tmp_path: Path) -> None:
    """Text shorter than 200 chars causes extract_manifest to return None."""
    transcript = tmp_path / "t.jsonl"
    short_text = "Q" * 100
    _write_jsonl(transcript, [_make_assistant_record(text_blocks=[short_text])])
    assert listener.extract_manifest(str(transcript)) is None


def test_extract_manifest_exactly_200_chars_allowed(tmp_path: Path) -> None:
    """Text of exactly 200 chars (after head cap) is allowed (>= 200)."""
    transcript = tmp_path / "t.jsonl"
    text = "R" * 200
    _write_jsonl(transcript, [_make_assistant_record(text_blocks=[text])])
    result = listener.extract_manifest(str(transcript))
    assert result is not None
    assert len(result[0]) == 200


def test_extract_manifest_199_chars_returns_none(tmp_path: Path) -> None:
    """Text of 199 chars (< 200) returns None."""
    transcript = tmp_path / "t.jsonl"
    text = "S" * 199
    _write_jsonl(transcript, [_make_assistant_record(text_blocks=[text])])
    assert listener.extract_manifest(str(transcript)) is None


# ---------------------------------------------------------------------------
# extract_manifest — missing/corrupt transcript
# ---------------------------------------------------------------------------


def test_extract_manifest_missing_file_returns_none(tmp_path: Path) -> None:
    """Non-existent transcript file returns None."""
    assert listener.extract_manifest(str(tmp_path / "no_such_file.jsonl")) is None


def test_extract_manifest_empty_file_returns_none(tmp_path: Path) -> None:
    """Empty transcript returns None (no assistant records)."""
    transcript = tmp_path / "empty.jsonl"
    transcript.write_text("", encoding="utf-8")
    assert listener.extract_manifest(str(transcript)) is None


def test_extract_manifest_no_assistant_records_returns_none(tmp_path: Path) -> None:
    """File with only non-assistant records returns None."""
    transcript = tmp_path / "t.jsonl"
    _write_jsonl(
        transcript,
        [
            {"type": "user", "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"type": "system", "content": "system message"},
        ],
    )
    assert listener.extract_manifest(str(transcript)) is None


# ---------------------------------------------------------------------------
# Cache read / write tolerance + dedup
# ---------------------------------------------------------------------------


def test_read_listener_cache_missing_returns_fresh(tmp_path: Path) -> None:
    """Missing cache file returns fresh state (P2 schema: pending=[])."""
    path = tmp_path / "listener-xyz.json"
    result = listener.read_listener_cache(path)
    assert result["pending"] == []
    assert result["whispered_map_ids"] == []


def test_read_listener_cache_corrupt_returns_fresh(tmp_path: Path) -> None:
    """Corrupt (non-JSON) cache file returns fresh state (P2 schema: pending=[])."""
    path = tmp_path / "listener-xyz.json"
    path.write_text("{{bad json", encoding="utf-8")
    result = listener.read_listener_cache(path)
    assert result["pending"] == []
    assert result["whispered_map_ids"] == []


def test_read_listener_cache_non_dict_returns_fresh(tmp_path: Path) -> None:
    """Non-dict JSON in cache file returns fresh state (P2 schema: pending=[])."""
    path = tmp_path / "listener-xyz.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    result = listener.read_listener_cache(path)
    assert result["pending"] == []
    assert result["whispered_map_ids"] == []


def test_write_read_listener_cache_roundtrip(tmp_path: Path) -> None:
    """Write then read returns the same data."""
    path = tmp_path / "personal_kb" / "listener-abc.json"
    data: dict[str, Any] = {
        "pending": {"id": "kb-00001", "short_title": "Auth", "long_title": "Auth map"},
        "whispered_map_ids": ["kb-00002"],
    }
    listener.write_listener_cache(path, data)
    result = listener.read_listener_cache(path)
    assert result["pending"] == data["pending"]
    assert result["whispered_map_ids"] == ["kb-00002"]


def test_write_listener_cache_atomic(tmp_path: Path) -> None:
    """write_listener_cache does not leave a partial file on success."""
    path = tmp_path / "personal_kb" / "listener-abc.json"
    listener.write_listener_cache(path, {"pending": None, "whispered_map_ids": []})
    # No .tmp file should remain
    tmp_files = list(path.parent.glob("*.tmp"))
    assert tmp_files == []


def test_write_listener_cache_preserves_dir(tmp_path: Path) -> None:
    """write_listener_cache creates parent dirs if needed."""
    path = tmp_path / "deep" / "nested" / "listener-s.json"
    listener.write_listener_cache(path, {"pending": None, "whispered_map_ids": []})
    assert path.exists()


# ---------------------------------------------------------------------------
# render_whisper — format and BANNED_TOKENS disjointness
# ---------------------------------------------------------------------------


def test_render_whisper_with_long_title() -> None:
    """Full whisper line with long_title."""
    entry: dict[str, object] = {
        "id": "kb-00001",
        "short_title": "Authentication",
        "long_title": "OAuth2 flow diagram",
    }
    result = render_whisper(entry)
    assert result == "Possibly relevant map — [kb-00001] Authentication: OAuth2 flow diagram"


def test_render_whisper_without_long_title() -> None:
    """Whisper line omits ': ...' suffix when long_title is empty."""
    entry: dict[str, object] = {
        "id": "kb-00002",
        "short_title": "Ingestion",
        "long_title": "",
    }
    result = render_whisper(entry)
    assert result == "Possibly relevant map — [kb-00002] Ingestion"
    assert ": " not in result


def test_render_whisper_none_long_title() -> None:
    """long_title=None treated as empty (no suffix)."""
    entry: dict[str, object] = {
        "id": "kb-00003",
        "short_title": "Graph",
        "long_title": None,  # type: ignore[typeddict-item]
    }
    result = render_whisper(entry)
    assert result == "Possibly relevant map — [kb-00003] Graph"


def test_render_whisper_em_dash_codepoint() -> None:
    """Whisper uses U+2014 EM DASH, not a hyphen."""
    entry: dict[str, object] = {"id": "kb-1", "short_title": "X", "long_title": ""}
    result = render_whisper(entry)
    assert "—" in result


def test_render_whisper_scaffold_disjoint_from_banned_tokens() -> None:
    """The fixed scaffold of the whisper contains no banned imperative tokens.

    Uses banned-token-free fixture titles (mirroring test_hook_cli.py:252).
    """
    entry: dict[str, object] = {
        "id": "kb-00001",
        "short_title": "Authentication",
        "long_title": "OAuth2 flow",
    }
    result = render_whisper(entry)
    lowered_words = set(result.lower().split())
    assert lowered_words.isdisjoint(BANNED_TOKENS), (
        f"Whisper contains banned token(s): {lowered_words & BANNED_TOKENS}"
    )


# ---------------------------------------------------------------------------
# get_listener_cache_path — path formula
# ---------------------------------------------------------------------------


def test_get_listener_cache_path_formula(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns ~/.cache/personal_kb/listener-{session_id}.json."""
    monkeypatch.setenv("HOME", str(tmp_path))
    path = get_listener_cache_path("my-session-123")
    assert path.name == "listener-my-session-123.json"
    assert path.parent.name == "personal_kb"
    assert path.parent.parent.name == ".cache"


def test_get_listener_cache_path_different_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different session ids produce different paths."""
    monkeypatch.setenv("HOME", str(tmp_path))
    p1 = get_listener_cache_path("session-A")
    p2 = get_listener_cache_path("session-B")
    assert p1 != p2
