"""``personal-kb-hook`` console script entry point.

Reads a JSON payload from stdin, branches on ``hook_event_name``:

* **SessionStart** / **UserPromptSubmit** — resolves the project via the
  committed ``.kb_project`` walk-up, looks the resolved project up in the
  on-disk JSONL / HTTP maps index, applies per-session suppression, and
  prints a factual directory string. On ``UserPromptSubmit``, also checks
  the listener cache for a pending whisper and appends it (whisper-last).
* **Stop** — when the listener gate is enabled, extracts the assistant
  manifest from the transcript and spawns a detached listener-worker
  subprocess. Never produces stdout; never calls :func:`http_index.load_index`.

Tolerant by design: any error path — empty stdin, malformed JSON, an
unsupported event, no ``.kb_project`` anywhere on the walk, no maps for the
resolved project, the same maps already surfaced this session — results in
``exit 0`` with NO stdout. The hook must never raise into the harness.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from personal_kb_hook import http_index, listener
from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import compose_directory, render_claude_json, render_whisper
from personal_kb_hook.resolver import resolve_project
from personal_kb_hook.roster import load_roster
from personal_kb_hook.suppression import mark_emitted, should_emit

_SUPPORTED_EVENTS = frozenset({"SessionStart", "UserPromptSubmit", "Stop"})


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI flags. The only supported flag is ``--format``."""
    parser = argparse.ArgumentParser(
        prog="personal-kb-hook",
        description="Surface a project's mental_map directory into a Claude Code session.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "claude-json"),
        default="text",
        help="Output format. 'text' prints the bare directory string; "
        "'claude-json' wraps it in the hookSpecificOutput envelope.",
    )
    return parser.parse_args(argv)


def _read_payload() -> dict[str, object] | None:
    """Read stdin and parse it as a single JSON object. None on any error."""
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return None
    if not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _emit(args: argparse.Namespace, event_name: str, text: str) -> None:
    """Write ``text`` to stdout in the chosen format."""
    if args.format == "claude-json":
        sys.stdout.write(render_claude_json(event_name, text))
    else:
        sys.stdout.write(text)


def main(argv: list[str] | None = None) -> None:
    """Entrypoint registered as ``personal-kb-hook = ...``.

    Top-level try/except ensures the hook never raises into the harness —
    any unexpected exception becomes a silent ``exit 0``.
    """
    try:
        args = _parse_args(list(sys.argv[1:]) if argv is None else argv)
    except SystemExit:
        # argparse errors print to stderr; the hook must still exit cleanly
        # and produce no stdout.
        return

    try:
        payload = _read_payload()
        if payload is None:
            return

        event_name = payload.get("hook_event_name")
        if not isinstance(event_name, str) or event_name not in _SUPPORTED_EVENTS:
            return

        cwd = payload.get("cwd")
        cwd_str = cwd if isinstance(cwd, str) else None

        # ------------------------------------------------------------------ #
        # Stop event: spawn listener worker; NEVER call http_index.load_index #
        # ------------------------------------------------------------------ #
        if event_name == "Stop":
            if not listener.is_listener_enabled():
                return

            transcript_path = payload.get("transcript_path")
            if not isinstance(transcript_path, str) or not transcript_path:
                return

            session_id_stop = payload.get("session_id")
            if not isinstance(session_id_stop, str) or not session_id_stop:
                return

            project_ref_stop = resolve_project(cwd_str)  # null allowed

            manifest = listener.extract_manifest(transcript_path)
            if manifest is None:
                return

            text_content, operated = manifest
            request_data: dict[str, Any] = {
                "text": text_content,
                "cwd_project": project_ref_stop,
                "operating": operated,
                "source_label": project_ref_stop,
            }

            # Write request to a NamedTemporaryFile (worker will delete it)
            req_tmp_path: str
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".json",
                delete=False,
            ) as fh:
                req_tmp_path = fh.name
                json.dump(request_data, fh)

            cache_path = get_listener_cache_path(session_id_stop)
            listener.spawn_worker(req_tmp_path, str(cache_path))
            return

        # ------------------------------------------------------------------ #
        # SessionStart / UserPromptSubmit: directory pipeline + whisper       #
        # ------------------------------------------------------------------ #

        # Pending whisper check — UserPromptSubmit only, independent of the
        # directory pipeline's early returns (lines below).  Wrapped in its
        # own try/except so that any listener failure leaves directory intact.
        whisper: str | None = None
        whisper_id: str | None = None
        whisper_cache_path: Path | None = None
        whisper_pre_ids: list[str] = []

        if event_name == "UserPromptSubmit":
            try:
                session_id_w = payload.get("session_id")
                if (
                    isinstance(session_id_w, str)
                    and session_id_w
                    and listener.is_listener_enabled()
                ):
                    _wcp = get_listener_cache_path(session_id_w)
                    cache_data = listener.read_listener_cache(_wcp)
                    pending: Any = cache_data.get("pending")
                    raw_whispered: Any = cache_data.get("whispered_map_ids")
                    pre_ids: list[str] = []
                    if isinstance(raw_whispered, list):
                        pre_ids = [i for i in raw_whispered if isinstance(i, str)]
                    if isinstance(pending, dict):
                        p_id: Any = pending.get("id")
                        if isinstance(p_id, str) and p_id and p_id not in pre_ids:
                            whisper = render_whisper(pending)
                            whisper_id = p_id
                            whisper_cache_path = _wcp
                            whisper_pre_ids = pre_ids
            except Exception:
                # Listener failure must never prevent directory emission.
                whisper = None
                whisper_id = None
                whisper_cache_path = None
                whisper_pre_ids = []

        # Helper: emit whisper-only and update cache
        def _flush_whisper() -> None:
            if whisper and whisper_id and whisper_cache_path is not None:
                _emit(args, event_name, whisper)
                listener.write_listener_cache(
                    whisper_cache_path,
                    {
                        "pending": None,
                        "whispered_map_ids": [*whisper_pre_ids, whisper_id],
                    },
                )

        # Early return (a): no project resolved
        project_ref = resolve_project(cwd_str)
        if not project_ref:
            _flush_whisper()
            return

        # Roster fan-out: load_index queries each (label,url,key) entry
        # concurrently and returns the merged shape
        # ``dict[str, list[tuple[label, MapEntry]]]``. Absent roster
        # synthesis (the single legacy 'personal' entry from
        # PERSONAL_KB_URL/KEY) is the responsibility of P0's load_roster.
        roster = load_roster()
        index = http_index.load_index(roster)

        # Maps OWNED by the resolved project — drop the label, since Line 1
        # never carries source attribution (a single project's maps cannot
        # come from more than one KB in v1; if they did, label-merging
        # would happen at the project level, but Line 1's API is still
        # ``render_directory(project_ref, list[MapEntry])``).
        own_pairs = index.get(project_ref) or []
        maps = [entry for (_label, entry) in own_pairs]

        # Collect cross-project map KEYS (all projects except the resolved
        # one), carrying the source label so suppression sets cannot
        # collide across KBs that share an id namespace.
        cross_map_ids: list[MapKey] = [
            MapKey(label=label, id=entry["id"])
            for proj, pairs in index.items()
            if proj != project_ref
            for (label, entry) in pairs
        ]

        # Emit only when at least one line has content (own maps OR cross roster).
        map_ids: list[MapKey] = [
            MapKey(label=label, id=entry["id"]) for (label, entry) in own_pairs
        ]
        all_map_ids = map_ids + cross_map_ids

        # Early return (b): empty index
        if not all_map_ids:
            _flush_whisper()
            return

        session_id = payload.get("session_id")
        session_id_str = session_id if isinstance(session_id, str) else ""

        source = payload.get("source")
        source_str = source if isinstance(source, str) else None

        # Early return (c): suppression says no
        if not should_emit(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=all_map_ids,
            source=source_str,
        ):
            _flush_whisper()
            return

        directory = compose_directory(project_ref, maps, index)
        if directory is None:
            _flush_whisper()
            return

        # Compose final output: directory + optional whisper (whisper last)
        combined = directory + "\n" + whisper if whisper else directory

        _emit(args, event_name, combined)

        mark_emitted(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=all_map_ids,
        )

        # Commit whisper cache update after successful directory+whisper emission
        if whisper and whisper_id and whisper_cache_path is not None:
            listener.write_listener_cache(
                whisper_cache_path,
                {
                    "pending": None,
                    "whispered_map_ids": [*whisper_pre_ids, whisper_id],
                },
            )

    except Exception:
        # Intentional broad catch: the hook must NEVER raise into the harness.
        return


if __name__ == "__main__":
    main()
