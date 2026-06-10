"""``personal-kb-hook`` console script entry point.

Reads a JSON payload from stdin, branches on ``hook_event_name``
(``SessionStart`` and ``UserPromptSubmit`` only), resolves the project via
the committed ``.kb_project`` walk-up, looks the resolved project up in the
on-disk JSONL maps index, applies per-session suppression, and prints a
factual directory string.

Tolerant by design: any error path — empty stdin, malformed JSON, an
unsupported event, no ``.kb_project`` anywhere on the walk, no maps for the
resolved project, the same maps already surfaced this session — results in
``exit 0`` with NO stdout. The hook must never raise into the harness.
"""

from __future__ import annotations

import argparse
import json
import sys

from personal_kb_hook import http_index
from personal_kb_hook.render import compose_directory, render_claude_json
from personal_kb_hook.resolver import resolve_project
from personal_kb_hook.suppression import mark_emitted, should_emit

_SUPPORTED_EVENTS = frozenset({"SessionStart", "UserPromptSubmit"})


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

        project_ref = resolve_project(cwd_str)
        if not project_ref:
            return

        index = http_index.load_index()
        maps = index.get(project_ref) or []

        # Collect cross-project map IDs (all projects except the resolved one).
        cross_map_ids: list[str] = [
            m["id"] for proj, proj_maps in index.items() if proj != project_ref for m in proj_maps
        ]

        # Emit only when at least one line has content (own maps OR cross roster).
        map_ids = [m["id"] for m in maps]
        all_map_ids = map_ids + cross_map_ids
        if not all_map_ids:
            return

        session_id = payload.get("session_id")
        session_id_str = session_id if isinstance(session_id, str) else ""

        source = payload.get("source")
        source_str = source if isinstance(source, str) else None

        if not should_emit(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=all_map_ids,
            source=source_str,
        ):
            return

        directory = compose_directory(project_ref, maps, index)
        if directory is None:
            return

        if args.format == "claude-json":
            output = render_claude_json(event_name, directory)
        else:
            output = directory
        sys.stdout.write(output)

        mark_emitted(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=all_map_ids,
        )
    except Exception:
        # Intentional broad catch: the hook must NEVER raise into the harness.
        return


if __name__ == "__main__":
    main()
