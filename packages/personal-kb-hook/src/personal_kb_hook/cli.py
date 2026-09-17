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
import socket
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from personal_kb_hook import http_index, listener, telemetry, whisper_debug
from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import compose_directory, render_claude_json, render_whisper
from personal_kb_hook.resolver import resolve_project
from personal_kb_hook.roster import load_roster
from personal_kb_hook.suppression import mark_emitted, should_emit

_SUPPORTED_EVENTS = frozenset({"SessionStart", "UserPromptSubmit", "Stop", "PostToolUse"})
_KB_GET_TOOL_NAMES = frozenset({"mcp__personal-kb__kb_get", "mcp__team-kb__team_kb_get"})


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
        # PostToolUse event: whisper-telemetry consume; NEVER touches index   #
        # nor falls through to the directory pipeline. Silent-on-failure     #
        # (the outer try/except in main() and telemetry.mark_consumed's own  #
        # internal guard cover any error), always produces zero stdout,      #
        # fires on EVERY kb_get and team_kb_get — must be ultra-cheap (no   #
        # http_index.load_index, no resolve_project).                        #
        # ------------------------------------------------------------------ #
        if event_name == "PostToolUse":
            tool_name = payload.get("tool_name")
            if not isinstance(tool_name, str) or tool_name not in _KB_GET_TOOL_NAMES:
                return
            session_id_pt = payload.get("session_id")
            if not isinstance(session_id_pt, str) or not session_id_pt:
                return
            tool_input = payload.get("tool_input")
            if not isinstance(tool_input, dict):
                return
            entry_id_raw: Any = tool_input.get("entry_id")
            # entry_id is str | list[str] per kb_get.py:49-50 — normalize.
            consume_ids: set[str]
            if isinstance(entry_id_raw, str):
                if not entry_id_raw:
                    return
                consume_ids = {entry_id_raw}
            elif isinstance(entry_id_raw, list):
                consume_ids = {item for item in entry_id_raw if isinstance(item, str) and item}
                if not consume_ids:
                    return
            else:
                return
            telemetry.mark_consumed(session_id_pt, consume_ids, telemetry.now_ts())
            return

        # ------------------------------------------------------------------ #
        # Stop event: spawn listener worker; NEVER call http_index.load_index #
        # ------------------------------------------------------------------ #
        if event_name == "Stop":
            # UNCONDITIONAL whisper-telemetry flush runs OUTSIDE / BEFORE the
            # listener-enabled guard — roster rows accrue regardless of the
            # gate, which is OFF in most sessions today. flush_session is
            # internally silent-on-failure; the outer try/except in main()
            # backstops anything that slips through.
            session_id_stop_raw = payload.get("session_id")
            if isinstance(session_id_stop_raw, str) and session_id_stop_raw:
                telemetry.flush_session(session_id_stop_raw)

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
                "session_id": session_id_stop,
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

        # SessionStart whisper-telemetry orphan sweep (safety net for sessions
        # where Stop did not fire). orphan_sweep is internally silent-on-failure
        # and the outer try/except in main() backstops anything left.
        if event_name == "SessionStart":
            session_id_ss = payload.get("session_id")
            if isinstance(session_id_ss, str) and session_id_ss:
                telemetry.orphan_sweep(session_id_ss)

        # Pending whisper check — UserPromptSubmit only, independent of the
        # directory pipeline's early returns (lines below).  Wrapped in its
        # own try/except so that any listener failure leaves directory intact.
        #
        # P2 schema: ``pending`` is a LIST of per-KB pointer objects
        # ``{label, id, short_title, long_title}`` (≤ 1 per label after the
        # worker's arbitration); ``whispered_map_ids`` is a list of
        # two-element ``[label, id]`` lists. Pre-P2 cache files used
        # ``pending`` as a single dict and ``whispered_map_ids`` as bare-id
        # strings — both are back-parsed tolerantly to label ``'personal'``.
        whisper: str | None = None
        whisper_cache_path: Path | None = None
        whisper_pre_pairs: list[list[str]] = []
        whisper_emitted_pairs: list[list[str]] = []

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
                    pending_raw: Any = cache_data.get("pending")
                    raw_whispered: Any = cache_data.get("whispered_map_ids")

                    # Back-parse whispered_map_ids: bare str → ['personal', s];
                    # [label, id] kept verbatim; anything else dropped.
                    pre_pairs: list[list[str]] = []
                    if isinstance(raw_whispered, list):
                        for item in raw_whispered:
                            if isinstance(item, str) and item:
                                pre_pairs.append(["personal", item])
                            elif isinstance(item, list) and len(item) == 2:
                                pl, pi = item[0], item[1]
                                if isinstance(pl, str) and pl and isinstance(pi, str) and pi:
                                    pre_pairs.append([pl, pi])

                    # Back-parse pending: single dict → wrap with label='personal';
                    # list-of-dicts kept (label defaulted to 'personal' if absent).
                    pending_items: list[dict[str, str]] = []
                    if isinstance(pending_raw, dict):
                        d_id = pending_raw.get("id")
                        if isinstance(d_id, str) and d_id:
                            d_st = pending_raw.get("short_title")
                            d_lt = pending_raw.get("long_title")
                            pending_items.append(
                                {
                                    "label": "personal",
                                    "id": d_id,
                                    "short_title": d_st if isinstance(d_st, str) else "",
                                    "long_title": d_lt if isinstance(d_lt, str) else "",
                                }
                            )
                    elif isinstance(pending_raw, list):
                        for raw_item in pending_raw:
                            if not isinstance(raw_item, dict):
                                continue
                            d_id = raw_item.get("id")
                            if not isinstance(d_id, str) or not d_id:
                                continue
                            d_label = raw_item.get("label")
                            if not isinstance(d_label, str) or not d_label:
                                d_label = "personal"
                            d_st = raw_item.get("short_title")
                            d_lt = raw_item.get("long_title")
                            pending_items.append(
                                {
                                    "label": d_label,
                                    "id": d_id,
                                    "short_title": d_st if isinstance(d_st, str) else "",
                                    "long_title": d_lt if isinstance(d_lt, str) else "",
                                }
                            )

                    # Filter already-whispered (label, id) pairs.
                    pre_set = {(p[0], p[1]) for p in pre_pairs}
                    filtered = [p for p in pending_items if (p["label"], p["id"]) not in pre_set]

                    # Whisper-debug PROMPT-path lines: one `suppress` per entry
                    # the pre_set filter dropped (NOT the cap step — out of scope
                    # per AC). Silent-on-failure inside whisper_debug; the outer
                    # whisper try/except already backstops anything left.
                    for _p in pending_items:
                        if (_p["label"], _p["id"]) in pre_set:
                            whisper_debug.append_prompt_suppress(session_id_w, _p["id"])

                    # Defensive one-per-KB cap (preserve first per label).
                    seen_labels: set[str] = set()
                    capped: list[dict[str, str]] = []
                    for p in filtered:
                        if p["label"] in seen_labels:
                            continue
                        seen_labels.add(p["label"])
                        capped.append(p)

                    if capped:
                        roster_for_whisper = load_roster()
                        roster_labels_w = [e.label for e in roster_for_whisper]
                        multi_kb_w = len(roster_for_whisper) > 1

                        # Tie-break winner label (UserPromptSubmit-time mirror
                        # of the worker's AC-6 rule). source_label here is the
                        # currently-resolved project_ref, NOT a KB label, so
                        # step (1) typically does not match in practice.
                        src_label = resolve_project(cwd_str)
                        if src_label and src_label in roster_labels_w:
                            winner_label: str | None = src_label
                        elif "personal" in roster_labels_w:
                            winner_label = "personal"
                        elif roster_labels_w:
                            winner_label = roster_labels_w[0]
                        else:
                            winner_label = None

                        def _order_key(p: dict[str, str]) -> tuple[int, str]:
                            if winner_label is not None and p["label"] == winner_label:
                                return (0, p["label"])
                            return (1, p["label"])

                        ordered = sorted(capped, key=_order_key)

                        lines = [
                            render_whisper(
                                p,
                                label=p["label"],
                                multi_kb=multi_kb_w,
                            )
                            for p in ordered
                        ]
                        whisper = "\n".join(lines)
                        whisper_cache_path = _wcp
                        whisper_pre_pairs = pre_pairs
                        whisper_emitted_pairs = [[p["label"], p["id"]] for p in ordered]

                        # Whisper-debug PROMPT-path inject lines: one per
                        # entry in `ordered` (matches the emit set 1:1 — at
                        # this point whisper_emitted_pairs is non-empty iff
                        # `ordered` is non-empty, so a logged inject always
                        # corresponds to a real emission). Silent-on-failure.
                        for _p in ordered:
                            whisper_debug.append_prompt_inject(
                                session_id_w, _p["id"], _p["short_title"]
                            )
            except Exception:
                # Listener failure must never prevent directory emission.
                whisper = None
                whisper_cache_path = None
                whisper_pre_pairs = []
                whisper_emitted_pairs = []

        # Helper: emit whisper-only and update cache
        def _flush_whisper() -> None:
            if whisper and whisper_emitted_pairs and whisper_cache_path is not None:
                _emit(args, event_name, whisper)
                listener.write_listener_cache(
                    whisper_cache_path,
                    {
                        "pending": [],
                        "whispered_map_ids": [*whisper_pre_pairs, *whisper_emitted_pairs],
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
        # Preserve (label, MapEntry) tuples so the roster telemetry emit
        # can carry each map's ``pointers`` list into the appended jsonl
        # row (used by mark_consumed to chain-credit map → detail fetches;
        # GTD 88441f9c).
        cross_pairs: list[tuple[str, Any]] = [
            (label, entry)
            for proj, pairs in index.items()
            if proj != project_ref
            for (label, entry) in pairs
        ]
        cross_map_ids: list[MapKey] = [
            MapKey(label=label, id=entry["id"]) for (label, entry) in cross_pairs
        ]

        # Emit only when at least one line has content (own maps OR cross roster).
        map_ids: list[MapKey] = [
            MapKey(label=label, id=entry["id"]) for (label, entry) in own_pairs
        ]
        all_map_ids = map_ids + cross_map_ids
        all_pairs: list[tuple[str, Any]] = list(own_pairs) + cross_pairs

        # Early return (b): empty index
        if not all_map_ids:
            _flush_whisper()
            return

        session_id = payload.get("session_id")
        session_id_str = session_id if isinstance(session_id, str) else ""

        source = payload.get("source")
        source_str = source if isinstance(source, str) else None

        # Early return (c): suppression says no
        emit_reason = should_emit(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=all_map_ids,
            source=source_str,
        )
        if emit_reason is None:
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

        # Whisper-telemetry roster emit: one jsonl row per shown MapKey
        # (own + cross-project). append_row is internally silent-on-failure;
        # the outer try/except in main() backstops anything else.
        # build_engine reads HEADLESS_BUILD_ENGINE defensively; unset => null
        # (interactive/control-plane).
        #
        # Each row carries the map's ``pointers`` list — the kb-ids the map's
        # body mentions — so telemetry.mark_consumed can chain-credit this
        # map row when a kb_get fetches one of its detail entries rather
        # than the map itself (GTD 88441f9c). Pointers are stripped by the
        # server's pydantic model on flush (extra="ignore" by default), so
        # the wire format stays backward compatible with the pre-pointers
        # WhisperTelemetryRow — the direct-vs-chain signal reaches the
        # server inside ``trigger_context.consumed_via`` instead.
        if session_id_str:
            _host = socket.gethostname()
            _engine = telemetry.build_engine()
            _ts = telemetry.now_ts()
            for _label, _entry in all_pairs:
                _raw_ptrs = _entry.get("pointers", [])
                _row_pointers: list[str] = (
                    [p for p in _raw_ptrs if isinstance(p, str) and p]
                    if isinstance(_raw_ptrs, list)
                    else []
                )
                telemetry.append_row(
                    session_id_str,
                    {
                        "session_id": session_id_str,
                        "host": _host,
                        "surface": "roster",
                        "map_id": _entry["id"],
                        "source_kb": _label,
                        "cwd_project": project_ref,
                        "trigger_context": {
                            "cwd_project": project_ref,
                            "emit_reason": emit_reason.value,
                        },
                        "emitted_ts": _ts,
                        "consumed": False,
                        "consumed_ts": None,
                        "build_engine": _engine,
                        "pointers": _row_pointers,
                    },
                )

        # Commit whisper cache update after successful directory+whisper emission
        if whisper and whisper_emitted_pairs and whisper_cache_path is not None:
            listener.write_listener_cache(
                whisper_cache_path,
                {
                    "pending": [],
                    "whispered_map_ids": [*whisper_pre_pairs, *whisper_emitted_pairs],
                },
            )

    except Exception:
        # Intentional broad catch: the hook must NEVER raise into the harness.
        return


if __name__ == "__main__":
    main()
