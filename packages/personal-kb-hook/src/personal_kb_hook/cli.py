"""``personal-kb-hook`` console script entry point.

Reads a JSON payload from stdin, branches on ``hook_event_name``:

* **PreToolUse** — the prevention soft gate
  (:func:`personal_kb_hook.prevention.pre_tool`): reads the session's cached
  Bash cue index, NO network, and on a match prints a one-time ``deny``
  envelope verbatim (regardless of ``--format``); otherwise no stdout.
* **SessionStart** / **UserPromptSubmit** — resolves the project via the
  committed ``.kb_project`` walk-up, looks the resolved project up in the
  on-disk JSONL / HTTP maps index, applies per-session suppression, and
  prints a factual directory string. On ``UserPromptSubmit``, also checks
  the listener cache for a pending whisper and appends it (whisper-last).
  On ``SessionStart``, also fetches the prevention payload
  (:func:`personal_kb_hook.prevention.session_start`); a non-empty gotcha
  slice is emitted in the same single output, before the directory.
* **PostToolUseFailure** — forwards the failed tool call as a record-only
  ``post_tool`` event to ``POST /api/kb/event`` (the failure-cue index) via
  :func:`personal_kb_hook.events.post_failure` (unchanged, record-only).
  Before that POST, the opt-in (``KB_FAILURE_CONTEXT``), cache-only failure
  context (:func:`personal_kb_hook.prevention.failure_context`) checks a
  failed Bash call against the session's cached gate index; on a match the
  ``additionalContext`` envelope is printed verbatim (regardless of
  ``--format``), at most once per resolution per session.
* **PostToolUse** — whisper-telemetry consume on ``kb_get``. Never produces
  stdout.
* **Stop** — records/flushes the soft-gate decision log and refreshes the
  prevention cache (unconditionally); ships a turn digest when the cached
  ``surprise_capture`` is shadow or on (independent of the listener gate); then,
  when the listener gate is enabled, extracts the assistant manifest from the
  transcript and spawns a detached listener-worker subprocess. Never produces
  stdout; never calls :func:`http_index.load_index`.

SessionStart additionally appends a tool inventory of the personal script
directories (``KB_TOOL_DIRS``, opt-in with no default; see
:mod:`personal_kb_hook.tool_inventory`) after the directory text.

Tolerant by design: any error path — empty stdin, malformed JSON, an
unsupported event, no ``.kb_project`` anywhere on the walk, no maps for the
resolved project, the same maps already surfaced this session — results in
``exit 0`` with NO stdout, except that on SessionStart those quiet paths
still emit the tool inventory alone when it is non-empty. The hook must
never raise into the harness.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import socket
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

from personal_kb_hook import (
    events,
    http_index,
    listener,
    prevention,
    telemetry,
    tool_inventory,
    turn_digest,
    whisper_debug,
)
from personal_kb_hook.index_reader import MapKey
from personal_kb_hook.listener_worker import _MAX_POINTERS_PER_KB
from personal_kb_hook.paths import get_listener_cache_path
from personal_kb_hook.render import (
    compose_directory,
    render_claude_json,
    render_new_maps,
    render_whisper,
)
from personal_kb_hook.resolver import resolve_project
from personal_kb_hook.roster import load_roster
from personal_kb_hook.suppression import EmitReason, get_surfaced_map_ids, mark_emitted, should_emit

_SUPPORTED_EVENTS = frozenset(
    {
        "PreToolUse",
        "SessionStart",
        "UserPromptSubmit",
        "Stop",
        "PostToolUse",
        "PostToolUseFailure",
    }
)
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


def read_prompt_text(payload: dict[str, object]) -> str | None:
    """Read the user prompt text from a ``UserPromptSubmit`` payload.

    Checks ``prompt`` FIRST — the key empirically observed on a real
    headless run on 2026-09-19 (payload keys were exactly ``cwd``,
    ``hook_event_name``, ``permission_mode``, ``prompt``, ``prompt_id``,
    ``session_id``, ``transcript_path``) — then falls back to
    ``user_input``, the spelling the official docs claim. When BOTH are
    present, ``prompt`` wins. Returns ``None`` when neither key is
    present, or when the only present value(s) are not a non-empty str.

    Not wired into any behavior today — it exists so the upcoming
    inline-listener work cannot get the key wrong.
    """
    for key in ("prompt", "user_input"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


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

    # SessionStart gotcha slice: once fetched it must reach stdout exactly once
    # on EVERY exit path of the directory pipeline (incl. an unexpected error).
    slice_state: dict[str, Any] = {"text": None, "emitted": False}
    # SessionStart tool inventory: same single-write discipline. It rides in
    # the same output as the slice/directory (appended last), or alone on the
    # quiet paths. ``slice_state["emitted"]`` doubles as the "stdout written"
    # flag for every main-path emit.
    inv_state: dict[str, Any] = {"text": None}

    try:
        payload = _read_payload()
        if payload is None:
            return

        event_name = payload.get("hook_event_name")
        if not isinstance(event_name, str) or event_name not in _SUPPORTED_EVENTS:
            return

        # PreToolUse: prevention soft gate. Cache-only — NO network, no index,
        # no roster, no resolver. The deny envelope is written verbatim.
        if event_name == "PreToolUse":
            decision = prevention.pre_tool(payload)
            if decision is not None:
                sys.stdout.write(decision)
            return

        cwd = payload.get("cwd")
        cwd_str = cwd if isinstance(cwd, str) else None

        # PostToolUseFailure: cache-only failure context (stdout envelope on a match) + record-only POST.  # noqa: E501
        if event_name == "PostToolUseFailure":
            context = prevention.failure_context(payload)
            events.post_failure(payload)
            if context is not None:
                sys.stdout.write(context)
            return

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
                # Prevention: abandoned-retry + summary rows, flush the gate
                # log, then refresh the cache (delivers server switch flips).
                prevention.stop(payload)
                prevention.flush_gate_log(session_id_stop_raw)
                prevention.refresh(payload)
                # Turn digest: gated only by the cached surprise_capture mode,
                # independent of the listener gates below (decision R10).
                turn_digest.stop(payload)

            if not listener.is_listener_enabled():
                return

            # Headless dispatch runs never get another UserPromptSubmit (the
            # only whisper delivery path), so judging would burn Sonnet votes
            # on an undeliverable whisper. KB_LISTENER_HEADLESS=TRUE opts back in.
            if (
                telemetry.build_engine() is not None
                and os.environ.get("KB_LISTENER_HEADLESS", "").upper() != "TRUE"
            ):
                print("personal-kb-hook: headless run, skipping listener", file=sys.stderr)
                return

            transcript_path = payload.get("transcript_path")
            if not isinstance(transcript_path, str) or not transcript_path:
                return

            session_id_stop = payload.get("session_id")
            if not isinstance(session_id_stop, str) or not session_id_stop:
                return

            project_ref_stop = resolve_project(cwd_str)  # null allowed

            scanned = listener.scan_transcript(transcript_path)
            if scanned is None:
                return
            transcript_text, operated = scanned

            # Prefer the harness-provided final assistant text: the transcript
            # is written asynchronously and may lag the current turn. The
            # transcript is still the source of the ``operating`` list.
            text_source = "transcript"
            lam = payload.get("last_assistant_message")
            if isinstance(lam, str) and lam:
                text_content = listener.normalize_text(lam)
                text_source = "last_assistant_message"
            else:
                text_content = listener.normalize_text(transcript_text) if transcript_text else None
            if text_content is None:
                return
            request_data: dict[str, Any] = {
                "text": text_content,
                "cwd_project": project_ref_stop,
                "operating": operated,
                "source_label": project_ref_stop,
                "session_id": session_id_stop,
                "hook_event_name": event_name,
                "text_source": text_source,
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
                prevention.orphan_sweep(session_id_ss)
            slice_state["text"] = prevention.session_start(payload)
            try:
                _sid_inv = payload.get("session_id")
                _src_inv = payload.get("source")
                inv_state["text"] = tool_inventory.build_inventory(
                    session_id=_sid_inv if isinstance(_sid_inv, str) else None,
                    source=_src_inv if isinstance(_src_inv, str) else None,
                )
            except Exception:
                inv_state["text"] = None

        # Pending whisper check — UserPromptSubmit only, independent of the
        # directory pipeline's early returns (lines below).  Wrapped in its
        # own try/except so that any listener failure leaves directory intact.
        #
        # P2 schema: ``pending`` is a LIST of per-KB pointer objects
        # ``{label, id, short_title, long_title}`` (≤ _MAX_POINTERS_PER_KB=2
        # per label after the worker's arbitration, GTD 66ea1fe4);
        # ``whispered_map_ids`` is a list of two-element ``[label, id]``
        # lists. Pre-P2 cache files used ``pending`` as a single dict and
        # ``whispered_map_ids`` as bare-id strings — both are back-parsed
        # tolerantly to label ``'personal'``.
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

                    # Per-KB cap of _MAX_POINTERS_PER_KB (GTD 66ea1fe4: up to
                    # 2 pointers may survive per label — preserve the first
                    # _MAX_POINTERS_PER_KB, in their cache-file (server
                    # evidence) order).
                    label_drain_counts: dict[str, int] = {}
                    capped: list[dict[str, str]] = []
                    for p in filtered:
                        if label_drain_counts.get(p["label"], 0) >= _MAX_POINTERS_PER_KB:
                            continue
                        label_drain_counts[p["label"]] = label_drain_counts.get(p["label"], 0) + 1
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

                        # Group by label (stable — preserves `ordered`'s
                        # winner-label-first, then each label's own
                        # evidence order) so render_whisper renders a KB's
                        # 1..2 pointers on ONE line (GTD 66ea1fe4).
                        grouped_by_label: dict[str, list[dict[str, str]]] = {}
                        label_order: list[str] = []
                        for p in ordered:
                            if p["label"] not in grouped_by_label:
                                grouped_by_label[p["label"]] = []
                                label_order.append(p["label"])
                            grouped_by_label[p["label"]].append(p)

                        lines = [
                            render_whisper(
                                grouped_by_label[lbl],
                                label=lbl,
                                multi_kb=multi_kb_w,
                            )
                            for lbl in label_order
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

        # Helper: emit whisper-only (UserPromptSubmit) or slice-only
        # (SessionStart) on an early return, and update the whisper cache.
        def _flush_whisper() -> None:
            if slice_state["text"] is not None:
                slice_state["emitted"] = True
                slice_text: str = slice_state["text"]
                if inv_state["text"] is not None:
                    slice_text = slice_text + "\n\n" + inv_state["text"]
                _emit(args, event_name, slice_text)
                return
            if inv_state["text"] is not None and not slice_state["emitted"]:
                slice_state["emitted"] = True
                _emit(args, event_name, inv_state["text"])
                return
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

        # Orientation reasons (first-emission, compact, scope-change) get the
        # FULL roster, byte-identical to the pre-delta behavior. new-maps
        # gets ONLY a delta FYI naming the maps this session has not already
        # seen — never the full directory (that re-injection is the defect
        # this switch fixes). telemetry_pairs mirrors exactly what gets
        # announced-as-new in THIS emission, so the telemetry loop below stays
        # branch-agnostic; mark_ids mirrors what mark_emitted() unions in.
        telemetry_pairs: list[tuple[str, Any]]
        mark_ids: list[MapKey]

        if emit_reason == EmitReason.NEW_MAPS:
            surfaced = get_surfaced_map_ids(session_id=session_id_str)
            delta_pairs: list[tuple[MapKey, Any]] = [
                (key, entry)
                for key, (_label, entry) in zip(all_map_ids, all_pairs, strict=True)
                if key not in surfaced
            ]
            # Empty-delta guard: should_emit() only returns NEW_MAPS when the
            # resolved set is NOT a subset of surfaced_map_ids, so this is
            # normally unreachable. If it happens anyway, emit NOTHING and
            # write NO telemetry — never fall back to the full directory.
            if not delta_pairs:
                _flush_whisper()
                return
            directory = render_new_maps(delta_pairs)
            if directory is None:
                _flush_whisper()
                return
            telemetry_pairs = [(key.label, entry) for key, entry in delta_pairs]
            mark_ids = [key for key, _entry in delta_pairs]
        else:
            directory = compose_directory(project_ref, maps, index)
            if directory is None:
                _flush_whisper()
                return
            telemetry_pairs = all_pairs
            mark_ids = all_map_ids

        # Compose final output: directory + optional whisper (whisper last)
        combined = directory + "\n" + whisper if whisper else directory
        if slice_state["text"] is not None:
            combined = slice_state["text"] + "\n" + combined
        if event_name == "SessionStart" and inv_state["text"] is not None:
            combined = combined + "\n\n" + inv_state["text"]

        slice_state["emitted"] = True
        _emit(args, event_name, combined)

        mark_emitted(
            session_id=session_id_str,
            scope=project_ref,
            map_ids=mark_ids,
        )

        # Whisper-telemetry roster emit: one jsonl row per map ANNOUNCED-AS-NEW
        # in this emission — the full set (own + cross-project) for the
        # orientation reasons, or just the delta for new-maps. append_row is
        # internally silent-on-failure; the outer try/except in main()
        # backstops anything else. build_engine reads HEADLESS_BUILD_ENGINE
        # defensively; unset => null (interactive/control-plane).
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
            for _label, _entry in telemetry_pairs:
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
        if not slice_state["emitted"]:
            parts = [t for t in (slice_state["text"], inv_state["text"]) if t is not None]
            if parts:
                with contextlib.suppress(Exception):
                    _emit(args, "SessionStart", "\n\n".join(parts))
        return


if __name__ == "__main__":
    main()
