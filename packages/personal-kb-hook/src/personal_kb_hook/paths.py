"""Stdlib-only on-disk path helpers used by the standalone hook.

``get_hook_scratch_path`` DUPLICATES the equivalent in
``personal_kb.config.get_hook_scratch_path``. The duplication is deliberate
— the standalone hook package must not import the main ``personal_kb``
distribution (that is the whole point of the split). ``get_listener_cache_path``
is hook-only.
"""

from __future__ import annotations

from pathlib import Path


def get_hook_scratch_path(session_id: str) -> Path:
    """Return the per-session hook scratch file path.

    Used by the hook to suppress re-injection of the same map directory in
    a session. Stored under ``~/.cache/personal_kb/`` so it survives across
    the pair of ``SessionStart`` / ``UserPromptSubmit`` invocations within
    one session, but is naturally torn down with the cache.
    """
    return Path(f"~/.cache/personal_kb/injected-{session_id}.json").expanduser()


def get_listener_cache_path(session_id: str) -> Path:
    """Return the per-session listener cache file path.

    Used by the listener worker to store pending whisper entries and the set
    of already-whispered map ids for the session. Stored under
    ``~/.cache/personal_kb/`` as a sibling of the hook scratch file.
    """
    return Path(f"~/.cache/personal_kb/listener-{session_id}.json").expanduser()


def get_whisper_log_path(session_id: str) -> Path:
    """Return the per-session whisper-telemetry jsonl log path.

    Whisper-efficacy telemetry (GTD ccc05354): the hook appends one jsonl row
    per shown roster map and per listener whisper to this file as it happens,
    then Stop batch-POSTs the whole file to ``/api/kb/telemetry/whispers``
    and SessionStart sweeps orphaned prior-session logs from the same
    directory.

    The file lives FLAT under ``~/.cache/personal_kb/`` — a sibling of the
    listener cache and hook scratch files, NOT in its own subdirectory. The
    flat layout means SessionStart's orphan sweep can ``glob('whisper-log-*.jsonl')``
    in one shot, and ``unlink`` after a successful POST is one ``os.replace``-free
    operation.
    """
    return Path(f"~/.cache/personal_kb/whisper-log-{session_id}.jsonl").expanduser()


def get_whisper_debug_log_path(session_id: str) -> Path:
    """Return the per-session whisper-debug plaintext log path.

    Local real-time debug log (separate from whisper telemetry / the
    Postgres analytics sink). The whisper-decision listener pipeline
    appends one block per listener RUN and one line per PROMPT-path
    inject/suppress as it happens, so the operator can ``tail -f`` this
    file to see IF/WHEN a whisper fires, WHAT was whispered, and WHY.

    Plaintext ``.log`` (NOT ``.jsonl``) — flat under
    ``~/.cache/personal_kb/`` as a sibling of the listener cache, hook
    scratch, and whisper-telemetry log files. Ephemeral; not flushed to
    the server; no orphan-sweep; no rotation.
    """
    return Path(f"~/.cache/personal_kb/whisper-debug-{session_id}.log").expanduser()


def get_event_drop_log_path() -> Path:
    """Return the harness-event drop log path (hook-only, no drift-guard twin).

    ``events.post_failure`` appends one jsonl line here for every
    ``PostToolUseFailure`` event it could not deliver to ``POST /api/kb/event``
    (missing fields, no URL/key, timeout, URL error, non-2xx). The file is
    unlinked before an append once it exceeds 256 KB. A zero server-side
    failure heartbeat should be cross-checked against this log.
    """
    return Path("~/.cache/personal_kb/event-drops.jsonl").expanduser()


def get_prevention_cache_path(session_id: str) -> Path:
    """Return the per-session prevention cache path (hook-only, no drift-guard twin).

    Holds the soft-gate settings and Bash cue index fetched from
    ``GET /api/kb/prevention`` plus the session's deny-once state.
    """
    return Path(f"~/.cache/personal_kb/prevention-{session_id}.json").expanduser()


def get_gate_log_path(session_id: str) -> Path:
    """Return the per-session soft-gate decision log (hook-only).

    PreToolUse appends one jsonl row per gate decision here (it never touches
    the network); Stop batch-POSTs the file to ``/api/kb/prevention/decisions``.
    """
    return Path(f"~/.cache/personal_kb/gate-log-{session_id}.jsonl").expanduser()


def get_tool_inventory_log_path() -> Path:
    """Return the SessionStart tool-inventory decision log path (hook-only).

    One jsonl file shared across sessions; local observability only, never
    sent over the network.
    """
    return Path("~/.cache/personal_kb/tool-inventory.jsonl").expanduser()


def get_turn_state_path(session_id: str) -> Path:
    """Return the per-session turn-state file path (hook-only, no drift-guard twin).

    Holds the per-session Stop counter (``next_turn_index``) and the uuid of
    the newest transcript record the last digest saw (``last_uuid``).
    """
    return Path(f"~/.cache/personal_kb/turn-state-{session_id}.json").expanduser()


def get_turn_digest_log_path(session_id: str) -> Path:
    """Return the per-session local turn-digest decision log path (hook-only).

    Local decision log of Stop digests and sender outcomes; it is never POSTed
    anywhere, is capped at 256 KiB and is removed after 7 days.
    """
    return Path(f"~/.cache/personal_kb/turn-digest-log-{session_id}.jsonl").expanduser()
