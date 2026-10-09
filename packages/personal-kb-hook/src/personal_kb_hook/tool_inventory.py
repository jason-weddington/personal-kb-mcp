"""SessionStart inventory of personal script directories (stdlib-only).

Lists the executables found in the directories named by ``KB_TOOL_DIRS``
(opt-in: unset or blank means no inventory and no scan) as
``<name> — <description>`` lines so the agent
knows which personal tools exist before it improvises one. Purely a local
filesystem scan: no network, no model call, no KB access. Fails open —
:func:`build_inventory` never raises.

Every call also appends one decision row to
``~/.cache/personal_kb/tool-inventory.jsonl`` (local observability only).
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from personal_kb_hook import telemetry
from personal_kb_hook.paths import get_tool_inventory_log_path
from personal_kb_hook.render import _EM_DASH

if TYPE_CHECKING:
    from collections.abc import Callable

MAX_TOOLS = 40
MAX_CHARS = 2000
RESERVE = 24
MAX_DESC_CHARS = 120
MAX_FILE_BYTES = 65536
HEAD_BYTES = 4096
MAX_HEADER_LINES = 30
TIME_BUDGET_S = 0.050
HEADER_PREFIX = "Personal tools in "
OVERFLOW_FMT = "(+{n} more not shown)"
# "budget" would contain the banned token "get", hence "limit".
TRUNCATED_LINE = "(list truncated: scan time limit reached)"
LOG_MAX_BYTES = 1_048_576

_OFF_VALUES = frozenset({"0", "false", "no", "off"})
_SKIPPED_NAMES_LOGGED = 20

# The ONLY way this module reads time (test seam).
_clock: Callable[[], float] = time.monotonic


def describe(path: Path) -> str | None:
    """Return a one-line description from the script's header, or None."""
    try:
        if os.stat(path).st_size > MAX_FILE_BYTES:
            return None
        with open(path, "rb") as fh:
            head = fh.read(HEAD_BYTES)
    except OSError:
        return None
    if b"\x00" in head:
        return None
    lines = head.decode("utf-8", errors="replace").splitlines()[:MAX_HEADER_LINES]
    candidate: str | None = None
    want_next = False
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if want_next:
            if not stripped:
                continue
            candidate = stripped
            break
        if idx == 0 and stripped.startswith("#!"):
            continue
        if not stripped or stripped.startswith("set "):
            continue
        if stripped.startswith("#"):
            text = stripped.lstrip("#").strip()
            if not text or text.startswith("shellcheck") or text.startswith("-*-"):
                continue
            candidate = text
            break
        if stripped.startswith(('"""', "'''")):
            text = stripped[3:]
            for q in ('"""', "'''"):
                if text.endswith(q):
                    text = text[: -len(q)]
                    break
            text = text.strip()
            if text:
                candidate = text
                break
            want_next = True
            continue
        return None
    if candidate is None:
        return None
    name = path.name
    if candidate.startswith(name):
        rest = candidate[len(name) :]
        for sep in (f" {_EM_DASH} ", " - ", ": "):
            if rest.startswith(sep):
                candidate = rest[len(sep) :]
                break
    candidate = " ".join(candidate.split())
    if not candidate:
        return None
    if len(candidate) > MAX_DESC_CHARS:
        candidate = candidate[: MAX_DESC_CHARS - 1] + "…"
    return candidate


def _is_off() -> bool:
    return os.environ.get("KB_TOOL_INVENTORY", "").strip().lower() in _OFF_VALUES


def _requested_dirs() -> list[str]:
    raw = os.environ.get("KB_TOOL_DIRS")
    if raw is None or not raw.strip():
        return []
    dirs: list[str] = []
    for part in raw.split(":"):
        part = part.strip()
        if not part:
            continue
        expanded = os.path.expanduser(part)
        if not os.path.isabs(expanded):
            continue
        dirs.append(os.path.abspath(expanded))
    return dirs


def _display_dir(p: str) -> str:
    home = os.path.expanduser("~")
    if p == home or p.startswith(home + os.sep):
        return "~" + p[len(home) :]
    return p


def _eligible(entry: os.DirEntry[str]) -> bool:
    name = entry.name
    if name.startswith(".") or name.lower().endswith(".md") or name.endswith("~"):
        return False
    return entry.is_file() and os.access(entry.path, os.X_OK)


def _new_stats() -> dict[str, Any]:
    return {
        "outcome": "disabled",
        "dirs_requested": [],
        "dirs_scanned": [],
        "eligible_count": 0,
        "rendered_count": 0,
        "overflow_count": 0,
        "name_only_count": 0,
        "budget_exhausted": False,
        "elapsed_ms": 0.0,
        "chars": 0,
        "error_type": None,
        "tools": [],
        "skipped_names": [],
    }


def _build(stats: dict[str, Any]) -> str | None:
    if _is_off():
        stats["outcome"] = "disabled"
        return None
    start = _clock()
    last = start
    dirs = _requested_dirs()
    stats["dirs_requested"] = list(dirs)

    seen: set[str] = set()
    groups: list[list[str]] = []  # each: [header, tool line, ...]
    cur_len = 0
    capped = False
    exhausted = False
    eligible = 0
    rendered = 0
    name_only = 0
    skipped: list[str] = []

    for d in dirs:
        if exhausted:
            break
        if not os.path.isdir(d):
            continue
        stats["dirs_scanned"].append(d)
        try:
            with os.scandir(d) as it:
                entries = sorted(it, key=lambda e: e.name.casefold())
        except OSError:
            continue
        header = HEADER_PREFIX + _display_dir(d) + ":"
        group: list[str] | None = None
        for entry in entries:
            last = _clock()
            if last - start > TIME_BUDGET_S:
                exhausted = True
                break
            try:
                if not _eligible(entry):
                    continue
            except OSError:
                continue
            name = entry.name
            if name in seen:
                continue
            seen.add(name)
            eligible += 1
            if not capped:
                if rendered >= MAX_TOOLS:
                    capped = True
                else:
                    desc = describe(Path(entry.path))
                    line = name if desc is None else f"{name} {_EM_DASH} {desc}"
                    extra = (len(header) + 1) if group is None else 0
                    total = cur_len + (1 if cur_len else 0) + extra + len(line)
                    if total > MAX_CHARS - RESERVE:
                        capped = True
                    else:
                        if group is None:
                            group = [header]
                            groups.append(group)
                        group.append(line)
                        cur_len = total
                        rendered += 1
                        if desc is None:
                            name_only += 1
                        stats["tools"].append(
                            {"name": name, "dir": d, "described": desc is not None}
                        )
                        continue
            if len(skipped) < _SKIPPED_NAMES_LOGGED:
                skipped.append(name)

    overflow = eligible - rendered
    stats.update(
        eligible_count=eligible,
        rendered_count=rendered,
        overflow_count=overflow,
        name_only_count=name_only,
        budget_exhausted=exhausted,
        skipped_names=skipped,
        elapsed_ms=round((last - start) * 1000.0, 3),
    )

    def _join() -> str:
        return "\n".join("\n".join(g) for g in groups)

    if rendered == 0:
        if not stats["dirs_scanned"]:
            stats["outcome"] = "no_dirs"
        elif exhausted:
            stats["outcome"] = "budget_empty"
        else:
            stats["outcome"] = "no_tools"
        return None

    tail: list[str] = []
    if overflow > 0:
        tail.append(OVERFLOW_FMT.format(n=overflow))
    if exhausted:
        tail.append(TRUNCATED_LINE)
    text = "\n".join([_join(), *tail])
    # Safety net: the trailer lines are meant to fit inside RESERVE.
    while len(text) > MAX_CHARS and groups:
        g = groups[-1]
        g.pop()
        if len(g) == 1:
            groups.pop()
        stats["tools"].pop()
        rendered -= 1
        overflow += 1
        tail = [OVERFLOW_FMT.format(n=overflow)] + ([TRUNCATED_LINE] if exhausted else [])
        text = "\n".join([_join(), *tail]) if groups else ""
    if not text:
        stats["outcome"] = "no_tools"
        return None
    stats.update(rendered_count=rendered, overflow_count=overflow, outcome="emitted")
    return text


def _append_log(row: dict[str, Any]) -> None:
    path = get_tool_inventory_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and os.stat(path).st_size > LOG_MAX_BYTES:
            os.replace(path, path.with_name("tool-inventory.jsonl.1"))
    except (OSError, TypeError, ValueError):
        pass
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except (OSError, TypeError, ValueError):
        pass


def build_inventory(*, session_id: str | None = None, source: str | None = None) -> str | None:
    """Return the tool-inventory text, or None. Never raises."""
    stats = _new_stats()
    text: str | None = None
    try:
        text = _build(stats)
    except Exception as exc:
        text = None
        stats["outcome"] = "error"
        stats["error_type"] = type(exc).__name__
    try:
        stats["chars"] = len(text) if text else 0
        row: dict[str, Any] = {
            "ts": telemetry.now_ts(),
            "session_id": session_id,
            "source": source,
            "build_engine": telemetry.build_engine(),
            "host": socket.gethostname(),
            **stats,
        }
        _append_log(row)
    except Exception:  # noqa: S110 - logging must never affect the result
        pass
    return text
