#!/usr/bin/env python3
"""SessionStart hook for the replay experiment harness (``slice`` arm only).

Invoked by Claude Code as ``python3 session_slice.py <hook-config.json>``.
Prints the configured ``slice_text`` as SessionStart ``additionalContext`` and
appends one ``slice_delivered`` line to the run's hook log.

It fails OPEN: on any exception it exits 0 with no stdout and no
``slice_delivered`` line, which the harness then marks
``slice_not_delivered`` (invalid run).

Stdlib only. Never imports ``replay.py`` or ``kb_core``.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def main(argv: list[str]) -> int:
    """Run the hook.

    Args:
        argv: Process argv; ``argv[1]`` is the hook-config path.

    Returns:
        Always 0.
    """
    try:
        config = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        slice_text = config["slice_text"]
        if not isinstance(slice_text, str):
            raise TypeError("slice_text is not a string")
        envelope = json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": slice_text,
                }
            }
        )
        line = {
            "ts": datetime.now(UTC).isoformat(),
            "task_id": config.get("task_id"),
            "arm": config.get("arm"),
            "rep": config.get("rep"),
            "event": "SessionStart",
            "decision": "slice_delivered",
        }
        with Path(config["hook_log"]).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")
        print(envelope)
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
