# personal-kb-hook

Stdlib-only Claude Code hook that surfaces a project's `mental_map` directory
and listener whispers. See the repo root `README.md` for install and wiring.

## Environment

| Variable | Effect |
|----------|--------|
| `PERSONAL_KB_LISTENER` | `1`/`true` enables the listener (also needs a resolvable URL/key). |
| `KB_LISTENER_HEADLESS` | `TRUE` runs the listener on `Stop` even in headless dispatch runs (`HEADLESS_BUILD_ENGINE` set). Default: skipped, because a headless run never gets another `UserPromptSubmit`, the only whisper delivery path, so the Sonnet votes would be wasted. |

## Failure-cue feed (`PostToolUseFailure`)

The hook can forward every failed tool call to the KB service's failure-cue index (`POST /api/kb/event`). Wire it with no matcher, so every tool is covered, and with `async: true`, so it never blocks the agent:

```json
{
  "hooks": {
    "PostToolUseFailure": [
      {"hooks": [{"type": "command", "command": "personal-kb-hook --format=claude-json", "async": true}]}
    ]
  }
}
```

This is **record-only**: no delivery yet. The hook writes nothing to stdout (no `additionalContext`, no decision); it sends one `post_tool` event per failure (1.5 s timeout, no retry) using the same `PERSONAL_KB_URL` / `PERSONAL_KB_API_KEY` resolution as the other surfaces. Claude Code does not fire `PostToolUseFailure` for validation rejections, permission denials or cancellations.

Any event that could not be delivered (missing fields, no URL/key, timeout, URL error, non-2xx response) is appended as one jsonl line to the drop log `~/.cache/personal_kb/event-drops.jsonl`. The log is reset once it exceeds 256 KB. The service's `GET /api/kb/event/heartbeat` counts recorded failures only, so a zero heartbeat must be checked against this drop log before you conclude that nothing failed.
