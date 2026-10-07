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

## Prevention channels (`SessionStart` slice + `PreToolUse` soft gate)

**Gotcha slice.** On `SessionStart` the hook makes one `GET /api/kb/prevention` call (1.5 s timeout, no retry; personal KB only). A non-empty `slice_text`, the "Known gotchas for <project>" list of corrected facts from the KB, is injected in the same single SessionStart output, ahead of the maps directory. It is never written to `CLAUDE.md`, `AGENTS.md`, memory files or any file in the repo.

**Soft gate.** The same response carries the gate settings and a Bash cue index, cached per session. Wire `PreToolUse` with `"matcher": "Bash"` and `"timeout": 5` (see the root README). When a Bash call's two-word class (such as `git push`) exactly matches a cue, the hook denies it once, with the corrected fact as the reason; retrying the identical call is allowed. There are at most 2 denies per session. PreToolUse never makes a network call.

**Server switches** (set on the KB service, read per request):

| Variable | Default | Effect |
|----------|---------|--------|
| `KB_SOFT_GATE_ENABLED` | off | Only the literal `TRUE` (any case) sends a gate index. |
| `KB_SOFT_GATE_SHADOW` | shadow on | Only `FALSE` turns on real denies; in shadow mode the hook records `would_deny` and prints nothing. |
| `KB_SOFT_GATE_DISABLED_PROJECTS` | empty | Comma-separated projects whose gate is off. |
| `KB_DELIVER_OBSERVED_ONCE` | on | `FALSE` withholds autonomous resolutions seen in only one session. |

A switch flip reaches a live session at its next `Stop`, when the hook re-fetches the settings.

**Files** under `~/.cache/personal_kb/`: `prevention-<session>.json` is the cached settings, index and deny-once state; caches older than 7 days are removed. `gate-log-<session>.jsonl` is the local decision log, flushed at `Stop` to `POST /api/kb/prevention/decisions` (stale logs from other sessions are swept at `SessionStart`). `event-drops.jsonl` is the drop log shared with the failure-cue feed, where failed fetches and flushes are recorded (`op` = `prevention_fetch`, `gate_flush` or `gate_log_rotated`).

The gate is inert until resolutions exist in the KB in the `hints.resolution` format documented in `packages/kb-service/src/kb_service/prevention.py`. Until then the slice is fed only by supersedes corrections.
