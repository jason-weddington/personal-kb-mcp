# personal-kb-hook

Stdlib-only Claude Code hook that surfaces a project's `mental_map` directory
and listener whispers. See the repo root `README.md` for install and wiring.

## Environment

| Variable | Effect |
|----------|--------|
| `PERSONAL_KB_LISTENER` | `1`/`true` enables the listener (also needs a resolvable URL/key). |
| `KB_LISTENER_HEADLESS` | `TRUE` runs the listener on `Stop` even in headless dispatch runs (`HEADLESS_BUILD_ENGINE` set). Default: skipped, because a headless run never gets another `UserPromptSubmit`, the only whisper delivery path, so the Sonnet votes would be wasted. |
| `KB_TOOL_DIRS` | Colon-separated directories scanned for the SessionStart tool inventory. Required to enable the inventory; there is no default, and when unset or blank nothing is scanned. Missing or relative entries are skipped; the first dir wins on duplicate names. |
| `KB_TOOL_INVENTORY` | Default on. `0`/`false`/`no`/`off` disables the SessionStart tool inventory. |
| `KB_GOTCHA_SLICE` | Default on. `0`/`false`/`no`/`off` suppresses the SessionStart gotcha slice text while the soft gate is still armed (gate index fetched and cached). |
| `KB_ROSTER_OTHER_DOMAINS` | Default off. `1`/`true`/`yes`/`on` adds the `Maps in other domains` roster line (other projects' maps) to the SessionStart/UserPromptSubmit directory. Paused by default: no measured consumption, pending a kb-bench arm. Own-project maps are unaffected. |
| `KB_FAILURE_CONTEXT` | Default off. `1`/`true`/`yes`/`on` enables the `PostToolUseFailure` failure context (the matching corrected fact delivered next to a failed Bash call). The record-only failure-cue POST runs either way. |

## SessionStart tool inventory

On every `SessionStart` the hook lists the executables found in `KB_TOOL_DIRS` as `<name> — <description>` lines grouped under `Personal tools in <dir>:`, appended after the maps directory (or alone when there is nothing else to say), so the agent knows which personal scripts exist before it improvises one. The description is the first header comment, docstring or `Usage:` line of the script; scripts with no recognisable header are listed by name only. The list is capped at 40 tools, 2,000 characters and a 50 ms scan budget; `(+N more not shown)` marks overflow and `(list truncated: scan time limit reached)` marks a scan cut short. Each call appends one decision row to the local log `~/.cache/personal_kb/tool-inventory.jsonl` (never sent over the network; rotated at 1 MiB).

## Failure-cue feed (`PostToolUseFailure`)

The hook forwards every failed tool call to the KB service's failure-cue index (`POST /api/kb/event`) and, when opted in, hands the agent the matching correction right next to the failure. Wire it with no matcher, so every tool is covered, as a synchronous entry with the opt-in:

```json
{
  "hooks": {
    "PostToolUseFailure": [
      {"hooks": [{"type": "command", "command": "KB_FAILURE_CONTEXT=1 personal-kb-hook --format=claude-json", "timeout": 5}]}
    ]
  }
}
```

**Failure context.** When a failed Bash call matches the session's cached gate index (the same two-word class and args-prefix matcher the `PreToolUse` soft gate uses), the hook prints the corrected fact as `additionalContext`, so it lands next to the tool error in the same turn. Each resolution is delivered at most once per session; a later failure that matches an already-delivered resolution is recorded as `failure_context_repeat` and prints nothing. Delivery is independent of the deny limits and of shadow mode. It reads only the local cache and makes no network call. The index is empty unless the service sets `KB_SOFT_GATE_ENABLED`, so nothing is ever delivered while the gate is off.

**Put the opt-in only on a synchronous entry.** Under `async: true`, Claude Code delivers a hook's `additionalContext` on the next conversation turn (or at the next user interaction if the session is idle), detached from the failure it explains. Without `KB_FAILURE_CONTEXT`, an `async: true` entry remains the record-only feed.

**Latency.** Synchronous wiring adds the record-only POST, at most 1.5 s, to every failed call's latency. A `timeout` drop line in `event-drops.jsonl` is the local sign that bound is being hit.

The record-only POST sends one `post_tool` event per failure (1.5 s timeout, no retry) using the same `PERSONAL_KB_URL` / `PERSONAL_KB_API_KEY` resolution as the other surfaces, whether or not the failure context is enabled. Claude Code does not fire `PostToolUseFailure` for validation rejections, permission denials or cancellations.

Any event that could not be delivered (missing fields, no URL/key, timeout, URL error, non-2xx response) is appended as one jsonl line to the drop log `~/.cache/personal_kb/event-drops.jsonl`. The log is reset once it exceeds 256 KB. The service's `GET /api/kb/event/heartbeat` counts recorded failures only, so a zero heartbeat must be checked against this drop log before you conclude that nothing failed.

## Prevention channels (`SessionStart` slice + `PreToolUse` soft gate)

**Gotcha slice.** On `SessionStart` the hook makes one `GET /api/kb/prevention` call (1.5 s timeout, no retry; personal KB only). A non-empty `slice_text`, the "Known gotchas for <project>" list of corrected facts from the KB, is injected in the same single SessionStart output, ahead of the maps directory. It is never written to `CLAUDE.md`, `AGENTS.md`, memory files or any file in the repo.

**Soft gate.** The same response carries the gate settings and a Bash cue index, cached per session. Wire `PreToolUse` with `"matcher": "Bash"` and `"timeout": 5` (see the root README). When a Bash call's two-word class (such as `git push`) exactly matches a cue, the hook denies it, with the corrected fact as the reason; retrying the identical call is allowed. PreToolUse never makes a network call.

**Deny limits and re-arming.** There is no per-session budget, so a session that runs for weeks keeps the gate. A lesson that denied stays quiet for `rearm_hours` (default 24), then may deny again. A `SessionStart` whose `source` is `compact`, `resume` or `clear` re-arms every lesson at once and logs a `rearmed` row, because the agent may have lost the earlier deny reason from its context; a fresh `startup` does not. Retrying the identical call right after a deny marks that lesson `overridden` (logged as an `overridden` row), and it stays quiet until its next re-arm. Denies are rate limited to `max_denies_per_turn` per turn (default 1; the count resets on `UserPromptSubmit` and `Stop`) and `max_denies_per_hour` in any 60 minutes (default 6). An over-limit match is logged as `skipped_cap` with reason `per_turn` or `per_hour`. Shadow mode consumes the same limits.

**Server switches** (set on the KB service, read per request):

| Variable | Default | Effect |
|----------|---------|--------|
| `KB_SOFT_GATE_ENABLED` | off | Only the literal `TRUE` (any case) sends a gate index. |
| `KB_SOFT_GATE_SHADOW` | shadow on | Only `FALSE` turns on real denies; in shadow mode the hook records `would_deny` and prints nothing. |
| `KB_SOFT_GATE_DISABLED_PROJECTS` | empty | Comma-separated projects whose gate is off. |
| `KB_DELIVER_OBSERVED_ONCE` | on | `FALSE` withholds autonomous resolutions seen in only one session. |
| `KB_SURPRISE_CAPTURE` | off | `shadow` or `on` makes the hook send a turn digest at every `Stop`; any other value means off. |
| `KB_SOFT_GATE_MAX_DENIES_PER_TURN` | 1 | Most denies (or `would_deny` in shadow) in one turn; integer 1..1000, anything else means the default. |
| `KB_SOFT_GATE_MAX_DENIES_PER_HOUR` | 6 | Most denies in any rolling 60 minutes; integer 1..1000, anything else means the default. |
| `KB_SOFT_GATE_REARM_HOURS` | 24 | Hours a lesson stays quiet after it denies before it may deny again; integer 1..1000, anything else means the default. |

The response still carries a legacy `max_denies` field, fixed at 1000, so hooks that predate these limits are no longer capped at 2 denies per session.

A switch flip reaches a live session at its next `Stop`, when the hook re-fetches the settings.

**Files** under `~/.cache/personal_kb/`: `prevention-<session>.json` is the cached settings, index, per-lesson deny state and rate-limit counters; caches older than 7 days are removed. `gate-log-<session>.jsonl` is the local decision log, flushed at `Stop` to `POST /api/kb/prevention/decisions` (stale logs from other sessions are swept at `SessionStart`). `event-drops.jsonl` is the drop log shared with the failure-cue feed, where failed fetches and flushes are recorded (`op` = `prevention_fetch`, `gate_flush`, `gate_log_rotated`, `turn_digest` or `failure_context`, the last with reason `state_write`, `record_failed` or an exception class name). `failure-context-<session>.json` holds the resolution ids already delivered as failure context this session (removed after 7 days). `turn-state-<session>.json` is the Stop counter and last digested record id (removed after 31 days). `turn-digest-log-<session>.jsonl` is the local digest decision log (capped at 256 KiB, removed after 7 days).

The gate is inert until resolutions exist in the KB in the `hints.resolution` format documented in `packages/kb-service/src/kb_service/prevention.py`. Until then the slice is fed only by supersedes corrections.

## Surprise capture (`Stop` turn digest)

The switch is read from the top-level `surprise_capture` of `GET /api/kb/prevention` and cached per session, so a flip lands at the next `Stop`. The cached value is the only gate on the send, independent of `PERSONAL_KB_LISTENER` and `KB_LISTENER_HEADLESS`. It only decides whether a digest is sent: whether the service just detects (`shadow`) or also distills (`on`) is decided by its own `KB_SURPRISE_CAPTURE` at processing time.

A digest holds the user prompt (4000 chars), the ordered assistant text (2000 chars each), tool calls (target 500 chars) and tool results (1500-char head/tail excerpt) of the turn, capped at 200 items, plus the final message (4000 chars) taken from `last_assistant_message`, because the transcript can lag. The turn window is every transcript record newer than the later of the previous digest's last record and the latest human prompt (task notifications, slash-command echoes, meta and sidechain records are not prompts). `event_id` is `<session_id>:<turn_index>`, and `turn_index` counts every `Stop` in the session whatever the mode.

A body over 64 KiB is truncated oldest-first; if even an empty item list does not fit it is dropped as `too_large` (fail closed). Text is sent unredacted and the service redacts secrets before storage. A detached `python -m personal_kb_hook.turn_sender` POSTs the body with a 10 s timeout and no retry.

Drop-log reasons (`event-drops.jsonl`, `op` = `turn_digest`): `no_url_key`, `transcript_unreadable`, `too_large`, `spawn_failed`, `error`, `state_write_failed`, `body_unreadable`, `http_<code>`, `timeout`, `urlerror`, `rejected_write-failed`, `rejected_redaction-unavailable`, `rejected_duplicate-mismatch`.

The local `turn-digest-log-<session>.jsonl` (never sent anywhere) has two row kinds. `op` `stop`: `ts`, `session_id`, `turn_index`, `event_id`, `capture`, `action` (`sent`, `no_url_key`, `too_large`, `spawn_failed` or `error`), `error`, `state`, `boundary`, `last_uuid_missing`, `cut`, `records_parsed`, `unknown_block_types`, `items_built`, `items_sent`, `bytes`, `truncated`, `user_prompt_null`, `final_message_null`, `elapsed_ms`. `op` `send`: `ts`, `session_id`, `event_id`, `http_status`, `server_reason`, `redactions`, `drop_reason`, `elapsed_ms`.

To reconcile a low server heartbeat: (1) the server's `prevention_fetch` INFO lines show the `surprise_capture` each session got; (2) `stop` rows with action `sent` count spawned digests; (3) `send` rows with `server_reason` `recorded`, summed over every session's log on that host for the same window, should equal `count` in the `GET /api/kb/turn/heartbeat` row for harness `claude-code`, that mode and that host (the heartbeat groups by harness, mode and host with no per-session breakdown; `sessions_with_gaps` flags lost turns inside a session); (4) any gap is explained by a `send` row's `drop_reason` or an `event-drops.jsonl` line with op `turn_digest`. Two tripwires: `last_uuid_missing` true, and a non-empty `unknown_block_types` (a transcript format change).
