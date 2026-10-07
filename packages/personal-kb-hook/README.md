# personal-kb-hook

Stdlib-only Claude Code hook that surfaces a project's `mental_map` directory
and listener whispers. See the repo root `README.md` for install and wiring.

## Environment

| Variable | Effect |
|----------|--------|
| `PERSONAL_KB_LISTENER` | `1`/`true` enables the listener (also needs a resolvable URL/key). |
| `KB_LISTENER_HEADLESS` | `TRUE` runs the listener on `Stop` even in headless dispatch runs (`HEADLESS_BUILD_ENGINE` set). Default: skipped, because a headless run never gets another `UserPromptSubmit`, the only whisper delivery path, so the Sonnet votes would be wasted. |
