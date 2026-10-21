# Wire contracts

## What this is

Golden request/response fixtures for the endpoints a harness calls, `GET /api/kb/prevention`, `POST /api/kb/turn`, `POST /api/kb/event` and `POST /api/kb/belief-check`. They are used today by the Claude Code hook and are intended for talos to vendor (adoption is a harness-design item), so a schema change breaks the other side's tests instead of drifting silently.

## File format

`prevention.json`, `turn.json` and `belief_check.json` each carry the top-level keys `contract`, `contract_version`, `generated_by`, `method`, `path`, `request` and `response`; `turn.json` also has `request.max_body_bytes`.

Examples are listed in `examples` arrays: request and response examples are `{name, body}` envelopes, and the prevention request examples are `{name, query}` envelopes.

The schemas are pydantic `model_json_schema()` output (JSON Schema draft 2020-12) for `PreventionResponse`, `TurnDigestRequest`, `TurnDigestResponse`, `BeliefCheckRequest` and `BeliefCheckResponse`, including the docstring-derived `description` and `title` keys.

`request.query_schema` in `prevention.json` is derived from FastAPI's OpenAPI parameters and carries `additionalProperties: false`; body schemas do not, because pydantic silently drops unknown body keys.

## Invariants a schema cannot express

POST /api/kb/turn checks in this order: 422 (schema-invalid body), then 413 (raw request bytes above `max_body_bytes`, whitespace included, in any capture mode), then 200 with a `reason`; a DB failure is reason `write-failed` at 200.

`event_id` must equal `<session_id>:<turn_index>` when `agent_id` is absent, and `<session_id>:agent:<agent_id>` when it is present (the SubagentStop digest form); anything else is answered 422.

The `<session_id>:agent:<agent_id>` event_id form and the SubagentStop digest REQUIRE the kb-service release that knows them to be deployed before any sender uses them: against a pre-release service a SubagentStop digest fails the `event_id` check and is answered 422, landing in the hook's drop log as `http_422`, whereas a new hook's null `agent_id`/`agent_type` on a normal Stop digest are silently dropped by an older server (unknown body keys are dropped). Same for the optional `agent_id`/`agent_type` on `/api/kb/event`.

Unknown body keys are dropped silently, so a consumer must assert exact key sets per `$defs`, as `test_contracts_hook.py` does, and may omit an optional field only by listing it explicitly (`HOOK_OMITTED_REQUEST_FIELDS`).

GET /api/kb/prevention resolves the project as the non-blank `project` stripped, else the `cwd` basename lowercased with `_` turned into `-` and any dispatch run-id suffix stripped, else `""` (empty index and slice; gate settings and `surprise_capture` are still env-driven).

GET /api/kb/prevention never answers 5xx; on failure it returns an inert body (gate enabled false, shadow true, empty index and slice, zero diagnostics) that a client cannot tell from an empty result.

`harness=talos` returns `tool_map` (native name to canonical name) while the index stays in canonical Claude Code names, so a consumer translates its native names through `tool_map` before matching; without `harness`, `tool_map` is `{}`.

`gate.enabled` and `gate.shadow` reflect only the Claude Code gate switch, and a mapped harness receives the index regardless and applies its own mode.

An absent `tool_map` key means a server older than tool-name normalization, so the harness must not post native-named digests to it.

Turn `event_id`s must be unique per session within the server's turn-digest retention (90 days).

Undeclared query params are silently ignored, so a consumer must validate its query against `query_schema`.

Auth is `Authorization: Bearer <api key>`; with `KB_AUTH_MODE=none` the token is ignored (the hook sends `local-no-auth` on a loopback URL).

A request-side addition such as a new TurnItem kind or query param needs the kb-service release deployed before any sender uses it, because an older server answers 422 for the whole digest.

The talos sender rules for `harness_correction` are in the `TurnHarnessCorrectionItem` `$defs` description.

## POST /api/kb/event

`event.json` lists under `fields` which fields are client-supplied (the `EventRequest` properties) and which are server-computed (`cue_key`, `normalizer_version`, `host_class`, `project_source`, `target`, `target_class`, `normalized_error`, `error_rule`, `anomaly`, `raw_error_excerpt`, `received_ts`); a client never sends the computed ones.

Each request example carries its `stored` computed fields under the top-level `stored` map and its response under `response.examples`, and the route always answers 200 with the `EventResponse` shape.

A mapped harness (today talos) sends its native `tool_name` and native `tool_input` argument names (`edit_file` with `path`, for instance), and the KB normalizes both to the canonical Claude Code names before computing the cue, so the stored `tool` is `Edit` and `target` is the path.

Mapped harnesses send their native error text (talos opens with `exit 1 (0.01s)` and `--- stdout ---` / `--- stderr ---` markers, Claude Code with `Exit code 1`) and the KB normalizes it, so the same failure gets the same `cue_key` from either.

`event_id` must be unique per session; a repeated `event_id` is answered `duplicate` and not stored again.

## POST /api/kb/belief-check

Auth is the same as GET /api/kb/prevention.

The route never answers 5xx: a schema-invalid body is 422, everything else is 200 with a `reason`, and `matches` is `[]` on every reason except `checked`.

The ten reasons in check order: `no-project` (the stripped project is blank), `project-disabled` (the project is in `KB_SOFT_GATE_DISABLED_PROJECTS`), `no-embedder` (the KB is FTS-only), `no-chunks` (text and reasoning yield no chunk of 16 or more characters), `no-candidates` (no trusted resolution with a wrong belief applies), `timeout` (the server deadline passed before the resolution load or the chunk embed finished), then `warming` (no candidate vector was cached yet because a warm task is still running), `embed-failed` (the embedder is configured but failed or cached nothing; kb-bench with embeddings off answers `embed-failed`, not `no-embedder`) or `checked` (scoring ran); `error` on any exception.

`matches` are the top 3 by score, descending, INCLUDING below-threshold ones; a consumer injects only when `matches[0].score >= threshold`.

`checked` with `scored < candidates` is a partial check and must not be counted as a clean negative.

`score` is raw cosine similarity, unrounded, and depends on the server's embedder model; `matcher_version` does not change with the model, so scores and threshold are comparable only between responses with equal (`matcher_version`, `embed_model`).

`matcher_version` changes on any threshold, chunking, candidate or scoring change, with no `contract_version` bump.

`text` and `reasoning` are truncated server-side to their last 4096 characters and never rejected for length.

The project is the `project` field stripped; it is matched case-sensitively against resolutions (the kill-switch list is case-insensitive), there is no `cwd` fallback, and blank is `no-project`.

`pending_calls` tool names are native, not canonicalized, and are not scored in matcher_version 1.

The server keeps no per-session state, so a consumer injects a given `entry_id` at most once per session, or once per its own re-arm window, because a negated restatement of a wrong belief can score like the belief itself.

A server start empties the vector cache; each uncached candidate costs about 8 ms on the production embedder (qwen3-embedding:0.6b, dim 1024), up to 500 candidates, embedded in background batches of 32, so checks right after a start answer `warming` or a partial `checked`, and a consumer sends one warm-up check at memory connect (any text of 16 or more characters) and discards the result.

`session_id` and `turn_index` are only logged; they are the join key (`session_id`, `turn_index`) to `turn_events` for auditing a match.

A server older than this endpoint answers 404; a consumer treats 404, any non-200 and a timeout as no match (fail open).

A consumer that measures records per call the full response body, the `session_id` and `turn_index` it sent, and its own client-side timeouts.

A new `BeliefCheckReason` member is a breaking response change (enum-added) and needs a `CONTRACT_VERSIONS` bump.

## Seeing drift in production

The server logs `turn_event reason=invalid` (422, loc and type only) and `turn_event reason=too-large` (413), and counts both in `route_outcomes` (plus `unmapped_tools`) on GET /api/kb/turn/heartbeat.

The hook's turn-digest log records `http_status` and `server_reason` per send.

A /api/kb/prevention failure shows only as the server WARNING `prevention_fetch failed`.

A /api/kb/belief-check failure shows as the server WARNING `belief_check failed`; every request logs one INFO `belief_check ` line with `reason`, `detail`, `top_id` and `top_score`, and warm-ups log `belief_check_warm`.

## Regenerating

Run `uv run python packages/kb-service/scripts/gen_contracts.py` from the repo root, with `--check` to verify, and never hand-edit the JSON.

A pydantic or fastapi upgrade or a model docstring edit can change the rendered schema and needs a regenerate.

The enforcing tests are `packages/kb-service/tests/test_contracts_service.py` and `packages/personal-kb-hook/tests/test_contracts_hook.py`.

## contract_version

`contract_version` is an integer per file, edited only in `CONTRACT_VERSIONS` in the generator, and bumped for a breaking change.

For requests a breaking change is a removed or renamed field, a narrowed type, a lower `max*`/`max_body_bytes` or higher `min*`, a newly required field or query param, a removed enum member or TurnItem kind, or a changed path or method.

For responses it is a removed or renamed field, a field becoming optional or nullable, a widened type, or a new member of any response enum (`TurnDigestResponse.reason`, `PreventionResponse.surprise_capture`, `BeliefCheckResponse.reason`).

These regenerate with no bump: a new optional request field or query param, a new TurnItem kind, a new response field with a default, and description or title-only changes.

The generator detects property-removed, required-added/removed, enum-removed/added, limit-tightened/added, kind-removed, def-removed, type-changed and endpoint-changed, and refuses to write until the version is bumped.

Those detectors are a floor: any other change in the breaking list (for example a response field gaining null in anyOf) needs a manual bump, and a model rename trips def-removed and is bumped anyway.

## Vendoring

Copy this directory verbatim, and re-copy whenever its content changes.
