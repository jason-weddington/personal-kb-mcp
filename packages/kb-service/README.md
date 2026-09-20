# personal-kb-web-service

Hosted home-lab service for the Personal KB. Wraps the **`kb-core`** engine
(hybrid search + knowledge graph + agentic retrieval) behind an authenticated
HTTP API and a React SPA. The `personal-kb` MCP server is a **thin client** of
this service, and the **anticipatory listener** pushes knowledge pointers into
agent sessions.

## What it is

- **FastAPI service** over a singleton `kb_core.KnowledgeBase`: the full KB
  surface under `/api/kb/*` (search/get/store/ask/summarize/ingest/preflight/
  graph/maps-index), SSE-streamed queries and chat, JWT-or-API-key auth with
  invite-gated registration and an admin pane.
- **React 19 / MUI 7 SPA** (`frontend/`): login, search, entry detail,
  force-graph visualization, streaming Ask, chat, settings (API-key minting
  with MCP-config snippet), admin (invites/users).
- **Anticipatory listener** (`POST /api/kb/listener`): maps-only cross-project
  relevance engine — rule A (drop the session project's own maps), rule B
  (drop maps for systems the agent is operating, via `operated_via` hints on
  map entries), unanimous-3 Sonnet retrieve-and-cite. The `personal-kb-hook`
  fans this call out across a **roster of KBs** (personal + team), unions the
  per-KB winners via a client-side suppress-only arbitration step, and whispers
  one pointer per KB into the next agent turn. Single-KB Gate-1 validated at
  1.00 precision / 0 false injections; the multi-KB go-live gate
  (`evals/listener/run_p3.py`) gates on **0 union false-injections** (precision
  informational) and **passed live 2026-06-15** — multi-KB whisper is live
  (kb-01839).

## Topology

One **stack** = one KB: a Postgres data DB (kb-core owns it, `KB_DATABASE_URL`)
plus a service DB for auth/app-config (`KB_SERVICE_DATABASE_URL`). The personal
and team KBs are two deployments of identical code pointed at different DBs.

## Dev deployment

`kb-host-1` runs the full stack under the `kb-service` systemd user unit at
`http://kb-host-1:8000`; `./deploy.sh` pulls main there, rebuilds the
frontend, and restarts. Runbooks: dev server **kb-01746**, production cutover
**kb-01765**, per-machine hook/listener setup **kb-01784**.

## Installing somnus on the KB hosts

`somnus`, the nightly map-maintenance loop binary (see
`docs/nightly-map-maintenance-design.md`), is delivered to the KB hosts by `scripts/install-somnus.sh`.
It resolves the target token by fetching `latest` from the artifact host (artifact-host's Caddy, default `https://artifacts.lab.example.com`), verifies the artifact exists at that token, and installs it on each of `kb-host-1`, `kb-host-2` and `kb-host-3` (overridable via `KB_HOSTS`), skipping hosts already current.

Run it as `./scripts/install-somnus.sh` — or pin a token with `--version <TOKEN>` and retarget with `KB_HOSTS="kb-host-2"`.
It is idempotent and safe on a fresh host, attempts every host even when one is powered off, prints a per-host summary (installed / skipped-already-current / unreachable / failed), and exits non-zero unless every host ended at the target token.
It restarts nothing: somnus is a timer-invoked subprocess, so there is no daemon to bounce and the next timer fire picks up the new binary.
On a failed post-install version check it rolls back to the previous binary rather than leaving a broken one, because the run is unattended.

This path is deliberately independent of the dispatch fleet's installer (`agent-gtd-dispatch/talos-update.sh`, same artifact host, different session).
Per Jason's ruling of 2026-09-19, every session owns its own release tooling, so the KB hosts and the dispatch fleet can legitimately run different tokens at different times — nothing may assume fleet-wide version uniformity, and changing one installer must not force a change on the other.

## Development

See `CLAUDE.md` for commands, layout, and the KB entries that hold the
architecture (`kb-01744`, `kb-01742`) and listener design (`kb-01725`).
