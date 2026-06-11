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
  whispers the resulting pointer into the next agent turn. Validated at 1.00
  precision / 0 false injections on the eval harness (`evals/listener/`).

## Topology

One **stack** = one KB: a Postgres data DB (kb-core owns it, `KB_DATABASE_URL`)
plus a service DB for auth/app-config (`KB_SERVICE_DATABASE_URL`). The personal
and team KBs are two deployments of identical code pointed at different DBs.

## Dev deployment

`kb-host-1` runs the full stack under the `kb-service` systemd user unit at
`http://kb-host-1:8000`; `./deploy.sh` pulls main there, rebuilds the
frontend, and restarts. Runbooks: dev server **kb-01746**, production cutover
**kb-01765**, per-machine hook/listener setup **kb-01784**.

## Development

See `CLAUDE.md` for commands, layout, and the KB entries that hold the
architecture (`kb-01744`, `kb-01742`) and listener design (`kb-01725`).
