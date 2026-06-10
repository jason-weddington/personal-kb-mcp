# personal-kb-web-service

Hosted home-lab service for the Personal KB. Wraps the **`kb-core`** engine
(hybrid search + knowledge graph + agentic retrieval) behind an authenticated
HTTP API + web app, so the local `personal-kb` MCP server becomes a **thin
client** (config collapses to URL + API key + `KB_INSTANCE_ROLE`).

## What it is

- **FastAPI service** over a singleton `kb_core.KnowledgeBase` (`create_postgres`),
  modeled on the **Agent GTD** server: dual JWT-or-API-key auth, an API-key
  creation flow in the UI, and user-systemd deploy on `git-host`.
- **Authenticated web app** — a React/MUI SPA bootstrapped from the `fullstack`
  starter (auth, settings pane, Postgres), absorbing the old local KB **explorer**
  (graph viz, SSE-streamed `ask`/`summarize`, chat) under login.
- **API-key → MCP config**: the web app mints a key; that key goes in the MCP
  client config; the thin client calls this service.

## Topology

Two physical scopes mirroring today's two MCP instances — `personal` and `team` —
each pointing at its existing Postgres DB on the `git-host` pgvector cluster
(`personal_kb`, `team_kb`). No data migration: the DBs already exist.

## Status

P1–P3 done (backend API). P4a done: Vite/React 19/MUI 7 frontend chassis with auth plumbing
(kb-01449 guards), FastAPI SPA serving. Pending: P4b (settings/admin), P4c (explorer/chat),
P5 thin MCP client, P6 deploy/cutover.

Architecture and the phased build plan are in KB entry **`kb-01742`**; engine seam is **`kb-01727`**.

## Frontend dev setup

```bash
npm --prefix frontend install
npm --prefix frontend run dev     # Vite on :5173 — proxies /api to :8000
npm --prefix frontend run build   # produces frontend/dist (served by FastAPI)
npm --prefix frontend run test    # vitest
npm --prefix frontend run lint    # eslint
```

## Build sources

- **server skeleton + auth/API-keys + thin-client `HttpBackend`** ← `agent_gtd`
- **frontend chassis (auth, settings, Postgres)** ← `fullstack`
- **engine** ← `kb-core` (in the `personal_kb` repo's `packages/kb-core/`)
