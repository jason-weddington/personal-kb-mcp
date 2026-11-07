# Runbook: freezing a hosted KB when it stops being the system of record

Use this when a hosted kb-service KB keeps serving reads but its owner moves writes to another system, for example when the data is migrated to a new KB implementation. The freeze is reversible until the new system has taken writes the old one never saw.

Host names, URLs and keys for a specific deployment do not belong here; keep them in the deployment's private ops notes. Every switch named below is documented in `how_it_works.md` (configuration reference) or `docs/intelligent-kb.md` (section 22, production state and switches).

## 0. Before the day

- Deploy a kb-service release that has `KB_READ_ONLY` and `KB_READ_ONLY_MESSAGE`, with both left unset so nothing changes yet. Confirm the release is live on the KB host.
- Agree the dump format with the importer, then dry-run the dump against the live KB and check the importer accepts it. Record row counts per table and the highest entry id.
- Decide what the session hooks will read after the freeze (see step 6) and have the new endpoint and key ready if they move.
- Check that no dispatch runs are in flight, and tell live peer sessions, because a kb-service restart briefly degrades hooks for every session on the fleet (the deploy preflight guard lists both).

## 1. Stop the unattended writers

Each of these writes the KB without an agent asking. Stop them first, so the final dump is not racing them.

- **Nightly map maintenance (somnus):** disable its timer on the KB host, or set its kill switch in somnus's own env file. Confirm the timer no longer lists a next run.
- **Surprise capture:** set `KB_SURPRISE_CAPTURE=off`. The background worker reads it only at startup, so this needs the restart in step 3. Live sessions stop sending turn digests at their next Stop, once `/api/kb/prevention` reports `surprise_capture` as off.
- **Listener:** set `KB_LISTENER_ENABLED=FALSE`. It is read per request.
- **Embedding retry worker:** set `KB_EMBED_WORKER_ENABLED=FALSE`, since it writes embeddings to the data DB.

## 2. Take the final dump

- Run the dump in the agreed format (entries, versions and edges at minimum) after step 1, not before.
- Compare its row counts and highest entry id with the live database, and keep the dump file and its counts beside the importer's run record.

## 3. Freeze

- Set `KB_READ_ONLY=TRUE` and `KB_READ_ONLY_MESSAGE` to text that tells a writer where writes go now, then restart kb-service.
- Verify: an MCP `kb_store` returns the read-only error with the message, an HTTP write route returns 403, and `kb_search` and `kb_get` still return results. Telemetry and hook-ingest routes are not covered by this switch; they are off because of step 1.

## 4. Import

- Run the importer against the dump and compare its counts with step 2.
- Spot-check a few entries by id in both systems, including the most recent one and one with a long version history.

## 5. Repoint MCP clients

- On each machine, point the user-scope MCP server entry for this KB at the new system's endpoint and key. Keep the same server name, so agents' tool names do not change.
- Do the same for every other harness that is wired to the KB (for example an Antigravity `hooks.json` and its MCP config).
- Start a fresh session on each machine and confirm a search reaches the new system.

## 6. Session hooks

The hook (`personal-kb-hook`) calls more than search and store: `/api/kb/prevention` (the SessionStart gotcha slice and the PreToolUse soft gate), `/api/kb/maps-index`, `/api/kb/event`, `/api/kb/turn`, `/api/kb/listener`, `/api/kb/telemetry/whispers` and `/api/kb/prevention/decisions`. Its URL and key come from `PERSONAL_KB_URL` and `PERSONAL_KB_API_KEY` in each machine's hook command, so moving it is a per-machine config change.

- If the new system serves a compatible `/api/kb/prevention` (see `contracts/prevention.json`), repoint the hook's URL and key to it. The other endpoints may 404 there; the hook fails soft on every non-2xx.
- If it does not, and the owner accepts losing the slice, gate, failure context, listener and surprise capture outside the new system, remove the hook's entries from every harness on every machine: Claude Code's hook settings for all seven events, and any other harness wired to it (for example an Antigravity `hooks.json`). Where hook settings come from a shared base file that is merged into each machine's own settings, removing them from the base file does not remove them from machines that already merged it, so delete them from each machine's settings too. Then start a fresh session per machine and confirm no hook output and no KB calls in the hook's logs.
- Leaving the hook pointed at the frozen KB also works, since it fails soft, but its slice and gate stop learning and its digest and event posts keep arriving at a KB that ignores them.

## Rollback

Before the new system has taken writes the old one lacks: set `KB_READ_ONLY=FALSE`, restore the step 1 switches to their previous values, restart kb-service, and point the MCP clients and hooks back. After that point, rolling back loses the new writes unless they are exported from the new system and re-imported.
