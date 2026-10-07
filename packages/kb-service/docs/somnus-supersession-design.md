# somnus iteration 1: don't point a map at a superseded entry

Status: DRAFT for Jason's review, 2026-10-07. Nothing is built. GTD: fb8a4f52 (design umbrella).

## The problem, measured

The 2026-10-07 full catch-up wrote 29 maps over 24 projects for $15.15. Three independent reviews read every map plus every pointed entry and checked load-bearing claims against the repos. About a quarter were good, half were useful but flawed, and 7 actively misled and were deleted. Nearly every serious defect had one cause: **somnus pointed a map at an entry that a newer entry had already corrected, and nothing told it.** Examples from the review:

- kb-03134 says outright that it retracts kb-03029's numbers. The discovery-crawler map pointed at both, with no warning.
- kb-01598 says the token file in kb-01318 is a leftover that nothing reads. The auth map pointed at kb-01318.
- kb-01726 records that the smithy-json bug was fixed upstream. The deleted map kb-03605 led with kb-00075's workaround, which would double-escape today.
- kb-01262 records that kb-01261's file-naming scheme was dropped. The sidecar map listed kb-01261 first.

A map is fact-free routing, so it cannot be wrong on its own. It is wrong when it routes an agent to a stale fact without saying a newer one exists. That makes the fix a pointer-selection problem, which is why iteration 1 is narrow.

## Why the existing mechanism does not save us

kb-core already models supersession explicitly. An entry's `hints.supersedes` creates a `supersedes` graph edge, and there is a `superseded_by` column. `kb_ask`'s decision_trace walks the chains. In practice it is almost unused: **25 `supersedes` edges across 3,372 active entries, and `superseded_by` is set on none.** Agents correct a fact by writing a new entry, not by declaring what it replaces. Explicit edges are a free first check, but the design has to find implicit supersession, which is what Jason proposed: walk the project's entries and look for newer ones that replace the one a map is about to point at.

## Scope of iteration 1

In scope:

- For every pointer somnus is about to write (`create_map` pointers and `add_pointer`), check whether a newer entry in the same project supersedes it.
- If one does, point at the newer entry instead, or point at both with the newer one first.
- Record every check and verdict in the run report, so precision can be read from real nights.

Out of scope, deliberately:

- **Writing to fact entries.** Iteration 1 does not set `superseded_by`, add edges, or edit entries. Flagging or fixing facts for a human is iteration 2, once we know how often the verdicts are right.
- **Re-checking pointers on existing maps.** That is the same check run over a different list. Add it once the check is trusted.
- **Drift between code and KB** (kb-00778's numbers vs `ranking.py`). A KB-only check cannot see it.
- **Cross-project supersession** (cleanr's cache entry superseded by a dispatch-log entry in another project). The walk is project-scoped, as Jason specified.

## Design

### Step 1: candidate newer entries, server side, no model

Extend `GET /api/kb/map-loop-input` so each `MapLoopEntry` carries the evidence somnus currently lacks:

- `updated_at` (normalised to UTC; the live corpus mixes naive and aware timestamps, see kb-03449)
- `explicit_superseded_by`: ids from incoming `supersedes` edges whose source entry is active
- `newer_neighbors`: up to 5 `{id, similarity, updated_at}` for active same-project entries that are newer than this one and have cosine similarity at or above a floor

The pair query already exists for pockets (pgvector `<=>` over `knowledge_vec`, restricted to the project), so this is one more pass over the same pairs, filtered to "neighbour is newer". The floor needs calibrating; start at the pocket observation floor and tune it against the fixtures below. No inference happens here; it is SQL.

Why similarity and not title or tag matching: kb-03134 and kb-03029 share a topic, not a title. Embedding neighbours are the cheapest signal that two entries are about the same thing.

### Step 2: the supersession check, a new somnus step between rung 2 and apply

For each pointer about to be written:

1. **Explicit edge.** If `explicit_superseded_by` is non-empty, the newest active superseder wins. No model call.
2. **No newer neighbours.** If `newer_neighbors` is empty, keep the pointer. No model call. Expect this to be most pointers.
3. **Otherwise, one model call** with the candidate's full text and each newer neighbour's full text. The tool has a closed verdict and **required evidence in the signature.** This follows kb-03486: put the thing you need in the tool's shape, not in the prompt.
   - `verdict`, one of:
     - `supersedes`: the newer entry replaces the older one entirely
     - `amends`: the newer entry corrects or extends part of it
     - `independent`: same topic, no conflict
   - `superseding_id`: required unless the verdict is `independent`.
   - `old_claim` and `new_claim`: required verbatim quotes, one from each entry, showing the conflict. Required-but-empty is allowed only for `independent`, which turns an omission into an explicit decision.

**Code verifies the quotes.** Each quote must be a substring of the entry it claims to come from (after whitespace normalisation). A verdict whose quotes don't verify is downgraded to `independent` and counted. This is the guard against an invented supersession: the model cannot claim a conflict it cannot quote.

### Step 3: the action

| Verdict | Pointer change | Note on the map |
|---|---|---|
| explicit edge, or `supersedes` | Replace the old pointer with the superseding entry. If that entry is already a pointer, drop the old one. | none needed |
| `amends` | Keep both, newer first. | A code-composed gloss on the old pointer: "partly superseded by kb-X". The model never writes this text. |
| `independent` | unchanged | none |

Glosses on swapped pointers are written by rung 2 as today, but for the entry actually pointed at.

### Step 4: record everything

The run report gains a `supersession_checks` array: candidate, path taken (explicit / no-neighbours / model), verdict, superseding id, quote-verification result and action taken. This is the instrument for deciding iteration 2. It runs alongside the feature; it doesn't gate it.

## Companion fix: a deleted map must not come back

Deleting a map today records nothing in the cluster ledger (`map_delete_routes.py`), so somnus can write the same cluster again on a later night. That includes the 7 deleted tonight. `DELETE /api/kb/maps/{id}` already requires a reason. It should also write a decline row for the map's member set, keyed the way the ledger keys clusters (member-set overlap). Then the loop treats the cluster as declined until membership changes enough to reopen it. It's small, it's server side, and it should ship with or before iteration 1.

## Calibration fixtures

These are pairs from the 2026-10-07 review where the right answer is known:

| Older | Newer | Expected |
|---|---|---|
| kb-03029 | kb-03134 | supersedes (it says it retracts the numbers) |
| kb-01318 | kb-01598 | amends at least (the token-file location) |
| kb-01434 / kb-01593 | kb-01598 | supersedes (key injection retired) |
| kb-00075 | kb-01726 | supersedes (fixed upstream) |
| kb-01261 | kb-01262 | amends (naming scheme dropped) |
| kb-00775 | kb-03026 | amends (owner penalty softened) |

Also include two negative pairs from the same projects, related but compatible, to measure false positives. The fixture test runs the real check against frozen entry text and fails on any wrong verdict.

## Cost

The model call only fires for pointers with a newer neighbour above the floor. Tonight wrote about 130 pointers across 29 maps. Even if a third had candidates, that is about 40 calls of roughly 3–5k input tokens. That's well under $1 for a full catch-up and pennies per normal night. The real cost is latency on the Pi; this is not a budget question.

## Work items once approved

1. **kb-service:** `updated_at`, `explicit_superseded_by` and `newer_neighbors` on map-loop-input, plus the ledger decline on delete. personal_kb; mirrors existing route patterns.
2. **somnus:** the supersession step, its tool schema, quote verification, the pointer action and the run-report section. harness-design; it depends on item 1's fields.
3. **Fixtures:** the table above as a test, plus a fixture-backed offline run before the first live night.

Item 2 should also fix bug e2a307d7 (the rung-2 prompt lists `no_change`, but the tool set has no such tool), since it touches the same file.

## Questions for Jason

1. Is `amends` worth having in iteration 1, or should it be binary (supersedes or not) for the first measurement?
2. "Newer" means `updated_at`. An old entry that was substantially rewritten counts as newer, which seems right. Metadata-only updates also bump it, which adds noise. Accept that for now?
3. Should iteration 2 (writing a flag or `superseded_by` onto the old entry) wait for a precision number from the run reports, or is the quote-verification guard enough to go straight there?
