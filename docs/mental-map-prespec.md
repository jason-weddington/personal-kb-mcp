# Pre-Spec: Mental Map Nodes for Personal KB

**Status:** Debated 2026-06-03. Sections 1–6 are the original *starting positions* — preserved as the record of what went into the debate. **The settled design is §7, which supersedes the starting positions where they conflict.** §7 also corrects several factual errors in the starting positions (the prespec made claims about the codebase that turned out to be false — see §7.1).

**Author/owner:** Jason
**Date:** 2026-06-03

---

## 1. The problem

Human developers carry a *mental model* of a system they've worked on: not the details, but a map of where the details live and how the pieces fit. "I don't know what port the firewall accepts Wireguard on, but I know to check the pfSense UI." That's not knowledge — it's a **pointer to where knowledge lives**. Cognitive science calls this **transactive memory**: you offload the data and retain a directory entry for retrieving it. Most of what a senior engineer "knows" about a large system is this kind of pointer, not recall.

This mental model has two properties that matter here:

1. **It is the thing that makes onboarding hard.** Picking up a codebase you haven't touched in a year is painful precisely because the *map* faded, not the facts — the facts were always in the code.
2. **It decays faster than the underlying knowledge, and stale maps are actively harmful** — a confident pointer to the wrong place is worse than no pointer at all.

Personal KB today is excellent at storing and retrieving **facts** (the data tier). It has no representation of the **map** (the directory tier). Agents get good recall but no orientation. They can answer "what's the Wireguard port?" if it's stored, but they have no cheap way to *know that the firewall subsystem exists, what it's responsible for, and which entries describe it* — the way a human who's worked on it would.

Claude Code's native memory (`MEMORY.md` as a list of pointers to detail files) gestures at this, but it's project-local, single-machine, and not graph-native. Personal KB and the Team KB variant are the right substrate to do it properly: portable across agent harnesses, multi-project, decay-aware, graph-connected.

## 2. What we're building

A new node type — the **mental map** — that is a small, deliberately fact-free **orientation node**. Its entire job is to be cheap to load and rich in pointers. It encodes the directory tier of transactive memory: *what exists, what it's for, and where the details are.* The details continue to live in ordinary KB entries; the map points at them via graph edges.

The design has three tiers, mirroring the transactive structure:

```
SYSTEM MAP   →  "how-to-AWS docs platform: knowledge-graph service, ingestion, MCP surface, ..."
   │ contains
COMPONENT MAP →  "ingestion pipeline: chunker, dedup agent, extractor, ... see entries below"
   │ references
DETAIL ENTRIES → kb-00421 (chunk overlap rationale), kb-00422 (dedup threshold), ...
```

- **System map** = directory of components.
- **Component map** = directory of detail entries (and sub-components).
- **Detail entries** = the existing data tier, unchanged.

An agent dropped into the system loads the system map (cheap), sees which component is relevant, pulls that component map (cheap), then follows edges to the specific detail entries it needs (paid only for what it reads). This is the human "build context for a system" move, made mechanical.

## 3. Core design decisions (starting positions)

### 3.1 Structure: a new `entry_type`, not a new storage primitive

A mental map is a new **`entry_type`** (`mental_map`), reusing the entire existing entry pipeline — storage, versioning, FTS, embeddings, graph edges, the explorer render. Its special-ness lives in **retrieval behavior** (index surfacing, greedy injection) and **render**, not in storage.

*Rationale:* every feature in this codebase that survived did so by reusing the pipeline (per `how_it_works.md` and the ROADMAP's "rejected" entries). A parallel `map:` node primitive parallel to `note:` would force re-implementation of search, decay, and versioning for no benefit. The map's pointer edges are already expressible with existing edge machinery (`references`, `hints.related_entities`, plus two new edge types below).

### 3.2 Maps hold edges, not facts (structural enforcement)

**A map node structurally cannot contain factual content — only orientation prose plus pointers.** This is the load-bearing discipline. The transactive test: *if a line states the fact, it's miscategorized; it should state where the fact lives.* "Wireguard port is 51820" is a `factual_reference` entry. "Wireguard config → see the firewall component map / pfSense entries" is a map line.

We enforce this structurally rather than by trusting steering, because agents authoring maps will smuggle facts in for convenience. Proposed enforcement (debate to refine the mechanism):

- A map entry's body is **orientation prose** (what this system/component is and does, at a glance) plus a **pointer list** (edges to component maps and detail entries, each with a one-line "what you'll find there").
- A **hard cap** on body size (orientation prose) — small enough to force pointer-not-content discipline. Starting number: ~1500 chars of prose, debate to set.
- Pointers are graph edges, not inline facts. A map with zero outbound pointer edges is invalid (it's not a map, it's an orphan note).
- Optional author-time linter: flag map bodies that look like they're stating specifics (numbers, config values, code identifiers in assertion position) rather than pointing.

### 3.3 Scope: nested system → component

Maps nest via two new edge types:

- `contains` (system map → component map, or component → sub-component)
- `part_of` (the reverse)

Detail entries attach to the component (or system) map they orient via the existing `references` edge (map → entry) — semantically "this map points you to this entry." A `mental_map` may point to other `mental_map`s (nesting) and to ordinary entries (leaves).

Scope is anchored to the existing `project_ref` and tag machinery, so a system map for `how-to-aws` and its component maps share a project scope and are discoverable together.

### 3.4 Authoring: human + agent, with provenance; auto-synthesis as a track

Maps are authored and maintained by **both humans and agents**, with `authored_by` provenance (`human` / `agent` / `synthesized`) on the node so trust and maintenance can be reasoned about differently per source.

- **Human-authored:** curated orientation, the "this is how I think about this system" map.
- **Agent-authored:** an agent that just built deep context for a component writes the map before the session ends, capturing orientation while it still has it (the same "capture while you still have context" principle already in the steering doc).
- **Auto-synthesized (stretch track):** generate a *draft* map from the cluster of existing entries in a scope, using the graph + `kb_summarize` machinery already present. Draft maps are clearly marked `synthesized` and require human/agent promotion before they're treated as authoritative. This is explicitly a later phase — flagged as an open exploration, not a v1 commitment.

### 3.5 Injection: hybrid push + pull, plus a `kb_preflight` extension

The mechanism is **hybrid**, because the two halves solve different problems:

- **Pull alone fails on cold-start.** An agent can't ask for the auth map if it doesn't know the auth subsystem exists. This is the "unknown unknowns" problem, and it's exactly the gap human onboarding suffers.
- **Push alone bloats context.** Pushing full maps on every scope change burns the context window.

So the labor splits:

**Push (discovery) — hooks surface a thin *map index*, not full maps.** The index is a few lines: available map titles + IDs + a one-line description each, scoped to the detected system/component. Cheap, bounded.

The Claude Code hook surface supports this directly:

| Hook | Role | Signal it carries |
|---|---|---|
| `SessionStart` | Surface the system/component map index at session open. stdout is added to context; also supports `mcp_tool` hooks so the index can come straight from a `kb_*` tool. | `cwd`, `source` (startup/resume/clear/compact) |
| `CwdChanged` | Re-surface the index when Claude moves into a different component (e.g. `cd services/billing`). The scope-shift trigger. | `cwd` |
| `UserPromptSubmit` | Refine scope from conversation ("the auth flow") even when CWD doesn't move. Runs every turn → must be cheap (30s timeout). | `prompt` text |

*Critical constraint discovered in the hook docs:* a hook can only **inject a string into context** — it cannot make the agent call a tool. So the push half delivers the **map index as text**; the agent, seeing the index, decides whether to fetch. This *enforces* the hybrid division of labor rather than just enabling it.

*Critical phrasing constraint:* injected `additionalContext` must be **factual statements, not imperative instructions** — imperative phrasing ("load these maps") trips Claude's prompt-injection defenses and gets surfaced to the user instead of used. The index must read as a directory: *"Maps available for this system: [kb-00310] Ingestion pipeline — chunking, dedup, extraction. [kb-00311] MCP surface — tool definitions and routing."*

**Pull (depth) — the agent fetches full maps via a tool.** Seeing the index, the agent calls a new tool (`kb_map_get`, or an extension of `kb_get`) to pull the full orientation node(s) it judges relevant, then traverses edges to detail entries as needed.

**`kb_preflight` extension.** The same map-index surfacing is also exposed through `kb_preflight`, so the tool-triggered path works when no hook fires or the agent re-orients mid-session. `kb_preflight` already returns a project context primer; map index becomes part of that primer.

**Scope detection** is a ranked combination, not CWD alone (CWD proved unreliable in the `kb_preflight` history): CWD (`SessionStart.cwd` + `CwdChanged`) as the coarse signal, conversation keywords (`UserPromptSubmit.prompt`) as the refiner, optionally files-touched (`PostToolUse` matcher on `Edit|Write`) as a third signal. The index query is a cheap KB lookup keyed on whichever signals fired.

### 3.6 Staleness: all three layers

A stale map is worse than no map, so maps get stronger staleness handling than data entries — three layers:

1. **Stalest-pointer freshness inheritance.** A map is only as fresh as the entries it points to. Its effective confidence is bounded by (or derived from) the freshness of its outbound pointer targets. A map pointing mostly at stale entries is itself surfaced as stale, even if its prose was edited recently.
2. **Re-validation prompts on access.** When a map is pulled and it (or its pointer set) is past a freshness threshold, the agent is prompted to confirm the map still reflects reality before relying on it — and to update it if not. Maps are the natural place to spend a re-validation turn because they're cheap and high-leverage.
3. **Pointer-rot detection.** When a pointer target is deactivated or superseded (the graph already tracks both), the map carries a flag: *"this map points at kb-00422, which has been superseded by kb-00510."* Surfaced as a `[POINTER-ROT]`-style badge, analogous to the existing `[CONFLICTING]` / `[EXPIRES]` badges, so a human or agent can repair the map.

These compose: inheritance sets baseline freshness, rot detection catches structural breakage, re-validation is the human/agent repair loop.

## 4. Desired outcome

An agent (any harness, via MCP) dropped into an unfamiliar-to-it system gets the same "I don't know, but I know where to look" orientation a human who's worked on it would have — without the human having to re-explain the system, and without burning the context window on full detail. Specifically:

- On session start / scope shift, the agent sees a thin directory of what exists in this system and where the details live.
- It pulls only the maps it needs, and from there only the detail entries it needs.
- The map compounds over time and travels across agent tools, the same way the rest of the KB does.
- Maps don't silently rot into confident-but-wrong pointers.

## 5. Open questions for the adversarial debate

These are the seams most likely to fail. The debate should attack them directly.

1. **Is `mental_map`-as-`entry_type` actually sufficient,** or does the index-surfacing + nesting behavior need first-class support that a mere entry type can't carry cleanly? (Starting position: entry type is enough. Break it.)
2. **What exactly is the size cap, and is "structural" enforcement real or theater?** A char cap is crude. Is there a sharper invariant (e.g. "every assertion in a map body must resolve to an outbound edge")? Can a linter detect fact-smuggling without false positives that frustrate authors?
3. **Freshness inheritance math.** If a map's freshness derives from its pointers, what's the function — min (stalest-pointer), mean, weighted? Does inheritance interact badly with the existing per-type decay half-lives? Could a map get stuck permanently stale because of one rarely-accessed leaf?
4. **Index relevance without per-turn LLM cost.** `UserPromptSubmit` runs every turn under a 30s budget. How does the index lookup pick the right scope cheaply? Pure keyword/FTS? A tiny classifier? When does it stay silent to avoid noise?
5. **Context-injection noise.** Surfacing an index every scope change risks training the agent (and annoying the user) to ignore it. What's the suppression policy — only on scope *change*, only when maps exist, dedup against what's already in context (the original "greedily inject maps not already in context" idea)?
6. **The `SessionStart` MCP-not-connected race.** Docs warn MCP servers may not be connected when `SessionStart` fires. Does the index push need a fallback (retry on first `UserPromptSubmit`, or a cached index on disk)?
7. **Who maintains agent-authored maps, and how do we prevent map sprawl** — dozens of overlapping half-maps? Does map authoring need the same dedup discipline `kb_ingest` has for entries?
8. **Auto-synthesis trust boundary.** If a synthesized draft map is wrong, it's a confident-but-wrong pointer at scale. What promotion gate keeps synthesized maps from being trusted prematurely?
9. **Does the three-tier nesting hold for real systems,** or do systems have messier topologies (cross-cutting concerns, components shared across systems) that a strict `contains` tree can't represent? Should edges be a DAG, not a tree?

## 6. Explicitly out of scope (for v1)

- Auto-synthesis beyond a clearly-marked draft mechanism (own track).
- Cross-system shared-component topology (revisit after the tree-vs-DAG question is settled).
- Any change to how detail entries themselves are stored or decayed.

---

## 7. Settled design (post-debate, 2026-06-03)

A 4-angle × 2-round adversarial debate (Minimalist / Knowledge Architect / Retrieval-Harness Engineer / Lifecycle Steward), each grounded in the actual code, ran against §§1–6. This section records the resolution. Jason made the two irreducible calls (§7.6).

### 7.1 Factual corrections to the starting positions

All four agents independently verified these against the repo. The starting positions were wrong on each:

1. **No `[CONFLICTING]` badge exists.** §3.6 justified `[POINTER-ROT]` as "analogous to the existing `[CONFLICTING]`/`[EXPIRES]` badges." Only `[EXPIRED]`/`[EXPIRES Nd]` render (`formatters.py`). Conflict detection is unbuilt (ROADMAP "Later", line 41). The precedent is a phantom — `[POINTER-ROT]` would be net-new, not a parallel render.
2. **A new `entry_type` is not free pipeline reuse.** `HALF_LIVES[entry_type]` (`confidence/decay.py:56`) is hard-keyed with no default → a new enum member **KeyErrors on every decay call**. `kb_preflight` hardcodes `entry_type IN ('decision','lesson_learned')` and `= 'pattern_convention'` (`preflight.py:48,64`) → a new type is **invisible to the index surface (the headline use case)** until those WHERE clauses are edited. §3.1's "reuses the entire pipeline" omits these.
3. **`contains`/`part_of` are not new edge types.** `graph/builder.py::_add_edge` persists arbitrary `edge_type` strings; `graph/queries.py` traverses arbitrary edge sets. The substrate is **already a DAG**. Zero schema work — and zero enforcement/traversal semantics. §3.3 oversells them both directions.
4. **CWD scope detection was already tried, failed, and removed** ("MCP server CWD unreliable" — ROADMAP:47, `how_it_works.md:430`). §3.5's `CwdChanged` trigger resurrects the dead signal. CWD is at most a low-weight hint.
5. **`kb_store` has zero dedup** (dedup lives only in `ingest/dedup_agent.py`, called only from `kb_ingest`). "Agents author maps while they still have context" → **guaranteed overlapping half-maps on day one** (Q7), not a someday-risk.
6. **The decay self-heal trap.** Decay anchors to `max(created_at, last_accessed)` and `touch_accessed` bumps `last_accessed` on every read. A map's freshness therefore tracks **read frequency, not correctness** — a constantly-surfaced map stays "fresh" while its pointers rot; a correct-but-cold map trips stale. This is backwards for an orientation node and reframes the whole staleness design (§7.4).

### 7.2 Structure — new `mental_map` entry_type (Jason's call)

A new `mental_map` `EntryType`, reusing storage/versioning/FTS/edges. It earns enum identity (over a `factual_reference` tagged `map`) as the clean dispatch home for (a) a decoupled freshness policy and (b) the fact-free lint — a tagged `factual_reference` would decay on the wrong 90-day clock and is *allowed* to contain retrievable values, the one thing a map must never do.

**Same-commit guard rails (mandatory — without these the type crashes or is invisible):**
- `HALF_LIVES.get(entry_type, default)` (or special-case maps out of value-decay entirely, per §7.4) so a new member can never KeyError `confidence/decay.py`.
- Add `mental_map` to `kb_preflight`'s hardcoded `entry_type` WHERE clauses (`preflight.py:48,64`) or maps never surface in the primer.

### 7.3 The fact-free invariant — "no retrievable value in assertion position"

The starting position's "every assertion resolves to an outbound edge" is **falsified by its own example**: *"the pipeline has three stages"* is a structural claim with no edge target, yet it can rot. Replace it with the sharper, enforceable boundary:

> A map body may assert **structure and relationships** (counts of parts, "X calls Y", "look in Z") and **pointers**. It may **not** contain a **retrievable value** — a port, threshold, config string, signature. Those follow a pointer to a `factual_reference`.

Discriminator: *would a reader act on this number/string directly, or follow it to a source?* Counts-of-parts are pointers-in-disguise (they tell you how many edges to expect); config values are facts. Enforce as a **soft lint heuristic** (numeral/identifier in assertion position), not a graph invariant. The ~1500-char cap is demoted to an **advisory proxy**, not the definition. The required-outbound-pointer rule stays (a map with zero pointers is an orphan note).

### 7.4 Freshness = pointer-validity, not value-decay

Because a map holds no retrievable values of its own (§7.3), value-decay is meaningless for it, and the read-frequency self-heal (§7.1.6) makes a decay clock actively misleading. So:

- **Exempt `mental_map` from value-decay** (the `.get()` default returns no-decay / effectively-infinite half-life).
- **Stale = a pointer no longer resolves.** On GET, resolve the map's outbound edges and surface inline any target that is `superseded_by` or deactivated — reusing the supersedes edges already in `graph/queries.py:252-269`. This is the **on-GET check**, not a new always-on badge subsystem.
- **Cut freshness-inheritance entirely.** The `min()`/stalest-pointer function manufactures the prespec's own nightmare: one cold 90-day-half-life leaf pins a map below the 0.5 threshold permanently. It fights the existing model.
- **Acceptance test:** reading a map keeps its effective state above threshold independent of leaf freshness (proves no permanent-staleness trap).

### 7.5 Topology — DAG, not tree

Ship `contains`/`part_of` as ordinary free-form edges (zero code). **Specify the model as a DAG:** a map node may have multiple `contains`-parents (shared component) and orient multiple components (cross-cutting concern like auth). Do **not** encode a single-parent tree constraint — it forces a falsehood on day one. Named three-tier nesting is *not* enforced in v1: ship flat-with-DAG-edges; add tier semantics only when a real map overflows one screen and demonstrably needs them.

**Content bias (hard authoring rule):** maps should hold the **non-re-derivable** — cross-repo wiring, infra/runtime state, "why" rationale. An agent can re-`grep` local code a human can't, so restating-the-repo prose is low-value *and* a sprawl magnet. Note the Steward's sharpening: non-re-derivable content is exactly the content with **no self-correcting feedback loop** (a stale infra pointer is invisible until it misleads) — which is the strongest reason for human review and a low map count (§7.6).

### 7.6 Authoring & sprawl — human-only for v1

- **v1 = human-authored maps only.** No agent auto-authoring, no auto-synthesis in *any* phase. `kb_store` has no dedup gate, so capping authors caps sprawl for free.
- **Before any agent-authored or synthesized maps:** build a store-time dedup gate for `mental_map` (hybrid-search same `project_ref`/scope on store, refuse-or-update if a same-component map exists) with a **named reconciliation owner** (update-existing vs create-new must be a decision, not an accident).
- **Auto-synthesis is cut from every v1 phase** — not deferred as a "stretch track." A wrong synthesized draft is a confident-wrong pointer at scope-scale, the highest-blast-radius failure. It gets its own future spec and its own debate.

### 7.7 Surfacing — pull now, push as packaged opt-in CLI hook tools (Jason's call)

Both halves ship, but the labor splits the way the harness constraints force.

**Pull (in-process, deterministic, v1 first):** extend `kb_preflight` to return a tag/type-filtered **Maps index** for an explicit `project_ref` — a few lines, in-process, testable against the eval corpus. The agent, seeing the index, calls `kb_get` (or `kb_map_get`) to pull full maps it judges relevant, then traverses edges to detail entries.

**Push (packaged CLI hook tools — Jason's reframe):** the genuine cold-start gap is real and pull *cannot* close it — an agent can't request the auth map for a subsystem it doesn't know exists. But the push surface is **not** in-MCP-server hooks and **not** bespoke per-user `settings.json` glue. It's a small set of **CLI entrypoints shipped in the package** (`uv tool install`-able), which users wire into whatever harness hooks they have. Portable (any harness that can shell out), opt-in, documented. Rationale: a new entry_type changes what's *storable*; the push is what changes whether agents *proactively orient* — which is the actual observed gap.

The CLI hook tools must obey the constraints the debate pinned down:
- **Inject a factual, directory-style string** ("Maps for this system: [kb-00310] Ingestion — chunking, dedup, extraction. …"), **never imperative** — imperative phrasing trips prompt-injection defenses and gets surfaced to the user instead of used. (A hook can only inject a *string*; it cannot force a tool call — which is *why* push delivers an index and pull is the agent's choice.)
- **Scope via FTS keyword match on map titles** — no embeddings (Ollama hop has fallback gaps), no per-turn LLM classifier. **Not CWD as anchor** (§7.1.4); CWD is at most a low-weight hint.
- **Stay silent** unless: maps exist for the resolved scope ∧ the scope *changed* ∧ the map IDs aren't already in context.
- **Cross-turn suppression via a per-session scratch file** (`~/.cache/personal_kb/injected-<session_id>`: surfaced map IDs + last resolved scope). Hooks are stateless between turns; without this file "only on change" is unimplementable.
- **Read a denormalized on-disk maps index** that the MCP server writes on store/update, so the hook never needs a live MCP call — sidestepping the SessionStart-MCP-not-connected race. First `UserPromptSubmit` re-attempts as fallback.
- **Push only the non-re-derivable maps** (§7.5). Local-code-structure maps are pull-on-demand at most — never pushed — which is the single biggest lever on the per-turn noise budget.

### 7.8 v1 scope checklist

- [ ] `mental_map` `EntryType` + same-commit guards: `HALF_LIVES` default, `kb_preflight` WHERE clauses.
- [ ] Maps exempt from value-decay; on-GET pointer-validity check (reuse supersedes edges).
- [ ] `contains`/`part_of` as free-form DAG edges (multi-parent allowed); required-outbound-pointer validation; no enforced tier nesting.
- [ ] Fact-free **advisory** lint (retrievable-value-in-assertion heuristic); char cap advisory.
- [ ] Human-authored only; auto-synthesis and agent auto-authoring out until a store-time dedup gate exists.
- [ ] `kb_preflight` Maps-index (pull) — ship + eval first.
- [ ] Packaged CLI hook tools (push): FTS-title scope, per-session scratch-file suppression, on-disk index, factual-not-imperative injection, non-re-derivable-only.
- [ ] Acceptance test: a map cannot get stuck permanently stale.

### 7.9 Still open (not blocking v1)

- **Tier nesting semantics** — ship flat-with-DAG-edges; revisit named tiers when a real map overflows.
- **Auto-synthesis** — its own future spec + debate; cut from all v1 phases.
- **Per-turn noise tolerance** — only Jason's lived judgment after dogfooding can set the acceptable injection rate; the suppression gate is built tight by default.
