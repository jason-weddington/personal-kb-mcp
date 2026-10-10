# Eval-coverage audit: which KB memory features have proof

2026-10-10. Tracking item: GTD 758a3cd4. Asked for by Jason before any more net-new memory feature work: audit what kb-bench actually covers, find features with no evidence and overlapping features, then expand kb-bench. Sources: three read-only research passes (kb-bench and eval assets, the feature inventory in code, and the KB's efficacy record), with every claim cited there to file:line or KB id; the lead spot-checked the ones acted on below.

## Conclusion

Only three features have controlled evidence that they change agent behaviour: the gotcha slice, the soft gate (Bash-shaped traps only), and surprise capture (mostly shape 2, with a scripted best-case correction). Nothing else in the memory and delivery space has any, and the two oldest push channels, the maps roster and the listener whisper, have negative production telemetry: 9 of 531 interactive roster pushes consumed and 0 of 5,797 headless (2026-10-07), and 0 of 20 whispers consumed over 14 days.

The slice result is about the fact being in context, not about the KB channel. The native_memory arm (the same seed rendered into Claude Code's MEMORY.md) scored 3/3 in every cell where the slice did. So the KB's distinct value is not how it formats a delivery; it is getting the right fact into the store in the first place (capture) and carrying it across harnesses, models and machines. That matches the mining result: the KB held the needed fact beforehand in only 14% of 90 real episodes, and none of those 13 facts was a resolution, the only kind the slice and gate deliver (kb-03676).

kb-bench cannot see the older features as built. Every seed holds 0 or 1 entry and no mental_map, the listener is forced off, UserPromptSubmit is never wired, no arm has an MCP server, failure context is never enabled, and every trial is a short `--print` session. The bench measures the resolution channel in an otherwise empty KB.

## Evidence by feature

Grades: A = a kb-bench arm isolates it; B = offline component metric or replay; C = production telemetry without a control; D = design rationale or anecdote only.

| feature | channel | grade | key numbers | kb-bench coverage today |
|---|---|---|---|---|
| Gotcha slice | SessionStart, content | A | Opus k=3: kb_off 1/12 → slice 12/12 on four scenarios; add-remote 0/10 → 10/10 (kb-03669, kb-03659) | yes; but native_memory ties it everywhere |
| Soft gate | PreToolUse Bash deny | A for Bash traps | gate_only rescued 6 of 12, regressed 0; on top of the slice it added nothing and cost a turn (kb-03673); lexical matcher P 0.78 R 0.36 on 1,677 real commands (kb-03721) | yes; rate limits and re-arm postdate the images |
| Surprise capture | Stop digests, server detectors, distiller, critic | A (shape 2), B, C | two-session learn_off → learn_on rescued 3-6 of 15 per job, 0 regressed; qwen 0/8 → 7/8 (kb-03710); lesson precision 14/19 good with the Opus critic (kb-03709); shadow 2026-10-10: 3 of 3 would-writes correct | yes; critic, TTL and lesson classes postdate the images; no kb_off two-session baseline |
| Tool inventory | SessionStart, content | A (one scenario) | inventory_only rescued add-remote 10/10; 1/3 to 2/3 elsewhere | partly; it rides along in the slice arms, so "slice" is slice plus inventory |
| Failure context | PostToolUseFailure, content | D | shipped and enabled in v1.5.0 with no isolating eval | never enabled in any arm (KB_FAILURE_CONTEXT unset; the full arm wires the hook async and record-only) |
| Maps roster and delta | SessionStart, UserPromptSubmit, pointers | C (negative) | 9/531 interactive and 0/5,797 headless consumed; about 37 of 38 pushed maps out of domain (kb-02938) | none: no seed contains a map |
| Mental maps in general | pull and push | D for behaviour, B for retrieval ranking | 3.8% coverage before somnus; anecdotes only (kb-02937) | none |
| Listener whisper | Stop computes, UserPromptSubmit delivers, pointers | B for gate precision, C (negative) for use | gate P 0.71-0.80 on n=17-30; 0 of 20 whispers consumed in 14 days; about 150 Sonnet calls a day at the last measurement | none: forced off, and single-turn trials have no next prompt |
| kb_preflight | MCP pull | D | no usage telemetry or eval | none: no MCP in the bench |
| kb_search, kb_ask, kb_summarize | MCP pull | B | search MRR 0.91 NDCG 0.93; agent baseline 1.00 on 13 queries | none |
| somnus nightly maps | background job | D | output counts only; its own kill conditions (new maps retrieved, jury acceptance, consumption) have not been checked since it went live 2026-10-07 | none |
| Supersession, near-duplicate guard | write-time | B for calibration, D for behaviour | 0.88 floor calibrated on 339 creates; the supersedes replay harness exists but has never been run | none |

## Overlaps

These are the places where features do the same job through different channels, which is the cruft risk Jason named.

1. **Which maps are relevant, four ways.** The SessionStart roster, the UserPromptSubmit new-maps delta, the listener whisper and kb_preflight's Maps section all deliver titles of the same mental_map rows. Only the roster and listener are measured, and both are near zero.
2. **The same corrected fact, up to four times a session.** The slice at SessionStart, the gate before the call, failure context after it, and kb_preflight's Recent list. There is no cross-channel dedup, and in kb-bench the gate added nothing once the slice had delivered the same fact.
3. **Write-time hygiene split by caller.** Required `supersedes` lives only in the MCP schema, the near-duplicate 409 only in the HTTP store route, the distiller runs its own duplicate check, and ingest has its own dedup agent.
4. **Five staleness lifecycles that disagree.** Confidence decay, TTL, superseded_by, the surprise 30-day expiry, and the staleness warning. Concrete defect found: kb_preflight's Recent, Conventions and Related lists do not filter expired entries, so expired autonomous lessons leak into preflight (fix: GTD f2331adc).
5. **Three pipelines over the same Bash failures.** Surprise shape 1 (turn digests), failure_events (repeat-rate metric) and failure context (the gate index) each read the same incidents separately.
6. **Measurement in eight places, with holes.** The slice has no delivery or consumption record at all; whisper "consumed" counts only a kb_get, never a kb_search or an agent acting on a title; the tool inventory has no consumption signal.

## kb-bench expansion, in order

Each step names the result that would retire a feature, so none of these is a one-branch gate.

1. **Bench hygiene.** Stamp the exact wheel git sha into every report (three different dev builds reported as 1.3.0-1.5.0), add a kb_off arm to two-session jobs, and rebuild images on current main so the critic, TTL, rate limits and re-arm are what gets measured.
2. **Busy-KB seeds.** A seed generator that surrounds each trap fact with hundreds of realistic entries: distractors, maps shaped like somnus output, supersession chains, expired entries. Also a variant where the trap fact is a plain entry rather than a resolution, which is what production looks like. Without this the roster, maps, listener, preflight, slice caps and project precedence have nothing to do.
3. **Pull arm.** MCP server plus the standard steering ("search before acting"), no push. Answers whether pull works at all, and whether SessionStart push is needed when it does.
4. **Maps arms.** Roster-only and preflight-maps arms on busy seeds where the trap fact is reachable only through a map pointer, with and without somnus-written maps. If maps rescue nothing, the roster push and somnus retire and maps stay as pull-only orientation; if they rescue, map quality gets funded.
5. **Listener arm.** A multi-turn scenario (follow-ups already exist) with UserPromptSubmit and Stop wired, the listener enabled with a daemon key, and the trap reached on turn two or later. If it rescues nothing, the listener retires.
6. **Failure-context arm.** Synchronous PostToolUseFailure with KB_FAILURE_CONTEXT=1 on failure-shaped traps, against gate_only and slice.
7. **KB learning vs harness-native learning.** Two-session trials with Claude Code's own memory on and the KB off, against learn_on, and a session 2 on a different harness or model (the qwen lane now, talos once wired). This is the north-star comparison: does the KB beat harness memory, and does it carry across harnesses and machines.
8. **Long session.** A scenario that compacts mid-session, to test gate and failure-context re-arming and slice re-delivery (absorbs the deferred long-session item in GTD 30ff4dc2).

## Held until the results land

- The harness-memory up-sync (GTD ae89d2ea): steps 2 and 7 measure whether memory copies in the KB starve surprise capture.
- Standing guidance (GTD ce2cd311): steps 3 and 4 show where pushed content belongs relative to the roster and slice.

## Decisions for Jason

1. Switch the listener off on Jason's KB now, and let step 5 decide whether it comes back. It costs Sonnet calls on every turn that passes retrieval and has had 0 whispers consumed in 14 days. Recommendation: yes; the code stays and re-enabling is one env line.
2. Drop the roster's "Maps in other domains" line now, keeping own-project maps until step 4. About 37 of 38 pushed titles are out of domain, it costs context in every session including every headless run, and no push from it has been measured as consumed. Recommendation: yes.
