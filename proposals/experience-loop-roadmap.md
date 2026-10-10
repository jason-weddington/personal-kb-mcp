# Experience loop: status and roadmap

Living document, last updated 2026-10-10. Tracking item: GTD 45e54c5b. This is the place to resume from: what has shipped toward "an agent that learns from experience", what the evals say, and what comes next, in order. Every open item carries a GTD id on the Personal-KB board (or the kb-bench board where noted). The design sources are `experience-loop.md` (Jason's ideas and the amendment that the KB is an agent tool) and `experience-loop-debate-synthesis.md` (the multi-angle debate); where they disagree with this file, this file reflects later decisions.

## Where we are

The loop is notice, write, correct, deliver, measure.

Correct (v1.1.0): corrections are first-class. `kb_store` requires `supersedes`, the superseded_by invariant is maintained, a near-duplicate create returns 409, updates need a change_reason, map pointers cannot be superseded, and reads hide superseded entries.

Deliver (v1.2.0 and v1.3.0): resolutions (`hints.resolution`: corrected fact, wrong belief, cue, scope, provenance), the session-start gotcha slice, the PreToolUse soft gate (since v1.3.0 it matches every segment of a compound command), the session-start tool inventory, and listener fixes. `KB_GOTCHA_SLICE=0` turns the slice off while keeping the gate armed, for evals.

Measure: kb-bench (Harbor scenarios with a seeded offline KB per attempt, arms compared pairwise, single-session and two-session learning trials), the replay harness in `scripts/replay/`, and the failure-cue index that records every failed tool call with a normalised cue.

Notice and write (on main, unreleased as of this update): surprise capture. The hook sends a turn digest at Stop when the server's `KB_SURPRISE_CAPTURE` is shadow or on. kb-service stores digests (`turn_events`, secrets redacted) and detects three shapes: (1) a failed Bash command later corrected in the same session, deterministic; (2) the user's next prompt corrects the previous assistant turn, one model call; (3) a tool output contradicts a claim earlier in the turn, one model call. In mode on, a distiller writes autonomous, observed resolutions and merges repeat sightings by bumping `observed_sessions`. Observed-once captures reach the slice with a hedged label and become gate-eligible at two sightings. The default is off. `POST /api/kb/surprise/drain` processes everything pending synchronously, for evals. Detector eval harness: `scripts/surprise_eval/`.

## What the evals say so far

Single-session kb-bench on Opus (k=3): the slice rescues all four trapping scenarios (caddy-apt, ollama-num-ctx, stop-hook-transcript, wireguard-reachability) from a near-zero KB-off baseline. add-remote and github-push do not trap Opus and serve as regression checks.

Soft gate on its own (gate_only arm): it rescues the two scenarios whose trap is a Bash command, 3/3 each from 0/3, and cannot help the two whose trap is a belief or a config edit. On top of the slice it adds nothing in short sessions and costs one turn.

Transcript mining (three machines, about 30 days, 1,910 records, private data in the evals repo): 345 wrong-assumption episodes (129 user corrections, 216 contradictions by tool output), 61 recurring lesson clusters. In 5 of the 8 top episodes, verified against the source transcripts, the KB already held the corrected fact before the episode. That is a delivery failure, not a capture failure.

Detector eval (833-case stratified sample of the mined set, silver labels from a Sonnet judge, confidence floor 0.7):

- Shape 2 with Sonnet 5.5: precision 0.81 on the sample (about 0.69 reweighted to the population), recall 0.48. Before the confidence floor recall is 0.79, so many true positives sit at confidence 0.5 to 0.62.
- Shape 2 with Opus 5.5: precision 0.86, recall 0.51, but Opus returns `stop_reason: refusal` with empty content on about 20% of these benign prompts.
- Shape 3: recall 0.08 to 0.09 on both models. 141 of 439 cases are out of the detector's scope (no assistant claim before the tool result in the window), and the model says no on most of the rest.

Two-session learning trials (session 1 makes the mistake, gets corrected or recovers, and the loop captures it; session 2 is a fresh agent in a reset container facing the same trap; learn_off has the same hooks with capture off). Session-2 passes out of 3, learn_off → learn_on:

| scenario | shape | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| ollama-num-ctx-learn | 2 | 0 → 3 | 0 → 1 |
| stop-hook-transcript-learn | 2 | 0 → 3 | 1 → 3 |
| caddy-apt-learn | 2 | 0 → 0 | 3 → 3 |
| wireguard-reachability-learn | 2 | 0 → 0 | 0 → 0 |
| review-push-learn | 1 | 0 → 0 | 0 → 0 |

What the table shows:

- Learning works for shape 2: a user correction in session 1 became a delivered resolution and session 2 avoided the trap (Sonnet 6/6 on the two scenarios where it captured, Opus up from 1 to 4 of 6).
- A delivered slice lesson does not stop a habitual command. On review-push-learn with Sonnet, the shape-1 lesson was written and delivered in all three trials and session 2 still made the rejected push first every time. This motivates making shape-1 and shape-2 captures gate-eligible at first sighting (step 3).
- Shape-1 detection had the first-segment bug: Opus chained `git show ...; git push origin main`, which was classified as `git show`, so no lesson was written. Fix dispatched as GTD f3d06c8d.
- Opus often suspected the problem itself in session 1 (for example, the truncated context on ollama), so the user's follow-up confirmed rather than corrected and nothing was captured, yet a fresh Opus fell for the trap again in session 2. Knowledge the agent arrives at on its own is lost; that is the case for automatic capture beyond corrections.
- caddy-apt-learn does not trap Opus at all, and wireguard-reachability-learn is not rescued by either model; check its capture and lesson quality before reading more into it.

Jobs: learn-matrix-opus-5-5-k3 and learn-matrix-sonnet-5-5-k3 in the kb-bench jobs dir; detector eval in the private evals repo (surprise/run-2026-10-10).

## Next, in order

1. **Ship and enable for Jason's KB** (GTD cd878c09). Release with surprise capture default off on every instance, then set `KB_SURPRISE_CAPTURE=on` on Jason's personal KB only. No shadow-only phase: the offline detector eval already measures precision on his own transcripts, so read cost as it runs instead. The other instances stay off until his has run for a while.
2. **Detector cost, Jason's call** (GTD 0dbe49a3). Up to two Sonnet calls per turn (shape 2 on turns that start with a human prompt, shape 3 on every turn). Turn the eval's per-call token counts into dollars per day at real volume; the levers are a cheaper detector model (Haiku, or Clef locally once the GPU is free) and the shape-2 confidence floor, both measured by the eval harness for their recall cost.
3. **Gate-eligible first sightings for shapes 1 and 2** (GTD 03ce88c7, dispatched with the shape-1 fix f3d06c8d). A shape-1 capture is detected deterministically from tool output and a shape-2 capture comes from the user's own words, so both become gate-eligible at the first sighting; shape 3, a model judgment, keeps waiting for a second sighting. The two-session matrix showed a delivered slice lesson failing to stop a habitual wrong push 3/3.
4. **Headless fleet digests** (GTD 60cd0a1c). Enable the turn-digest hook for the dispatch user on the fleet, so headless runs learn shapes 1 and 3 (they never receive a human correction).
5. **Relevance-aware delivery** (GTD 10d811fa). The slice is the 20 newest corrections per project, blind to the task, and it will crowd out as autonomous captures accumulate. The mined "the KB knew but did not deliver" episodes are the eval that must show lexical and recency delivery falling short before a better matcher is built.
6. **Somnus consolidation of autonomous captures** (GTD 5399b5b0; overlaps fb8a4f52). Nightly: merge duplicate lessons, retire ones that stop being true (the debate's use-based strength: a success after delivery strengthens, the same failure after delivery weakens and flags), and promote recurring ones. This extends what somnus already does for maps and overlaps the existing inconsistency item.

## Remaining roadmap after that

Items from the experience-loop plan and the debate that are not covered above, with their state:

- **Shape-3 detector recall** (GTD 13079367). At 0.09 it is not yet useful; widen the window the detector sees, revisit the scope rule, and check the silver labels before tuning the prompt. Mined scenarios are in a kb-bench rollout (efe29fbc): gateway-bearer-learn, uv-stale-install-learn and tail-masks-failure-learn (shape 3) and removed-remote-learn (shape 2).
- **Shape-2 confidence floor** (GTD 65bb7c2b). Re-run the detector eval at floors 0.5 and 0.6 and pick the floor from the precision and recall trade, after confirming whether the silver labels undercount real corrections.
- **Episodes on failure, the debate's first bet** (GTD c5c57405). On PostToolUseFailure, deliver the matching resolution as same-turn context. The failure-cue index already records failures; delivery is not built.
- **Automatic capture beyond corrections** (GTD 78a2997a). A System 1 check per turn for new facts, decisions with their reasons and gotchas, written by a System 2 model or somnus. Surprise capture covers only corrections.
- **Write-time conflict detection** (GTD 87cafc8a). On every store, classify the new entry against its neighbours as supports, refines, conflicts or unrelated, and turn conflicts into supersedes edges or flags.
- **Decision triggers** (GTD b43b65a6). Deliver a recorded decision when an agent is about to reverse or relitigate it.
- **Harness-agnostic delivery** (GTD 506d0b53). A KB delivery channel inside Talos and a native-memory sync (AGENTS.md or equivalent) for harnesses other than Claude Code.
- **Listener and roster decision** (GTD c2d96250). Production consumption of whispers and roster pushes is near zero; refetch the consumption detector once, then switch off or redesign those channels.
- **Repeat-mistake metric in production** (GTD a3a7b307). A weekly cross-session repeat rate per cue from `failure_events`, cut by harness and machine, so the north star is read from real traffic. Related: L4 search telemetry (GTD cf855f71).
- **Deferred until an eval demands it** (GTD 30ff4dc2, someday): a first-class resolutions table (triggers: the gate proves its value in long sessions, or autonomous captures need their own lifecycle), a semantic gate matcher (the current lexical matcher also false-matches heredoc body lines), a long-session gate scenario, and the one-event-stream sidecar (wait for a third consumer).
- **Small** (GTD 2f89e8f2): log the provider's stop_reason when a response has no text block, so a refusal is distinguishable from an empty answer.
