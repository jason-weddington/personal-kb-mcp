# Experience loop: status and roadmap

Living document, last updated 2026-10-10. Tracking item: GTD 45e54c5b. This is the place to resume from: what has shipped toward "an agent that learns from experience", what the evals say, and what comes next, in order. Every open item carries a GTD id on the Personal-KB board (or the kb-bench board where noted). The design sources are `experience-loop.md` (Jason's ideas and the amendment that the KB is an agent tool) and `experience-loop-debate-synthesis.md` (the multi-angle debate); where they disagree with this file, this file reflects later decisions.

## Where we are

The loop is notice, write, correct, deliver, measure.

Correct (v1.1.0): corrections are first-class. `kb_store` requires `supersedes`, the superseded_by invariant is maintained, a near-duplicate create returns 409, updates need a change_reason, map pointers cannot be superseded, and reads hide superseded entries.

Deliver (v1.2.0 and v1.3.0): resolutions (`hints.resolution`: corrected fact, wrong belief, cue, scope, provenance), the session-start gotcha slice, the PreToolUse soft gate (since v1.3.0 it matches every segment of a compound command), the session-start tool inventory, and listener fixes. `KB_GOTCHA_SLICE=0` turns the slice off while keeping the gate armed, for evals.

Measure: kb-bench (Harbor scenarios with a seeded offline KB per attempt, arms compared pairwise, single-session and two-session learning trials), the replay harness in `scripts/replay/`, and the failure-cue index that records every failed tool call with a normalised cue.

Notice and write (released in v1.4.0, enabled on Jason's KB 2026-10-10 with the soft gate live): surprise capture. The hook sends a turn digest at Stop when the server's `KB_SURPRISE_CAPTURE` is shadow or on. kb-service stores digests (`turn_events`, secrets redacted) and detects three shapes: (1) a failed Bash command later corrected in the same session, deterministic; (2) the user's next prompt corrects the previous assistant turn, one model call; (3) a tool output contradicts a claim earlier in the turn, one model call. In mode on, a distiller writes autonomous, observed resolutions and merges repeat sightings by bumping `observed_sessions`. Observed-once captures reach the slice with a hedged label and become gate-eligible at two sightings. The default is off. `POST /api/kb/surprise/drain` processes everything pending synchronously, for evals. Detector eval harness: `scripts/surprise_eval/`.

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

Second round, on v1.4.0 (shape-1 compound fix and first-sighting gate eligibility in), four scenarios mined from Jason's transcripts plus review-push-learn again. Session-2 passes out of 3, learn_off → learn_on:

| scenario | shape | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| review-push-learn | 1 | 0 → 0 | 0 → 3 |
| removed-remote-learn | 2 | 0 → 3 | 0 → 2 |
| uv-stale-install-learn | 3 | 0 → 0 | 2 → 3 |
| gateway-bearer-learn | 3 | 0 → 0 | 0 → 0 |
| tail-masks-failure-learn | 3 | 3 → 3 | 3 → 3 |

- Opus review-push went 0 → 3: the shape-1 fix plus the gate catch a habitual wrong push that the slice alone could not.
- Sonnet got the same correct deny 3/3 and retried the identical command each time. The deny reason still said "observed once, unconfirmed", contradicting the step-3 decision that first-sighting shape-1 and shape-2 captures are trusted. Fix dispatched as GTD 6fe0b2c7: trusted wording for shapes 1 and 2.
- Shape 2 keeps working on a mined episode (removed-remote-learn).
- Shape 3 as defined misses discovered knowledge. On gateway-bearer, Opus fixed the problem in session 1 and wrote "Root cause: ... That was wrong", but it never made a wrong claim for a tool output to contradict, so nothing was captured and session 2 fell for the trap. This matches the 0.09 detector recall; the redesign is GTD 13079367, overlapping 78a2997a.
- tail-masks-failure-learn does not trap either model and serves as a regression check.

Jobs: learn-matrix-*-k3 and learn-mined-*-k3 in the kb-bench jobs dir; detector eval in the private evals repo (surprise/run-2026-10-10).

## Overnight 2026-10-10 (v1.5.0 and v1.6.0)

First night with capture on, and the incident. Within about four hours the distiller wrote three autonomous lessons, all shape 1 from headless dispatch runs. One was false: a dispatch clone's momentary lint errors became "the repo has lint errors, run ruff only on touched files". Two had gate cues as broad as a whole command family ({uv run}, {git checkout}). The {uv run} cue denied an unrelated interactive command within minutes. All three were deactivated and capture went back to shadow (kb-03707). Fixes in v1.5.0:

- cue args_prefix is the shortest prefix of the failed command's arguments that differs from the fix's, and there is no cue when they cannot be told apart;
- the distiller rejects lessons rooted in the session's own in-progress state or task-specific narrowing;
- headless shape-1 lessons wait for a second sighting before gating.

The roadmap's "no shadow phase" call was wrong: the bench scenarios contain only the intended trap, so only real traffic shows the noise.

v1.6.0 makes shadow useful. It dry-runs the full distiller and cue pipeline into `surprise_dry_runs`, and `GET /api/kb/surprise/candidates` exposes candidates with their would-be outcome, so an agent can audit before capture goes back on. First shadow output: both headless shape-1 candidates were rejected as not durable (correctly: the session's own in-progress edits), and the one would-write was a correct shape-3 lesson from the lead session's own investigation.

Detector results (private evals repo, surprise/):

- Shape-3 redesign, v2: the trigger is the agent reaching a corrected understanding or root cause backed by tool output. On Sonnet, recall went from 0.08 to 0.41 at population precision 0.62 (v1: 0.66), with no cases out of scope.
- Per-shape confidence floor: shape 2 now defaults to 0.5, which moves recall from 0.48 to 0.76 for population precision 0.69 to 0.64; shape 3 stays at 0.7.

Delivery gap, measured on a random 90 durable mined episodes: the KB held the fact BEFORE the mistake in only 14%, held it after in 34%, and lacked it in 49% (about 9 of those are steering facts that belong in CLAUDE.md). None of the known-before facts was a correction or resolution, 10 of 13 sat under another project_ref, and most were one line in a run log. Capture, not delivery, is the bigger hole, which demotes step 5.

Clef as a System-1 prefilter (odin; context only, not a production dependency):

- clef-flash, shape 2, threshold 0.3: recall 0.74 while passing only 18% of turns to Sonnet.
- clef-flash, shape 3 (v1-style question): recall 0.69 at threshold 0.5, passing 31%.
- clef 27B is no better and about 3x slower.
- clef-flash cannot write the lesson text; it would only gate the Sonnet call.

Local extraction models on odin (shape 2, same 80-case stratified sample as the API models; latency under load from other jobs): qwen3.8:27b reached precision 0.86 and recall 0.60 on the raw sample at about 45 s per call. The other two models are recorded in the private repo once their runs finish.

Cost estimate for step 2. Assuming Sonnet-class pricing of $3/M input and $15/M output (check current pricing), a detector call is about 2-2.5k input tokens and under 100 output tokens, roughly $0.008-0.01. Interactive volume averages 15 prompts per active day (164 on the busiest), plus every tool-using turn for shape 3. A headless dispatch run is one turn. That comes to about $0.5-1 on a typical day and $5 on a heavy day. A clef-flash prefilter would cut shape-2 calls by about 80%.

Lesson precision, the number that decides whether capture goes back on. personal-kb-evals/surprise/replay_pipeline.py replays labelled mined cases through a local shadow-mode kb-service, and an Opus judge rates every lesson the pipeline would write against the source and current reality (kb-03709).

- v1.6.0: 39 would-write lessons, 19 good, 12 harmful. Shape-2 lessons overgeneralise narrow user corrections; shape-3 positives were 6/8 good.
- With a scope-faithful distiller and a critic pass (55676e7): 14 would-write lessons, 7 good, 6 harmful. Volume fell, but precision stayed at about 50%.
- Of the remaining harm, a third is time-staleness: true when captured, false after a later change. That argues for expiry (dispatched: a 30-day TTL unless re-observed, permanent at three sessions; the critic also rejects hedged user claims).
- Duplicates of existing entries cannot show in an empty-KB replay; production compares against the real KB.

Model-agnostic learning (kb-03710):

- Claude Code on local qwen3.8:27b (the 5090) learned from session-1 corrections in 7 of 8 two-session trials, against 0 of 8 with capture off.
- On shape-2 detection over the same 80 cases, local qwen models on odin match Sonnet (qwen3.8:27b-mtp: population recall 0.67 and precision 0.45, vs Sonnet's 0.67 and 0.43) at about 30x the latency.
- A fully local detector (clef-flash prefilter, then qwen) is quality-viable for the async worker. The distiller and critic were not evaluated locally.

Decisions for Jason:

1. The quality bar for turning autonomous capture back on. The options are: wait for better precision; turn it on with autonomous lessons delivered only as hedged slice context that never gates; or turn it on for shape 3 only, the best class.
2. The detector and distiller model: Sonnet API spend of roughly $0.5-5 a day, or a local path on odin.

Status of the six steps:

1. Release and enable: done (v1.4.0 to v1.6.0). On Jason's KB the soft gate is live and capture is in shadow; turning it back on is decision 1 above.
2. Cost: estimate above; the choice of detector model and prefilter is Jason's.
3. First-sighting gate trust: done, then narrowed after the incident. Interactive shape 1 and all shape 2 are trusted at first sighting with precise cues; headless shape 1 and shape 3 wait for a second sighting. Trusted lessons are delivered without the "unconfirmed" hedge (Sonnet ignored hedged denies).
4. Headless fleet digests: done. The dispatch user has hook 1.4.1+ with Stop wired on all four hosts, and headless digests are arriving.
5. Relevance-aware delivery: demoted by the delivery-gap result. If picked up, trigger at the moment of action with the assumption as the query, and drop project scoping.
6. Somnus consolidation: open, needed once autonomous lessons accumulate.

Also shipped overnight:

- delivery on tool failure: PostToolUseFailure returns the matching correction next to the failure (KB_FAILURE_CONTEXT=1, synchronous wiring; on in the synced settings);
- the weekly cross-session repeat-rate metric (`GET /api/kb/metrics/repeat-rate`, `kb-service metrics repeat-rate`);
- stop_reason logging on empty provider responses;
- the per-shape floor.

kb-bench qwen lane (Claude Code on qwen3.8:27b-256k, the 5090 now free): two-session trials on removed-remote, stop-hook, ollama-num-ctx and review-push, k=2, running overnight.

## Remaining roadmap after that

Items from the experience-loop plan and the debate that are not covered above, with their state:

- **Shape-3 detector recall** (GTD 13079367): done, v2 above. At 0.09 it is not yet useful; widen the window the detector sees, revisit the scope rule, and check the silver labels before tuning the prompt. Mined scenarios are in a kb-bench rollout (efe29fbc): gateway-bearer-learn, uv-stale-install-learn and tail-masks-failure-learn (shape 3) and removed-remote-learn (shape 2).
- **Shape-2 confidence floor** (GTD 65bb7c2b): done (0.5). Re-run the detector eval at floors 0.5 and 0.6 and pick the floor from the precision and recall trade, after confirming whether the silver labels undercount real corrections.
- **Episodes on failure, the debate's first bet** (GTD c5c57405): done (v1.5.0). On PostToolUseFailure, deliver the matching resolution as same-turn context. The failure-cue index already records failures; delivery is not built.
- **Automatic capture beyond corrections** (GTD 78a2997a). A System 1 check per turn for new facts, decisions with their reasons and gotchas, written by a System 2 model or somnus. Surprise capture covers only corrections.
- **Write-time conflict detection** (GTD 87cafc8a). On every store, classify the new entry against its neighbours as supports, refines, conflicts or unrelated, and turn conflicts into supersedes edges or flags.
- **Decision triggers** (GTD b43b65a6). Deliver a recorded decision when an agent is about to reverse or relitigate it.
- **Harness-agnostic delivery** (GTD 506d0b53). A KB delivery channel inside Talos and a native-memory sync (AGENTS.md or equivalent) for harnesses other than Claude Code.
- **Listener and roster decision** (GTD c2d96250). Production consumption of whispers and roster pushes is near zero; refetch the consumption detector once, then switch off or redesign those channels.
- **Repeat-mistake metric in production** (GTD a3a7b307): done (v1.5.0). A weekly cross-session repeat rate per cue from `failure_events`, cut by harness and machine, so the north star is read from real traffic. Related: L4 search telemetry (GTD cf855f71).
- **Deferred until an eval demands it** (GTD 30ff4dc2, someday): a first-class resolutions table (triggers: the gate proves its value in long sessions, or autonomous captures need their own lifecycle), a semantic gate matcher (the current lexical matcher also false-matches heredoc body lines), a long-session gate scenario, and the one-event-stream sidecar (wait for a third consumer).
- **Small** (GTD 2f89e8f2, done): log the provider's stop_reason when a response has no text block, so a refusal is distinguishable from an empty answer.
