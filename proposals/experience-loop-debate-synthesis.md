# Experience loop: moderator synthesis


> **Amendment 2 (Jason, 2026-10-07): repeat mistakes are days apart.** Recurrence within 5 tool calls measures in-session recovery, not learning. The primary metric is the **cross-session repeat rate**: for each failure cue, the share of later sessions (days apart, on any harness or machine) that hit the same cue after it was first captured and resolved. It is reported per cue, per week, by harness, with the baseline backfilled from historical transcripts and Talos run records. A cue that never recurs is a success.
>
> Consequences:
>
> - Failure-time delivery only fires after the repeat has happened. It is the **recovery** channel, measured separately by turns-to-recovery, and it is not the first bet.
> - The **prevention** channels become the first bet, both fed by the same cue index:
>   - the cue-keyed soft gate on PreToolUse (tool + target, deny once with the procedure attached);
>   - the session-start slice of known gotchas.
> - The live 20% holdout is dropped. At cross-session volume it would take months to reach a signal. The replay experiment (known mistakes, KB on vs off) is the causal test.

> **Amendment (Jason, 2026-10-07) — read this before anything below.** The KB is an agent tool. Jason never writes, reviews, confirms or curates entries; agents are his interface to the KB. Every "human-authored", "human-confirmed", "your confirmation", "your queue" or "review minutes" mechanism below is void. Read the provenance tiers as follows:
>
> - **Deliberate** replaces human-authored: an agent stored the entry because the session's human asked it to.
> - **Autonomous**: a hook, somnus or another unattended capture wrote the entry.
> - **Observed** means grounded in tool output; **asserted** means only claimed by an agent.
>
> Promotion and quality control are automated:
>
> - Recurrence in lineage-independent sessions.
> - Re-verification, where a later agent (or somnus) re-checks the claim against the primitive and reproduces the observation.
> - An agent-run audit: a sampled judge workflow that writes its precision estimate to the KB, with no human queue.
>
> "A machine entry never supersedes a human entry" becomes: **an autonomous capture may supersede a deliberate entry only with an observed contradiction**, meaning tool-output evidence carried as an event_id.

## 1. Verdict

- The north star stays the same: an agent makes a mistake once and does not repeat it, on any harness or machine. All four angles now judge each piece by whether it changes what the agent does next. "Consumed" is retired as a success metric.
- Delivery comes before more capture. With 20 whispers and 0 consumed, and 5,797 headless pushes and 0 consumed, the reading half is the bottleneck. The first build is delivery at the one moment agents demonstrably attend to: right after their own failure.
- Recall uses the same cue as encoding: a normalized failure signature, matched lexically. No model call runs on the hot path, and no model ever runs on PreToolUse.
- Machine memory is two-tier. Episodes are captured in shadow with checkable provenance (an adapter-stamped event_id). They become semantic entries only by recurrence or human confirmation. A machine-written entry never supersedes a human-authored one.
- The platform stays thin. Generalize the existing `/api/kb/listener` into `/api/kb/event`, keep the logic on the server, and keep adapters stdlib-only and fail-open. No sidecar, no Dogwood embed and no System 1 question library until a second shipped question proves its precision.

## 2. The first bet

**Failure-cued episodic delivery on the post_tool seam, shipped in Claude Code and Talos together.**

- **How it works.** On a tool failure, the adapter POSTs a `post_tool` event to `/api/kb/event`. The server normalizes the event into a cue_key: tool, target_class, and the error with paths, ids, line numbers and timestamps stripped, plus project and host_class. The server returns at most one flat corrective statement as same-turn context: the fix plus the wrong belief it replaces.
- **Holdout.** On 20% of eligible events, chosen by `hash(session_id + cue_key) mod 5`, the server withholds delivery. Destructive and outward-facing procedures (deploy, push) are exempt.
- **On a miss.** The server shadow-writes the episode and attaches the next successful same-target call in that session as a candidate resolution.
- **Instrumentation first.** p95 hook latency per event type, a per-(harness, mode) event-count heartbeat, and session_id, harness, engine and host on every event. These ship before delivery does.
- **Success:** after 300 treated events, same-cue re-failure within 5 calls is lower in the treated group than in the holdout, and turns-to-next-success is lower too.
- **Kill or redesign if any of these happen:**
  - treated is no better than holdout after 300 treated events;
  - delivered-then-same-failure is above 20%, which means cue keys are interfering and need narrowing;
  - p95 latency on failure lookup goes above the 800 ms hard timeout.

## 3. Settled design points

- No model call on PreToolUse. The soft gate is a lexical (tool, normalized target) match against a procedure index cached at session_start.
- The harness-native slice goes in as SessionStart additionalContext or the Talos system prompt. It is never written into CLAUDE.md, AGENTS.md or memory files, because those dirty dispatch clones or tie memory to one machine.
- Adapters stamp observable provenance (event_id, a result hash, harness, machine). The server rejects machine candidates that carry no event_id.
- At most one memory is delivered per cue (the fan-effect cap).
- Lineage exclusion: a delivered memory that the agent re-asserts never counts as corroboration of itself.
- A machine entry never supersedes a human-authored or human-confirmed entry without a human. Machine supersedes may target machine entries only.
- Every correction stores a `(wrong_belief, corrected_fact, evidence)` triple.
- Per-turn automatic capture (idea 1) is cut this cycle.
- Surprise capture is shadow-only and is not one raw-text question. A deterministic signal comes first, then a Clef `noul` question asked only against an explicit prior claim. It runs in Talos first, and in Claude Code at turn_end against `last_assistant_message`.
- Talos artifacts are keyed per run, so a re-dispatch stops erasing the evidence.
- Pairing, holdout assignment and per-session deny state all live on the server, keyed by session_id.
- The 26-supersedes replay runs alongside the build, not as a gate (the skeptic conceded this in round 2).

## 4. Near-convergences I resolve

- **Precision bar for delivering machine-derived episodes: 0.9, not 0.8.** The bar applies to a 50-item hand-labelled sample of resolution pairings. These are delivered as corrections, and the proposal's own principle is precision over recall wherever a correction is delivered. Weekly audits continue afterwards at 20 items.
- **Starting delivery corpus.** Weeks 1–4 deliver human-authored or human-confirmed entries matched to cue_keys. Observed episodes join once they clear the 0.9 bar, subject to irreducible #2 below. The bet's holdout can then run before any machine write is in the path.
- **Time-based TTL versus use-based strength: both apply, at different tiers.** An unpromoted shadow candidate expires after 30 days if it is never matched. A delivered memory changes strength with use: success after delivery adds 1, the same failure after delivery subtracts 2 and flags the memory, and 60 days unused archives it. Hand-authored entries keep today's decay.
- **The four-event envelope versus post_tool only.** Define the envelope fields once, because writing them down costs nothing. Implement only `session_start` and `post_tool` now. `pre_tool` arrives with the soft gate and `turn_end` with surprise capture.
- **Covered-versus-uncovered comparison versus a holdout: the holdout.** Covered signatures are the common, easy-to-fix ones and regress to the mean, so that comparison can show improvement when delivery does nothing.
- **Soft gate bounds.** Deny once per (session, procedure), at most 2 denies per session, never on an identical retry, with a kill switch per project. Kill the gate if fewer than 30% of retries after a deny change the command. It is deferred until failure delivery shows a holdout delta.
- **Listener and roster channels.** First refetch the consumption detector once, since 0 may be a broken detector. If the detector is sound, switch off headless roster pushes now: headless runs cannot receive whispers anyway. Give the interactive channels 14 more days, then apply the same test.

## 5. Irreducible disagreements for Jason

1. **Withhold a known fix on 20% of eligible failures, or don't.**
   - *Yes:* it is the only causal design at natural volume, and without it "learning" cannot be told apart from regression to the mean.
   - *No:* every held-out event is a repeat mistake we knowingly allowed. The 26-record replay plus the before/after trend may be evidence enough for a one-person lab.
2. **Can an observed but unconfirmed episode be delivered, or not?** That is, an episode that has cleared the 0.9 precision gate but has neither recurred nor been confirmed by you.
   - *Yes (evaluator):* waiting for recurrence in 2 or more sessions means the second mistake must happen before the system helps, which defeats "make it once."
   - *No (memory-science, skeptic):* one observation is not evidence for a generalization. Premature generalization is exactly how one bad night becomes a "fact."
3. **How many minutes a week will you spend reviewing?**
   - *About 10 minutes (skeptic):* cap machine promotions at about 3 per project per day, and lower the cap if the queue overflows.
   - *About 2 minutes (evaluator):* a 20-item audit, which forces promotion to rely almost entirely on recurrence and slows learning.

## 6. Roadmap

1. **Instrumentation and backfill.** Add cue_key normalization, tag every event with session, harness, engine and host, key Talos artifacts per run, and backfill failure signatures from CC transcripts and Talos `run.sqlite`. *Why:* there is no metric without these. *Worked if:* the backfilled repeat rate is computable per signature and every (harness, mode) heartbeat is non-zero.
2. **M1, the known-mistake share of quality misses.** Find what fraction of the logged quality misses concerned knowledge the KB already held when the run started (entry created_at earlier than the run). *Why:* this number decides whether delivery or capture is the bigger hole. *Worked if:* the number exists, with your labels on the ambiguous matches.
3. **The first bet** (section 2). *Worked if:* treated beats holdout after 300 events.
4. **The 26-supersedes replay (M3), run alongside item 3.**
   - Three arms (KB off, session-start slice, failure-time delivery) at 3 reps each, scored with McNemar, plus 10 control tasks (decoy or nothing relevant).
   - *Worked if:* delivery cuts repeats by at least 25 points absolute. Stop machine corrections if control tasks lose more than 5 points.
5. **Shadow episodes, then somnus consolidation.** Label 50 resolution pairings. Somnus promotes an episode on recurrence in 2 or more lineage-independent sessions, or on your confirmation. Conflicts with human entries go to your queue. *Worked if:* precision is at least 0.9 and no machine supersede of a human entry happens.
6. **Session-start slice.** At most 20 confirmed corrections or decisions per project, delivered as context, regenerated by somnus. *Worked if:* the replay's slice arm beats KB-off.
7. **Procedures through the soft gate**, sourced from human-authored procedures or failure clusters of 2 or more surfaced by item 5. *Worked if:* at least 30% of post-deny retries change the command and no run ends within 3 turns of a deny.
8. **Surprise capture in shadow, Talos first.** *Worked if:* precision is at least 0.9 on 100 labelled positives. Only then is per-turn capture reconsidered.

## 7. The repeat-mistake metric

- **Repeat event:** a failure on cue_key *c* in session *s* at a point when a resolution for *c* already existed, whether delivered or held out.
- **Primary (causal):** P(same-cue re-failure within the next 5 tool calls | eligible failure). Compare treated against holdout, and use turns-to-next-success as the secondary outcome.
- **Secondary (field trend):** cross-session repeat rate = sessions with a repeat event on *c* ÷ sessions in which *c* occurs at all, per signature, per week, cut by harness and machine.
- **Tertiary (beliefs):**
  - M2: Clef `noul` probes of each stored `wrong_belief` over the Stop `last_assistant_message` text, per exposure. Expect fewer than 10 events a month, so it is a trend line, not proof.
  - M3: the replay arm from roadmap item 4.
- **Data sources:**
  - `post_tool` event logs (cue_key, event_id, holdout flag, delivered id);
  - the backfill from CC transcripts and Talos `run.sqlite`;
  - quality-miss entries in dispatch-performance-log;
  - the 26 supersedes records.
- **Baseline:**
  - computed retroactively from the backfill, so no waiting window;
  - the concurrent holdout arm;
  - the KB-off arm of M3.
- **Change course if any of these happen:**
  - treated is not below holdout after 300 treated events;
  - delivered-then-same-failure is above 20%;
  - M3 improves by less than 25 points absolute after two iterations of delivery form;
  - control-task pass rate drops by more than 5 points;
  - a machine-written entry is superseded within 14 days more than once in 50 deliveries.
