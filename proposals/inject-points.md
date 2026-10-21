# Inject points: where the KB can put a fact in front of an agent

2026-10-21 · Status: research input for applied science (the inject and relevance areas). Not a spec. Nothing here is decided; it maps the space so studies can be designed against it.

## Why this exists

Jason, 2026-10-20: "the intelligent KB is a dead simple idea. an out of band process needs to listen in and save durable facts that could inform a future action that an agent would take. Then, at the right time, it needs to surface relevant facts to the agent. That's it." All of the difficulty is in three places: capture (which facts are durable), inject (when is the right time) and relevance (which facts apply). Applied science researches only those three.

This document covers inject, and the parts of relevance that depend on where a fact is injected. It records three things so they survive the session that worked them out: the delivery channels the KB has today, every Claude Code hook that can carry a fact to the model (checked against the hooks reference), and a thought experiment on where facts should go if we owned the whole agent loop. That last one applies directly to talos, which we do own.

## What the production evidence says about inject today

Read-only checks on the maintainer's KB, 2026-10-20 and 10-21:

- **Listener whispers are not consumed.** Since the listener was switched back on (2026-10-13 04:30Z), 37 whispers were delivered and 0 consumed, at any time and so also within 10 minutes (`whisper_telemetry`). From 06-18 to 10-10: 111 delivered, 3 consumed, 1 of them within 10 minutes. An independent check against turn digests found no later direct fetch of any of the 37 maps, although those sessions made 59 `kb_get` calls. In kb-bench, by contrast, study PR-001-c found the listener rescues traps (discordant pairs Sonnet b=9 c=0, Opus b=39 c=0). The gap between lab and production is the open question.
- **The listener proposes the same few maps.** 98 server-side whisper decisions named only 20 distinct maps, one of them 38 times. The hook delivers each (session, map) pair at most once, so the 98 decisions came to about 50 deliveries on the main interactive host. Each repeat proposal still costs three model votes.
- **Session-start maps are not consumed either.** 173 roster deliveries since 10-13, 0 consumed; 19 of 25,547 before that.
- **The gate is precise only when its cue is.** 64 of the 73 production gate denials came from one lesson whose cue (`git checkout`, no `args_prefix`) matched every checkout in its project. After a deny, agents retried the identical command 67 of 73 times. The cue was removed on 2026-10-20 (kb-04091 v3). A channel that cries wolf trains agents to ignore it.
- **The gate covers almost nothing.** Of 33 active resolutions with a Bash cue, 6 were trusted with a two-word target class, and only those can fire the gate or a failure hint. That left 5 after kb-04091's cue was removed. Failure hints work: they fired 7 times on 10-18, and no failure has matched a cue since.
- **The session-start gotcha slice is cut by its text cap.** The server selects up to 20 resolutions, but `render_slice` drops lines to fit 4,000 characters. On 10-20 the slice rendered 8 of 20 lines for personal-kb, 7 of 20 for applied-science and 8 of 15 for agent-gtd. Session start is best effort and is the place for orientation maps, so the slice is not being tuned.
- **Relevance fails on paraphrase.** kb-02938 had documented whisper consumption since July, yet two sessions told the user it could not be measured. Search drops an entry when the query contains one word the entry lacks: the full-text leg requires every word, and the ratio filter after RRF fusion keeps an entry found by only one leg only if it is that leg's #1. The fix is GTD 91487b2f.

## Today's delivery channels (Claude Code)

| Channel | Hook | Fires | Payload | Notes |
|---|---|---|---|---|
| Gotcha slice and maps directory | `SessionStart` | startup, resume, clear, compact | one-line resolutions, map pointers | Best effort. Re-fetched and re-armed after compaction. |
| Soft gate | `PreToolUse` (Bash) | before a matching command | denies once with the corrected fact | Only trusted two-word Bash cues. |
| Failure hint | `PostToolUseFailure` (Bash) | after a matching command fails | the corrected fact next to the error | Same index as the gate. Must be a synchronous hook. |
| Listener whisper | judged at `Stop`, delivered at the next `UserPromptSubmit` | after the turn | up to 2 map pointers per KB | Since 2026-10-07 (GTD 83030636) it judges the Stop payload's `last_assistant_message`, because the transcript can lag. It skips headless runs, which never get a next prompt, unless `KB_LISTENER_HEADLESS=TRUE`. So headless sessions receive no whispers at all. |

## Claude Code hook events that can carry a fact to the model

Checked against the raw hooks reference (https://code.claude.com/docs/en/hooks.md) on 2026-10-21. Text a hook returns as `additionalContext` is wrapped in a system reminder, and the reference says where it lands for each event: "SessionStart and SubagentStart: at the start of the conversation, before the first prompt; UserPromptSubmit and UserPromptExpansion: alongside the submitted prompt; PreToolUse, PostToolUse, PostToolUseFailure, and PostToolBatch: next to the tool result; Stop and SubagentStop: at the end of the turn. The conversation continues so Claude can act on the feedback; PostModelSwitch: with the next request after the switch." The limit is 10,000 characters per hook.

**Mid-turn: the model reads the text on its next call.**

| Hook | Fires | Use for facts |
|---|---|---|
| `PreToolUse` | before each tool call; matcher on tool name | Can also block the call (the gate). The text lands next to the tool result. |
| `PostToolUse` | after a tool call succeeds; matcher on tool name | React to what the agent just read or produced, e.g. facts about the file it opened. |
| `PostToolUseFailure` | after a tool call fails | Failure hints. |
| `PostToolBatch` | "after a full batch of parallel tool calls resolves, before the next model call"; no matcher | The best general mid-turn slot: one delivery per model step rather than one per tool. |
| `SubagentStart` | when a subagent is spawned; matcher on agent type | The text lands before the subagent's first prompt. Subagents receive no KB facts today. |
| `SubagentStop` | when a subagent finishes | The parent sees the text and continues. |

**End of turn.**

| Hook | Use for facts |
|---|---|
| `Stop` | The text lands at the end of the turn, and the conversation continues so the model can act on it. This is the only hook that reaches a headless `claude -p` run before it exits. Each use costs an extra model call and needs a loop guard; the input carries `stop_hook_active`. |

**Between turns.**

| Hook | Use for facts |
|---|---|
| `UserPromptSubmit` | Today's whisper delivery. It also fires on turns Claude Code starts on its own, for example a background subagent reporting back. A hook that hits its timeout (30 s default for this event) has its context discarded. |
| `UserPromptExpansion` | When a slash command expands into a prompt. |
| `SessionStart` | Startup, resume, clear, compact. |
| `PostModelSwitch` | With the next request after a model switch. Niche. |

**Cannot carry facts to the model:** `PreCompact`, `PostCompact`, `Notification`, `InstructionsLoaded`, `CwdChanged`, `FileChanged`, `DirectoryAdded`, `ConfigChange`, `SessionEnd`, `StopFailure`, the worktree, task and teammate events, and `MessageDisplay`, which changes only what is shown on screen.

**Async hooks are not a timely channel.** The reference says: "Hook output is delivered on the next conversation turn. If the session is idle, the response waits until the next user interaction." With `-p`, Claude Code kills any async hook still running at teardown. `asyncRewake` wakes the model on exit code 2, but that is error semantics, not a delivery channel. This matches the failure-hint wiring rule in `docs/intelligent-kb.md` section 15.

## Thought experiment: if we owned the whole agent loop

A generic loop:

1. Assemble the session: system prompt, tools, project context.
2. Receive the task: a user prompt or a dispatch spec.
3. Think: the model writes reasoning and a plan.
4. Choose an action: the model emits one or more tool calls.
5. Check before execution: permissions and validation.
6. Execute the tool.
7. Observe the result, update beliefs, and return to step 3.
8. Declare done: the final message.
9. Verify: the gate command and tests.
10. Commit and report; then end or compact.

Where facts would go, and what each point matches on:

- **A. At task intake (step 2), matched against the task text.** This is the strongest relevance signal available, especially for a groomed dispatch spec. Put the facts in the first user message, after the cache breakpoint, so the cached prefix stays byte-stable.
- **B. Before the action (step 5), matched against the concrete call: command, file path, arguments.** Owning the loop, the harness can hold the call and add the fact, not only deny or allow it. This is a precise gate.
- **C. After the result (step 7), matched against the outcome.** A failure gets the known fix (failure hints). Reading or editing a file gets the facts tied to that file or module.
- **D. Mid-thought (between steps 3 and 4), matched against the model's stated beliefs.** The harness sees the reasoning as it is produced. When the model asserts something the KB records as a wrong belief (the `wrong_belief` field of a resolution), inject the corrected fact before the action executes. This is the listener's real shape: correct the belief at the moment it drives an action. Claude Code can only approximate it at `PostToolBatch`, after a step has already run.
- **E. Before accepting "done" (between steps 8 and 9), matched against the diff and the files touched.** Check against the conventions recorded for those files, for example "docs change in the same commit".
- **F. At compaction (step 10), matched against the current subgoal.** Re-inject what the model just lost.

Capture points fall out of the same loop: failure-then-recovery pairs at step 7, verifier failures at step 9, and user corrections at step 2. Turn digests and surprise capture already cover these for Claude Code and talos.

Precision matters more at points B, C and D than at A, because they interrupt an action in progress. The gate evidence above is the cautionary case.

## Talos should get different treatment

- **Talos is headless only.** It never receives a next user prompt, so delivery with the next prompt is useless there by construction. Only points A to F apply.
- **We own its loop**, so B and D can be synchronous and exact: no hook timeout, no async lag, no transcript lag.
- **It already has a verify step** (the project `gate_command`), so E slots in naturally.
- **It runs cheaper models** (the glm and qwen lanes). They may follow injected facts more literally, which raises the precision bar.
- **Talos is the cleanest place to test D**, the mid-thought belief check. It is the injection point with the most potential and the one Claude Code cannot express.

## Claude Code mapping

| Point | Claude Code today | Talos |
|---|---|---|
| A. Task intake | `UserPromptSubmit` on the first prompt, or `SessionStart` (no task text yet) | inject into the first user message |
| B. Before action | `PreToolUse` (deny plus reason, or `additionalContext`) | hold the call, add the fact |
| C. After result | `PostToolUse`, `PostToolUseFailure` | same, synchronous |
| D. Mid-thought | approximate at `PostToolBatch` from the latest assistant text | between the reasoning and the tool call |
| E. Before done | `Stop` with `additionalContext` (the conversation continues) | before `gate_command` |
| F. Compaction | `SessionStart` with source `compact` | in the compaction step |
| Subagents | `SubagentStart` | n/a |

## Measurement already available

- `whisper_telemetry` (consumption per session, surface and map, with chain credit). It is served by `GET /api/kb/observe/whisper-telemetry` since v1.23.0. From v1.23.0 the orphan sweep no longer erases logs from parallel live sessions, so earlier rows undercount.
- Turn digests record the entry ids a `kb_get` requested (v1.22.0 and later), so a delivery can be joined exactly to a later fetch.
- `gate_decisions` (`denied`, `retry`, `retry_changed_command`, `failure_context`), which says whether an injected fact changed the next action.
- `listener_decisions` (server proposals and votes; these are not deliveries).

## Related

- `proposals/listener-system-one-gate.md` (2026-10-07): a fast inline decision model so the whisper lands when the agent acts, not the turn after.
- `docs/intelligent-kb.md` sections 13 to 17: the slice, the gate, failure context, orientation push and the listener as built.
- KB: kb-04267 (inject baseline), kb-04269 (listener baseline), kb-04297 (the kb-02938 miss), kb-02938 (whisper telemetry validity).
