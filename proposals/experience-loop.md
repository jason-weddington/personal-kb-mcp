# The experience loop: an agent that learns from experience


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

2026-10-07 · Jason Weddington (goals, ideas) + KB lead session (synthesis, grounding) · Status: PROPOSED — input to a multi-angle design debate

## The goal

**An agent that learns from experience.** It should accumulate new facts, episodic memory (what happened and what went wrong), how things work, and decisions with their reasons. Experience gained in one session should be available to the next session, on any harness and any machine. With finite context windows, the only way to do this today is an outside knowledge base plus some out-of-band channel that puts the right memory in front of the agent at the right moment.

Two constraints define what this system is for. Harnesses will grow their own memory (Claude Code already has project memories), so the KB has to be:

- **harness-agnostic:** only a thin hook or adapter is harness-specific;
- **machine-agnostic:** not tied to one machine's project directory.

Whispers, maps, the listener, preflight and roster pushes are implementation details. The north-star question for every proposal below is: **does the agent make a mistake once and not repeat it, wherever it runs next?**

## Where we are (measured 2026-10-07)

**Reading half.**
- The listener (the Stop hook, then a 3-vote Sonnet jury, then a pointer on the next prompt) made 332 decisions in the last 14 days. It whispered 20 times, wrote 11 telemetry rows, and had 0 consumed.
- Roster pushes went to interactive sessions 531 times (9 consumed) and to headless runs 5,797 times (0 consumed).
- The listener reads a transcript that may lag the turn. A fix is in flight: use `last_assistant_message`.
- Headless runs cannot receive listener whispers at all.

**Writing half.**
- Entries are written only when an agent or the human decides to.
- Maps cover 3.8% of entries.
- 20% of entries carry LLM-enrichment edges, and those edges are low-yield and low-harm (emergent-edge study, GTD 0041a9da).

**Correction half, just shipped in v1.1.0.**
- `kb_store` requires `supersedes` (a list of ids, or `"none"`).
- A near-duplicate create returns 409 with three escapes: update, supersedes, or distinct_from.
- `superseded_by` is maintained as an invariant, and search hides superseded entries.
- Map writes may not point at superseded entries.
- Corrections are now first-class records. 26 exist today.

**Constraints verified against the docs and the code.**
- Claude Code's `PreToolUse` `additionalContext` lands *next to the tool result*, not before the tool runs. The only pre-action channel is `permissionDecision: deny | ask`.
- Stop and SubagentStop carry `last_assistant_message`. The transcript can lag.
- About 89% of `thinking` blocks in recent Opus transcripts are empty. Talos records full reasoning.
- Decision ("System One") models: Clef 27B is local on jason-desktop's 5090 (Cloudflare, Apache 2.0, about 209 ms median, vendor-reported); Clef-flash is 9B at about 39 ms; Jev is hosted.
  - API: Ollama `/v1/systemone` (Ollama ≥0.35.1).
  - Question types: `noul` (probability true), `choice` (up to 26 options on Clef) and `score`.
  - Up to 64 questions per call.
  - `confidence` is entropy-based, not calibrated correctness.
- Dogwood (AWS, Apache 2.0) is a temporal policy engine: Cedar plus MFOTL operators such as "formerly" and "within 15m". It returns allow or deny per tool call, before execution, as a Rust library embedded in the harness.

## Decisions already made (Jason, 2026-10-07)

1. **Soft gate for procedures.** Procedural memory (how to deploy, how to run a migration) needs to land before the action. That means `deny` once with the procedure attached, then allow the retry.
2. **"Same turn, after the tool" is good enough** for other delivery via PreToolUse/PostToolUse.
3. **Decision model:** run Clef 27B on the 5090 for experiments. Production hosting gets revisited when new hardware arrives.
4. **Keep emergent graph edges.**

## The ideas (Jason's, lightly organized)

The organizing frame is one store with two halves:

- **System 1:** a cheap decision model decides *whether* something matters at each moment.
- **System 2:** a frontier model, or the existing nightly writer somnus, decides *what* to write or say.

### A. Capture (writing)

1. **Automatic capture with a System 1 check.** On every turn, ask whether anything is worth remembering: a new fact observed, a belief corrected, a decision made with its reasoning, a gotcha hit. Only passing turns go to a frontier model, or to a nightly batch, which writes the entry.
2. **Learn from surprises (prediction errors).** On PostToolUse, ask: "Does this result contradict something the agent asserted earlier in this turn?" A yes becomes an episodic memory plus a `supersedes`/`conflicts` edge. Today's mistake becomes tomorrow's correction. This is the closest thing to learning in the literal sense.
3. **Ground memories in observation, not assertion.** Label every machine-written candidate as *observed in tool output*, *claimed by the agent*, or *human-confirmed*. Deliver only observed or human-confirmed memories as corrections; agent-claimed ones are low-trust context. Write-time conflict detection is a four-way choice per neighbour: supports, refines, conflicts, unrelated.

### B. Delivery (reading)

4. **Match the trigger to the kind of memory:**
   - **Facts:** when the agent asserts something (Stop).
   - **Episodes:** right after a failure (PostToolUse / PostToolUseFailure). Lexical lookup works on error strings, and attention is highest then.
   - **Procedures:** before the action. Keyed on tool plus target, via the soft gate.
   - **Decisions:** when the agent is about to reverse or relitigate a recorded decision.
5. **Use harness-native memory as a delivery channel.** At session start, sync a small curated per-project slice into CLAUDE.md, AGENTS.md or Claude Code memory. The KB stays the source of truth; each harness reads the facts in its native form.
6. **Whisper the corrective fact, not a pointer** (from the listener proposal). A busy agent skips "possibly relevant map — kb-X".

### C. Platform

7. **One event stream feeds both policy and memory.** A small protocol: turn text, reasoning, tool request, tool result in; advice or verdicts out. That is essentially Dogwood's event model. A sidecar could serve governance and memory from the same stream, with a thin adapter per harness.

### D. Measurement

8. **Measure learning directly: the repeat-mistake rate.** Keep the fact behind every correction. Count how often agents assert things the KB already corrects (superseded claims are now first-class), and whether that rate falls. In the factory, a tier-2 arm with the KB on versus off, on tasks whose known mistakes are already in the KB. "Whispers consumed" measures plumbing; this measures the goal.

## The lead session's starting position (for the debate to attack)

- **Order:**
  1. Measure (8), running alongside the work.
  2. Surprise capture (2), in shadow first, carrying provenance (3).
  3. Episodes-on-failure delivery (4).
  4. Harness-native sync (5).
  5. Automatic capture (1), only once (2)+(3) show precision.
  6. Procedures via the soft gate.
  7. The platform (7) deferred until a third consumer exists. kb-service's hook API already is a small event protocol, and Talos can be a second client without a sidecar.
- **One shared System 1 layer.** A question library, shadow logging, a kill switch per question, and an offline replay harness, serving every question above. Reuse somnus as the System 2 writer.
- **Precision over recall everywhere a memory is *delivered as a correction*.** A wrong correction corrupts what the agent reasons from next. Capture can be liberal, because shadow-logged candidates cost nothing.

## Questions the debate must resolve

1. **Order and first bet.** Which single piece most directly reduces repeated mistakes soonest: surprise capture, episodic delivery on failure, harness-native sync, or something else?
2. **Write-side quality.** How do we keep machine-written memory from flooding the KB with plausible noise or locking in self-consistent misconceptions?
   - Is provenance labelling enough?
   - What review or decay applies to machine-written entries?
   - Who confirms?
3. **Surprise detection.** Does "this tool result contradicts something the agent asserted this turn" work as a single decision-model question? What is the unit of memory written: the error, the wrong belief, or the corrected fact?
4. **Delivery channels.** Which moments matter most for each memory kind, given the verified hook constraints (PreToolUse context lands with the result; only deny is pre-action)? How should the soft gate be bounded so it never stalls a run?
5. **Harness-agnostic boundary.** What is the minimal contract a new harness adapter must implement? Is the sidecar or event protocol (7) needed now, or is the kb-service API enough?
6. **Measurement.** How do we compute the repeat-mistake rate concretely from data we have or can cheaply add? What is the baseline, and what result would make us stop or change course?
7. **Failure modes.** What does it look like if this goes wrong, and what is the earliest signal?
