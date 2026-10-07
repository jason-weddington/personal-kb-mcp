# Replay experiment harness

Does the KB stop an agent from repeating a known mistake? This harness replays real corrections (an old KB entry that a newer entry showed to be wrong) as tempting tasks, runs each task under three arms, and has an arm-blind LLM judge decide whether the agent acted on the wrong belief.

`replay.py` is stdlib-only. It calls `claude` and `psql` as subprocesses and never imports `kb_core` or `personal_kb`. The two hook scripts under `hooks/` are standalone stdlib scripts.

## Where the data goes

Real exports, generated tasks and results contain private KB content. They must go to a private, local-only eval repository outside this repo. Every subcommand takes a required `--out DIR` and refuses (exit 2, `refusing to write replay data inside the repo`) when `DIR` is this repo or lies under it. This repo ships code and the synthetic fixture in `fixtures/synthetic_pairs.json` only; every id in it starts with `syn-`.

## Pipeline

Run from the repo root, with `$OUT` pointing at a directory in your private eval repo:

1. `python3 scripts/replay/replay.py export --out "$OUT" --dsn "$KB_DSN"` (or `--sqlite /path/to/knowledge.db`) — reads `supersedes` edges into `pairs.json`, keeping only qualifying superseders (the rule in `kb_core.supersession`, re-implemented locally): LLM-inferred edges, inactive superseders and mental maps are dropped and counted in `manifest.json`.
2. `python3 scripts/replay/replay.py generate --out "$OUT" --limit 5 --controls 2` — an LLM (default `opus`) decides per pair whether it is a real correction, drops pairs already covered by your steering files (`--steering`, default the user `CLAUDE.md` plus `rules/*.md`), and writes a tempting task plus an optional per-task gate regex. One more call writes control tasks to which no correction applies. Every decision is logged to `generate-log.jsonl`; kept tasks go to `tasks.jsonl`.
3. `python3 scripts/replay/replay.py run --out "$OUT" --reps 1` — runs every task under each arm (default `sonnet`), writing `runs/<task_id>/<arm>/<rep>/` with `stream.jsonl`, `hook-log.jsonl` and `result.json`.
4. `python3 scripts/replay/replay.py judge --out "$OUT"` — one arm-blind judge call (default `sonnet`) per ok run, appended to `judgments.jsonl`.
5. `python3 scripts/replay/replay.py report --out "$OUT"` — writes `report.json` and `report.md`.

`run` and `judge` are resumable: a run with status `ok` or `invalid` is never rerun (`error` and `skipped_budget` are), and a run with a clean judgment is never re-judged.

## The three arms

- `kb_off` — no KB channel at all. The baseline.
- `slice` — a SessionStart hook injects the same unscoped slice of up to 20 corrections into every task (controls included).
- `soft_gate` — a PreToolUse hook denies, once, the first tool call whose target matches the task's own gate regex, with the stored correction as the reason. Later matching calls fall through to the sandbox.

Each arm tests the IDEAL of its channel — a perfect per-task regex gate, an unscoped 20-correction slice — not the eventual production matcher or retrieval. A null result here means the channel cannot work even when perfectly targeted; a positive result bounds what production can achieve.

## Sandbox

Each run gets a fresh `tempfile.mkdtemp()` scratch directory holding the task files and a project `.claude/settings.json` that wires the hooks. The task agent runs with `--setting-sources project`, `--strict-mcp-config` with an empty MCP config, the tools `Bash,Read,Write,Edit,Glob,Grep`, `--max-turns 15`, and no session persistence.

The PreToolUse hook (`hooks/pretool_gate.py`) is the sandbox. Bash is always denied (the command is recorded, never executed); file tools are denied outside the scratch directory; every other tool is denied. It FAILS CLOSED: any exception — malformed stdin, an unreadable config, a bad regex — prints a deny and logs `hook_error`. Every invocation logs one line to the run's `hook-log.jsonl` before printing.

Before any task runs, a canary asks the agent to `touch CANARY` under the `kb_off` setup. If the hook log has no `sandbox_deny` or the file exists, `run` prints `sandbox canary failed` and exits 3.

Residual risk: a hook that hits its 5 s timeout lets the tool call through. That is detected after the fact as `unhooked_tool_call` (the run is marked invalid), not prevented.

## Isolation checks

A run is marked `invalid` with the first of these that fires:

- `mcp_servers_present` — the init event lists any MCP server.
- `hook_leak` — a hook event fired outside the arm's allow-list (`PreToolUse` for every arm, plus exactly one `SessionStart` in `slice`).
- `slice_not_delivered` — the slice arm logged other than exactly one `slice_delivered`.
- `hook_error` — the PreToolUse hook hit an exception.
- `unhooked_tool_call` — a tool call has no hook-log line.
- `gate_in_wrong_arm` — a gate denial outside `soft_gate`.
- `slice_in_wrong_arm` — a slice delivery outside `slice`.

A run is `error` on `timeout`, `nonzero_exit`, `no_result_event` or `bad_files` (a task file path that escapes the scratch directory).

The judge is blind to the arm: its digest excludes tool results and gate-denied calls (a prevented call is not a repeat), redacts the slice header and the `KB correction:` prefix, and a runtime audit refuses to call the judge (`blinding_violation`) if the prompt still contains an arm name, the slice header, the gate prefix or the sandbox message. Each judge prompt is saved as `judge_prompt.txt` with the raw output in `judge_raw.json`.

## Confounds

- The user `CLAUDE.md` still loads in every arm under `--setting-sources project`. Mitigation: `generate` drops any correction already stated or implied in the steering text (`in_steering`).
- The agent's own text can still reveal the arm semantically (for example by paraphrasing an injected correction) despite redaction. The report flags blinding violations but cannot catch paraphrase.

## Billing and cost bounds

By default every `claude` child runs on subscription billing: `CHILD_ENV_STRIP` removes `ANTHROPIC_API_KEY` (plus `CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`, `HEADLESS_BUILD_ENGINE`, `PERSONAL_KB_URL` and `PERSONAL_KB_API_KEY`) from the child environment. Pass `run --use-api-key` to keep the key. Under subscription billing `total_cost_usd` is notional.

Each call is capped with `--max-budget-usd`: 0.50 for run and judge calls, 1.00 for generate calls. All phases (and the canary) append to one `cost-ledger.jsonl`; when its total reaches `--budget-usd` (default 20.0) no further call is made, remaining runs are written as `skipped_budget` (judgments as `budget`), and the command exits 0.

## Decision rule

Paired units are (task, rep) pairs where both `kb_off` and the arm were judged on a correction task.

- Success: the arm's repeat rate is at least 25 points below `kb_off` (`meets`).
- Stop: the arm's control pass rate drops more than 5 points below `kb_off` (`stop-control`), checked before success.
- `insufficient` when there are no paired units or no judged controls in either arm; otherwise `below`.
- The exact McNemar `p_value` is reported only; the verdict does not use it.

With fewer than 10 correction tasks the report is marked `PILOT - underpowered; verdicts are indicative only`. `soft_gate_applicable` restricts the correction units to tasks with a gate (controls are shared with `soft_gate`).

## Tests

- `tests/test_replay_harness.py` — hermetic; both subprocess seams are stubbed and the hooks are run as real subprocesses. Runs in the default `-m "not eval"` gate.
- `tests/eval/test_replay_live.py` — `@pytest.mark.eval` live smoke against the real `claude` CLI on the synthetic fixture. Run deliberately: `uv run pytest tests/eval/test_replay_live.py -m eval -s`.
