# Surprise detector eval harness

How good is the surprise-capture detector? This harness measures the production detector, `kb_service.surprise_worker.detect_digest`, per shape (1: a failed Bash command later corrected, 2: the human corrects the assistant's previous turn, 3: a tool result contradicts an earlier claim in the same turn) on a labelled case set, and reports precision and recall with Wilson 95% intervals.

Every case goes through the same path as a live digest before a model sees it: `TurnDigestRequest` validation and its field caps, the 64 KiB digest check, `redact_turn_digest` secret redaction, then `digest_from_row`. It skips project and ts normalisation and the DB JSON round trip, because the detector reads neither, and it re-implements nothing: prompts, parsing, the confidence floor and the grounding check all come from `kb_service.surprise` and `kb_service.surprise_worker`. It writes nothing to the KB, GTD or any database.

## Where the data goes

Private labelled cases and results contain real transcript text. They live in a private, local-only eval repository outside this repo, never in this one.

`run` takes a required `--out DIR` and refuses (exit 2, `refusing to write surprise eval data inside the repo`) when `DIR` is this repo or lies under it; nothing is created. Both `validate` and `run` refuse (exit 2, `refusing to read non-synthetic cases from inside the repo`) a cases file inside this repo when any of its ids does not start with `syn-`.

This repo ships code and synthetic fixtures only: `fixtures/synthetic_cases.jsonl` (digests form, one positive and one negative per shape) and `fixtures/synthetic_mined_cases.jsonl` (mined form). Every id in them starts with `syn-`.

## Case formats

A cases file is JSONL, one case per line; blank lines are skipped. Unknown keys are rejected in every form, and validation errors name the case id and the field location only, never the case text.

Common keys: `id` (required, `^[A-Za-z0-9._:-]{1,128}$`, unique), `shape` (required, 1, 2 or 3), `label` (required bool: is this a real corrected belief?), `hard_negative` (bool, default false; only on negatives — a negative chosen because it looks like a correction), `expected` (positives only: `{"wrong_belief": str, "corrected_fact": str, "durable": bool?}`), `note`, `host` and `project` (str), `label_source` and `frame` (str of 1-64 chars, each defaulting to `unspecified`).

`label_source` says who labelled the case (for example `human`, or `silver:sonnet-judge` for a model judge). `frame` says how the cases were sampled (`unfiltered` for a random sample of applicable turns, or for example `marker-prefiltered` when a keyword prefilter chose them). Both feed the precision bar: a cell's bar is `not-gating` unless every case in it has frame `unfiltered` and a label_source that is neither `unspecified` nor starts with `silver`.

Digests form: the case carries `digests`, a list of turn digests shaped like the hook's `POST /api/kb/turn` body (`session_id`, `turn_index` and `items` required; `event_id`, `project`, `user_prompt`, `final_message`, `truncated` and `ts` optional). `event_id` defaults to `<session_id>:<turn_index>`. All digests share one session and ascend in `turn_index`; the last digest is the one under test. Shape 2 needs exactly two consecutive turns with no tool_result in the second; shape 3 needs exactly one. Shape-1 cases are digests-form only.

Example (digests form): `{"id": "syn-s3-pos", "shape": 3, "label": true, "label_source": "synthetic", "frame": "synthetic", "digests": [{"session_id": "syn-sess-5", "turn_index": 0, "project": "syn", "user_prompt": "fix the config", "items": [{"kind": "assistant_text", "text": "The config lives in /etc/foo.conf."}, {"kind": "tool_call", "tool_use_id": "t1", "tool": "Bash", "target": "cat /etc/foo.conf", "target_class": "cat"}, {"kind": "tool_result", "tool_use_id": "t1", "is_error": true, "excerpt": "No such file: /etc/foo.conf; config is at /etc/foo/main.conf"}]}]}`

Mined form (shapes 2 and 3 only), the flat form a transcript miner writes. Shape 2 carries `prev_final_message` (str, required), `user_prompt` (str, required) and `prev_assistant_texts` (list of str, optional). Shape 3 carries `items`, a non-empty flat list of `{kind: assistant_text, text}`, `{kind: tool_call, tool, target}` and `{kind: tool_result, is_error, excerpt}` objects with exactly those keys. In shape 3, an `assistant_text` whose text starts with `[user] ` marks a human message, i.e. a turn boundary.

Conversion of shape 2: turn 0 holds one assistant_text per non-blank entry of `prev_assistant_texts` (none when the key is absent) and `final_message` = `prev_final_message` (null when blank), with no user prompt; turn 1 holds `user_prompt` and no items. Session id is the case id, project the case's `project`.

Conversion of shape 3: one turn 0. The last `[user] ` item sets `user_prompt` (the text after the prefix) and everything up to and including it is dropped; with no such item `user_prompt` is null and every item is kept. Tool calls get ids `t1`, `t2`, … in order, a target taken the way production extracts it (`kb_core.cues.extract_target`: the command for Bash, the path for file tools, the pattern for Glob/Grep, and `""` for any other tool, MCP tools included) and `target_class` from `kb_core.cues.target_class`. Each tool result pairs with the oldest still-unpaired call (FIFO); a result with no unpaired call gets `orphan1`, `orphan2`, …. `final_message` is null, `truncated` false, and no field is sliced: an over-cap value fails validation.

Example (mined form): `{"id": "syn-m2-pos", "shape": 2, "label": true, "hard_negative": false, "host": "syn-host", "project": "syn", "expected": {"wrong_belief": "Port 8080 is free", "corrected_fact": "8080 is taken by caddy", "durable": true}, "prev_final_message": "Port 8080 is free; the dashboard will listen there.", "prev_assistant_texts": ["Checking ports.", "  ", "Port 8080 is free; the dashboard will listen there."], "user_prompt": "no, 8080 is taken by caddy, use 8081"}`

## Running

Run from the repo root, with `$CASES` and `$OUT` pointing into your private eval repo:

1. `uv run python scripts/surprise_eval/surprise_eval.py validate --cases "$CASES"` — validates and redacts every case, makes no model call, writes no file, and prints one JSON line of counts (`by_label_source`, `by_shape`, `cases`, `redacted_cases`). Exit 0, or 2 with `validate: <reason>`.
2. `uv run python scripts/surprise_eval/surprise_eval.py run --cases "$CASES" --out "$OUT" --model claude-sonnet-4-6 --model claude-sonnet-5-5 --model claude-opus-5-5` — writes `results.jsonl`, `report.json` and `report.md` to `$OUT`.

`run` flags: `--model M` (repeatable, distinct; required when any shape-2/3 case remains; shape-1 cases are rule-based and evaluated once as `rule:shape1`), `--shapes` (default `1,2,3`), `--limit N` (the first N cases after the shape filter, in file order; default no limit — shuffle a copy of the file for a quick random sample), `--concurrency` (default 4), `--timeout SECONDS` (default production's timeout) and `--force` (overwrite existing result files; without it `run` refuses to overwrite).

Each `--model` is built exactly as production builds the detector: an `AnthropicLLMClient` from `kb_service.config.build_anthropic_config(model=...)`. So the default per-request timeout is production's `KB_ANTHROPIC_TIMEOUT` (30 s unless set) and the API key resolves as in the service. Passing `--timeout` (for example `--timeout 120`) separates model quality from the production timeout; calls slower than the production timeout are counted either way.

The production default detector is `claude-sonnet-5-5` (`surprise_worker.SURPRISE_DETECTOR_DEFAULT_MODEL`). `KB_SURPRISE_DETECTOR_MODEL` overrides it in the service and is ignored by the harness, which takes `--model`; the report warns when the production default was not among the measured models.

The confidence floor resolves per shape exactly as in the service: `KB_SURPRISE_MIN_CONFIDENCE_SHAPE2` / `KB_SURPRISE_MIN_CONFIDENCE_SHAPE3` (per shape), then `KB_SURPRISE_MIN_CONFIDENCE` (global override, both shapes), then the defaults (shape 2: 0.5, shape 3: 0.7). Invalid values are ignored. The floors are read once per run and recorded as `min_confidence_by_shape` (plus the global `min_confidence`) in `report.json` and `report.md`. There is no threshold sweep; rerun with a different value instead.

## What is measured

Every case is scored at case level. A shape-2/3 case is predicted positive when the detector's record for that shape is a `candidate`; a shape-1 case when any shape-1 candidate fires. Only shape 1 compares the extracted belief with `expected` (`expected_match`).

`predicted` vs `model+`: `model+` means the model said surprise before the confidence floor and the grounding check (`candidate`, `low_confidence` or `ungrounded`). `lost_to_gates` counts positives the model caught but the floor or the grounding check dropped.

Detector errors (`llm_error`, `unparseable`, `invalid_fields`) count as negatives, as they do in production, and are reported as `errors`.

`not_applicable`: the detector's own skip rules (shape 2 needs a previous turn with text and a non-empty user prompt; shape 3 needs an assistant claim before a tool result) decide that no model call is made. `R applicable` is recall with those positives excluded from the denominator; plain recall keeps them. The gap is detector scope, not model quality.

Hard negatives are counted per cell (`hard_negatives`) with their false positives (`fp_hard_negative`). `recall_durable` is recall over positives whose `expected.durable` is true.

Cross-shape candidates: a case labelled for one shape can still trigger another shape's detector, which production would write. `cross_shape_fp` counts negatives where that happened.

Precision, recall and R applicable carry Wilson 95% intervals. The precision bar is `precision >= 0.9 once tp+fp >= 100`, our reading of "precision is at least 0.9 on 100 labelled positives" in `proposals/experience-loop-debate-synthesis.md` (item 8): the 100 counts detector positives. A cell reads `meets`, `below`, `insufficient` (fewer than 100 predicted positives) or `not-gating` (see `label_source` and `frame` above).

## Reading the report

`report.md` has the per-(shape, model) table, outcome counts, a Disagreements table of every false positive and false negative, the warnings and a few notes. `report.json` carries the same cells with every metric, plus provenance: detector version, cases file sha256, git commit and dirty flag, flags, package versions, timeouts, `min_confidence` and a `prompt_set_sha256` over every prompt sent.

Warnings flag what can mislead: detector errors, missing text blocks, provider failures by exception type, not_applicable positives, positives lost to the gates, calls over the production timeout, anomalies, cross-shape candidates, harness errors, shape-1 belief mismatches, cells with no positives, non-gating precision, and an unmeasured production default model.

Anomalies mean the detector broke an invariant the harness relies on (one record per shape, at most one model call per case, the prompt the harness rebuilds matching the one sent: `own_records=N`, `llm_calls=N`, `prompt_mismatch`). A harness error is an exception raised while evaluating one case; the row is kept, marked and excluded from the metrics.

Provider warnings are the WARNING lines `kb_core.llm.anthropic` logs during each evaluation (at most 5 per row): `no text block` when a response had no text block, and `generation failed` (with ` exc=<Type>`) when the call raised.

Exit codes: 0 on success; 2 for any refusal or invalid input; 3 when some model's detector was unavailable (every call `llm_error`) or made no call at all (every case not_applicable); 4 when any row has an anomaly or a harness error. 4 wins over 3, and every applicable message is printed. All three files are written for exit 0, 3 and 4.

## Billing

Every shape-2/3 evaluation is one Anthropic API call through kb_core's `AnthropicLLMClient`, billed to the API key rather than a subscription. Token cost is not recorded, because `LLMProvider.generate` drops usage, so `prompt_chars` and `response_chars` are the size proxy.

## Tests

`tests/test_surprise_eval_harness.py` is hermetic (a stub model through the `make_llm` seam, or the real client with its SDK client faked) and runs in the default `pytest -m "not eval"` gate. This README is the harness's only documentation: `how_it_works.md` is not edited.
