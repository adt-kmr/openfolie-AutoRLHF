# Agent C — OpenRouter, Jev, and the data pipeline

Read `docs/ARCHITECTURE.md` (frozen interfaces — implement them exactly),
`docs/UPSTREAM_API_NOTES.md` (verified APIs) and `docs/DATA_FORMATS.md` (frozen record
formats). You own exactly:

- `src/autorlhf/llm/client.py`, `src/autorlhf/llm/jev.py`
- `src/autorlhf/pipeline/**`
- `tests/test_llm_*.py`, `tests/test_pipeline_*.py`

Do not edit any other file. (`src/autorlhf/llm/schemas.py` is lead-owned and frozen; ask
the lead via `write agent://<lead-id>` if you need a field added.)

## `llm/client.py` — `OpenRouterClient`

The single module that touches the network.

- Wrap the official `openrouter` SDK (`from openrouter import OpenRouter`). Construct one
  client; `client.alpha.decisions.create(...)` needs
  `server_url="https://openrouter.ai"` because the Decisions path is `/api/alpha/decisions`
  and the SDK's default server is already `/api/v1`. Chat keeps the default server.
- `chat(...)` returns an `autorlhf.core.types.GenerationResult`, mapping
  `usage.prompt_tokens`/`completion_tokens`/`cost`, `finish_reason == "length"` →
  `truncated=True`. Support `response_format={"type": "json_object"}` and a
  `chat_json(...)` helper that extracts the first JSON object from the text (handling
  ```json fences) and raises `GenerationError` when none can be parsed.
- `decide(request: DecisionsRequest) -> DecisionsResponse`: validate the response with the
  frozen wire models. Treat a missing answer key or an unexpected `type` as an error.
- `decide_many(requests, *, on_progress=None, cancel_event=None)` with a bounded
  `ThreadPoolExecutor(max_concurrency)`.
- Retries: bounded, exponential backoff with jitter, honouring `Retry-After` on 429 →
  `RateLimitedError`, `RemoteAPIError` otherwise. Map SDK `errors.*`/HTTP status to the
  typed exceptions (`CredentialError` for 401/402/403, `RemoteAPIError` for the rest).
  Never swallow an error into an empty result.
- Caching: consult `ResponseCache` (`autorlhf.storage.cache`) before every call, keyed on
  the full content-hash of the request (model + messages + decoding params, or
  model + state + questions). Store the parsed response plus tokens/cost. Expose
  `usage_summary()` (calls, tokens, cost) for the dashboard.
- `verify_credentials()` uses a cheap live endpoint
  (`client.api_keys.get_current_key_metadata()`); return `(ok, message)` and translate
  failures into an actionable message. Never log the key.
- Weave tracing must be **optional and explicit**: if `autorlhf.integrations.tracking`
  exposes an enabled Weave client, call it per request; otherwise do nothing. Guard the
  import.

## `llm/jev.py` — `JevEvaluator`

Jev is a typed decision model, not an LLM. Use only `score`, `choice`, `noul`.

- `score_response(...) -> Judgement`: one decisions request whose `state` contains the
  character specification, the relevant conversation/history, the user prompt, and the
  candidate response; `questions` are built from the rubric
  (`Rubric.questions_for_judge()`), one `score` question per criterion with ordered,
  clearly described levels. Keep the request within Jev's 32k context by truncating the
  _response_ (never the rubric) and recording that in `Judgement.raw`.
- Map every answer into `CriterionScore` (score, `probabilities`, `legend`, `confidence`,
  `answer_type`). Missing/failed criteria keep `score=None` and an `error` string. Use
  `Rubric.overall(...)` for `overall`. Record `judge_build` from the response's `model`
  field, `usage`, `provider`, and the verbatim response under `raw`.
- `compare_responses(...) -> PairDecision`: a `choice` question with options `a` and `b`
  plus an explicit tie option (`tie`, described as "no meaningful preference"), state
  containing the character spec, context, prompt, both responses, and both responses'
  criterion scores when available. Record `probabilities`, `confidence`, `margin`
  (the overall score difference, when both responses were scored) and the raw choice key.
- `rank_presets(*, state, options) -> (chosen_key | None, info)`: a `choice` decision over
  the feasible preset ids with their real descriptions; return the probabilities,
  confidence, and the exact model build. Never present these as probabilities of training
  success (the TUI states this; your docstring should too).
- `check_condition(...) -> (probability, info)`: a `noul` decision.
- All three primitives must be exercised, and each returns the judge's real numbers.

## `pipeline/**`

### `models.py`

`DatasetBundle`, `ReviewItem`, `ReviewDecision` exactly as specified in
`docs/ARCHITECTURE.md`, plus small helpers:
`DatasetBundle.to_summary()`, `ReviewItem.apply_human(decision)`.

### `prompts.py` — synthetic prompt generation

- Build a diverse prompt set from the character spec + seeds with the teacher model:
  ordinary interactions, varied phrasings, context-following (a prior turn plus a
  follow-up), multi-turn continuations of a seed, edge cases, and explicit boundary probes
  (attempts to make the character break its constraints). Honour
  `GenerationConfig.edge_case_fraction` and `category` tags.
- Ask the teacher for a JSON array of objects
  (`{"text": ..., "category": ..., "context": [...], "avoids": ...}`); validate, reject
  malformed/duplicate entries and regenerate the shortfall, bounded.
- Deterministic prompt ids (`p0001`…) and stable hashing so resume reuses cached prompts.
- Also support the `seed` origin (a held-out prompt derived from a seed conversation's user
  turn, when `EvaluationConfig.include_seed_prompts`).

### `candidates.py` — candidate responses

- Generate `candidates_per_prompt` responses per prompt from `teacher` (OpenRouter),
  `local` (the HF model through `autorlhf.training.models.generate_text`, loaded once and
  kept alive), or `mixed`.
- Deterministic candidate ids (`{prompt_id}-c{n}`), stable request hashes, cached results
  reused on resume, per-candidate temperature variation for diversity.
- A failed generation is recorded with `status="failed"` and `error`, never as an empty
  response, and never becomes training data.
- Emit progress events; respect `cancel_event`/`pause_event`; bound concurrency.

### `splits.py` — prompt-level splits

- Assign prompts to `train`/`validation`/`eval` **before** any response is generated, by
  prompt id, honouring `eval_fraction`, `validation_fraction`, `min_eval_prompts`,
  `min_train_prompts`, `seed`. Stratify across categories so edge cases and boundary
  probes appear in every split.
- Persist `splits.json` per `docs/DATA_FORMATS.md`; on resume reuse the assignment when
  `dataset_hash` still matches, and re-split only when the fingerprint changed.
- Add `validate_split(split, prompts)` proving no prompt id appears in more than one split
  (this is the anti-leakage invariant).

### `dataset.py` — records and validation

- Build the `dataset.jsonl` / `sft.jsonl` / `preferences.jsonl` records per
  `docs/DATA_FORMATS.md`; write atomically via `autorlhf.storage.runs.append_jsonl`, and
  support rewriting a record after an edit (rewrite the file atomically).
- `validate_dataset(records) -> list[str]` covering: empty responses, duplicate records
  (same prompt+candidate+text), malformed conversations (broken role alternation), invalid
  preferences (chosen == rejected, either side missing/failed), inconsistent prompt ids,
  and truncation. Return human-readable problems, and persist the summary in counters.
- Counters to maintain via `RunStore.bump_counter`/`set_counter`: `prompts_generated`,
  `candidates_generated`, `generation_failed`, `judgements_ok`, `judgements_failed`,
  `pairs_accepted`, `pairs_rejected_tie`, `pairs_rejected_low_confidence`,
  `pairs_rejected_invalid`, `accepted`, `edited`, `rejected`, `sft_examples`,
  per-split counts.
- `load_dataset(paths) -> DatasetBundle` for resume and for the TUI's dataset screen.

### `preferences.py` — preference construction

- For each prompt, take its accepted candidates, score them, then compare the best two (or
  the two highest-margin) with `JevEvaluator.compare_responses`.
- Accept a pair only when the evidence supports it: the judge's winner is decisive, the
  overall margin exceeds `rubric.tie_epsilon`, and confidence (when returned) is at least
  `rubric.min_confidence_for_preference`. Otherwise record a rejection with the specific
  reason in the audit trail.
- Human decisions **always** win over Jev in interactive mode; write `source="human"` and
  keep Jev's recommendation in the payload.
- Never manufacture a preference label from a tiny difference; a tie is a valid outcome.

### `review.py` — `ReviewQueue`

Exactly the class in `docs/ARCHITECTURE.md`, backed by the `reviews` table.
`submit_items` persists each item (`status="pending"`) and emits a bus event
(`kind="sample"`, `data={"review_item": item.model_dump(mode="json")}`) so the TUI can
render it. `wait_for_decision` blocks on a per-item `threading.Event` (with a timeout and
cancellation check) until the TUI calls `decide`. `pending()`/`load()`/`stats()` let a
fresh process resume an interrupted review session. Human edits must be applied to the
stored item and to the dataset records.

### `evaluate.py` — baseline/final evaluation and comparison

- `evaluate_model(...)`: generate a response for every held-out `EvalPrompt` from a named
  variant (`base` or `finetuned`), then judge each with the same rubric. Same prompts, same
  rubric, same judge model for both variants. Support loading the adapter
  (`PeftModel.from_pretrained`) and, for the base variant, either the raw base model or the
  adapter with `disable_adapter()`.
- Aggregate per-criterion means over _scored_ responses only, count
  `failed_judgements`/`requested_judgements`, and list prompts with no usable judgement.
- `compare_reports(base, final, *, pairwise_pairs) -> ComparisonReport`: per-criterion
  means, deltas, overall, pairwise win/tie/inconclusive counts, and representative
  examples (prompt, base text, tuned text, both judgements) for the UI.
- Warn (in `warnings` and `inconclusive=True`) when coverage is below
  `min_responses_for_claims`, when a judge failed, or when the differences are within the
  rubric's noise — **never claim a statistically significant improvement from a tiny
  sample**.
- Save `baseline_eval.json`, `final_eval.json`, `comparison.json`.

### `orchestrate.py` — `PipelineRunner`

Exactly the methods in `docs/ARCHITECTURE.md`. Each one:

- validates its preconditions first (credentials present, config valid, model metadata
  resolvable) and raises `CredentialError`/`ConfigError`/`InsufficientDataError` with an
  actionable hint _before_ spending money;
- advances `RunStatus` through `RunStore.update_status` (transitions are validated — check
  `autorlhf.core.enums.can_transition` if a step seems stuck);
- updates counters, persists artifacts, refreshes the manifest
  (`autorlhf.storage.runs.refresh_manifest`);
- is **idempotent on resume**: reuse cached prompts/judgements and existing
  `dataset.jsonl`/`preferences.jsonl` rather than regenerating them.
- `build_training_plan(bundle, *, adapter_path=None)` selects feasible presets
  (`autorlhf.config.presets.feasible_presets`), asks Jev to rank them
  (`preset_ranking_state` + `JevEvaluator.rank_presets`) when there are two or more and a
  key is available, falls back to `recommend_preset` otherwise, and returns a
  `TrainingPlan` with the resolved hyperparameters, the rationale the app authored, Jev's
  distribution, the resource estimate, and the SFT/DPO record lists. When there are too few
  valid pairs for DPO it must switch the objective to SFT (when supervised examples exist)
  or raise `InsufficientDataError` — with the counts in the message.

## Tests you own

No network, no downloads. Use `unittest.mock`/`monkeypatch` on the SDK client and on
`OpenRouterClient` internals.

- Jev request construction: the `questions` object for a rubric matches the documented
  shape (score criteria ordered lowest-first, choice criteria as a name→description map,
  noul with exactly `true`/`false`); `state` contains character + prompt + response +
  context; the decisions call is made with `server_url="https://openrouter.ai"`.
- Jev response parsing: real-shaped score/choice/noul responses (use the exact shapes in
  `docs/UPSTREAM_API_NOTES.md`), including a response with `probabilities`/`confidence`
  omitted (must not raise, must not invent values), a missing answer key (must raise), and
  an answer with the wrong `type` (must raise).
- Chat handling: mocked SDK response → `GenerationResult` with tokens/cost and
  `truncated` from `finish_reason="length"`; a JSON-mode response with a ```json fence
  parses; a malformed body raises `GenerationError`.
- Retries/backoff: a 429 then success retries; exhausted retries raise `RateLimitedError`;
  a 401 raises `CredentialError`.
- Caching: the second identical call makes no SDK call; different decoding params produce
  a different key; costs are accumulated.
- Preference construction: pair accepted above thresholds; tie rejected with the reason
  recorded; low confidence rejected; human decision overrides Jev and is recorded with
  `source="human"`.
- Splits: no prompt id appears in two splits for many random seeds; `splits.json` is reused
  when the dataset fingerprint matches and regenerated when it changes.
- Dataset validation: each problem class in the list above is detected once.
- Comparison maths: per-criterion deltas, overall, pairwise counts, and
  `inconclusive=True` for a tiny sample; failed judgements excluded from means.
- `PipelineRunner.build_training_plan` falling back to SFT when there are no valid pairs
  and raising `InsufficientDataError` when there is nothing at all.
- `ReviewQueue`: submit → pending → decide → `wait_for_decision` returns the decision;
  a persisted decision survives a restart of the queue object.

## Definition of done

- `uv run pytest tests/test_llm_*.py tests/test_pipeline_*.py` passes.
- `import autorlhf.pipeline.orchestrate` does not import textual or torch.
- `uv run ruff check src/autorlhf/llm src/autorlhf/pipeline` is clean.
