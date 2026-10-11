# Agent E — foundation tests

Read `docs/ARCHITECTURE.md` first. You own exactly `tests/test_foundation_*.py`.
Do not edit any file under `src/`. If you find a real bug in `src/`, fix nothing — report
it to the lead by writing a concise message to `agent://<lead-id>` and write the failing
test with `pytest.mark.xfail(strict=False)` plus a comment naming the defect.

The lead-owned foundation you are covering:

- `src/autorlhf/core/enums.py` — status transitions
- `src/autorlhf/core/ids.py`, `core/events.py`, `core/logging.py`, `core/types.py`
- `src/autorlhf/config/paths.py`, `config/settings.py`, `config/character.py`,
  `config/rubric.py`, `config/run_config.py`, `config/presets.py`
- `src/autorlhf/storage/db.py`, `storage/runs.py`, `storage/cache.py`
- `src/autorlhf/llm/schemas.py`

## Required coverage

1. **Configuration serialization and secret redaction**
   - `RunConfig` / `AppSettings` / `CharacterSpec` / `Rubric` round-trip through JSON and
     back unchanged, and `RunConfig.fingerprint()` is stable across a round-trip while
     `verbose`/`created_at`/`slug` do not affect it.
   - `SecretStore` is rooted in a temp dir: `update` preserves an unrelated line and a
     comment, the file mode is `0600` on POSIX, `load` prefers the real environment over
     the file, `redacted()` and `describe()` never contain the raw secret, and
     `mask_secret` hides short values entirely.
   - `parse_env_text`/`render_env_text` handle quotes, `export ` prefixes, blank lines and
     duplicates.
   - `core.logging.redact_text`/`redact_mapping` strip `OPENROUTER_API_KEY=...`,
     `sk-or-v1-...`, `hf_...` shapes and any key whose name contains `token`/`key`/`secret`.
2. **Character prompt and conversation validation** — every rule in
   `SeedConversation.validate_conversation` and `CharacterSpec.validate_spec`: alternation,
   missing assistant turn, empty content, truncated prompt, seed count limit, empty
   constraint, and a valid spec producing no problems. `DEFAULT_CHARACTER` must be valid.
3. **Rubric + hyperparameter preflight rules** — `Rubric.validate_rubric` (duplicate ids,
   non-positive weight, no included criterion, out-of-range confidence threshold),
   `Rubric.overall` ignoring missing criteria but weighting included ones,
   `Hyperparameters.validate_hyperparameters` for every documented conflict (bf16+fp16,
   QLoRA without 4-bit, FFT+4-bit, DPO effective batch < 2, bad scheduler/optim,
   `max_steps` semantics), `signature checks` for `RunConfig.validate_config` including the
   Weave-consent rule, and `estimate_training_memory` monotonicity (bigger batch/seq/rank
   never estimates less VRAM than the smaller one; QLoRA estimates less than LoRA for the
   same model).
4. **Preset feasibility** — hand-built `HardwareInfo` fixtures: CPU-only makes QLoRA and
   FFT infeasible with a specific reason and makes the CPU preset the recommendation;
   CUDA + bitsandbytes makes QLoRA feasible; a small-VRAM CUDA device rejects the
   large-GPU presets by name; `feasible_presets` orders feasible before infeasible;
   `preset_choice_options` only contains feasible ids.
5. **Status machine** — `can_transition`/`assert_transition` for representative legal and
   illegal pairs (`created → preparing_data`, `completed → anything` illegal,
   `paused → training_dpo` legal, `created → completed` illegal).
6. **SQLite run store** — create/get/list; transitions recorded and illegal transitions
   rejected with `ValueError`; counters accumulate; metrics append and `latest_metric`
   works; datasets, reviews (upsert + `review_stats`), preferences (upsert on `pair_id`),
   checkpoints (`complete_only`, `last_complete_checkpoint`), artifacts; `delete_run`
   removes every row; `incomplete_runs` excludes `completed`/`cancelled` and includes
   `interrupted`/`failed`.
7. **Artifacts and exports** — `write_json`/`read_json` round-trip and atomicity
   (a `.tmp` file must never survive), `append_jsonl`/`read_jsonl`/`count_jsonl`,
   `file_hash` stability, `is_checkpoint_complete` false without the sentinel / with an
   empty `trainer_state.json` / with no weight file, true only when all are present;
   `select_resume_checkpoint` ignoring an incomplete newer checkpoint and picking the
   highest complete one; `export_run` producing a directory that contains the config and
   datasets, excludes a planted `config.env` and `.env`, and whose `manifest.json` contains
   no key material.
8. **Cache** — same payload hits, changed payload misses, the `enabled=False` path performs
   no writes, and `lookup_or_compute` stores usage metadata.
9. **Event bus** — sequence numbers increase monotonically, a listener raising does not
   break other listeners (`errors()` records it), the sink receives every event, history is
   bounded by `history_size`, and unsubscribe works.
10. **Wire schemas** — `DecisionsResponse` parses the exact documented response from
    `docs/UPSTREAM_API_NOTES.md` including `legend`, `probabilities` and `confidence`;
    parsing a response with `probabilities`/`confidence` omitted succeeds and leaves them
    `None`; an unknown answer `type` raises `ValidationError`; `CriterionScore.score` is
    `None` (never `0`) for a failed judgement; `ComparisonReport` round-trips.

Use `tmp_path` (and `AUTORHLF_HOME`/`AUTORHLF_CONFIG_HOME` via `monkeypatch` where
relevant). No network, no torch, no downloads, no sleeps over ~0.2 s.

## Definition of done

- `uv run pytest tests/test_foundation_*.py` passes and covers every numbered item.
- No test asserts incidental implementation details (private attribute names, exact log
  wording) — assert behaviour and documented contracts.
- `uv run ruff check tests/test_foundation_*.py` is clean.