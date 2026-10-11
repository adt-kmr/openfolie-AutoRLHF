# Agent B — fine-tuning engine

Read `docs/ARCHITECTURE.md` (frozen interfaces — implement the `training/*` signatures
exactly) and `docs/UPSTREAM_API_NOTES.md` (verified library APIs). You own exactly:

- `src/autorlhf/training/**`
- `tests/test_training_*.py`

Do not edit any other file.

## Design constraints

- `training/**` must import **without importing textual**.
- `detect_hardware()` must import `torch` lazily (inside the function) so the TUI can open
  the dashboard fast. `import autorlhf.training.jobs` must not pull in `torch`/`trl` at
  module import time either — import them inside `TrainingJob.run()`.
- On this dev machine there is **no GPU**: LoRA must be runnable on CPU for tiny models,
  and QLoRA/FFT must be _honestly disabled_ with a specific reason.
- Never use `bitsandbytes` unless the preflight confirmed it is usable.

## Modules

### `hardware.py`

- `detect_hardware(*, with_bitsandbytes=True) -> HardwareInfo` (from
  `autorlhf.core.types`). Detect: device type (cuda/rocm/mps/xpu/cpu), device name and
  count, total/free VRAM via `torch.cuda.mem_get_info` when available, system RAM from
  `/proc/meminfo` on Linux and `psutil`-free fallbacks elsewhere, CPU count, torch/CUDA
  versions, bf16/fp16 support (`torch.cuda.is_bf16_supported`, MPS checks), whether
  bitsandbytes imports _and_ can quantize on this device, flash-attn availability, and
  importable package versions (`packages`). Record caveats in `estimate_notes` (e.g.
  "VRAM totals are measured, per-run estimates are modelled") and problems in `warnings`.
  Distinguish measured from estimated everywhere. Never raise for a missing optional
  dependency — degrade and report.
- `estimate_resources(hw, hp, *, param_count, gpu_count=1) -> ResourceEstimate`: delegate
  the arithmetic to `autorlhf.config.presets.estimate_training_memory` so there is one
  implementation.

### `models.py`

- `load_tokenizer(cfg: HFModelConfig, *, hf_token, allow_fallback=None) -> (tokenizer, warnings)`.
  Clear `ChatTemplateError` when the tokenizer ships no chat template and the fallback was
  not opted into; when it is, install a **documented generic** template and say so in
  `warnings` (the fallback must be visible in the UI, the manifest and the log).
  Validate that the tokenizer has a pad token (set `pad_token = eos_token` and say so).
- `load_model(cfg, hp, *, hf_token, for_training=False, device=None) -> LoadedModel`
  (`dtype=`, `attn_implementation=`, `device_map`/`.to(device)`, `revision=`).
  For QLoRA build `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=...,
bnb_4bit_use_double_quant=..., bnb_4bit_compute_dtype=torch.bfloat16/float16)` and
  refuse to build it when `hw` says unsupported. Raise `OutOfMemoryError` with an
  actionable hint on OOM, `UnsupportedArchitectureError` when the config is not a causal
  LM, `ModelLoadError` for unknown/gated repo ids (mention `HF_TOKEN` when the error
  mentions auth/401/403/404-gated).
- `build_chat_prompt(tokenizer, messages) -> str` (add_generation_prompt=True, tokenize=False).
- `count_parameters(model) -> int`; `model_param_count_from_metadata(cfg, *, hf_token)` —
  **metadata only** via `AutoConfig.from_pretrained`, summing `config.hidden_size`-derived
  estimates or `config.num_parameters` style fields without downloading weights. Must not
  fetch a multi-GB model.
- Optional helper for local generation used by the pipeline:
  `generate_text(model, tokenizer, messages, *, max_new_tokens, temperature, top_p, seed,
device) -> (text, meta)` with `model.generate` and a do_sample/temperature combination
  that is valid when temperature <= 0.

### `lora.py`

- `infer_family(model) -> str` from `model.config.model_type` / architecture name.
- `discover_target_modules(model, *, preferred=None) -> (targets, notes)`: walk the real
  module names (`dict(model.named_modules())`), infer sensible targets per family
  (Llama/Qwen/Mistral-style `q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`;
  GPT-2-style `c_attn c_proj` with `fan_in_fan_out=True`; fall back to a scored heuristic
  on linear-layer suffixes). Return notes naming which family and which modules were found.
- `validate_target_modules(model, targets) -> None` raising `TargetModulesError` listing
  the available candidates when nothing matches — **never** let training proceed with zero
  matched modules.
- Anything the loader needs to know about GPT-2's `fan_in_fan_out` must be returned, not
  buried.

### `plan.py`

`TrainingPlan` exactly as specified in `docs/ARCHITECTURE.md` (pydantic model, `to_json`,
`save`). It is the object the TUI shows before training and the trainers consume.

### `trainers.py`

- `build_sft_config(hp, paths, *, run_name, report_to) -> SFTConfig` and
  `build_dpo_config(hp, paths, *, run_name, report_to) -> DPOConfig`, mapping every field
  of `Hyperparameters` (`max_length`, not `max_seq_length`; `completion_only_loss`,
  `assistant_only_loss`; `report_to` only when tracking is enabled).
- `build_lora_config(hp, model, *, fan_in_fan_out=False) -> LoraConfig`.
- `prepare_sft_dataset(records) -> Dataset` and `prepare_dpo_dataset(records) -> Dataset`
  from the JSONL records documented in `docs/DATA_FORMATS.md` (agent C writes the file;
  the format is frozen — SFT records carry `messages`, DPO records carry `prompt`,
  `chosen`, `rejected`).
- `TokenMetricsCallback(TrainerCallback)` publishing `metric`/`progress` events
  (`loss`, `eval_loss`, `learning_rate`, `grad_norm`, `epoch`, `step`) to the bus and
  persisting to `RunStore` when given one. Include a memory-usage metric where
  `torch.cuda` is available.
- `run_sft(...) -> TrainingResult` and `run_dpo(...) -> TrainingResult` implementing:
  checkpoint save cadence from `hp.save_steps`, `save_total_limit`, resume via
  `trainer.train(resume_from_checkpoint=...)`, graceful interruption (catch
  `KeyboardInterrupt`, and poll `cancel_event` from the callback to stop training and save
  a final checkpoint), pause (block the callback at a step boundary while
  `pause_event` is set, then continue), `mark_checkpoint_complete()` after every
  successful save, and `Trainer.save_model()` into `paths.adapter_dir` at the end.
  Register the checkpoint and adapter in `RunStore` (`save_checkpoint`, `save_artifact`,
  `set_last_checkpoint`).
- `find_resume_checkpoint(paths, *, objective) -> Path | None` using
  `autorlhf.storage.runs.select_resume_checkpoint` on the objective's checkpoint dir
  (`paths.checkpoints_dir / objective.value`).

### `jobs.py`

`TrainingJob` and `TrainingResult` exactly as specified. `TrainingJob.run()`:

- computes `find_resume_checkpoint` when `resume_from` is not given and the run is being
  resumed,
- runs the SFT warm-up first when `plan.hyperparameters.sft_warmup` and there are enough
  supervised examples, then DPO on the adapter produced by the warm-up,
- raises `InsufficientDataError` with a clear explanation when neither objective has
  enough data (never present a fake successful run),
- emits `status`/`phase`/`progress`/`metric`/`sample`/`warning`/`error` events,
- returns a `TrainingResult` for user-requested stops (`status="interrupted"`) instead of
  raising, and sets `status="failed"` + `error` for real failures,
- records `wandb_url` when W&B is active (via `autorlhf.integrations.tracking`, imported
  lazily and only when enabled — if that module does not exist yet, guard the import).

## Tests you own (`tests/test_training_*.py`)

No GPU, no downloads. Minimum: target-module discovery and validation on a tiny
`AutoModelForCausalLM.from_config(...)` model (GPT-2-style and Llama-style configs);
`discover_target_modules` + `validate_target_modules` raising `TargetModulesError` for a
nonsense target; hardware preflight feasibility rules (hand-built `HardwareInfo`: no
accelerator → QLoRA/FFT infeasible with a reason, CUDA+bitsandbytes → QLoRA feasible);
`build_sft_config`/`build_dpo_config` mapping every `Hyperparameters` field to the TRL
config (assert `max_length`, `completion_only_loss`, `report_to=[]` when tracking is off);
`prepare_sft_dataset`/`prepare_dpo_dataset` producing the expected columns;
`find_resume_checkpoint` ignoring an incomplete checkpoint and picking the highest
complete one; and a real end-to-end **smoke test of one SFT step** on a tiny
`from_config` model with an offline tokenizer (`PreTrainedTokenizerFast` with a
hand-built `tokenizers.Tokenizer`, or a tiny tokenizer written to a temp dir) — mark it
with `@pytest.mark.slow` and keep it in the default run only if it takes under ~30 s.

If you cannot make the offline tokenizer work, create a tiny tokenizer directory by hand
(`tokenizer_config.json` + `vocab.json` + `merges.txt` for a WordLevel/BPE) so
`AutoTokenizer.from_pretrained(tmp)` succeeds with no network. Do not skip this: an
end-to-end LoRA SFT step on CPU is the acceptance criterion for the whole project.

## Definition of done

- `uv run python -c "import autorlhf.training.jobs"` is fast (<1 s) and does not import torch.
- A one-step LoRA SFT run on a tiny local model actually completes and writes an adapter.
- QLoRA and FFT are gated with specific reasons on this machine (assert in a test).
- `uv run pytest tests/test_training_*.py` passes.
- `uv run ruff check src/autorlhf/training` is clean.
