# Upstream API notes (verified against the *installed* versions)

Probed on the real virtualenv, not from memory. Re-probe with a throwaway script if
you doubt anything here; do not code from an assumption.

Installed versions:

| package | version | notes |
|---|---|---|
| torch | 2.14.1 | CPU wheel works; no GPU on the dev machine |
| transformers | **5.19.0** | major-version API differences below |
| trl | **1.15.0** | `processing_class`, `max_length`, `SFTConfig`/`DPOConfig` |
| peft | 0.21.2 | `get_peft_model`, `prepare_model_for_kbit_training`, `PeftModel` |
| datasets | 5.1.0 | |
| accelerate | 1.15.0 | |
| bitsandbytes | 0.50.2 | installed, but CUDA-only at runtime |
| textual | **8.2.8** | |
| openrouter | 1.3.43 | official SDK; see below |
| wandb / weave | 0.30.0 / 0.53.12 | optional extras |
| pydantic | 2.12.5 | |
| rich | 15.0.0 | |

## transformers 5.19

- `PreTrainedModel.from_pretrained(*args, **kwargs)`. **Use `dtype=`**; `torch_dtype=` still
  works but logs `"torch_dtype" is deprecated! Use "dtype" instead!`.
  Also accepted: `attn_implementation=`, `device_map=`, `quantization_config=`,
  `revision=`, `token=`, `trust_remote_code=`, `low_cpu_mem_usage=`.
- Use the enum `transformers.utils.quantization_config.BitsAndBytesConfig`
  (import path: `from transformers import BitsAndBytesConfig`).
- Chat templates live on the tokenizer instance:
  `tokenizer.apply_chat_template(conversation, tools=None, documents=None, chat_template=None,
  add_generation_prompt=False, continue_final_message=False, tokenize=True, padding=False,
  truncation=False, max_length=None, return_tensors=None, return_dict=True,
  return_assistant_tokens_mask=False, tokenizer_kwargs=None, **kwargs)`.
  `tokenizer.chat_template` is `None` for tokenizers that ship none → that must be an
  explicit, explained fallback, never a silent default.
- Offline unit tests: `AutoConfig.for_model("gpt2", ...)` /
  `AutoModelForCausalLM.from_config(config)` construct a tiny model with no download.
  `AutoTokenizer.from_pretrained` needs a local dir. For tests, build a tiny
  GPT-2-style model + a hand-written `tokenizer.json`-less tokenizer is painful — prefer
  an option that avoids the network: `transformers.PreTrainedTokenizerFast(tokenizer_object=...)`
  with a hand-built `tokenizers.Tokenizer` (BPE/WordLevel) works offline.

## trl 1.15

```python
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer
SFTTrainer(model, args=SFTConfig(...), train_dataset=..., eval_dataset=...,
           processing_class=tokenizer, peft_config=lora_config, callbacks=[...])
DPOTrainer(model, ref_model=None, args=DPOConfig(...), train_dataset=..., eval_dataset=...,
           processing_class=tokenizer, peft_config=lora_config, callbacks=[...])
```

- The tokenizer argument is `processing_class`, **not** `tokenizer`.
- Sequence length is `max_length` on both configs (**not** `max_seq_length`).
  `DPOConfig` has **no** `max_prompt_length`.
- `SFTConfig` useful fields: `max_length`, `completion_only_loss`, `assistant_only_loss`,
  `packing`, `packing_strategy`, `dataset_text_field`, `loss_type`, `max_steps`,
  `eval_packing`, `chat_template_path`.
- `DPOConfig` fields include `beta`, `loss_type` (+ the usual TrainingArguments set).
- Standard `TrainingArguments` fields are inherited: `output_dir`, `per_device_train_batch_size`,
  `gradient_accumulation_steps`, `learning_rate`, `num_train_epochs`, `max_steps`,
  `lr_scheduler_type`, `warmup_ratio`, `weight_decay`, `max_grad_norm`,
  `gradient_checkpointing`, `optim`, `seed`, `bf16`, `fp16`, `save_steps`, `eval_steps`,
  `logging_steps`, `save_total_limit`, `report_to`, `run_name`, `remove_unused_columns`,
  `dataloader_num_workers`.
- DPO with a PEFT adapter: pass `ref_model=None`; TRL uses the adapter-disabled model as
  the reference (do **not** build a second copy of the base model).
- `Trainer.train(resume_from_checkpoint=path)` restores optimizer/scheduler/RNG state;
  `Trainer.save_model()`, `Trainer.save_state()`, `Trainer.state.global_step`.
- Subclass `transformers.TrainerCallback` and override `on_log`, `on_step_end`,
  `on_save` to publish metrics/checkpoints to the `EventBus`.

## peft 0.21

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
LoraConfig(r=..., lora_alpha=..., lora_dropout=..., bias="none",
           target_modules=["q_proj", ...], task_type="CAUSAL_LM", use_dora=False)
model = get_peft_model(base, lora_config)
model.print_trainable_parameters()
model.disable_adapter()   # -> base-model behaviour, useful for comparisons
PeftModel.from_pretrained(base, "path/to/adapter")
```
`LoraConfig` has `target_parameters`, `rank_pattern`, `alpha_pattern`, `modules_to_save`,
`exclude_modules`. `fan_in_fan_out=True` is required for GPT-2's `c_attn` fused
projection. `prepare_model_for_kbit_training(model, use_gradient_checkpointing=...)`
exists.

## OpenRouter SDK 1.3.43 (official, `openrouter`)

```python
from openrouter import OpenRouter
client = OpenRouter(api_key=..., http_referer=..., x_open_router_title="autorlhf",
                    timeout_ms=120_000)
client.chat.send(model="openai/gpt-4o-mini", messages=[...], temperature=..., top_p=...,
                 max_tokens=..., seed=..., response_format=..., server_url=None)
client.alpha.decisions.create(model="typesafe/jev-1.13", state={...}, questions={...},
                              session_id=..., user=..., server_url="https://openrouter.ai")
client.models.list(...)                       # catalogue / model existence checks
client.api_keys.get_current_key_metadata()    # cheap credential validation
client.credits.get_credits()                  # balance
client.close()
```

**Critical:** the SDK's default server is `https://openrouter.ai/api/v1`
(`openrouter.SERVERS["production"]`) and the Decisions operation's path is
`/api/alpha/decisions`. Every method accepts a per-call `server_url`, so pass
`server_url="https://openrouter.ai"` to `alpha.decisions.create` and leave chat on the
default. (Confirmed by reading `openrouter/decisions.py`: `path="/api/alpha/decisions"`.)

Wire format for decisions (from the OpenRouter docs):

```json
{"model":"typesafe/jev-1.13",
 "state":{"any":"json"},
 "questions":{"is_bug":{"type":"noul","instructions":"...","criteria":{"true":"...","false":"..."}},
              "team":{"type":"choice","instructions":"...","criteria":{"a":"...","b":"..."}},
              "urgency":{"type":"score","instructions":"...","criteria":["lowest","middle","highest"]}}}
```
Response: `{"id","model","provider","answers":{...},"usage":{"input_tokens","output_tokens","cost"}}`
where each answer carries `type`; `score` has `score` + `legend` (``"0"``,``"1"``… → level
text) + `probabilities` + `confidence`; `choice` has `choice` + `probabilities` +
`confidence`; `noul` has `noul` (P(true)).

Rules: output tokens are free, input tokens are billed; `usage.cost` is USD. The schema
marks `probabilities`/`confidence`/`legend` **optional** — read them behind a presence
check even though current models return them. A missing key or unexpected `type` is an
error, never a default. Jev is **not** an LLM: it returns no prose. Never emulate it by
prompting a chat model.

## TRL 1.15 blocker (already solved — do not re-solve)

`SFTTrainer`/`DPOTrainer` unconditionally patch the model with a fused LM head and
always call it with `fused_lm_head=True`, routing the loss through
`trl.kernels.ChunkedLogProbFunction` — a **Triton** kernel. With no GPU driver this
raises `RuntimeError: 0 active drivers ([])` on the first training step, and there is no
config flag to disable it. Call `autorlhf.core.trl_compat.install_cpu_logprob_fallback()`
once, before constructing any trainer. It substitutes only the numerical kernel with a
numerically-verified pure-PyTorch implementation and is a no-op on machines where the
Triton kernel works. The tokenizer's chat template must also contain `{% generation %}`
markers around assistant content, otherwise TRL's tokenize path corrupts the tokenizer's
Rust backend and later encodes fail with a misleading
`WordLevel error: Missing [UNK] token from the vocabulary`.

## Verified by the training agent (B), worth knowing

- transformers 5.19 removed `warmup_ratio` from `TrainingArguments`/`SFTConfig`/`DPOConfig` — only `warmup_steps` exists, and passing `warmup_ratio=` raises `TypeError`. `autorlhf.training.trainers` converts the ratio to steps once the dataset size is known.
- TRL configs auto-enable bf16: an `SFTConfig(output_dir=...)` without an explicit `bf16=` raises `ValueError: Your setup doesn't support bf16/gpu...` on a CUDA-built torch with no GPU. Always pass `bf16=`/`fp16=` explicitly (autorlhf does).
- Chat templates WITHOUT `{% generation %}` markers break TRL 1.15's conversational dataset preparation for both objectives and both `completion_only_loss` settings, surfacing as the misleading `WordLevel error: Missing [UNK] token from the vocabulary`. autorlhf's generic fallback template includes the markers, `load_tokenizer` warns, and `trainers.require_assistant_markers()` hard-fails with a clear error before training.

## textual 8.2.8

`App`, `Screen`, `Widget`, `ComposeResult`, `DataTable`, `TabbedContent`/`TabPane`,
`Input`, `TextArea`, `Select`, `RadioSet`, `ProgressBar`, `RichLog`, `Static`, `Label`,
`Footer`, `Header`, `Button`, `ListView`, `Collapsible`, `ContentSwitcher`, `ModalScreen`,
`work(thread=True, exclusive=True)`, `App.call_from_thread`, `App.run_test()` (async, for
headless tests) are all present. Test with `pytest-asyncio` (already configured
`asyncio_mode = "auto"`).