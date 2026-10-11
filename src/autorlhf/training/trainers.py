"""TRL trainer construction, dataset preparation and the training loop.

Design notes that matter:

* The engine never trains silently.  Checkpoints are marked complete only after
  ``Trainer._save_checkpoint`` finished writing ``trainer_state.json`` (the
  callback fires after it), so a killed process can never advertise a resumable
  checkpoint that is missing its optimiser state.
* Pause and stop are handled inside a callback at *step boundaries*, which is
  the only place where the optimiser state is consistent.  A user-requested stop
  returns a result with ``status="interrupted"`` and a saved adapter rather than
  raising.
* On this project's CPU-only machine TRL's fused LM head cannot run (Triton has
  no driver), so ``autorlhf.core.trl_compat.install_cpu_logprob_fallback()`` is
  installed before any trainer is constructed.  It substitutes only the
  numerical kernel.  :func:`ensure_trl_logprob_compat` adds one guard on top:
  TRL's ``DPOTrainer`` requires a boolean ``is_top1`` (it bitwise-ANDs it with a
  mask), which a pure-torch substitute must honour.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.errors import ChatTemplateError, ConfigError, InsufficientDataError
from autorlhf.core.events import EventBus
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import (
    RunPaths,
    directory_size,
    is_adapter_complete,
    list_checkpoint_dirs,
    mark_checkpoint_complete,
    select_resume_checkpoint,
)

Message = dict[str, str]
_VALID_ROLES = ("system", "user", "assistant")

#: Minimum number of usable records for an objective to be worth running.
MIN_SFT_EXAMPLES = 1
MIN_PREFERENCE_PAIRS = 1


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #


def checkpoint_root(paths: RunPaths, objective: TrainingObjective | str) -> Path:
    """``<run>/checkpoints/<objective>`` -- one checkpoint series per objective."""

    return paths.checkpoints_dir / TrainingObjective(objective).value


def find_resume_checkpoint(paths: RunPaths, *, objective: TrainingObjective | str) -> Path | None:
    """Highest-step *complete* checkpoint for ``objective``, or ``None``."""

    return select_resume_checkpoint(checkpoint_root(paths, objective))


# --------------------------------------------------------------------------- #
# records -> datasets
# --------------------------------------------------------------------------- #


def _clean_messages(raw: Any) -> list[Message]:
    cleaned: list[Message] = []
    if not isinstance(raw, list):
        return cleaned
    for entry in raw:
        if not isinstance(entry, dict):
            return []
        role = str(entry.get("role", ""))
        content = entry.get("content")
        if role not in _VALID_ROLES or not isinstance(content, str) or not content.strip():
            return []
        cleaned.append({"role": role, "content": content})
    return cleaned


def validate_sft_records(records: Iterable[dict]) -> list[str]:
    """Human-readable reasons any record will be dropped (empty when all are usable)."""

    problems: list[str] = []
    for index, record in enumerate(records):
        messages = _clean_messages(record.get("messages"))
        if not messages:
            problems.append(f"sft record {index}: missing or malformed 'messages'")
            continue
        if messages[-1]["role"] != "assistant":
            problems.append(
                f"sft record {index}: last message is '{messages[-1]['role']}', "
                "not an assistant turn"
            )
    return problems


def prepare_sft_dataset(records: Sequence[dict]) -> Any:
    """Build a TRL conversational SFT dataset from ``sft.jsonl`` records.

    Records that are not usable are dropped (and reported by
    :func:`validate_sft_records`); only the ``messages`` column is kept because
    everything else in the record is provenance the trainer does not need.
    """

    from datasets import Dataset

    rows = []
    for record in records:
        messages = _clean_messages(record.get("messages"))
        if not messages:
            continue
        if messages[-1]["role"] != "assistant":
            continue
        rows.append({"messages": messages})
    return Dataset.from_list(rows)


def validate_dpo_records(records: Iterable[dict]) -> list[str]:
    """Reasons any preference record will be dropped."""

    problems: list[str] = []
    for index, record in enumerate(records):
        prompt = str(record.get("prompt") or "").strip()
        chosen = str(record.get("chosen_text") or "").strip()
        rejected = str(record.get("rejected_text") or "").strip()
        if not prompt:
            problems.append(f"preference record {index}: empty 'prompt'")
        if not chosen or not rejected:
            problems.append(f"preference record {index}: empty chosen or rejected text")
        elif chosen == rejected:
            problems.append(
                f"preference record {index}: chosen and rejected text are identical "
                "(no preference signal)"
            )
    return problems


def prepare_dpo_dataset(records: Sequence[dict], *, system_prompt: str | None = None) -> Any:
    """Build a TRL conversational DPO dataset from ``preferences.jsonl`` records.

    ``prompt`` is the conversation *before* the assistant answer: the character
    system prompt, then ``prompt_context``, then the user turn.  ``chosen`` and
    ``rejected`` are single assistant turns.  The system prompt is prepended so
    the preference is judged under exactly the framing used at inference.
    """

    from datasets import Dataset

    rows = []
    system = (system_prompt or "").strip()
    for record in records:
        prompt = str(record.get("prompt") or "").strip()
        chosen = str(record.get("chosen_text") or "").strip()
        rejected = str(record.get("rejected_text") or "").strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        conversation: list[Message] = []
        if system:
            conversation.append({"role": "system", "content": system})
        context = record.get("prompt_context") or []
        if isinstance(context, list):
            for entry in context:
                if isinstance(entry, dict) and entry.get("role") in _VALID_ROLES:
                    content = entry.get("content")
                    if isinstance(content, str) and content.strip():
                        conversation.append({"role": str(entry["role"]), "content": content})
        conversation.append({"role": "user", "content": prompt})
        rows.append(
            {
                "prompt": conversation,
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# TRL configs
# --------------------------------------------------------------------------- #


def _pin_memory() -> bool:
    """Whether pinned host memory can actually help on this machine.

    Pinning is a copy-into-device optimisation; with no accelerator it only
    produces a torch warning, so the config says so explicitly.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency at train time
        return False
    if torch.cuda.is_available():
        return True
    xpu = getattr(torch, "xpu", None)
    return bool(xpu is not None and xpu.is_available())


def _warmup_steps(hp: Hyperparameters, planned_steps: int | None) -> int:
    """Convert the configured warm-up *ratio* into the step count TRL wants.

    transformers 5 removed ``warmup_ratio`` from ``TrainingArguments``, so the
    ratio must be resolved against a known total number of steps.  When the total
    is unknown (no ``max_steps``, unpinned epochs) this returns 0 rather than
    inventing a number; :func:`run_sft` / :func:`run_dpo` resolve it against the
    real dataset size once the dataset is built.
    """

    ratio = float(hp.warmup_ratio or 0.0)
    if ratio <= 0 or not planned_steps or planned_steps <= 0:
        return 0
    return max(1, round(ratio * planned_steps))


def _common_training_arguments(
    hp: Hyperparameters,
    output_dir: Path,
    *,
    run_name: str,
    report_to: Sequence[str],
    planned_steps: int | None = None,
) -> dict[str, Any]:
    """Fields shared by :class:`SFTConfig` and :class:`DPOConfig`."""

    return {
        "output_dir": str(output_dir),
        "run_name": run_name or hp.run_name or None,
        # Tracking is opt-in: an empty list means "log nowhere", which is what
        # the run configuration asks for when W&B is disabled.
        "report_to": list(report_to),
        "learning_rate": hp.learning_rate,
        "num_train_epochs": hp.num_train_epochs,
        "max_steps": hp.max_steps,
        "per_device_train_batch_size": hp.per_device_train_batch_size,
        "per_device_eval_batch_size": hp.per_device_eval_batch_size,
        "gradient_accumulation_steps": hp.gradient_accumulation_steps,
        "lr_scheduler_type": hp.lr_scheduler_type,
        "warmup_steps": _warmup_steps(hp, planned_steps),
        "weight_decay": hp.weight_decay,
        "max_grad_norm": hp.max_grad_norm,
        "gradient_checkpointing": hp.gradient_checkpointing,
        "optim": hp.optim,
        "seed": hp.seed,
        "bf16": hp.bf16,
        "fp16": hp.fp16,
        "save_steps": hp.save_steps,
        "save_total_limit": hp.save_total_limit,
        "logging_steps": hp.logging_steps,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": _pin_memory(),
    }


def build_sft_config(
    hp: Hyperparameters, paths: RunPaths, *, run_name: str = "", report_to: Sequence[str] = ()
) -> Any:
    """Map every :class:`Hyperparameters` field onto an ``SFTConfig``.

    ``max_length`` (not the removed ``max_seq_length``), ``completion_only_loss``
    from ``sft_completion_only_loss``, and ``assistant_only_loss=False``: the
    latter needs ``{% generation %}`` markers in the tokenizer's chat template,
    which a foreign tokenizer may not have, so it is never assumed.
    """

    from trl import SFTConfig

    arguments = _common_training_arguments(
        hp,
        checkpoint_root(paths, TrainingObjective.SFT),
        run_name=run_name,
        report_to=report_to,
        planned_steps=hp.max_steps if hp.max_steps > 0 else None,
    )
    arguments.update(
        {
            "max_length": hp.max_seq_length,
            "completion_only_loss": hp.sft_completion_only_loss,
            "assistant_only_loss": False,
            "packing": False,
            "eval_packing": False,
            # The SFT warm-up has its own epochs and (usually) a higher LR.
            "learning_rate": hp.resolved_sft_lr(),
            "num_train_epochs": hp.sft_epochs,
        }
    )
    return SFTConfig(**arguments)


def build_dpo_config(
    hp: Hyperparameters, paths: RunPaths, *, run_name: str = "", report_to: Sequence[str] = ()
) -> Any:
    """Map every :class:`Hyperparameters` field onto a ``DPOConfig``.

    ``DPOConfig`` has no ``max_prompt_length`` in TRL 1.15; ``max_length`` caps
    the whole prompt+completion sequence.
    """

    from trl import DPOConfig

    arguments = _common_training_arguments(
        hp,
        checkpoint_root(paths, TrainingObjective.DPO),
        run_name=run_name,
        report_to=report_to,
        planned_steps=hp.max_steps if hp.max_steps > 0 else None,
    )
    arguments.update(
        {
            "max_length": hp.max_seq_length,
            "beta": hp.dpo_beta,
            "loss_type": hp.dpo_loss_type,
            "eval_steps": hp.eval_steps,
        }
    )
    return DPOConfig(**arguments)


def require_assistant_markers(tokenizer: Any) -> None:
    """Refuse to train when the chat template cannot mark assistant spans.

    TRL 1.15 prepares every conversational dataset through
    ``apply_chat_template(..., tokenize=True, return_dict=True)``.  With a template
    that has no ``{% generation %}`` ... ``{% endgeneration %}`` blocks the
    tokenizer ends up in a broken state and the next encode fails with a cryptic
    ``WordLevel error: Missing [UNK] token from the vocabulary`` -- for both SFT
    and DPO, with and without ``completion_only_loss`` (verified against the
    installed versions on this machine).  Failing here, with the cause and the
    remedy, is the honest alternative.
    """

    from autorlhf.training.models import chat_template_has_generation_markers

    if chat_template_has_generation_markers(tokenizer):
        return
    raise ChatTemplateError(
        "the tokenizer's chat template cannot mark assistant spans, so no "
        "conversational dataset can be prepared",
        hint=(
            "TRL needs {% generation %}...{% endgeneration %} markers around assistant "
            "content. Add them to the tokenizer's chat_template, or pick a model whose "
            "template already has them."
        ),
    )


def build_lora_config(hp: Hyperparameters, model: Any, *, fan_in_fan_out: bool = False) -> Any:
    """Build the PEFT LoRA configuration for ``hp``.

    ``fan_in_fan_out`` must come from the model (see
    :func:`autorlhf.training.lora.needs_fan_in_fan_out`); GPT-2-style fused
    projections store their weights transposed and silently train nothing useful
    without it.
    """

    if hp.method is TrainingMethod.FFT:
        raise ConfigError(
            "full fine-tuning does not use LoRA",
            hint="Do not build a LoRA configuration for method='fft'.",
        )
    from peft import LoraConfig

    return LoraConfig(
        r=hp.lora_r,
        lora_alpha=hp.lora_alpha,
        lora_dropout=hp.lora_dropout,
        bias=hp.lora_bias,
        use_dora=hp.use_dora,
        task_type="CAUSAL_LM",
        target_modules=list(hp.lora_target_modules),
        fan_in_fan_out=fan_in_fan_out,
    )


# --------------------------------------------------------------------------- #
# callback
# --------------------------------------------------------------------------- #

_MEMORY_METRIC = "peak_gpu_memory_mb"


class TokenMetricsCallback(TrainerCallback):
    """Publishes TRL's metrics to the :class:`EventBus` and ``RunStore``.

    Metrics land in SQLite (when a store is given) *and* on the bus, so the TUI
    and a headless run see the same numbers.  The callback also owns pause/stop
    at step boundaries and marks each checkpoint complete once TRL has finished
    writing it.
    """

    def __init__(
        self,
        *,
        bus: EventBus,
        run_id: str,
        objective: TrainingObjective | str,
        store: RunStore | None = None,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
        total_steps: int | None = None,
        on_checkpoint: Callable[[Path, int], None] | None = None,
    ) -> None:
        self.bus = bus
        self.run_id = run_id
        self.objective = TrainingObjective(objective)
        self.store = store
        self.cancel_event = cancel_event
        self.pause_event = pause_event
        self.total_steps = total_steps
        self.on_checkpoint = on_checkpoint
        self.warnings: list[str] = []
        self.stop_requested = False
        self._pause_announced = False
        self._last_progress = -1

    # -- events ------------------------------------------------------------ #

    def _emit(self, kind: str, message: str, **kwargs: Any) -> None:
        self.bus.emit(
            kind,
            message,
            run_id=self.run_id,
            phase=f"train_{self.objective.value}",
            **kwargs,
        )

    def _record(self, name: str, value: float, step: int) -> None:
        if self.store is not None:
            try:
                self.store.add_metric(self.run_id, name, float(value), step)
            except Exception as exc:  # pragma: no cover - storage must not kill training
                self.warnings.append(f"could not persist metric '{name}': {exc}")
        self._emit(
            "metric",
            f"{name}={value:.6g}",
            step=step,
            level="info",
            verbose_only=name in {"num_tokens", "mean_token_accuracy", "entropy"},
            name=name,
            value=float(value),
            objective=self.objective.value,
        )

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> Any:
        logs = logs or {}
        step = int(getattr(state, "global_step", 0) or 0)
        for name, value in logs.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            self._record(str(name), float(value), step)
        memory = _peak_memory_mb()
        if memory is not None:
            self._record(_MEMORY_METRIC, memory, step)
        if self.store is not None:
            try:
                self.store.set_step(self.run_id, step)
            except Exception as exc:  # pragma: no cover - defensive
                self.warnings.append(f"could not record step: {exc}")
        total = int(getattr(state, "max_steps", 0) or self.total_steps or 0)
        if total > 0 and step != self._last_progress:
            self._last_progress = step
            self._emit(
                "progress",
                f"{self.objective.value}: step {step}/{total}",
                step=step,
                fraction=min(1.0, step / total),
                verbose_only=True,
            )
        return control

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.stop_requested = True
            self._emit(
                "warning",
                f"stop requested: ending {self.objective.value} at the next step boundary",
                step=int(getattr(state, "global_step", 0) or 0),
            )
            control.should_training_stop = True
            return control
        if self.pause_event is not None and self.pause_event.is_set():
            self._emit(
                "status",
                f"{self.objective.value} paused at step boundary",
                step=int(getattr(state, "global_step", 0) or 0),
            )
            while self.pause_event.is_set():
                if self.cancel_event is not None and self.cancel_event.is_set():
                    control.should_training_stop = True
                    self.stop_requested = True
                    break
                time.sleep(0.1)
            else:
                self._emit("status", f"{self.objective.value} resumed")
        return control

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        step = int(getattr(state, "global_step", 0) or 0)
        folder = Path(str(args.output_dir)) / f"checkpoint-{step}"
        if not folder.is_dir():
            self.warnings.append(f"expected checkpoint directory {folder} was not written")
            return control
        # The callback fires after Trainer._save_checkpoint wrote trainer_state.json
        # and the adapter weights, so the sentinel is only dropped when the
        # checkpoint is genuinely resumable.
        mark_checkpoint_complete(folder)
        if self.on_checkpoint is not None:
            self.on_checkpoint(folder, step)
        self._emit(
            "artifact",
            f"checkpoint {folder.name} complete",
            step=step,
            path=str(folder),
            size_bytes=directory_size(folder),
        )
        return control


def _peak_memory_mb() -> float | None:
    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


# --------------------------------------------------------------------------- #
# the shared training loop
# --------------------------------------------------------------------------- #


def ensure_trl_logprob_compat() -> bool:
    """Make TRL's log-prob substitute usable for DPO as well as SFT.

    ``autorlhf.core.trl_compat.install_cpu_logprob_fallback`` replaces TRL's
    Triton kernel with a pure-torch one.  Everything TRL's ``SFTTrainer`` does
    with ``is_top1`` (``outputs.is_top1.sum()``) works with a float tensor, but
    ``DPOTrainer`` combines it with a boolean mask::

        correct_predictions = chosen_is_top1 & chosen_mask.bool()

    which raises ``NotImplementedError: "bitwise_and_cpu" not implemented for
    'Float'`` for a float ``is_top1``.  The real kernel returns a boolean, so a
    substitute that returns floats is not faithful to it.

    This wraps a *project-owned* substitute so the fifth output is boolean.  It
    never touches the real Triton kernel (probed only when it is safe to call,
    i.e. when the installed callable comes from this project) and is a no-op
    once the substitute is fixed.  Returns ``True`` when a wrapper was installed.
    """

    import torch
    import trl.trainer.utils as trl_utils

    substitute = getattr(trl_utils, "_ChunkedLogProbFunction", None)
    if substitute is None:
        return False  # real Kernel unavailable or errored differently; leave alone
    if (getattr(substitute, "__module__", "") or "").split(".")[0] != "autorlhf":
        return False  # not ours: never probe the Triton kernel on a driver-less host

    try:
        probe = substitute.apply(
            torch.zeros(2, 4),
            torch.zeros(6, 4),
            None,
            torch.tensor([1, 2]),
            1.0,
            1024,
            None,
            1.0,
            ("is_top1",),
        )
    except Exception:  # pragma: no cover - a broken substitute is reported by training itself
        return False
    top1 = probe[4] if len(probe) > 4 else None
    if top1 is None or top1.dtype == torch.bool:
        return False

    class _BooleanTop1LogProb:
        """Delegates to the project substitute, coercing ``is_top1`` to bool."""

        @staticmethod
        def apply(  # type: ignore[override]
            hidden: Any,
            weight: Any,
            bias: Any,
            targets: Any,
            temperature: float,
            chunk_size: int,
            final_logit_softcapping: float | None = None,
            logit_scale: float = 1.0,
            outputs: tuple[str, ...] = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
        ) -> tuple[Any, Any, Any, Any, Any]:
            log_probs, entropy, log_sum_sq, mean_logits, top1 = substitute.apply(
                hidden,
                weight,
                bias,
                targets,
                temperature,
                chunk_size,
                final_logit_softcapping,
                logit_scale,
                outputs,
            )
            if top1 is not None and top1.dtype != torch.bool:
                top1 = top1 != 0
            return log_probs, entropy, log_sum_sq, mean_logits, top1

    trl_utils._ChunkedLogProbFunction = _BooleanTop1LogProb
    try:  # keep the kernels module consistent for code that imports it directly
        import trl.kernels as trl_kernels

        trl_kernels.ChunkedLogProbFunction = _BooleanTop1LogProb
    except ImportError:  # pragma: no cover - trl is a hard dependency
        pass
    return True


def _aggregate_metrics(log_history: Sequence[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for entry in log_history:
        for name, value in entry.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                continue
            metrics[str(name)] = float(value)
    return metrics


def _register_checkpoint(
    store: RunStore | None, run_id: str, folder: Path, step: int, metrics: dict[str, float]
) -> None:
    if store is None:
        return
    try:
        store.save_checkpoint(
            run_id,
            folder,
            step=step,
            complete=True,
            size_bytes=directory_size(folder),
            metrics=metrics,
        )
    except Exception:  # pragma: no cover - defensive
        pass


def _run_trainer(
    *,
    trainer_cls: Any,
    objective: TrainingObjective,
    config: Any,
    train_dataset: Any,
    eval_dataset: Any,
    model: Any,
    tokenizer: Any,
    peft_config: Any | None,
    trainer_kwargs: dict[str, Any] | None = None,
    run_id: str,
    paths: RunPaths,
    hp: Hyperparameters,
    bus: EventBus,
    cancel_event: threading.Event | None,
    pause_event: threading.Event | None,
    store: RunStore | None,
    resume_from: Path | None,
    adapter_dir: Path | None,
    artifact_name: str,
    epochs: float,
    warnings: list[str],
) -> Any:
    """Construct the TRL trainer, train, and save the final adapter."""

    from autorlhf.core.trl_compat import install_cpu_logprob_fallback
    from autorlhf.training.jobs import TrainingResult

    install_cpu_logprob_fallback()
    ensure_trl_logprob_compat()

    output_dir = Path(str(config.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    final_adapter = Path(adapter_dir) if adapter_dir else paths.adapter_dir
    metrics: dict[str, float] = {}
    result_warnings = list(warnings)

    def _on_checkpoint(folder: Path, step: int) -> None:
        _register_checkpoint(store, run_id, folder, step, metrics)

    estimated_steps = _estimate_total_steps(hp, len(train_dataset), epochs)
    callback = TokenMetricsCallback(
        bus=bus,
        run_id=run_id,
        objective=objective,
        store=store,
        cancel_event=cancel_event,
        pause_event=pause_event,
        total_steps=estimated_steps,
        on_checkpoint=_on_checkpoint,
    )

    bus.emit(
        "status",
        f"starting {objective.value} training",
        run_id=run_id,
        phase=f"train_{objective.value}",
        resumed_from=str(resume_from) if resume_from else None,
        steps=estimated_steps,
    )

    try:
        trainer = trainer_cls(
            model=model,
            args=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[callback],
            **(trainer_kwargs or {}),
        )
    except Exception as exc:  # noqa: BLE001 - reported as a failed result, not a crash
        return _failed_result(
            exc,
            run_id=run_id,
            objective=objective,
            hp=hp,
            output_dir=output_dir,
            resumed_from=resume_from,
            callback=callback,
            warnings=result_warnings,
            bus=bus,
        )

    try:
        trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    except KeyboardInterrupt:
        result_warnings.append("training interrupted; saving the in-progress state and adapter")
        bus.emit(
            "warning",
            "keyboard interrupt: saving state before exiting",
            run_id=run_id,
            phase=f"train_{objective.value}",
        )
        trainer.save_state()
        trainer.save_model(str(final_adapter))
        metrics = _aggregate_metrics(getattr(trainer.state, "log_history", []))
        _finalize(store, run_id, final_adapter, artifact_name, trainer, metrics)
        return TrainingResult(
            run_id=run_id,
            objective=objective,
            method=hp.method,
            status="interrupted",
            steps_completed=int(getattr(trainer.state, "global_step", 0) or 0),
            adapter_path=str(final_adapter),
            checkpoints=_complete_checkpoints(output_dir),
            resumed_from=str(resume_from) if resume_from else None,
            metrics=metrics,
            warnings=result_warnings + callback.warnings,
        )
    except Exception as exc:  # noqa: BLE001 - a mid-run failure keeps its completed steps
        metrics = _aggregate_metrics(getattr(trainer.state, "log_history", []))
        return _failed_result(
            exc,
            run_id=run_id,
            objective=objective,
            hp=hp,
            output_dir=output_dir,
            resumed_from=resume_from,
            callback=callback,
            warnings=result_warnings,
            bus=bus,
            steps=int(getattr(trainer.state, "global_step", 0) or 0),
            metrics=metrics,
        )

    steps = int(getattr(trainer.state, "global_step", 0) or 0)
    metrics = _aggregate_metrics(getattr(trainer.state, "log_history", []))
    status = "interrupted" if callback.stop_requested else "completed"
    try:
        trainer.save_state()
        trainer.save_model(str(final_adapter))
    except Exception as exc:  # noqa: BLE001 - failing to save is failing the run
        return _failed_result(
            exc,
            run_id=run_id,
            objective=objective,
            hp=hp,
            output_dir=output_dir,
            resumed_from=resume_from,
            callback=callback,
            warnings=result_warnings,
            bus=bus,
            steps=steps,
            metrics=metrics,
            prefix="training finished but saving the adapter failed",
        )

    _finalize(store, run_id, final_adapter, artifact_name, trainer, metrics)
    return TrainingResult(
        run_id=run_id,
        objective=objective,
        method=hp.method,
        status=status,
        steps_completed=steps,
        adapter_path=str(final_adapter),
        checkpoints=_complete_checkpoints(output_dir),
        resumed_from=str(resume_from) if resume_from else None,
        metrics=metrics,
        warnings=result_warnings + callback.warnings,
    )


def _failed_result(
    exc: BaseException,
    *,
    run_id: str,
    objective: TrainingObjective,
    hp: Hyperparameters,
    output_dir: Path,
    resumed_from: Path | None,
    callback: TokenMetricsCallback,
    warnings: list[str],
    bus: EventBus,
    steps: int = 0,
    metrics: dict[str, float] | None = None,
    prefix: str = "",
) -> Any:
    from autorlhf.training.jobs import TrainingResult

    message = str(exc)
    hint = getattr(exc, "hint", "")
    if hint:
        message = f"{message} ({hint})"
    if prefix:
        message = f"{prefix}: {message}"
    bus.emit("error", message, run_id=run_id, phase=f"train_{objective.value}", level="error")
    return TrainingResult(
        run_id=run_id,
        objective=objective,
        method=hp.method,
        status="failed",
        steps_completed=steps,
        adapter_path=None,
        checkpoints=_complete_checkpoints(output_dir),
        resumed_from=str(resumed_from) if resumed_from else None,
        metrics=metrics or {},
        error=message,
        warnings=warnings + callback.warnings,
    )


def _finalize(
    store: RunStore | None,
    run_id: str,
    final_adapter: Path,
    artifact_name: str,
    trainer: Any,
    metrics: dict[str, float],
) -> None:
    if not is_adapter_complete(final_adapter):
        return
    if store is None:
        return
    step = int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0)
    try:
        store.save_artifact(
            run_id,
            artifact_name,
            final_adapter,
            kind="adapter",
            meta={"objective": artifact_name, "step": step, "metrics": metrics},
        )
        store.set_last_checkpoint(run_id, str(final_adapter), step)
    except Exception:  # pragma: no cover - defensive
        pass


def _complete_checkpoints(output_dir: Path) -> list[str]:
    return [str(path) for path in list_checkpoint_dirs(output_dir) if _is_complete(path)]


def _is_complete(path: Path) -> bool:
    from autorlhf.storage.runs import is_checkpoint_complete

    return is_checkpoint_complete(path)


def _estimate_total_steps(hp: Hyperparameters, train_rows: int, epochs: float) -> int:
    """Planned optimiser steps; an estimate for progress reporting only."""

    if hp.max_steps and hp.max_steps > 0:
        return int(hp.max_steps)
    if train_rows <= 0:
        return 0
    effective = max(1, hp.effective_batch_size())
    return max(1, math.ceil(train_rows / effective * max(epochs, 1.0)))


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #


def _check_data(
    *, records: Sequence[dict], minimum: int, objective: TrainingObjective, label: str
) -> None:
    if len(records) < minimum:
        raise InsufficientDataError(
            f"{objective.value} needs at least {minimum} {label}, found {len(records)}",
            hint=(
                "Generate more prompts/candidates, or relax the review thresholds, before training."
            ),
        )


def run_sft(
    *,
    run_id: str,
    paths: RunPaths,
    hp: Hyperparameters,
    model: Any,
    tokenizer: Any,
    train_records: Sequence[dict],
    bus: EventBus,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    store: RunStore | None = None,
    eval_records: Sequence[dict] = (),
    resume_from: Path | None = None,
    report_to: Sequence[str] = (),
    run_name: str = "",
    adapter_dir: Path | None = None,
    peft_config: Any | None = None,
    apply_new_lora: bool = True,
    fan_in_fan_out: bool = False,
    extra_warnings: Sequence[str] = (),
) -> Any:
    """Supervised fine-tuning (the warm-up stage, or the whole job)."""

    from trl import SFTTrainer

    warnings = list(extra_warnings)
    require_assistant_markers(tokenizer)
    train_dataset = prepare_sft_dataset(train_records)
    _check_data(
        records=train_records,
        minimum=MIN_SFT_EXAMPLES,
        objective=TrainingObjective.SFT,
        label="usable supervised examples",
    )
    if len(train_dataset) < len(train_records):
        warnings.append(
            f"{len(train_records) - len(train_dataset)} sft record(s) were dropped as malformed"
        )
    if not train_dataset:
        raise InsufficientDataError(
            "no usable supervised examples after validation",
            hint="Every sft.jsonl record must carry a 'messages' list ending in an assistant turn.",
        )
    eval_dataset = prepare_sft_dataset(eval_records) if eval_records else None
    config = build_sft_config(hp, paths, run_name=run_name, report_to=report_to)
    config.warmup_steps = _warmup_steps(
        hp, _estimate_total_steps(hp, len(train_dataset), hp.sft_epochs)
    )
    if eval_dataset is not None and len(eval_dataset):
        config.eval_strategy = "steps"
    if apply_new_lora and peft_config is None:
        peft_config = build_lora_config(hp, model, fan_in_fan_out=fan_in_fan_out)
    elif not apply_new_lora:
        peft_config = None

    return _run_trainer(
        trainer_cls=SFTTrainer,
        objective=TrainingObjective.SFT,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        run_id=run_id,
        paths=paths,
        hp=hp,
        bus=bus,
        cancel_event=cancel_event,
        pause_event=pause_event,
        store=store,
        resume_from=resume_from,
        adapter_dir=adapter_dir,
        artifact_name="adapter-sft",
        epochs=hp.sft_epochs,
        warnings=warnings,
    )


def run_dpo(
    *,
    run_id: str,
    paths: RunPaths,
    hp: Hyperparameters,
    model: Any,
    tokenizer: Any,
    preference_records: Sequence[dict],
    bus: EventBus,
    system_prompt: str | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
    store: RunStore | None = None,
    eval_records: Sequence[dict] = (),
    resume_from: Path | None = None,
    report_to: Sequence[str] = (),
    run_name: str = "",
    adapter_dir: Path | None = None,
    peft_config: Any | None = None,
    apply_new_lora: bool = True,
    fan_in_fan_out: bool = False,
    extra_warnings: Sequence[str] = (),
) -> Any:
    """Preference optimisation (DPO).

    With ``ref_model=None`` (see :data:`trainer_kwargs`) TRL uses the
    adapter-disabled model as the reference, so no second copy of the base model
    is materialised -- which is what makes DPO affordable on one device.
    """

    from trl import DPOTrainer

    warnings = list(extra_warnings)
    require_assistant_markers(tokenizer)
    train_dataset = prepare_dpo_dataset(preference_records, system_prompt=system_prompt)
    _check_data(
        records=preference_records,
        minimum=MIN_PREFERENCE_PAIRS,
        objective=TrainingObjective.DPO,
        label="usable preference pairs",
    )
    if len(train_dataset) < len(preference_records):
        warnings.append(
            f"{len(preference_records) - len(train_dataset)} preference record(s) were "
            "dropped as malformed"
        )
    if not train_dataset:
        raise InsufficientDataError(
            "no usable preference pairs after validation",
            hint=(
                "Each preferences.jsonl record needs a prompt and non-identical "
                "chosen_text/rejected_text."
            ),
        )
    eval_dataset = (
        prepare_dpo_dataset(eval_records, system_prompt=system_prompt) if eval_records else None
    )
    config = build_dpo_config(hp, paths, run_name=run_name, report_to=report_to)
    config.warmup_steps = _warmup_steps(
        hp, _estimate_total_steps(hp, len(train_dataset), hp.num_train_epochs)
    )
    if eval_dataset is not None and len(eval_dataset):
        config.eval_strategy = "steps"
    if apply_new_lora and peft_config is None:
        peft_config = build_lora_config(hp, model, fan_in_fan_out=fan_in_fan_out)
    elif not apply_new_lora:
        peft_config = None

    return _run_trainer(
        trainer_cls=DPOTrainer,
        objective=TrainingObjective.DPO,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        trainer_kwargs={"ref_model": None},
        run_id=run_id,
        paths=paths,
        hp=hp,
        bus=bus,
        cancel_event=cancel_event,
        pause_event=pause_event,
        store=store,
        resume_from=resume_from,
        adapter_dir=adapter_dir,
        artifact_name="adapter",
        epochs=hp.num_train_epochs,
        warnings=warnings,
    )


__all__ = [
    "MIN_PREFERENCE_PAIRS",
    "MIN_SFT_EXAMPLES",
    "TokenMetricsCallback",
    "build_dpo_config",
    "build_lora_config",
    "build_sft_config",
    "checkpoint_root",
    "ensure_trl_logprob_compat",
    "find_resume_checkpoint",
    "prepare_dpo_dataset",
    "prepare_sft_dataset",
    "require_assistant_markers",
    "run_dpo",
    "run_sft",
    "validate_dpo_records",
    "validate_sft_records",
]
