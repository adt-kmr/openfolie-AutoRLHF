"""Hyperparameter presets, hardware-aware feasibility, and resource estimates.

Design rules (deliberate, keep them):

* autorlhf never asks the judge to invent a learning rate.  It generates a
  **closed set** of predefined configurations, filters them by measured
  hardware, and only then may ask Jev to *rank* the survivors with a ``choice``
  decision.  Jev's probabilities are shown as the judge's preference, never as a
  probability that training will succeed.
* Everything in this module is a pure function of its arguments, so the rules
  are unit-testable with a hand-built :class:`HardwareInfo` and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import PresetId, TrainingMethod, TrainingObjective
from autorlhf.core.errors import HardwareError
from autorlhf.core.types import HardwareInfo, ResourceEstimate

MIB = 1024 * 1024
GIB = 1024 * MIB


class TrainingPreset(BaseModel):
    """A named, self-consistent bundle of hyperparameters."""

    id: PresetId
    label: str
    summary: str
    method: TrainingMethod
    #: Objectives this preset is intended for.  Both are allowed for most
    #: presets; DPO-specific ones raise ``dpo_beta`` only.
    objectives: list[TrainingObjective] = Field(
        default_factory=lambda: [TrainingObjective.SFT, TrainingObjective.DPO]
    )
    #: Rough VRAM floor in MiB.  ``None`` means "CPU-capable / no floor".
    min_vram_mib: int | None = None
    requires_accelerator: bool = False
    experimental: bool = False
    warning: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    def to_hyperparameters(self, base: Hyperparameters | None = None) -> Hyperparameters:
        """Apply this preset's numerics onto ``base`` (or the defaults)."""

        data = (base or Hyperparameters()).model_dump(mode="json")
        data.update(self.params)
        data["method"] = self.method.value
        data["preset_id"] = self.id.value
        # Presets that do not opt into 4-bit explicitly must clear it, so a
        # preset switch is always a complete description of the method.
        data["load_in_4bit"] = self.method is TrainingMethod.QLORA
        merge = Hyperparameters.model_validate(data)
        # Target modules stay as discovered/edited by the user unless the preset
        # names them explicitly.
        if "lora_target_modules" not in self.params:
            merge.lora_target_modules = list((base or Hyperparameters()).lora_target_modules)
        return merge


PRESETS: list[TrainingPreset] = [
    TrainingPreset(
        id=PresetId.CPU_TINY_LORA,
        label="CPU / tiny LoRA",
        summary=(
            "Smallest viable LoRA run for CPU-only machines or a smoke test: rank 8, "
            "short sequences, batch 1 with heavy accumulation."
        ),
        method=TrainingMethod.LORA,
        min_vram_mib=None,
        params={
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 512,
            "gradient_checkpointing": False,
            "optim": "adamw_torch",
            "save_steps": 20,
            "eval_steps": 20,
            "logging_steps": 2,
            "bf16": False,
            "fp16": False,
        },
    ),
    TrainingPreset(
        id=PresetId.SINGLE_GPU_SAFE_LORA,
        label="Single GPU / safe LoRA",
        summary=(
            "Default single-GPU LoRA: rank 16, bf16 where supported, gradient "
            "checkpointing on, effective batch 8."
        ),
        method=TrainingMethod.LORA,
        min_vram_mib=6 * 1024,
        requires_accelerator=True,
        params={
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 1e-5,
            "num_train_epochs": 2.0,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "max_seq_length": 1024,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "bf16": True,
            "save_steps": 25,
            "eval_steps": 25,
            "logging_steps": 5,
        },
    ),
    TrainingPreset(
        id=PresetId.SINGLE_GPU_QLORA,
        label="Single GPU / QLoRA 4-bit",
        summary=(
            "4-bit NF4 base weights with bf16 compute: runs a much larger model than "
            "plain LoRA on the same VRAM, at a small quality cost."
        ),
        method=TrainingMethod.QLORA,
        min_vram_mib=6 * 1024,
        requires_accelerator=True,
        params={
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_train_epochs": 2.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bf16": True,
            "save_steps": 25,
            "eval_steps": 25,
            "logging_steps": 5,
        },
    ),
    TrainingPreset(
        id=PresetId.SINGLE_GPU_FAST_LORA,
        label="Single large GPU / fast LoRA",
        summary=(
            "Higher rank (64) and larger batches for >= 24 GiB accelerators; fastest "
            "wall-clock but the most VRAM-hungry LoRA preset."
        ),
        method=TrainingMethod.LORA,
        min_vram_mib=24 * 1024,
        requires_accelerator=True,
        params={
            "lora_r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "learning_rate": 8e-6,
            "num_train_epochs": 2.0,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 2,
            "max_seq_length": 2048,
            "gradient_checkpointing": False,
            "optim": "adamw_torch_fused",
            "bf16": True,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 5,
        },
    ),
    TrainingPreset(
        id=PresetId.MULTI_GPU_FFT,
        label="Full fine-tuning (experimental)",
        summary=(
            "Trains every weight. Highest quality ceiling, needs a large GPU or several "
            "of them, and is explicitly experimental."
        ),
        method=TrainingMethod.FFT,
        min_vram_mib=48 * 1024,
        requires_accelerator=True,
        experimental=True,
        warning=(
            "Full fine-tuning rewrites every weight: it needs tens of GiB of VRAM for a "
            "1B model and produces a multi-gigabyte checkpoint. Prefer LoRA unless you "
            "have confirmed the hardware headroom."
        ),
        params={
            "learning_rate": 2e-6,
            "num_train_epochs": 1.0,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "max_seq_length": 1024,
            "gradient_checkpointing": True,
            "optim": "adamw_torch",
            "bf16": True,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 5,
        },
    ),
]

PRESETS_BY_ID: dict[str, TrainingPreset] = {p.id.value: p for p in PRESETS}


def get_preset(preset_id: str | PresetId) -> TrainingPreset:
    key = preset_id.value if isinstance(preset_id, PresetId) else str(preset_id)
    try:
        return PRESETS_BY_ID[key]
    except KeyError:
        raise HardwareError(
            f"unknown training preset '{key}'",
            hint="Valid presets: " + ", ".join(sorted(PRESETS_BY_ID)),
        ) from None


# --------------------------------------------------------------------------- #
# feasibility
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PresetFeasibility:
    preset: TrainingPreset
    feasible: bool
    reason: str = ""
    estimate: ResourceEstimate | None = None
    #: True when it fits but only barely (< 15% VRAM headroom).
    tight: bool = False


def _headroom_ok(estimate: ResourceEstimate, hw: HardwareInfo, *, margin: float = 1.15) -> bool:
    if hw.total_vram_mib is None:
        return True
    return estimate.required_vram_mib * margin <= hw.total_vram_mib


def estimate_trainable_params(hp: Hyperparameters, param_count: int | None) -> int | None:
    """Approximate trainable parameter count.

    For LoRA this is a *rough* estimate: with ``r`` and hidden size ``h`` circa
    ``sqrt(param_count / 12 / layers)``, each adapted projection costs about
    ``2 * r * h``.  It is used only for display and preflight, with the
    assumptions recorded.
    """

    if param_count is None:
        return None
    if hp.method is TrainingMethod.FFT:
        return param_count
    # Coarse: 7 projections per layer (q,k,v,o,gate,up,down) is the Llama family
    # maximum; hidden size is approximated from the parameter count.
    hidden = max(64.0, (param_count / 12.0) ** 0.5)
    projections = max(1, len(hp.lora_target_modules) or 7)
    return int(2 * hp.lora_r * hidden * projections)


def estimate_training_memory(
    hp: Hyperparameters,
    *,
    param_count: int | None,
    gpu_count: int = 1,
) -> ResourceEstimate:
    """Coarse VRAM/RAM model.

    Honest about its limits: weights and optimizer state are computed from the
    parameter count, activation memory is a *fitted* rule of thumb keyed on
    parameter count, batch size and sequence length.  Every assumption is listed
    in :attr:`ResourceEstimate.assumptions`.
    """

    assumptions = [
        "weights and optimizer state derived from the parameter count",
        "activation memory is a rule-of-thumb estimate, not a measurement",
        f"single process, {max(1, gpu_count)} device(s)",
    ]
    if param_count is None:
        # Unknown model: report a placeholder that is clearly labelled.
        return ResourceEstimate(
            required_vram_mib=4 * 1024,
            required_ram_mib=8 * 1024,
            assumptions=assumptions + ["model size unknown; using a placeholder"],
        )

    if hp.method is TrainingMethod.QLORA:
        weight_bytes = param_count * 0.55  # 4-bit + scales/zeros overhead
        assumptions.append("4-bit NF4 weights ~0.55 bytes/parameter")
    else:
        dtype_bytes = 2 if (hp.bf16 or hp.fp16) else 4
        weight_bytes = param_count * dtype_bytes
        assumptions.append(f"{'bf16/fp16' if dtype_bytes == 2 else 'fp32'} weights")

    trainable = estimate_trainable_params(hp, param_count) or 0
    assumptions.append(f"~{trainable / 1e6:.1f}M trainable parameters")
    # AdamW fp32 m/v + fp32 master copy + grads of the same dtype as weights.
    trainable_state = trainable * (16 if hp.method is not TrainingMethod.FFT else 20)
    if hp.method is TrainingMethod.FFT:
        trainable_state += param_count * (2 if (hp.bf16 or hp.fp16) else 4)

    # Activations: fitted term, scaled by batch and sequence length, reduced 4x
    # by gradient checkpointing.
    seq = hp.max_seq_length
    batch = hp.per_device_train_batch_size
    base_params = param_count / 1e9
    act_bytes = 0.9 * GIB * (base_params**0.6) * batch * (seq / 1024.0)
    if hp.gradient_checkpointing:
        act_bytes *= 0.35
        assumptions.append("gradient checkpointing reduces activation memory")
    if hp.method is TrainingMethod.QLORA:
        act_bytes *= 0.6

    overhead = 0.9 * GIB  # CUDA context, cuBLAS workspaces, fragmentation
    fragmentation = 1.25
    total = (weight_bytes + trainable_state + act_bytes + overhead) * fragmentation

    # Host RAM: weights are read into host memory when there is no accelerator,
    # plus dataset/tokenizer and (for CPU runs) the full training state.
    if hp.method is TrainingMethod.QLORA:
        ram = weight_bytes * 1.2 + 2 * GIB
    else:
        ram = param_count * 2 + 2 * GIB
    if gpu_count <= 1:
        ram += 0.5 * GIB

    return ResourceEstimate(
        required_vram_mib=int(total / MIB),
        required_ram_mib=int(ram / MIB),
        trainable_parameters=trainable,
        total_parameters=param_count,
        assumptions=assumptions,
    )


def preset_feasibility(
    preset: TrainingPreset,
    hw: HardwareInfo,
    *,
    objective: TrainingObjective | None = None,
    param_count: int | None = None,
) -> PresetFeasibility:
    """Decide whether ``preset`` can run on ``hw``, with a specific reason."""

    if objective is not None and objective not in preset.objectives:
        return PresetFeasibility(
            preset, False, f"{preset.label} is not configured for {objective.value.upper()}."
        )

    supported, reason = hw.supports(preset.method)
    if not supported:
        return PresetFeasibility(preset, False, reason)

    if preset.requires_accelerator and not hw.has_accelerator:
        return PresetFeasibility(
            preset, False, "requires a CUDA/ROCm/MPS accelerator; none detected."
        )

    if preset.min_vram_mib is not None:
        if not hw.has_accelerator:
            return PresetFeasibility(
                preset, False, f"requires {preset.min_vram_mib // 1024} GiB of VRAM."
            )
        if hw.total_vram_mib is not None and hw.total_vram_mib < preset.min_vram_mib:
            return PresetFeasibility(
                preset,
                False,
                f"needs >= {preset.min_vram_mib // 1024} GiB VRAM, "
                f"detected {hw.total_vram_mib // 1024} GiB.",
            )

    hp = preset.to_hyperparameters()

    estimate = estimate_training_memory(
        hp, param_count=param_count, gpu_count=max(1, hw.device_count)
    )
    if hw.has_accelerator and not _headroom_ok(estimate, hw):
        return PresetFeasibility(
            preset,
            False,
            (
                f"estimated {estimate.required_vram_mib / 1024:.1f} GiB VRAM for "
                f"{preset.label}, detected {hw.total_vram_mib / 1024:.1f} GiB."
            ),
            estimate=estimate,
        )

    tight = False
    if hw.has_accelerator and hw.total_vram_mib:
        tight = estimate.required_vram_mib * 1.6 > hw.total_vram_mib

    return PresetFeasibility(preset, True, "", estimate=estimate, tight=tight)


def feasible_presets(
    hw: HardwareInfo,
    *,
    objective: TrainingObjective | None = None,
    param_count: int | None = None,
) -> list[PresetFeasibility]:
    """All presets, feasible ones first, in hardware-preference order."""

    results = [
        preset_feasibility(p, hw, objective=objective, param_count=param_count) for p in PRESETS
    ]
    order = {p.id.value: i for i, p in enumerate(_HARDWARE_PREFERENCE)}
    results.sort(
        key=lambda r: (
            not r.feasible,
            r.preset.experimental,
            order.get(r.preset.id.value, 99),
        )
    )
    return results


#: Preference order used for the *default* recommendation before any judge
#: ranking happens: cheapest safe thing first.
_HARDWARE_PREFERENCE: list[TrainingPreset] = [
    PRESETS_BY_ID[PresetId.CPU_TINY_LORA],
    PRESETS_BY_ID[PresetId.SINGLE_GPU_SAFE_LORA],
    PRESETS_BY_ID[PresetId.SINGLE_GPU_QLORA],
    PRESETS_BY_ID[PresetId.SINGLE_GPU_FAST_LORA],
    PRESETS_BY_ID[PresetId.MULTI_GPU_FFT],
]


def recommend_preset(
    hw: HardwareInfo,
    *,
    objective: TrainingObjective,
    param_count: int | None = None,
    preferred: str | None = None,
) -> PresetFeasibility | None:
    """Deterministic hardware-only recommendation (the pre-judge default)."""

    if preferred:
        candidate = PRESETS_BY_ID.get(preferred)
        if candidate is not None:
            feasibility = preset_feasibility(
                candidate, hw, objective=objective, param_count=param_count
            )
            if feasibility.feasible:
                return feasibility
    for feasibility in feasible_presets(hw, objective=objective, param_count=param_count):
        if feasibility.feasible:
            return feasibility
    return None


def preset_choice_options(feasibilities: list[PresetFeasibility]) -> dict[str, str]:
    """``option-key -> description`` for a Jev ``choice`` question.

    Keys are the preset ids so the answer maps straight back to a preset.
    """

    return {
        f.preset.id.value: (
            f"{f.preset.label} [{f.preset.method.value}] — {f.preset.summary}"
            + (f" (tight fit: {f.reason})" if f.tight else "")
        )
        for f in feasibilities
        if f.feasible
    }


def preset_ranking_state(
    feasibilities: list[PresetFeasibility],
    *,
    objective: TrainingObjective,
    hw: HardwareInfo,
    hp: Hyperparameters,
    model_id: str,
) -> dict[str, Any]:
    """State object handed to Jev so it ranks with real hardware context."""

    return {
        "task": "rank_training_presets",
        "model_id": model_id,
        "objective": objective.value,
        "hardware": {
            "device_type": hw.device_type,
            "device_name": hw.device_name,
            "device_count": hw.device_count,
            "total_vram_mib": hw.total_vram_mib,
            "total_ram_mib": hw.total_ram_mib,
            "bf16_supported": hw.bf16_supported,
            "bitsandbytes_available": hw.bitsandbytes_available,
        },
        "dataset_constraints": {
            "max_seq_length": hp.max_seq_length,
            "per_device_train_batch_size": hp.per_device_train_batch_size,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
        },
        "candidate_presets": [
            {
                "id": f.preset.id.value,
                "label": f.preset.label,
                "method": f.preset.method.value,
                "summary": f.preset.summary,
                "estimated_vram_mib": f.estimate.required_vram_mib if f.estimate else None,
                "note": f.reason,
            }
            for f in feasibilities
            if f.feasible
        ],
        "instructions": (
            "Pick the preset most likely to complete on this hardware without running out "
            "of memory, while still producing a meaningful improvement. Prefer the cheaper "
            "option when two options are close. This is a preference among the listed "
            "options only."
        ),
    }


def assert_method_runnable(hw: HardwareInfo, method: TrainingMethod) -> None:
    """Raise :class:`HardwareError` if ``method`` cannot run here."""

    ok, reason = hw.supports(method)
    if not ok:
        raise HardwareError(
            f"{method.value} is not available on this machine: {reason}",
            hint="Choose a different fine-tuning method, or install the missing extra.",
        )


__all__ = [
    "PRESETS",
    "PRESETS_BY_ID",
    "PresetFeasibility",
    "TrainingPreset",
    "assert_method_runnable",
    "estimate_trainable_params",
    "estimate_training_memory",
    "feasible_presets",
    "get_preset",
    "preset_choice_options",
    "preset_feasibility",
    "preset_ranking_state",
    "recommend_preset",
]
