"""LoRA target-module discovery and validation.

Discovery walks the *real* module names of a loaded model instead of guessing
from the model id, so a renamed or fused projection cannot silently produce a
run that trains nothing.  ``validate_target_modules`` is the hard gate: training
never starts with zero matched modules.

GPT-2-style models fuse their projections into ``c_attn``/``c_proj``, whose
weights are stored transposed.  PEFT needs ``fan_in_fan_out=True`` for those, so
:func:`needs_fan_in_fan_out` reports it explicitly and the notes returned by
:func:`discover_target_modules` name it as well.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from autorlhf.core.errors import TargetModulesError

#: Projection suffixes of the Llama/Mistral/Qwen/OLMo decoder family.
_LLAMA_LIKE_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

_LLAMA_LIKE_FAMILIES = frozenset(
    {
        "llama",
        "mistral",
        "mixtral",
        "qwen2",
        "qwen2_moe",
        "qwen3",
        "qwen3_moe",
        "gemma",
        "gemma2",
        "gemma3",
        "olmo",
        "olmo2",
        "olmo3",
        "stablelm",
        "internlm2",
        "deepseek_v2",
        "deepseek_v3",
        "smollm",
        "smollm2",
        "granite",
        "exaone",
        "glm4",
        "nemotron",
    }
)

#: family -> (ordered target suffixes, fan_in_fan_out)
_FAMILY_DEFAULTS: dict[str, tuple[list[str], bool]] = {
    "gpt2": (["c_attn", "c_proj"], True),
    "gpt_bigcode": (["c_attn", "c_proj"], True),
    "phi3": (["qkv_proj", "o_proj", "gate_up_proj", "down_proj"], False),
    "phi": (["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"], False),
    "falcon": (["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], False),
    "mpt": (["Wqkv", "out_proj", "up_proj", "down_proj"], False),
    "bloom": (["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], False),
    "opt": (["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"], False),
    "llama": (_LLAMA_LIKE_TARGETS, False),
    "qwen2": (_LLAMA_LIKE_TARGETS, False),
}

#: Suffixes that are very likely to be attention/MLP projections when the family
#: is unknown; used only by the scored heuristic.
_HEURISTIC_SUFFIXES = (
    *_LLAMA_LIKE_TARGETS,
    "c_attn",
    "c_proj",
    "qkv_proj",
    "query_key_value",
    "Wqkv",
    "out_proj",
    "dense",
    "fc1",
    "fc2",
    "dense_h_to_4h",
    "dense_4h_to_h",
    "gate_up_proj",
    "up_proj",
    "down_proj",
)


def infer_family(model: Any) -> str:
    """Best-effort architecture family of a loaded model.

    Prefers ``config.model_type`` and falls back to the class name, e.g.
    ``"llama"``, ``"qwen2"``, ``"gpt2"``.
    """

    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type:
        return model_type.lower()
    architectures = getattr(config, "architectures", None) or []
    for name in architectures:
        if isinstance(name, str):
            lowered = name.lower()
            for family in ("gpt2", "llama", "qwen", "mistral", "gemma", "falcon", "mpt", "opt"):
                if family in lowered:
                    return family
    class_name = type(model).__name__.lower()
    for family in _FAMILY_DEFAULTS:
        if family in class_name:
            return family
    return "unknown"


def family_defaults(family: str) -> tuple[list[str], bool]:
    """``(target suffixes, fan_in_fan_out)`` for a family, with fallbacks."""

    if family in _FAMILY_DEFAULTS:
        return list(_FAMILY_DEFAULTS[family][0]), _FAMILY_DEFAULTS[family][1]
    if family in _LLAMA_LIKE_FAMILIES:
        return list(_LLAMA_LIKE_TARGETS), False
    return [], False


def needs_fan_in_fan_out(model: Any) -> bool:
    """Whether LoRA on this model must set ``fan_in_fan_out=True``.

    True for GPT-2-style models, whose ``Conv1D`` projections store
    ``(in_features, out_features)`` transposed.
    """

    family = infer_family(model)
    _targets, fan_in_fan_out = family_defaults(family)
    if fan_in_fan_out:
        return True
    # Unknown family: detect a fused/transposed projection by module class name.
    conv1d = [name for name, module in model.named_modules() if type(module).__name__ == "Conv1D"]
    return bool(conv1d)


def _linear_like_names(model: Any) -> list[str]:
    """Full names of every leaf module that LoRA can realistically wrap."""

    names: list[str] = []
    for name, module in model.named_modules():
        # A bare top-level name (e.g. ``lm_head``) is never a projection inside
        # a transformer block, so it is not a sensible LoRA target.
        if "." not in name:
            continue
        if _is_linear_like(module):
            names.append(name)
    return names


def _is_linear_like(module: Any) -> bool:
    class_name = type(module).__name__
    if class_name in {"Linear", "Conv1D", "Linear4bit", "Linear8bitLt"}:
        return True
    return any(base.__name__ == "Linear" for base in type(module).__mro__)


def _present_suffixes(module_names: list[str]) -> dict[str, str]:
    """Map ``leaf suffix -> one example full module name``."""

    present: dict[str, str] = {}
    for name in module_names:
        suffix = name.rsplit(".", 1)[-1]
        present.setdefault(suffix, name)
    return present


def available_linear_modules(model: Any) -> dict[str, int]:
    """``leaf suffix -> number of modules`` for the linear-like modules present."""

    counts: Counter[str] = Counter()
    for name in _linear_like_names(model):
        counts[name.rsplit(".", 1)[-1]] += 1
    return dict(counts)


def discover_target_modules(
    model: Any, *, preferred: list[str] | None = None
) -> tuple[list[str], list[str]]:
    """Choose LoRA target modules from the model's real module names.

    Returns ``(targets, notes)``.  ``preferred`` (from the run configuration)
    wins when the modules exist; otherwise the family defaults are used, and if
    the family is unknown a scored heuristic picks the most plausible
    projection suffixes.  The notes always name the family, the matched modules
    and whether ``fan_in_fan_out`` is required.
    """

    family = infer_family(model)
    fan_in_fan_out = needs_fan_in_fan_out(model)
    module_names = _linear_like_names(model)
    present = _present_suffixes(module_names)
    notes: list[str] = [
        f"detected architecture family '{family}'",
        f"{len(module_names)} linear-like modules; {len(present)} distinct leaf names",
    ]

    chosen: list[str] = []
    if preferred:
        chosen = [name for name in preferred if name in present]
        missing = [name for name in preferred if name not in present]
        if missing:
            notes.append(
                "configured target modules not present in this model and ignored: "
                + ", ".join(missing)
            )
        if chosen:
            notes.append("using the configured target modules: " + ", ".join(chosen))

    if not chosen:
        defaults, _defaults_fan = family_defaults(family)
        chosen = [name for name in defaults if name in present]
        if chosen:
            notes.append("using the family defaults found in the model: " + ", ".join(chosen))

    if not chosen:
        scored = _score_suffixes(module_names)
        chosen = [name for name, _count in scored[:8]]
        if chosen:
            notes.append(
                "no family defaults matched; heuristic picked the most common projections: "
                + ", ".join(chosen)
            )

    notes.append(
        "fan_in_fan_out=True is required for this model (fused/transposed projections)"
        if fan_in_fan_out
        else "fan_in_fan_out=False is correct for this model"
    )
    return chosen, notes


def _score_suffixes(module_names: list[str]) -> list[tuple[str, int]]:
    """Rank leaf suffixes by how often they occur, known names first."""

    counts: Counter[str] = Counter(name.rsplit(".", 1)[-1] for name in module_names)
    known = [suffix for suffix in _HEURISTIC_SUFFIXES if suffix in counts]
    others = sorted(
        (suffix for suffix in counts if suffix not in known),
        key=lambda suffix: (-counts[suffix], suffix),
    )
    ranked = [*known, *others]
    return [(suffix, counts[suffix]) for suffix in ranked]


def validate_target_modules(model: Any, targets: list[str]) -> None:
    """Raise :class:`TargetModulesError` when no target matches a real module.

    Matching accepts either the leaf name (``q_proj``) or a dotted suffix
    (``model.layers.0.self_attn.q_proj``), which is how PEFT resolves them.
    """

    if not targets:
        raise TargetModulesError(
            "no LoRA target modules were configured",
            hint="Refresh target modules: discovery fills them from the loaded model.",
        )
    module_names = [name for name, _module in model.named_modules() if name]
    matched: list[str] = []
    for target in targets:
        if any(name == target or name.endswith("." + target) for name in module_names):
            matched.append(target)
    if not matched:
        available = available_linear_modules(model)
        listing = ", ".join(f"{name} ({count})" for name, count in sorted(available.items()))
        raise TargetModulesError(
            "none of the LoRA target modules "
            f"({', '.join(targets)}) exist in {type(model).__name__}",
            hint=(
                "Available linear module names: "
                + (listing or "none found")
                + ". Re-run target-module discovery."
            ),
        )


__all__ = [
    "available_linear_modules",
    "discover_target_modules",
    "family_defaults",
    "infer_family",
    "needs_fan_in_fan_out",
    "validate_target_modules",
]
