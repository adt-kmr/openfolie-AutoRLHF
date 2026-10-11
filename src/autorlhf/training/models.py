"""Model and tokenizer loading.

Heavy dependencies (``torch``, ``transformers``) are imported inside the
functions: the TUI must be able to import the pipeline without paying for them,
and a missing optional package must surface as a typed error rather than an
import failure.

Every fallback is explicit.  A tokenizer that ships no chat template produces a
:class:`ChatTemplateError` unless the run configuration opted into the
documented generic template, in which case the substitution is reported to the
caller as a warning and recorded on :class:`LoadedModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from autorlhf.config.run_config import HFModelConfig, Hyperparameters
from autorlhf.core.enums import TrainingMethod
from autorlhf.core.errors import (
    ChatTemplateError,
    ConfigError,
    ModelLoadError,
    OutOfMemoryError,
    QuantizationUnavailableError,
    UnsupportedArchitectureError,
)
from autorlhf.core.types import ChatMessage

#: The documented generic chat template, installed only when a tokenizer ships
#: none *and* the run configuration opted in.  It is deliberately plain text, but
#: it *does* wrap assistant content in ``{% generation %}`` markers: TRL 1.15
#: cannot prepare a conversational dataset without them (verified on this machine),
#: so the fallback has to be usable rather than merely pretty.
GENERIC_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: "
    "{% if message['role'] == 'assistant' %}"
    "{% generation %}{{ message['content'] }}{% endgeneration %}"
    "{% else %}{{ message['content'] }}{% endif %}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)

#: Substrings in a transformers error that mean "the repo is gated/private".
_AUTH_MARKERS = (
    "401",
    "403",
    "requires authentication",
    "gated",
    "private",
    "you must be authenticated",
    "access to model",
    "token is required",
)

#: Substrings that mean "this config is not a supported causal LM".
_ARCHITECTURE_MARKERS = (
    "unrecognized configuration class",
    "unrecognized model",
    "should have a `model_type`",
    "unsupported model type",
    "does not support",
    "not supported for",
    "is not a causal",
    "unsupported model",
)


@dataclass
class LoadedModel:
    """A ready-to-use model plus the provenance the manifest needs."""

    model_id: str
    resolved_revision: str
    tokenizer: Any
    model: Any
    param_count: int
    #: ``"tokenizer"`` or ``"generic-fallback"``.
    chat_template_source: str
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# error translation
# --------------------------------------------------------------------------- #


def _raise_model_error(exc: BaseException, cfg: HFModelConfig, *, context: str) -> None:
    """Translate a transformers/torch failure into a typed autorlhf error."""

    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered:
        raise OutOfMemoryError(
            f"{context} ran out of memory loading '{cfg.model_id}'",
            hint=(
                "Reduce max_seq_length or batch size, use a smaller base model, or "
                "enable QLoRA on a GPU with enough VRAM."
            ),
        ) from exc
    if any(marker in lowered for marker in _AUTH_MARKERS):
        raise ModelLoadError(
            f"{context} failed for '{cfg.model_id}': the repository is gated or private",
            hint="Set HF_TOKEN (autorlhf --setup) and confirm you accepted the model licence.",
        ) from exc
    if any(marker in lowered for marker in _ARCHITECTURE_MARKERS):
        raise UnsupportedArchitectureError(
            f"'{cfg.model_id}' is not a supported causal LM for AutoModelForCausalLM: {message}",
            hint="Choose a decoder-only causal LM, or check the model card for custom code.",
        ) from exc
    raise ModelLoadError(f"{context} failed for '{cfg.model_id}': {message}") from exc


def _resolve_dtype(name: str) -> Any:
    import torch

    mapping = {
        "auto": None,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError:
        raise ConfigError(
            f"unknown torch_dtype '{name}'",
            hint="Use one of: auto, float32, float16, bfloat16.",
        ) from None


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #


def load_tokenizer(
    cfg: HFModelConfig, *, hf_token: str | None, allow_fallback: bool | None = None
) -> tuple[Any, list[str]]:
    """Load ``cfg.model_id``'s tokenizer, with an explicit chat-template policy.

    Returns ``(tokenizer, warnings)``.  ``allow_fallback`` defaults to
    ``cfg.allow_generic_chat_template``; when a tokenizer ships no chat template
    and the fallback is declined, :class:`ChatTemplateError` is raised instead of
    silently using a template that is not the one the model was trained with.
    """

    from transformers import AutoTokenizer

    warnings: list[str] = []
    if allow_fallback is None:
        allow_fallback = cfg.allow_generic_chat_template

    kwargs: dict[str, Any] = dict(cfg.load_kwargs)
    kwargs.setdefault("revision", cfg.revision)
    kwargs.setdefault("token", hf_token)
    kwargs.setdefault("trust_remote_code", cfg.trust_remote_code)

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - translated into typed errors below
        _raise_model_error(exc, cfg, context="tokenizer load")

    template = getattr(tokenizer, "chat_template", None)
    if not template:
        if not allow_fallback:
            raise ChatTemplateError(
                f"tokenizer for '{cfg.model_id}' ships no chat template",
                hint=(
                    "Enable 'allow generic chat template' to use a documented plain-text "
                    "fallback, or pick an instruct model with a chat template."
                ),
            )
        tokenizer.chat_template = GENERIC_CHAT_TEMPLATE
        setattr(tokenizer, GENERIC_TEMPLATE_MARKER, True)
        warnings.append(
            "tokenizer ships no chat template; installed the documented generic template "
            "(plain 'role: content' formatting). Assistant-only loss masking is unavailable."
        )

    if not chat_template_has_generation_markers(tokenizer):
        warnings.append(
            "the chat template has no {% generation %} markers, so TRL cannot prepare a "
            "conversational dataset with it (either objective); supervised training will "
            "refuse to start until the template can mark assistant spans."
        )

    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise ChatTemplateError(
                f"tokenizer for '{cfg.model_id}' has neither a pad nor an eos token",
                hint="Use a tokenizer that defines pad_token or eos_token.",
            )
        tokenizer.pad_token = eos
        warnings.append(f"tokenizer has no pad token; using eos_token ('{eos}') as padding.")

    if getattr(tokenizer, "pad_token_id", None) is None:
        raise ChatTemplateError(
            f"tokenizer for '{cfg.model_id}' has no usable pad token id",
            hint="Set tokenizer.pad_token (or eos_token) to a token in the vocabulary.",
        )
    return tokenizer, warnings


def chat_template_has_generation_markers(tokenizer: Any) -> bool:
    """Whether the chat template can mark assistant spans.

    TRL builds ``assistant_masks`` only from templates containing
    ``{% generation %}`` ... ``{% endgeneration %}`` blocks.  Without them, TRL's
    conversational tokenisation fails outright (a confusing
    ``WordLevel error: Missing [UNK] token from the vocabulary``) or produces no
    mask at all, so this has to be known *before* a trainer is built.
    """

    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        return any(
            isinstance(value, str) and "{% generation %}" in value for value in template.values()
        )
    return isinstance(template, str) and "{% generation %}" in template


#: Instance marker set when :data:`GENERIC_CHAT_TEMPLATE` was installed.  A string
#: comparison against the template would be wrong: a real tokenizer may ship a
#: byte-identical template, and calling that a fallback would misreport provenance.
GENERIC_TEMPLATE_MARKER = "_autorlhf_generic_chat_template"


def chat_template_source(tokenizer: Any) -> str:
    """``"tokenizer"`` or ``"generic-fallback"`` for the template in use."""

    return (
        "generic-fallback"
        if getattr(tokenizer, GENERIC_TEMPLATE_MARKER, False)
        else "tokenizer"
    )


def build_chat_prompt(tokenizer: Any, messages: list[ChatMessage] | list[dict[str, str]]) -> str:
    """Render ``messages`` with the tokenizer's chat template, as plain text.

    ``add_generation_prompt=True`` so the model is asked to continue as the
    assistant, exactly as it was framed during training.
    """

    conversation = [
        {"role": m.role, "content": m.content}
        if isinstance(m, ChatMessage)
        else {"role": str(m["role"]), "content": str(m["content"])}
        for m in messages
    ]
    rendered = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    if not isinstance(rendered, str):
        raise ChatTemplateError(
            "the tokenizer's chat template did not render text",
            hint="Check the tokenizer's chat_template; it must support tokenize=False.",
        )
    return rendered


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


def count_parameters(model: Any) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _quantization_config(hp: Hyperparameters) -> Any:
    """Build the 4-bit config, or raise with the specific reason it cannot run."""

    from autorlhf.training.hardware import detect_hardware

    supported, reason = detect_hardware(with_bitsandbytes=True).supports(TrainingMethod.QLORA)
    if not supported:
        raise QuantizationUnavailableError(
            f"4-bit loading was requested but is not available: {reason}",
            hint="Train with plain LoRA, or run on a CUDA GPU with bitsandbytes installed.",
        )
    import torch
    from transformers import BitsAndBytesConfig

    compute_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    compute_dtype = compute_dtypes.get(hp.bnb_4bit_compute_dtype)
    if compute_dtype is None:
        raise ConfigError(
            f"unknown bnb_4bit_compute_dtype '{hp.bnb_4bit_compute_dtype}'",
            hint="Use bfloat16, float16 or float32.",
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=hp.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=hp.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_model(
    cfg: HFModelConfig,
    hp: Hyperparameters,
    *,
    hf_token: str | None,
    for_training: bool = False,
    device: str | None = None,
) -> LoadedModel:
    """Load the base model described by ``cfg`` in the shape ``hp`` requires."""

    from transformers import AutoModelForCausalLM

    tokenizer, warnings = load_tokenizer(cfg, hf_token=hf_token)

    kwargs: dict[str, Any] = dict(cfg.load_kwargs)
    kwargs.setdefault("revision", cfg.revision)
    kwargs.setdefault("token", hf_token)
    kwargs.setdefault("trust_remote_code", cfg.trust_remote_code)
    dtype = _resolve_dtype(cfg.torch_dtype)
    if dtype is not None:
        kwargs.setdefault("dtype", dtype)
    if cfg.attn_implementation:
        kwargs.setdefault("attn_implementation", cfg.attn_implementation)

    quantized = bool(hp.load_in_4bit) or hp.method is TrainingMethod.QLORA
    if quantized:
        kwargs["quantization_config"] = _quantization_config(hp)
        kwargs["device_map"] = device or "auto"
        warnings.append(
            f"loaded 4-bit ({hp.bnb_4bit_quant_type}, compute "
            f"{hp.bnb_4bit_compute_dtype}); base weights are quantized on device."
        )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - translated into typed errors below
        _raise_model_error(exc, cfg, context="model load")

    if not quantized and device and device != "auto":
        model = model.to(device)

    if quantized and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=hp.gradient_checkpointing
        )
    if not for_training:
        model.eval()

    revision = getattr(getattr(model, "config", None), "_commit_hash", None) or (
        cfg.revision or "main (unpinned)"
    )
    return LoadedModel(
        model_id=cfg.model_id,
        resolved_revision=str(revision),
        tokenizer=tokenizer,
        model=model,
        param_count=count_parameters(model),
        chat_template_source=chat_template_source(tokenizer),
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# metadata-only parameter estimate
# --------------------------------------------------------------------------- #


def _estimate_parameters_from_config(config: Any) -> int | None:
    """Parameter count from architecture fields only -- no weights downloaded."""

    if getattr(config, "is_encoder_decoder", False):
        return None
    hidden = (
        getattr(config, "hidden_size", None)
        or getattr(config, "n_embd", None)
        or getattr(config, "d_model", None)
    )
    layers = (
        getattr(config, "num_hidden_layers", None)
        or getattr(config, "n_layer", None)
        or getattr(config, "num_layers", None)
    )
    vocab = getattr(config, "vocab_size", None)
    if not hidden or not layers or not vocab:
        return None

    heads = getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None) or 1
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    head_dim = int(getattr(config, "head_dim", 0) or max(1, hidden // max(1, heads)))
    intermediate = (
        getattr(config, "intermediate_size", None) or getattr(config, "n_inner", None) or 4 * hidden
    )
    # Attention: q, k, v projections plus the output projection.
    attention = (
        hidden * head_dim * heads + 2 * hidden * head_dim * kv_heads + head_dim * heads * hidden
    )
    # MLP: 3 matrices for gated families (gate/up/down), 2 for classic ones.
    gated = getattr(config, "model_type", "") not in {"gpt2", "gpt_bigcode", "opt", "bloom"}
    mlp = (3 if gated else 2) * hidden * intermediate
    per_layer = attention + mlp + 2 * hidden  # two layernorms per block
    total = layers * per_layer + vocab * hidden + hidden
    if not getattr(config, "tie_word_embeddings", True):
        total += vocab * hidden
    return int(total)


def model_param_count_from_metadata(cfg: HFModelConfig, *, hf_token: str | None) -> int | None:
    """Approximate parameter count from the model's *config* only.

    Downloads the few-kilobyte config, never the weights.  Returns ``None`` when
    the architecture fields needed for the estimate are absent.
    """

    from transformers import AutoConfig

    kwargs: dict[str, Any] = {}
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    if hf_token:
        kwargs["token"] = hf_token
    if cfg.trust_remote_code:
        kwargs["trust_remote_code"] = True
    try:
        config = AutoConfig.from_pretrained(cfg.model_id, **kwargs)
    except Exception as exc:  # noqa: BLE001 - translated into typed errors below
        _raise_model_error(exc, cfg, context="config load")
    return _estimate_parameters_from_config(config)


# --------------------------------------------------------------------------- #
# local generation
# --------------------------------------------------------------------------- #


def generate_text(
    model: Any,
    tokenizer: Any,
    messages: list[ChatMessage] | list[dict[str, str]],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Generate one continuation locally.

    Sampling is enabled only for ``temperature > 0``; at ``temperature <= 0`` the
    call is greedy (``do_sample=False``), which is the only combination that is
    valid for ``model.generate``.  Returns ``(text, meta)``.
    """

    import torch

    prompt = build_chat_prompt(tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt")
    target_device = device or str(next(model.parameters()).device)
    encoded = {key: value.to(target_device) for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])

    do_sample = temperature > 0
    generation: dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation["temperature"] = float(temperature)
        if 0.0 < top_p <= 1.0:
            generation["top_p"] = float(top_p)

    if seed is not None:
        # Documented side effect: seeding the global RNG keeps sampling for a
        # given request reproducible at the cost of touching process state.
        torch.manual_seed(int(seed))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output = model.generate(**encoded, **generation)
    finally:
        if was_training:
            model.train()

    completion = output[0][prompt_tokens:]
    text = tokenizer.decode(completion, skip_special_tokens=True)
    meta: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion.shape[-1]),
        "truncated": int(completion.shape[-1]) >= int(max_new_tokens),
        "do_sample": do_sample,
        "temperature": float(temperature),
        "top_p": float(top_p) if do_sample else None,
        "seed": seed,
        "device": target_device,
        "dtype": str(getattr(model, "dtype", "")),
    }
    return text, meta


__all__ = [
    "GENERIC_CHAT_TEMPLATE",
    "LoadedModel",
    "GENERIC_TEMPLATE_MARKER",
    "build_chat_prompt",
    "chat_template_has_generation_markers",
    "chat_template_source",
    "count_parameters",
    "generate_text",
    "load_model",
    "load_tokenizer",
    "model_param_count_from_metadata",
]
