"""Compatibility shims for the installed TRL version.

TRL 1.15 patches the model's ``forward`` with a *fused LM head* in both
:class:`~trl.SFTTrainer` and :class:`~trl.DPOTrainer` and always calls it with
``fused_lm_head=True``.  That path computes per-token log-probabilities with
``trl.kernels.ChunkedLogProbFunction``, a **Triton** kernel.  Triton cannot
initialise a driver on a machine with no accelerator, so on CPU-only (and
otherwise driver-less) hosts every training step raises::

    RuntimeError: 0 active drivers ([]). There should only be one.

There is no configuration flag to opt out of the fused head, and replacing the
model's ``forward`` instead would duplicate TRL's masking, MoE-aux-loss and
label-mask semantics.  So this module substitutes *only the numerical kernel*
with an equivalent pure-PyTorch implementation.  Everything else in TRL's path
(``outputs`` tuple handling, ``masked_scatter`` onto the label mask, the loss
reduction, ``num_items_per_batch``) is untouched.

The substitute is deliberately faithful to the kernel's documented semantics:

* ``z = (hidden @ weight.T + bias) * logit_scale``
* if ``final_logit_softcapping`` is set, ``z = softcap * (2 * sigmoid(2z/softcap) - 1)``
* ``z = z / temperature``
* ``log_probs = z[target] - logsumexp(z)``
* ``entropy = logsumexp(z) - sum(softmax(z) * z)``
* ``log_sum_sq_probs = log(sum(softmax(z)**2))``
* ``mean_logits = mean(z)``
* ``is_top1 = target == argmax(z)`` as a **bool** tensor (DPO bitwise-ANDs it)

Because it is written with ordinary differentiable tensor ops, autograd handles
the backward pass exactly — including the soft-capping jacobian, which a
hand-written backward would be likely to get wrong.  Gradients are verified
against a direct reference computation in ``tests/test_foundation_trl_compat.py``.

Memory characteristic, stated honestly: the substitute materialises the logits
for one token chunk at a time and relies on autograd's graph for the backward
pass, so peak memory is a few multiples of ``chunk * vocab``.  The Triton kernel
is better at this.  On a CPU-only host, large-vocabulary models are out of reach
anyway, and this is what makes a real training run possible at all.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger("autorlhf.trl_compat")

#: Optional per-token outputs the kernel can return, in the kernel's order.
OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")

#: Tokens processed per projection chunk.  Smaller bounds peak memory.
TOKEN_CHUNK_SIZE = 512

_PATCHED = False


def triton_kernel_usable() -> bool:
    """Whether TRL's Triton kernel can actually run here.

    Requires an accelerator (the kernel is only correct on CUDA/XPU) *and* a
    Triton driver that initialises.
    """

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a hard dependency
        return False

    device = getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()
    if not device:
        xpu = getattr(getattr(torch, "xpu", None), "is_available", lambda: False)()
        if not xpu:
            return False
    try:
        import triton  # noqa: F401
        from triton.runtime import driver

        driver.active.get_current_device()
        return True
    except Exception:
        return False


class TorchChunkedLogProbFunction:
    """Drop-in replacement for ``trl.kernels.ChunkedLogProbFunction``.

    Only the ``apply`` entry point is used by TRL, and its argument order and
    return order match the kernel exactly.
    """

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
        outputs: tuple[str, ...] = OPTIONAL_OUTPUTS,
    ) -> tuple[Any, Any, Any, Any, Any]:
        import torch

        token_chunk = TOKEN_CHUNK_SIZE
        out_dtype = hidden.dtype
        # CPU matmul is fastest and most predictable in fp32; gradient flow is
        # unaffected by the upcast.
        compute_weight = weight.float() if not weight.is_floating_point() else weight
        compute_bias = None if bias is None else bias.float()

        log_probs_parts: list[Any] = []
        entropy_parts: list[Any] = []
        log_sum_sq_parts: list[Any] = []
        mean_logits_parts: list[Any] = []
        top1_parts: list[Any] = []

        total = hidden.shape[0]
        for start in range(0, total, token_chunk):
            end = min(start + token_chunk, total)
            hidden_chunk = hidden[start:end]
            if hidden_chunk.dtype != compute_weight.dtype:
                hidden_chunk = hidden_chunk.to(compute_weight.dtype)

            # [n, V] — this is the tensor TRL's kernel avoids materialising.
            logits = torch.nn.functional.linear(hidden_chunk, compute_weight, compute_bias)
            if logit_scale != 1.0:
                logits = logits * logit_scale
            if final_logit_softcapping:
                softcap = final_logit_softcapping
                logits = softcap * (2 * torch.sigmoid(2 * logits / softcap) - 1)
            if temperature != 1.0:
                logits = logits / temperature

            log_z = torch.logsumexp(logits, dim=-1)
            target_logits = logits.gather(-1, targets[start:end].unsqueeze(-1)).squeeze(-1)
            log_probs_parts.append((target_logits - log_z).to(out_dtype))

            with torch.no_grad():
                probabilities = torch.softmax(logits, dim=-1)
                # entropy = logsumexp(z) - sum(p * z), the stable form of
                # -sum(p * log p).  Not -(p * z).sum(-1), which is -E[z].
                entropy_parts.append((log_z - (probabilities * logits).sum(-1)).to(out_dtype))
                log_sum_sq_parts.append(
                    torch.log((probabilities * probabilities).sum(-1).clamp_min(1e-30)).to(
                        out_dtype
                    )
                )
                mean_logits_parts.append(logits.mean(-1).to(out_dtype))
                # Must stay bool: DPO does `is_top1 & chosen_mask.bool()`.  The
                # Triton kernel also returns a bool tensor here.  `argmax` already
                # returns the first maximum, matching the kernel's comparison.
                top1_parts.append(logits.argmax(-1) == targets[start:end])

        if not log_probs_parts:  # empty batch
            empty = hidden.new_zeros((0,))
            return (
                empty,
                empty.clone(),
                empty.clone(),
                empty.clone(),
                hidden.new_empty((0,), dtype=torch.bool),
            )

        def _join(parts: list[Any]) -> Any:
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

        return (
            _join(log_probs_parts),
            _join(entropy_parts),
            _join(log_sum_sq_parts),
            _join(mean_logits_parts),
            _join(top1_parts),
        )


def install_cpu_logprob_fallback(*, force: bool = False) -> bool:
    """Patch TRL's log-prob kernel with the pure-PyTorch substitute.

    Returns ``True`` when the substitute is installed (either now or from an
    earlier call), ``False`` when the real Triton kernel is usable and left
    alone.  Idempotent, and safe to call when TRL is not installed.
    """

    global _PATCHED
    if _PATCHED:
        return True
    if not force and triton_kernel_usable():
        return False

    try:
        kernels = importlib.import_module("trl.kernels")
        trainer_utils = importlib.import_module("trl.trainer.utils")
    except ImportError:  # pragma: no cover - TRL is a hard dependency
        return False

    # ``trl.trainer.utils`` binds the name at import time, so patching the
    # kernels module alone would not take effect.
    kernels.ChunkedLogProbFunction = TorchChunkedLogProbFunction  # type: ignore[attr-defined]
    if hasattr(trainer_utils, "_ChunkedLogProbFunction"):
        trainer_utils._ChunkedLogProbFunction = TorchChunkedLogProbFunction  # type: ignore[attr-defined]
    _PATCHED = True
    logger.info(
        "installed pure-PyTorch chunked-logprob fallback (TRL Triton kernel unusable "
        "on this device)"
    )
    return True


def trl_logprob_backend() -> str:
    """``"triton"``, ``"torch-fallback"``, or ``"unavailable"``."""

    if _PATCHED:
        return "torch-fallback"
    try:
        importlib.import_module("trl")
    except ImportError:
        return "unavailable"
    if not hasattr(importlib.import_module("trl.trainer.utils"), "_ChunkedLogProbFunction"):
        return "unavailable"
    return "triton" if triton_kernel_usable() else "torch"


__all__ = [
    "OPTIONAL_OUTPUTS",
    "TOKEN_CHUNK_SIZE",
    "TorchChunkedLogProbFunction",
    "install_cpu_logprob_fallback",
    "triton_kernel_usable",
    "trl_logprob_backend",
]
