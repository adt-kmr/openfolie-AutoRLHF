"""Foundation tests for the TRL log-probability compatibility shim.

Covers the pure-PyTorch substitute for TRL's Triton ``ChunkedLogProbFunction``:
numerical fidelity against a direct reference, gradient exactness, chunk-boundary
invariance, empty inputs, and the install/backend contract.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from autorlhf.core import trl_compat
from autorlhf.core.trl_compat import TorchChunkedLogProbFunction

VOCAB = 7
HIDDEN = 5

#: (temperature, final_logit_softcapping, logit_scale)
CONFIGURATIONS: list[dict[str, Any]] = [
    {"temperature": 1.0, "final_logit_softcapping": None, "logit_scale": 1.0},
    {"temperature": 1.0, "final_logit_softcapping": 8.0, "logit_scale": 1.0},
    {"temperature": 1.7, "final_logit_softcapping": None, "logit_scale": 0.5},
]
CONFIGURATION_IDS = ["plain", "softcap-8", "scale-0.5-temp-1.7"]


def _inputs(n_tokens: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(n_tokens, HIDDEN, generator=generator, requires_grad=True)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator, requires_grad=True)
    bias = torch.randn(VOCAB, generator=generator, requires_grad=True)
    targets = torch.randint(0, VOCAB, (n_tokens,), generator=generator)
    return hidden, weight, bias, targets


def _reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float,
    final_logit_softcapping: float | None,
    logit_scale: float,
) -> tuple[torch.Tensor, ...]:
    logits = hidden @ weight.T + bias
    logits = logits * logit_scale
    if final_logit_softcapping:
        softcap = final_logit_softcapping
        logits = softcap * (2 * torch.sigmoid(2 * logits / softcap) - 1)
    logits = logits / temperature

    log_z = torch.logsumexp(logits, dim=-1)
    log_probs = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - log_z
    probabilities = torch.softmax(logits, dim=-1)
    entropy = log_z - (probabilities * logits).sum(-1)
    log_sum_sq_probs = torch.log((probabilities * probabilities).sum(-1))
    mean_logits = logits.mean(-1)
    is_top1 = logits.argmax(-1) == targets
    return log_probs, entropy, log_sum_sq_probs, mean_logits, is_top1


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bool or expected.dtype == torch.bool:
        # `is_top1` must stay bool: DPO bitwise-ANDs it with the completion mask
        # (`is_top1 & chosen_mask.bool()`), which raises on a float tensor.
        assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
        assert torch.equal(actual, expected)
        return
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5), (
        (actual - expected).abs().max().item()
    )


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_matches_reference_computation(config: dict[str, Any]) -> None:
    hidden, weight, bias, targets = _inputs(11, seed=1)

    result = TorchChunkedLogProbFunction.apply(
        hidden,
        weight,
        bias,
        targets,
        config["temperature"],
        512,
        config["final_logit_softcapping"],
        config["logit_scale"],
    )
    expected = _reference(hidden, weight, bias, targets, **config)

    assert len(result) == 5
    for actual, reference in zip(result, expected, strict=True):
        _assert_close(actual, reference)


def test_entropy_is_positive_for_random_inputs() -> None:
    hidden, weight, bias, targets = _inputs(64, seed=2)
    _, entropy, _, _, _ = TorchChunkedLogProbFunction.apply(hidden, weight, bias, targets, 1.0, 512)

    assert torch.all(entropy > 0)


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=CONFIGURATION_IDS)
def test_gradients_match_reference(config: dict[str, Any]) -> None:
    hidden, weight, bias, targets = _inputs(9, seed=3)
    log_probs, *_ = TorchChunkedLogProbFunction.apply(
        hidden,
        weight,
        bias,
        targets,
        config["temperature"],
        512,
        config["final_logit_softcapping"],
        config["logit_scale"],
    )
    gradients = torch.autograd.grad(log_probs.sum(), [hidden, weight, bias])

    ref_hidden, ref_weight, ref_bias, ref_targets = _inputs(9, seed=3)
    ref_log_probs, *_ = _reference(ref_hidden, ref_weight, ref_bias, ref_targets, **config)
    ref_gradients = torch.autograd.grad(ref_log_probs.sum(), [ref_hidden, ref_weight, ref_bias])

    for actual, expected in zip(gradients, ref_gradients, strict=True):
        _assert_close(actual, expected)


@pytest.mark.parametrize(("n_tokens", "chunk"), [(7, 3), (1000, 333), (1000, 999)])
def test_token_chunking_does_not_change_values(monkeypatch, n_tokens: int, chunk: int) -> None:
    hidden, weight, bias, targets = _inputs(n_tokens, seed=4)
    expected = _reference(
        hidden, weight, bias, targets, temperature=1.0, final_logit_softcapping=8.0, logit_scale=1.0
    )

    monkeypatch.setattr(trl_compat, "TOKEN_CHUNK_SIZE", chunk)
    chunked = TorchChunkedLogProbFunction.apply(hidden, weight, bias, targets, 1.0, 512, 8.0, 1.0)

    monkeypatch.setattr(trl_compat, "TOKEN_CHUNK_SIZE", 1_000_000)
    single = TorchChunkedLogProbFunction.apply(hidden, weight, bias, targets, 1.0, 512, 8.0, 1.0)

    for actual, reference in zip(chunked, expected, strict=True):
        _assert_close(actual, reference)
    for actual, reference in zip(chunked, single, strict=True):
        _assert_close(actual, reference)


def test_empty_input_returns_five_empty_tensors() -> None:
    hidden = torch.zeros(0, HIDDEN)
    weight = torch.randn(VOCAB, HIDDEN)
    bias = torch.randn(VOCAB)
    targets = torch.zeros(0, dtype=torch.long)

    result = TorchChunkedLogProbFunction.apply(hidden, weight, bias, targets, 1.0, 512)

    assert len(result) == 5
    for tensor in result:
        assert isinstance(tensor, torch.Tensor)
        assert tensor.numel() == 0


def test_triton_kernel_usable_returns_bool_without_raising() -> None:
    usable = trl_compat.triton_kernel_usable()
    assert isinstance(usable, bool)

    has_accelerator = torch.cuda.is_available() or (
        hasattr(torch, "xpu") and torch.xpu.is_available()
    )
    if not has_accelerator:
        assert usable is False


def test_trl_logprob_backend_is_a_documented_value() -> None:
    assert trl_compat.trl_logprob_backend() in {"triton", "torch", "torch-fallback", "unavailable"}


def test_install_cpu_logprob_fallback_is_idempotent_and_patches_both_names(monkeypatch) -> None:
    import trl.kernels
    import trl.trainer.utils

    monkeypatch.setattr(trl_compat, "_PATCHED", False)
    monkeypatch.setattr(trl.kernels, "ChunkedLogProbFunction", object())
    monkeypatch.setattr(trl.trainer.utils, "_ChunkedLogProbFunction", object())

    assert trl_compat.install_cpu_logprob_fallback() is True
    assert trl.kernels.ChunkedLogProbFunction is TorchChunkedLogProbFunction
    assert trl.trainer.utils._ChunkedLogProbFunction is TorchChunkedLogProbFunction

    assert trl_compat.install_cpu_logprob_fallback() is True
    assert trl.kernels.ChunkedLogProbFunction is TorchChunkedLogProbFunction


def test_install_cpu_logprob_fallback_leaves_a_usable_triton_kernel_alone(monkeypatch) -> None:
    import trl.kernels
    import trl.trainer.utils

    monkeypatch.setattr(trl_compat, "_PATCHED", False)
    monkeypatch.setattr(trl_compat, "triton_kernel_usable", lambda: True)
    sentinel = object()
    monkeypatch.setattr(trl.kernels, "ChunkedLogProbFunction", sentinel)
    monkeypatch.setattr(trl.trainer.utils, "_ChunkedLogProbFunction", sentinel)

    assert trl_compat.install_cpu_logprob_fallback(force=False) is False
    assert trl.kernels.ChunkedLogProbFunction is sentinel
    assert trl.trainer.utils._ChunkedLogProbFunction is sentinel
    assert trl_compat.trl_logprob_backend() == "triton"
