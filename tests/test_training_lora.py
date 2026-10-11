"""LoRA target-module discovery and validation (no GPU, no downloads)."""

from __future__ import annotations

from typing import Any

import pytest

from autorlhf.core.errors import TargetModulesError
from autorlhf.training.lora import (
    available_linear_modules,
    discover_target_modules,
    family_defaults,
    infer_family,
    needs_fan_in_fan_out,
    validate_target_modules,
)


@pytest.fixture(scope="module")
def gpt2_model(tiny_model_dir) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(tiny_model_dir))


@pytest.fixture(scope="module")
def llama_model(tiny_llama_dir) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(tiny_llama_dir))


def test_infer_family_from_config(gpt2_model: Any, llama_model: Any) -> None:
    assert infer_family(gpt2_model) == "gpt2"
    assert infer_family(llama_model) == "llama"


def test_discover_gpt2_fused_projections(gpt2_model: Any) -> None:
    targets, notes = discover_target_modules(gpt2_model)
    assert targets == ["c_attn", "c_proj"]
    assert any("gpt2" in note for note in notes)
    assert any("fan_in_fan_out=True" in note for note in notes)
    assert needs_fan_in_fan_out(gpt2_model) is True


def test_discover_llama_projections(llama_model: Any) -> None:
    targets, notes = discover_target_modules(llama_model)
    assert targets == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    assert any("fan_in_fan_out=False" in note for note in notes)
    assert needs_fan_in_fan_out(llama_model) is False


def test_discover_prefers_configured_modules(gpt2_model: Any) -> None:
    targets, notes = discover_target_modules(gpt2_model, preferred=["c_attn"])
    assert targets == ["c_attn"]
    assert any("configured target modules" in note for note in notes)


def test_discover_ignores_missing_preferred_and_reports(gpt2_model: Any) -> None:
    targets, notes = discover_target_modules(gpt2_model, preferred=["not_a_module"])
    assert targets == ["c_attn", "c_proj"]
    assert any("not_a_module" in note for note in notes)


def test_available_linear_modules_counts(gpt2_model: Any) -> None:
    available = available_linear_modules(gpt2_model)
    assert available["c_attn"] == 2  # two transformer blocks
    assert "lm_head" not in available  # top-level modules are not targets


def test_validate_accepts_present_and_dotted_targets(gpt2_model: Any) -> None:
    validate_target_modules(gpt2_model, ["c_attn"])
    validate_target_modules(gpt2_model, ["transformer.h.0.attn.c_attn"])


def test_validate_rejects_nonsense_target(gpt2_model: Any) -> None:
    with pytest.raises(TargetModulesError) as info:
        validate_target_modules(gpt2_model, ["q_proj"])
    message = str(info.value)
    assert "q_proj" in message
    assert "c_attn" in info.value.hint  # the candidates are listed


def test_validate_rejects_empty_targets(gpt2_model: Any) -> None:
    with pytest.raises(TargetModulesError):
        validate_target_modules(gpt2_model, [])


def test_family_defaults_unknown_family_is_empty() -> None:
    assert family_defaults("some_brand_new_architecture") == ([], False)
