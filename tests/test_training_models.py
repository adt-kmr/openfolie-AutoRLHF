"""Model/tokenizer loading, chat template policy, metadata estimates, generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autorlhf.config.run_config import HFModelConfig, Hyperparameters
from autorlhf.core.enums import TrainingMethod
from autorlhf.core.errors import (
    ChatTemplateError,
    ConfigError,
    ModelLoadError,
    QuantizationUnavailableError,
)
from autorlhf.core.types import ChatMessage
from autorlhf.training.models import (
    GENERIC_CHAT_TEMPLATE,
    GENERIC_TEMPLATE_MARKER,
    build_chat_prompt,
    chat_template_has_generation_markers,
    chat_template_source,
    count_parameters,
    generate_text,
    load_model,
    load_tokenizer,
    model_param_count_from_metadata,
)


def _strip_chat_template(directory: Path) -> Path:
    """Remove every chat template artefact from a tokenizer directory."""

    for name in ("chat_template.jinja", "chat_template.json"):
        (directory / name).unlink(missing_ok=True)
    for name in ("tokenizer_config.json", "tokenizer.json"):
        path = directory / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.pop("chat_template", None)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _copy_directory(source: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            (target / item.name).write_bytes(item.read_bytes())
    return target


@pytest.fixture()
def template_less_tokenizer_dir(tmp_path, tiny_chat_model_dir) -> Path:
    """A full model directory whose tokenizer ships no chat template."""

    return _strip_chat_template(_copy_directory(tiny_chat_model_dir, tmp_path / "no-template"))


@pytest.fixture()
def marker_less_tokenizer_dir(tmp_path, tiny_chat_model_dir) -> Path:
    """A model directory whose tokenizer ships a template TRL cannot mask."""

    target = _copy_directory(tiny_chat_model_dir, tmp_path / "marker-less")
    for name in ("chat_template.jinja", "chat_template.json"):
        (target / name).unlink(missing_ok=True)
    path = target / "tokenizer_config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["chat_template"] = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    return target


def test_load_tokenizer_reports_the_real_template(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir), torch_dtype="float32")
    tokenizer, warnings = load_tokenizer(cfg, hf_token=None)
    assert tokenizer.chat_template
    assert chat_template_source(tokenizer) == "tokenizer"
    assert getattr(tokenizer, GENERIC_TEMPLATE_MARKER, False) is False
    assert warnings == []  # the fixture tokenizer has a pad token and a template


def test_load_tokenizer_without_template_raises(template_less_tokenizer_dir) -> None:
    cfg = HFModelConfig(model_id=str(template_less_tokenizer_dir))
    with pytest.raises(ChatTemplateError) as info:
        load_tokenizer(cfg, hf_token=None)
    assert "chat template" in str(info.value)
    assert info.value.hint


def test_load_tokenizer_generic_fallback_is_explicit(template_less_tokenizer_dir) -> None:
    cfg = HFModelConfig(model_id=str(template_less_tokenizer_dir))
    tokenizer, warnings = load_tokenizer(cfg, hf_token=None, allow_fallback=True)
    assert tokenizer.chat_template == GENERIC_CHAT_TEMPLATE
    assert warnings and "generic template" in warnings[0]
    # The fallback must be trainable: TRL needs {% generation %} markers, so the
    # generic template carries them around assistant content.
    assert chat_template_has_generation_markers(tokenizer) is True
    assert "{% generation %}" in GENERIC_CHAT_TEMPLATE


def test_load_tokenizer_warns_when_assistant_masks_are_impossible(
    marker_less_tokenizer_dir,
) -> None:
    cfg = HFModelConfig(model_id=str(marker_less_tokenizer_dir))
    tokenizer, warnings = load_tokenizer(cfg, hf_token=None)
    assert chat_template_has_generation_markers(tokenizer) is False
    assert chat_template_source(tokenizer) == "tokenizer"  # not replaced by the fallback
    assert any("{% generation %}" in warning for warning in warnings)


def test_generation_markers_are_detected_when_present(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir))
    tokenizer, warnings = load_tokenizer(cfg, hf_token=None)
    assert chat_template_has_generation_markers(tokenizer) is True
    assert not any("{% generation %}" in warning for warning in warnings)


def test_load_tokenizer_missing_repository_is_typed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = HFModelConfig(model_id=str(tmp_path / "does-not-exist"))
    with pytest.raises(ModelLoadError) as info:
        load_tokenizer(cfg, hf_token=None)
    assert "does-not-exist" in str(info.value)


def test_build_chat_prompt_renders_generation_prompt(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir))
    tokenizer, _ = load_tokenizer(cfg, hf_token=None)
    messages = [
        ChatMessage(role="system", content="You are terse."),
        ChatMessage(role="user", content="Say hi."),
    ]
    prompt = build_chat_prompt(tokenizer, messages)
    assert "You are terse." in prompt
    assert "Say hi." in prompt
    assert prompt.rstrip().endswith("assistant:")
    # dicts are accepted as well as ChatMessage objects
    as_dicts = build_chat_prompt(
        tokenizer,
        [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Say hi."}],
    )
    assert as_dicts == prompt


def test_load_model_reports_provenance(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(
        model_id=str(tiny_chat_model_dir), torch_dtype="float32", attn_implementation="eager"
    )
    hp = Hyperparameters(method=TrainingMethod.LORA, lora_target_modules=["c_attn"])
    loaded = load_model(cfg, hp, hf_token=None)

    assert loaded.model_id == str(tiny_chat_model_dir)
    assert loaded.param_count == count_parameters(loaded.model) > 0
    assert loaded.chat_template_source == "tokenizer"
    assert loaded.resolved_revision  # "main (unpinned)" for a local directory
    assert loaded.warnings == []
    assert loaded.model.training is False  # inference loading leaves eval mode on


def test_load_model_reports_generic_template_use(template_less_tokenizer_dir) -> None:
    cfg = HFModelConfig(
        model_id=str(template_less_tokenizer_dir),
        torch_dtype="float32",
        attn_implementation="eager",
        allow_generic_chat_template=True,
    )
    hp = Hyperparameters(method=TrainingMethod.LORA, lora_target_modules=["c_attn"])
    loaded = load_model(cfg, hp, hf_token=None)
    assert loaded.chat_template_source == "generic-fallback"
    assert any("generic template" in warning for warning in loaded.warnings)


def test_load_model_refuses_qlora_without_hardware(tiny_chat_model_dir, monkeypatch) -> None:
    import autorlhf.training.hardware as hardware_module
    from autorlhf.core.types import HardwareInfo

    monkeypatch.setattr(
        hardware_module,
        "detect_hardware",
        lambda **kwargs: HardwareInfo(device_type="cpu", device_count=0),
    )
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir), torch_dtype="float32")
    hp = Hyperparameters(
        method=TrainingMethod.QLORA, load_in_4bit=True, lora_target_modules=["c_attn"]
    )
    with pytest.raises(QuantizationUnavailableError) as info:
        load_model(cfg, hp, hf_token=None)
    assert "GPU" in str(info.value)


def test_load_model_rejects_unknown_dtype(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir), torch_dtype="float8")
    with pytest.raises(ConfigError):
        load_model(cfg, Hyperparameters(), hf_token=None)


def test_model_param_count_from_metadata(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(model_id=str(tiny_chat_model_dir), torch_dtype="float32")
    estimated = model_param_count_from_metadata(cfg, hf_token=None)
    assert estimated is not None and estimated > 0
    hp = Hyperparameters(method=TrainingMethod.LORA, lora_target_modules=["c_attn"])
    actual = load_model(cfg, hp, hf_token=None).param_count
    # Architecture-field arithmetic: close, never exact (position embeddings and
    # per-model extras are intentionally not enumerable without the weights).
    assert 0.5 * actual <= estimated <= 1.6 * actual


def test_model_param_count_from_metadata_missing_repo(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    cfg = HFModelConfig(model_id=str(tmp_path / "nope"))
    with pytest.raises(ModelLoadError):
        model_param_count_from_metadata(cfg, hf_token=None)


def test_generate_text_greedy_and_sampled(tiny_chat_model_dir) -> None:
    cfg = HFModelConfig(
        model_id=str(tiny_chat_model_dir), torch_dtype="float32", attn_implementation="eager"
    )
    hp = Hyperparameters(method=TrainingMethod.LORA, lora_target_modules=["c_attn"])
    loaded = load_model(cfg, hp, hf_token=None)
    messages = [{"role": "user", "content": "Say hi."}]

    text, meta = generate_text(
        loaded.model, loaded.tokenizer, messages, max_new_tokens=4, temperature=0.0
    )
    assert isinstance(text, str)
    assert meta["do_sample"] is False
    assert meta["prompt_tokens"] > 0
    assert 0 < meta["completion_tokens"] <= 4
    assert meta["temperature"] == 0.0

    _sampled, sampled_meta = generate_text(
        loaded.model,
        loaded.tokenizer,
        messages,
        max_new_tokens=4,
        temperature=0.7,
        top_p=0.9,
        seed=123,
    )
    assert sampled_meta["do_sample"] is True
    assert sampled_meta["top_p"] == pytest.approx(0.9)
    assert sampled_meta["seed"] == 123
