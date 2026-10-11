"""Foundation tests: config serialization, secret redaction, validation rules.

Covers spec items 1-4:

1. configuration serialization and secret redaction,
2. character prompt and conversation validation,
3. rubric + hyperparameter preflight rules,
4. preset feasibility and resource estimates.
"""

from __future__ import annotations

import json
import stat
import sys

import pytest
from pydantic import ValidationError

from autorlhf.config.character import (
    DEFAULT_CHARACTER,
    MAX_SEED_CONVERSATIONS,
    MAX_SYSTEM_PROMPT_CHARS,
    CharacterSpec,
    SeedConversation,
)
from autorlhf.config.presets import (
    assert_method_runnable,
    estimate_training_memory,
    feasible_presets,
    get_preset,
    preset_choice_options,
    preset_feasibility,
    recommend_preset,
)
from autorlhf.config.rubric import Criterion, Rubric, default_rubric
from autorlhf.config.run_config import Hyperparameters, RunConfig
from autorlhf.config.settings import (
    AppSettings,
    SecretStore,
    mask_secret,
    parse_env_text,
    render_env_text,
)
from autorlhf.core.enums import FeedbackMode, PresetId, TrainingMethod, TrainingObjective
from autorlhf.core.errors import ConfigError, CredentialError, HardwareError
from autorlhf.core.logging import REDACTED, redact_mapping, redact_text
from autorlhf.core.types import ChatMessage, HardwareInfo

# --------------------------------------------------------------------------- #
# item 1 - configuration serialization
# --------------------------------------------------------------------------- #


def test_run_config_json_round_trip_is_lossless(run_config: RunConfig) -> None:
    text = run_config.to_json()
    assert RunConfig.from_json(text) == run_config
    assert RunConfig.model_validate(json.loads(text)) == run_config


def test_run_config_fingerprint_stable_and_ignores_runtime_fields(run_config: RunConfig) -> None:
    fingerprint = run_config.fingerprint()
    assert RunConfig.from_json(run_config.to_json()).fingerprint() == fingerprint

    changed = run_config.model_copy(deep=True)
    changed.verbose = not changed.verbose
    changed.created_at = "1999-01-01T00:00:00+00:00"
    changed.slug = "a-completely-different-slug"
    assert changed.fingerprint() == fingerprint

    substantive = run_config.model_copy(deep=True)
    substantive.training.lora_r = run_config.training.lora_r + 1
    assert substantive.fingerprint() != fingerprint


def test_app_settings_json_round_trip(tmp_path) -> None:
    settings = AppSettings(
        verbose=True,
        max_concurrency=3,
        default_hf_model="some/model",
        last_preset=PresetId.CPU_TINY_LORA.value,
    )
    path = tmp_path / "settings.json"
    settings.save(path)
    assert AppSettings.load(path) == settings
    assert AppSettings.load(tmp_path / "missing.json") == AppSettings()


def test_app_settings_load_ignores_unknown_keys(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"verbose": True, "not_a_real_field": 1}))
    loaded = AppSettings.load(path)
    assert loaded.verbose is True
    assert not hasattr(loaded, "not_a_real_field")


def test_app_settings_public_view_redacts_secret_shapes() -> None:
    raw = "sk-or-v1-abcdefgh12345678"
    settings = AppSettings(wandb_project=raw)
    rendered = json.dumps(settings.public_view())
    assert raw not in rendered


def test_character_and_rubric_round_trip_through_json_and_files(tmp_path) -> None:
    character = DEFAULT_CHARACTER.model_copy(deep=True)
    assert CharacterSpec.from_json(character.to_json()) == character
    character_path = tmp_path / "character.json"
    character.save(character_path)
    assert CharacterSpec.load(character_path) == character

    rubric = default_rubric()
    assert Rubric.from_json(rubric.to_json()) == rubric
    rubric_path = tmp_path / "rubric.json"
    rubric.save(rubric_path)
    assert Rubric.load(rubric_path) == rubric


# --------------------------------------------------------------------------- #
# item 1 - secret store
# --------------------------------------------------------------------------- #


def test_secret_store_update_preserves_comments_and_unrelated_lines(app_paths) -> None:
    store = SecretStore(app_paths.env_path)
    app_paths.env_path.write_text("# keep me\nUNRELATED=value\nOPENROUTER_API_KEY=old\n")
    store.update({"OPENROUTER_API_KEY": "sk-or-v1-newvalue123456", "HF_TOKEN": "hf_newtoken"})

    text = app_paths.env_path.read_text(encoding="utf-8")
    assert "# keep me" in text
    assert "UNRELATED=value" in text
    assert "OPENROUTER_API_KEY=sk-or-v1-newvalue123456" in text
    assert "HF_TOKEN=hf_newtoken" in text
    assert parse_env_text(text)["UNRELATED"] == "value"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits only")
def test_secret_store_file_mode_is_0600(app_paths) -> None:
    store = SecretStore(app_paths.env_path)
    store.update({"OPENROUTER_API_KEY": "sk-or-v1-abcdefgh1234"})
    assert stat.S_IMODE(app_paths.env_path.stat().st_mode) == 0o600


def test_secret_store_load_prefers_environment_over_file(app_paths, monkeypatch) -> None:
    app_paths.env_path.write_text("OPENROUTER_API_KEY=sk-or-v1-fromfile0000\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fromenvironment00")
    store = SecretStore(app_paths.env_path)
    assert store.get("OPENROUTER_API_KEY") == "sk-or-v1-fromenvironment00"


def test_secret_store_fallback_file_has_lowest_precedence(app_paths, tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fallback = tmp_path / "project.env"
    fallback.write_text("HF_TOKEN=hf_fromfallback0000\n")
    app_paths.env_path.write_text("HF_TOKEN=hf_fromcanonical000\n")

    store = SecretStore(app_paths.env_path, fallback_paths=(fallback,))
    assert store.get("HF_TOKEN") == "hf_fromcanonical000"


def test_secret_store_redacted_and_describe_hide_the_raw_secret(app_paths) -> None:
    raw = "sk-or-v1-supersecretvalue12345"
    store = SecretStore(app_paths.env_path)
    store.update({"OPENROUTER_API_KEY": raw, "WANDB_PROJECT": "my-project"})

    redacted = store.redacted()
    assert raw not in json.dumps(redacted)
    assert redacted["OPENROUTER_API_KEY"] != raw
    assert redacted["WANDB_PROJECT"] == "my-project"

    described = store.describe()
    assert raw not in json.dumps(described)
    row = next(r for r in described if r["key"] == "OPENROUTER_API_KEY")
    assert row["present"] is True
    assert row["secret"] is True


def test_secret_store_require_raises_when_missing(app_paths, monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    store = SecretStore(app_paths.env_path)
    with pytest.raises(CredentialError):
        store.require("OPENROUTER_API_KEY", "Talking to OpenRouter")


def test_mask_secret_hides_short_values_entirely() -> None:
    assert mask_secret("abc") == "***"
    assert mask_secret("12345678") == "********"
    assert mask_secret("") == "(unset)"
    assert mask_secret(None) == "(unset)"
    long_mask = mask_secret("sk-or-v1-abcdefgh12345678")
    assert "abcdefgh" not in long_mask
    assert "sk-or-v1-abcdefgh12345678" not in long_mask


def test_parse_env_text_handles_quotes_exports_blanks_and_duplicates() -> None:
    text = (
        "# a comment\n"
        "\n"
        "PLAIN=value\n"
        "export EXPORTED=exported-value\n"
        'QUOTED="double quoted"\n'
        "SINGLE='single quoted'\n"
        "DUP=first\n"
        "DUP=second\n"
    )
    parsed = parse_env_text(text)
    assert parsed["PLAIN"] == "value"
    assert parsed["EXPORTED"] == "exported-value"
    assert parsed["QUOTED"] == "double quoted"
    assert parsed["SINGLE"] == "single quoted"
    assert parsed["DUP"] == "second"


def test_render_env_text_preserves_content_and_applies_values() -> None:
    existing = "# header comment\nUNRELATED=keep\nFOO=old\n\n"
    rendered = render_env_text({"FOO": "new", "BAR": "added"}, existing_text=existing)

    assert "# header comment" in rendered
    assert "UNRELATED=keep" in rendered
    assert "FOO=new" in rendered
    assert "FOO=old" not in rendered
    assert "BAR=added" in rendered

    parsed = parse_env_text(rendered)
    assert parsed == {"UNRELATED": "keep", "FOO": "new", "BAR": "added"}


# --------------------------------------------------------------------------- #
# item 1 - logging redaction
# --------------------------------------------------------------------------- #


def test_redact_text_strips_documented_secret_shapes() -> None:
    assert redact_text("sk-or-v1-abcdefgh12345678") == REDACTED
    assert REDACTED in redact_text("OPENROUTER_API_KEY=sk-or-v1-abcdefgh12345678")
    assert "hf_AbCdEfGhIjKlMnOpQrSt" not in redact_text("token=hf_AbCdEfGhIjKlMnOpQrSt")
    assert "sk-AbCdEfGhIjKlMnOpQrSt" not in redact_text("key=sk-AbCdEfGhIjKlMnOpQrSt")


@pytest.mark.parametrize(
    "text",
    [
        "my_token=abc",
        "app_secret=abc",
        "openrouter_api_key=abc",
        "hf_token=abc",
        "wandb_api_key=abc",
    ],
)
def test_redact_text_strips_token_secret_and_api_key_names(text: str) -> None:
    assert REDACTED in redact_text(text)


def test_redact_text_and_mapping_strip_bare_key_named_fields() -> None:
    assert REDACTED in redact_text("key=abc")
    assert REDACTED in redact_text("my_key=abc")
    assert redact_mapping({"some_key": "abc"})["some_key"] == REDACTED
    assert redact_mapping({"key_value": "abc"})["key_value"] == REDACTED
    assert redact_mapping({"access_token": "abc"})["access_token"] == REDACTED


def test_redact_mapping_does_not_over_redact_generic_words() -> None:
    assert redact_mapping({"keywords": "x"})["keywords"] != REDACTED
    assert redact_mapping({"monkey": "x"})["monkey"] != REDACTED


def test_redact_mapping_redacts_by_key_name_and_value_shape() -> None:
    data = {
        "openrouter_api_key": "sk-or-v1-abcdefgh12345678",
        "notes": "the token=hf_AbCdEfGhIjKlMnOpQrSt is here",
        "nested": {"wandb_api_key": "plainvalue"},
        "items": ["sk-or-v1-abcdefgh12345678", {"secret_thing": "x"}],
        "count": 3,
    }
    out = redact_mapping(data)

    assert out["openrouter_api_key"] == REDACTED
    assert "hf_AbCdEfGhIjKlMnOpQrSt" not in out["notes"]
    assert out["nested"]["wandb_api_key"] == REDACTED
    assert out["items"][0] == REDACTED
    assert out["items"][1]["secret_thing"] == REDACTED
    assert out["count"] == 3


# --------------------------------------------------------------------------- #
# item 2 - character / conversation validation
# --------------------------------------------------------------------------- #


def _conversation(*turns: tuple[str, str]) -> SeedConversation:
    return SeedConversation(
        messages=[ChatMessage(role=role, content=content) for role, content in turns]
    )


def test_valid_seed_conversation_has_no_problems() -> None:
    conversation = _conversation(("user", "hi"), ("assistant", "hello"))
    assert conversation.validate_conversation() == []
    assert conversation.is_valid() is True


def test_leading_system_turn_is_allowed_and_ignored() -> None:
    conversation = SeedConversation(
        messages=[
            ChatMessage(role="system", content="ignored preamble"),
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="hello"),
        ]
    )
    assert conversation.validate_conversation() == []
    assert [m.role for m in conversation.to_training_messages()] == ["user", "assistant"]


def test_conversation_must_start_with_a_user_turn() -> None:
    conversation = _conversation(("assistant", "hello"), ("user", "hi"), ("assistant", "bye"))
    problems = conversation.validate_conversation()
    assert any("start with a user turn" in p for p in problems)


def test_conversation_must_alternate_turns() -> None:
    conversation = _conversation(("user", "a"), ("user", "b"), ("assistant", "c"))
    problems = conversation.validate_conversation()
    assert any("alternate" in p for p in problems)


def test_conversation_missing_assistant_turn() -> None:
    conversation = _conversation(("user", "only a question"))
    problems = conversation.validate_conversation()
    assert any("no assistant turn" in p for p in problems)
    assert any("end with an assistant turn" in p for p in problems)


def test_conversation_without_user_or_assistant_turns() -> None:
    conversation = SeedConversation(messages=[ChatMessage(role="system", content="only system")])
    problems = conversation.validate_conversation()
    assert problems
    assert any("no user/assistant turns" in p for p in problems)


def test_empty_message_content_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        SeedConversation(messages=[ChatMessage(role="user", content="")])
    with pytest.raises(ValueError):
        ChatMessage(role="assistant", content="   ").validate_for_training()


def test_validate_conversation_reports_empty_content() -> None:
    conversation = SeedConversation.model_construct(
        messages=[
            ChatMessage(role="user", content=""),
            ChatMessage(role="assistant", content="hi"),
        ],
        label="",
    )
    problems = conversation.validate_conversation()
    assert any("is empty" in p for p in problems)


def test_default_character_is_valid() -> None:
    assert DEFAULT_CHARACTER.validate_spec() == []
    assert len(DEFAULT_CHARACTER.valid_seeds()) == len(DEFAULT_CHARACTER.seed_conversations)


def test_spec_with_valid_content_produces_no_problems(minimal_character: CharacterSpec) -> None:
    assert minimal_character.validate_spec() == []


def test_spec_flags_empty_system_prompt(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    spec.system_prompt = "   "
    assert "system prompt is empty" in spec.validate_spec()


def test_spec_flags_truncated_overlong_prompt(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    spec.system_prompt = "x" * (MAX_SYSTEM_PROMPT_CHARS + 1)
    problems = spec.validate_spec()
    assert any("system prompt" in p and "limit" in p for p in problems)


def test_spec_flags_seed_count_over_limit(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    valid = _conversation(("user", "u"), ("assistant", "a"))
    spec.seed_conversations = [
        valid.model_copy(deep=True) for _ in range(MAX_SEED_CONVERSATIONS + 1)
    ]
    problems = spec.validate_spec()
    assert any("seed conversations" in p and "limit" in p for p in problems)


def test_spec_flags_empty_constraint(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    spec.behavioral_constraints = ["fine", "   "]
    problems = spec.validate_spec()
    assert "constraint #2 is empty" in problems


def test_spec_requires_at_least_one_valid_seed(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    spec.seed_conversations = []
    problems = spec.validate_spec()
    assert any("at least 1 valid seed" in p for p in problems)
    assert spec.validate_spec(require_seeds=False) == []


def test_spec_prefixes_invalid_seed_problems_with_index(minimal_character: CharacterSpec) -> None:
    spec = minimal_character.model_copy(deep=True)
    spec.seed_conversations.append(
        SeedConversation.model_construct(
            messages=[ChatMessage(role="user", content="dangling")],
            label="",
        )
    )
    problems = spec.validate_spec()
    assert any(p.startswith("seed #2:") for p in problems)


def test_assert_valid_raises_config_error_for_empty_spec() -> None:
    with pytest.raises(ConfigError):
        CharacterSpec().assert_valid()


# --------------------------------------------------------------------------- #
# item 3 - rubric rules
# --------------------------------------------------------------------------- #


def _criterion(
    criterion_id: str,
    *,
    weight: float = 1.0,
    include: bool = True,
) -> Criterion:
    return Criterion(
        id=criterion_id,
        name=criterion_id,
        description="description",
        weight=weight,
        include_in_overall=include,
    )


def test_default_rubric_is_valid() -> None:
    assert default_rubric().validate_rubric() == []


def test_rubric_flags_duplicate_ids() -> None:
    rubric = Rubric(criteria=[_criterion("a"), _criterion("a")])
    assert any("duplicate criterion id" in p for p in rubric.validate_rubric())


def test_rubric_flags_non_positive_weight() -> None:
    rubric = Rubric(criteria=[_criterion("a", weight=0.0)])
    assert any("non-positive weight" in p for p in rubric.validate_rubric())


def test_rubric_flags_no_included_criterion() -> None:
    rubric = Rubric(criteria=[_criterion("a", include=False)])
    assert any("included in the overall" in p for p in rubric.validate_rubric())


def test_rubric_flags_out_of_range_confidence_threshold() -> None:
    rubric = default_rubric()
    rubric.min_confidence_for_preference = 1.5
    assert any("min_confidence_for_preference" in p for p in rubric.validate_rubric())
    rubric.min_confidence_for_preference = -0.1
    assert any("min_confidence_for_preference" in p for p in rubric.validate_rubric())


def test_rubric_flags_missing_criteria_and_negative_tie_epsilon() -> None:
    empty = Rubric()
    problems = empty.validate_rubric()
    assert "rubric has no criteria" in problems
    assert any("included in the overall" in p for p in problems)

    negative = default_rubric()
    negative.tie_epsilon = -1.0
    assert any("tie_epsilon" in p for p in negative.validate_rubric())


def test_rubric_assert_valid_raises() -> None:
    with pytest.raises(ConfigError):
        Rubric().assert_valid()


def test_rubric_overall_ignores_missing_but_weights_included() -> None:
    rubric = Rubric(criteria=[_criterion("a", weight=2.0), _criterion("b", weight=1.0)])
    assert rubric.overall({"a": 4.0}) == pytest.approx(4.0)
    assert rubric.overall({"a": 4.0, "b": 2.0}) == pytest.approx(10.0 / 3.0)
    assert rubric.overall({}) is None


def test_rubric_overall_skips_excluded_and_none_scores() -> None:
    rubric = Rubric(
        criteria=[_criterion("a", weight=1.0), _criterion("b", weight=5.0, include=False)]
    )
    assert rubric.overall({"a": 3.0, "b": 1.0}) == pytest.approx(3.0)
    assert rubric.overall({"a": None}) is None


# --------------------------------------------------------------------------- #
# item 3 - hyperparameter preflight rules
# --------------------------------------------------------------------------- #


def _hyperparameters(**overrides: object) -> Hyperparameters:
    data: dict[str, object] = {
        "method": TrainingMethod.LORA,
        "objective": TrainingObjective.SFT,
        "lora_target_modules": ["q_proj"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 1.0,
        "max_steps": -1,
    }
    data.update(overrides)
    return Hyperparameters(**data)


def test_hyperparameter_baseline_is_valid() -> None:
    assert _hyperparameters().validate_hyperparameters() == []


def test_hyperparameters_flag_bf16_and_fp16_together() -> None:
    problems = _hyperparameters(bf16=True, fp16=True).validate_hyperparameters()
    assert any("mutually exclusive" in p for p in problems)


def test_hyperparameters_flag_qlora_without_4bit() -> None:
    problems = _hyperparameters(
        method=TrainingMethod.QLORA, load_in_4bit=False
    ).validate_hyperparameters()
    assert any("QLoRA requires load_in_4bit=True" in p for p in problems)


def test_hyperparameters_qlora_with_4bit_is_valid() -> None:
    hp = _hyperparameters(method=TrainingMethod.QLORA, load_in_4bit=True)
    assert hp.validate_hyperparameters() == []


def test_hyperparameters_flag_fft_with_4bit() -> None:
    problems = _hyperparameters(
        method=TrainingMethod.FFT, load_in_4bit=True
    ).validate_hyperparameters()
    assert any("full fine-tuning cannot be combined with 4-bit" in p for p in problems)


def test_hyperparameters_flag_dpo_effective_batch_below_two() -> None:
    hp = _hyperparameters(
        objective=TrainingObjective.DPO,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
    )
    assert any("effective batch size >= 2" in p for p in hp.validate_hyperparameters())


def test_hyperparameters_dpo_effective_batch_of_two_is_valid() -> None:
    hp = _hyperparameters(
        objective=TrainingObjective.DPO,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
    )
    assert hp.validate_hyperparameters() == []
    assert hp.effective_batch_size() == 2


def test_hyperparameters_flag_unknown_scheduler_and_optimiser() -> None:
    assert any(
        "lr_scheduler_type" in p
        for p in _hyperparameters(lr_scheduler_type="not-a-scheduler").validate_hyperparameters()
    )
    assert any(
        "optim" in p for p in _hyperparameters(optim="not-an-optimiser").validate_hyperparameters()
    )


def test_hyperparameters_max_steps_semantics() -> None:
    assert any("max_steps" in p for p in _hyperparameters(max_steps=0).validate_hyperparameters())
    assert any("max_steps" in p for p in _hyperparameters(max_steps=-5).validate_hyperparameters())
    assert _hyperparameters(max_steps=10, num_train_epochs=1.0).validate_hyperparameters() == []
    assert _hyperparameters(max_steps=1, num_train_epochs=0.0).validate_hyperparameters() == []
    assert any(
        "num_train_epochs" in p
        for p in _hyperparameters(max_steps=-1, num_train_epochs=0.0).validate_hyperparameters()
    )


def test_hyperparameters_flag_unknown_bias_and_allow_empty_lora_targets() -> None:
    assert any(
        "lora_bias" in p for p in _hyperparameters(lora_bias="weird").validate_hyperparameters()
    )
    # An empty `lora_target_modules` is deliberately valid at configuration time:
    # the names can only be discovered from the real model, and the config screen
    # must not download weights.  The trainer discovers them and raises
    # TargetModulesError when nothing matches, which is tested in
    # tests/test_training_*.py.
    assert _hyperparameters(lora_target_modules=[]).validate_hyperparameters() == []


def test_hyperparameters_flag_qlora_fp16_with_bf16_compute() -> None:
    hp = _hyperparameters(method=TrainingMethod.QLORA, load_in_4bit=True, fp16=True)
    assert any("inconsistent" in p for p in hp.validate_hyperparameters())


def test_run_config_weave_requires_consent(run_config: RunConfig) -> None:
    config = run_config.model_copy(deep=True)
    config.weave_enabled = True
    config.weave_consent = False
    assert any("weave_consent" in p for p in config.validate_config())

    config.weave_consent = True
    assert not any("weave" in p for p in config.validate_config())


def test_run_config_validate_config_prefixes_subsystem_problems() -> None:
    problems = RunConfig().validate_config()
    assert any(p.startswith("character: ") for p in problems)
    assert any(p.startswith("rubric: ") for p in problems)


def test_run_config_flags_empty_model_id(run_config: RunConfig) -> None:
    config = run_config.model_copy(deep=True)
    config.model.model_id = "   "
    assert any("model id" in p for p in config.validate_config())


def test_run_config_interactive_mode_needs_a_prompt(run_config: RunConfig) -> None:
    config = run_config.model_copy(deep=True)
    config.feedback_mode = FeedbackMode.INTERACTIVE
    config.generation.num_prompts = 0
    assert any("interactive" in p for p in config.validate_config())


# --------------------------------------------------------------------------- #
# item 3 - resource estimates
# --------------------------------------------------------------------------- #

_PARAM_COUNT = 1_000_000_000


def test_estimate_memory_monotonic_in_batch_size() -> None:
    small = estimate_training_memory(
        _hyperparameters(per_device_train_batch_size=1), param_count=_PARAM_COUNT
    )
    large = estimate_training_memory(
        _hyperparameters(per_device_train_batch_size=8), param_count=_PARAM_COUNT
    )
    assert large.required_vram_mib > small.required_vram_mib


def test_estimate_memory_monotonic_in_sequence_length() -> None:
    short = estimate_training_memory(_hyperparameters(max_seq_length=512), param_count=_PARAM_COUNT)
    long = estimate_training_memory(_hyperparameters(max_seq_length=2048), param_count=_PARAM_COUNT)
    assert long.required_vram_mib > short.required_vram_mib


def test_estimate_memory_monotonic_in_lora_rank() -> None:
    low = estimate_training_memory(_hyperparameters(lora_r=8), param_count=_PARAM_COUNT)
    high = estimate_training_memory(_hyperparameters(lora_r=64), param_count=_PARAM_COUNT)
    assert high.required_vram_mib > low.required_vram_mib
    assert high.trainable_parameters > low.trainable_parameters


def test_estimate_memory_qlora_below_lora_for_same_model() -> None:
    lora = estimate_training_memory(
        _hyperparameters(method=TrainingMethod.LORA), param_count=_PARAM_COUNT
    )
    qlora = estimate_training_memory(
        _hyperparameters(method=TrainingMethod.QLORA, load_in_4bit=True), param_count=_PARAM_COUNT
    )
    assert qlora.required_vram_mib < lora.required_vram_mib


def test_estimate_memory_unknown_model_is_labelled_placeholder() -> None:
    estimate = estimate_training_memory(_hyperparameters(), param_count=None)
    assert estimate.required_vram_mib > 0
    assert any("unknown" in assumption for assumption in estimate.assumptions)


# --------------------------------------------------------------------------- #
# item 4 - preset feasibility
# --------------------------------------------------------------------------- #


def _cpu_hardware() -> HardwareInfo:
    return HardwareInfo(device_type="cpu", device_count=0, total_vram_mib=None, cpu_count=8)


def _cuda_hardware(*, vram_gib: int = 24, bitsandbytes: bool = True) -> HardwareInfo:
    return HardwareInfo(
        device_type="cuda",
        device_name="Test GPU",
        device_count=1,
        total_vram_mib=vram_gib * 1024,
        free_vram_mib=vram_gib * 1024,
        bf16_supported=True,
        fp16_supported=True,
        bitsandbytes_available=bitsandbytes,
    )


def test_cpu_only_makes_qlora_infeasible_with_a_reason() -> None:
    result = preset_feasibility(get_preset(PresetId.SINGLE_GPU_QLORA), _cpu_hardware())
    assert result.feasible is False
    assert "GPU" in result.reason


def test_cpu_only_makes_fft_infeasible_with_a_reason() -> None:
    result = preset_feasibility(get_preset(PresetId.MULTI_GPU_FFT), _cpu_hardware())
    assert result.feasible is False
    assert "GPU" in result.reason


def test_cpu_only_recommends_the_cpu_preset() -> None:
    recommendation = recommend_preset(_cpu_hardware(), objective=TrainingObjective.DPO)
    assert recommendation is not None
    assert recommendation.preset.id is PresetId.CPU_TINY_LORA
    assert recommendation.feasible is True


def test_cuda_with_bitsandbytes_makes_qlora_feasible() -> None:
    result = preset_feasibility(
        get_preset(PresetId.SINGLE_GPU_QLORA), _cuda_hardware(), param_count=500_000_000
    )
    assert result.feasible is True
    assert result.reason == ""
    assert result.estimate is not None


def test_cuda_without_bitsandbytes_makes_qlora_infeasible() -> None:
    result = preset_feasibility(
        get_preset(PresetId.SINGLE_GPU_QLORA), _cuda_hardware(bitsandbytes=False)
    )
    assert result.feasible is False
    assert "bitsandbytes" in result.reason


def test_small_vram_cuda_rejects_the_large_gpu_presets() -> None:
    hardware = _cuda_hardware(vram_gib=8)
    feasibilities = {f.preset.id: f for f in feasible_presets(hardware, param_count=500_000_000)}
    assert feasibilities[PresetId.SINGLE_GPU_FAST_LORA].feasible is False
    assert "24 GiB" in feasibilities[PresetId.SINGLE_GPU_FAST_LORA].reason
    assert feasibilities[PresetId.MULTI_GPU_FFT].feasible is False
    assert "GiB" in feasibilities[PresetId.MULTI_GPU_FFT].reason


def test_medium_vram_cuda_rejects_large_gpu_presets_on_min_vram() -> None:
    hardware = _cuda_hardware(vram_gib=20)
    fast = preset_feasibility(
        get_preset(PresetId.SINGLE_GPU_FAST_LORA), hardware, param_count=500_000_000
    )
    fft = preset_feasibility(get_preset(PresetId.MULTI_GPU_FFT), hardware, param_count=500_000_000)
    assert fast.feasible is False and "24 GiB" in fast.reason
    assert fft.feasible is False and "48 GiB" in fft.reason


def test_feasible_presets_orders_feasible_before_infeasible() -> None:
    results = feasible_presets(_cuda_hardware(vram_gib=8), param_count=500_000_000)
    flags = [r.feasible for r in results]
    assert flags == sorted(flags, reverse=True)
    assert flags[0] is True


def test_preset_choice_options_only_contains_feasible_ids() -> None:
    results = feasible_presets(_cuda_hardware(vram_gib=8), param_count=500_000_000)
    options = preset_choice_options(results)
    feasible_ids = {r.preset.id.value for r in results if r.feasible}
    assert set(options) == feasible_ids


def test_recommend_preset_honours_a_feasible_preference() -> None:
    recommendation = recommend_preset(
        _cuda_hardware(),
        objective=TrainingObjective.DPO,
        preferred=PresetId.SINGLE_GPU_QLORA.value,
        param_count=500_000_000,
    )
    assert recommendation is not None
    assert recommendation.preset.id is PresetId.SINGLE_GPU_QLORA


def test_recommend_preset_ignores_an_infeasible_preference() -> None:
    recommendation = recommend_preset(
        _cpu_hardware(),
        objective=TrainingObjective.DPO,
        preferred=PresetId.MULTI_GPU_FFT.value,
    )
    assert recommendation is not None
    assert recommendation.preset.id is PresetId.CPU_TINY_LORA


def test_preset_to_hyperparameters_sets_method_and_quantization() -> None:
    qlora = get_preset(PresetId.SINGLE_GPU_QLORA).to_hyperparameters()
    assert qlora.method is TrainingMethod.QLORA
    assert qlora.load_in_4bit is True
    assert qlora.preset_id == PresetId.SINGLE_GPU_QLORA.value

    lora = get_preset(PresetId.SINGLE_GPU_SAFE_LORA).to_hyperparameters()
    assert lora.method is TrainingMethod.LORA
    assert lora.load_in_4bit is False


def test_assert_method_runnable_raises_on_cpu() -> None:
    assert_method_runnable(_cpu_hardware(), TrainingMethod.LORA)
    with pytest.raises(HardwareError):
        assert_method_runnable(_cpu_hardware(), TrainingMethod.QLORA)
