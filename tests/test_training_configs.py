"""Hyperparameter -> TRL config mapping, dataset preparation, resume selection."""

from __future__ import annotations

import pytest

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.errors import ConfigError
from autorlhf.storage.runs import (
    CHECKPOINT_SENTINEL,
    RunPaths,
    mark_checkpoint_complete,
    write_json,
)
from autorlhf.training.trainers import (
    build_dpo_config,
    build_lora_config,
    build_sft_config,
    checkpoint_root,
    find_resume_checkpoint,
    prepare_dpo_dataset,
    prepare_sft_dataset,
    validate_dpo_records,
    validate_sft_records,
)

RUN_ID = "20260101-120000-test"


def _hp(**overrides) -> Hyperparameters:
    hp = Hyperparameters(
        method=TrainingMethod.LORA,
        objective=TrainingObjective.SFT,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_target_modules=["c_attn", "c_proj"],
        learning_rate=3e-4,
        num_train_epochs=2.0,
        max_steps=-1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        max_seq_length=256,
        lr_scheduler_type="linear",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=0.5,
        gradient_checkpointing=False,
        optim="adamw_torch",
        seed=7,
        bf16=False,
        fp16=False,
        save_steps=3,
        eval_steps=4,
        logging_steps=2,
        save_total_limit=2,
        dpo_beta=0.2,
        dpo_loss_type="hinge",
        sft_warmup=False,
        sft_epochs=3.0,
        sft_learning_rate=1e-3,
        sft_completion_only_loss=True,
    )
    for key, value in overrides.items():
        setattr(hp, key, value)
    return hp


def _record(index: int, *, split: str = "train", last_role: str = "assistant") -> dict:
    messages = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": f"Say hello {index}."},
        {"role": last_role, "content": "Hello."},
    ]
    return {
        "record_type": "sft",
        "dataset_version": "1",
        "run_id": RUN_ID,
        "example_id": f"s{index:04d}",
        "prompt_id": f"p{index:04d}",
        "split": split,
        "messages": messages,
        "source": "seed",
        "status": "accepted",
        "created_at": "2026-01-01T12:00:00",
    }


def _pair(index: int, *, split: str = "train", chosen: str = "Terse.", rejected: str = "Verbose."):
    return {
        "record_type": "preference",
        "dataset_version": "1",
        "run_id": RUN_ID,
        "pair_id": f"p{index:04d}-a-vs-b",
        "prompt_id": f"p{index:04d}",
        "split": split,
        "prompt": f"Will the fog lift {index}?",
        "prompt_context": [{"role": "user", "content": "Earlier question."}],
        "chosen_id": f"p{index:04d}-c1",
        "rejected_id": f"p{index:04d}-c0",
        "chosen_text": chosen,
        "rejected_text": rejected,
        "margin": 0.9,
        "judge_confidence": 0.81,
        "source": "jev",
        "created_at": "2026-01-01T12:00:00",
    }


# --------------------------------------------------------------------------- #
# configs
# --------------------------------------------------------------------------- #


def test_sft_config_maps_every_hyperparameter(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    hp = _hp()
    config = build_sft_config(hp, paths, run_name="unit-sft", report_to=())

    assert config.output_dir == str(paths.checkpoints_dir / "sft")
    assert config.run_name == "unit-sft"
    assert config.report_to == []  # tracking off -> nothing is reported anywhere
    assert config.max_length == hp.max_seq_length
    assert config.completion_only_loss is True
    assert config.assistant_only_loss is False  # never assumed: needs {% generation %}
    assert config.learning_rate == hp.resolved_sft_lr() == 1e-3
    assert config.num_train_epochs == hp.sft_epochs == 3.0
    assert config.max_steps == hp.max_steps == -1
    assert config.per_device_train_batch_size == 1
    assert config.per_device_eval_batch_size == 2
    assert config.gradient_accumulation_steps == 4
    assert config.lr_scheduler_type == "linear"
    # transformers 5 has no `warmup_ratio`: the ratio is resolved into steps, and
    # with an unpinned step count the builder honestly reports 0 until the dataset
    # size is known (run_sft/run_dpo set it then).
    assert config.warmup_steps == 0
    assert config.weight_decay == 0.01
    assert config.max_grad_norm == 0.5
    assert config.gradient_checkpointing is False
    assert config.optim == "adamw_torch"
    assert config.seed == 7
    assert config.bf16 is False and config.fp16 is False
    assert config.save_steps == 3
    assert config.logging_steps == 2
    assert config.save_total_limit == 2


def test_warmup_ratio_becomes_steps_when_the_total_is_known(tmp_path) -> None:
    hp = _hp(max_steps=100, warmup_ratio=0.05)
    config = build_sft_config(hp, RunPaths(tmp_path / "run"))
    assert config.warmup_steps == 5

    hp_zero = _hp(max_steps=100, warmup_ratio=0.0)
    assert build_sft_config(hp_zero, RunPaths(tmp_path / "run")).warmup_steps == 0


def test_dataloader_pinning_follows_the_real_device(tmp_path) -> None:
    import torch

    xpu = getattr(torch, "xpu", None)
    expected = bool(torch.cuda.is_available() or (xpu is not None and xpu.is_available()))
    config = build_sft_config(_hp(), RunPaths(tmp_path / "run"))
    assert config.dataloader_pin_memory is expected
    assert config.dataloader_num_workers == 0


def test_sft_config_report_to_tracking_enabled(tmp_path) -> None:
    config = build_sft_config(_hp(), RunPaths(tmp_path / "run"), report_to=["wandb"])
    assert config.report_to == ["wandb"]


def test_sft_config_completion_only_loss_follows_hyperparameters(tmp_path) -> None:
    hp = _hp(sft_completion_only_loss=False)
    config = build_sft_config(hp, RunPaths(tmp_path / "run"))
    assert config.completion_only_loss is False


def test_dpo_config_maps_every_hyperparameter(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    hp = _hp(objective=TrainingObjective.DPO)
    config = build_dpo_config(hp, paths, run_name="unit-dpo", report_to=())

    assert config.output_dir == str(paths.checkpoints_dir / "dpo")
    assert config.report_to == []
    assert config.max_length == hp.max_seq_length
    assert config.beta == 0.2
    assert list(config.loss_type) == ["hinge"]  # TRL normalises loss_type to a list
    assert config.learning_rate == hp.learning_rate == 3e-4
    assert config.num_train_epochs == hp.num_train_epochs == 2.0
    assert config.gradient_accumulation_steps == 4
    assert config.eval_steps == hp.eval_steps == 4
    assert config.save_steps == hp.save_steps == 3
    assert not hasattr(config, "max_prompt_length")  # removed in TRL 1.15


def test_build_lora_config_maps_fields() -> None:
    hp = _hp(lora_r=4, lora_alpha=8, lora_dropout=0.0, lora_bias="none", use_dora=False)
    config = build_lora_config(hp, object(), fan_in_fan_out=True)
    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.lora_dropout == 0.0
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"
    assert config.target_modules == ["c_attn", "c_proj"] or set(config.target_modules) == {
        "c_attn",
        "c_proj",
    }
    assert config.fan_in_fan_out is True


def test_build_lora_config_refuses_fft() -> None:
    with pytest.raises(ConfigError):
        build_lora_config(_hp(method=TrainingMethod.FFT), object())


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #


def test_prepare_sft_dataset_columns_and_drops() -> None:
    records = [
        _record(1),
        _record(2, split="validation"),
        {"example_id": "broken", "messages": []},
        {"example_id": "bad-role", "messages": [{"role": "tool", "content": "x"}]},
        {"example_id": "empty-content", "messages": [{"role": "assistant", "content": "  "}]},
        _record(3, last_role="user"),
    ]
    dataset = prepare_sft_dataset(records)
    assert dataset.column_names == ["messages"]
    assert len(dataset) == 2
    assert dataset[0]["messages"][0] == {"role": "system", "content": "You are terse."}
    assert dataset[0]["messages"][-1]["role"] == "assistant"


def test_validate_sft_records_reports_each_problem() -> None:
    problems = validate_sft_records([_record(1), {"messages": []}, _record(2, last_role="user")])
    assert len(problems) == 2
    assert "malformed" in problems[0]
    assert "not an assistant turn" in problems[1]


def test_prepare_dpo_dataset_columns_and_content() -> None:
    records = [_pair(1), _pair(2, split="validation")]
    dataset = prepare_dpo_dataset(records, system_prompt="You are terse.")
    assert dataset.column_names == ["prompt", "chosen", "rejected"]
    row = dataset[0]
    assert row["prompt"][0] == {"role": "system", "content": "You are terse."}
    assert row["prompt"][1] == {"role": "user", "content": "Earlier question."}
    assert row["prompt"][-1]["role"] == "user"
    assert row["prompt"][-1]["content"].startswith("Will the fog lift")
    assert row["chosen"] == [{"role": "assistant", "content": "Terse."}]
    assert row["rejected"] == [{"role": "assistant", "content": "Verbose."}]


def test_prepare_dpo_dataset_drops_degenerate_pairs() -> None:
    records = [
        _pair(1),
        _pair(2, chosen="Same.", rejected="Same."),
        _pair(3, chosen="", rejected="x"),
        dict(_pair(4), prompt=""),
    ]
    dataset = prepare_dpo_dataset(records)
    assert len(dataset) == 1
    problems = validate_dpo_records(records)
    assert len(problems) == 3
    assert any("identical" in problem for problem in problems)


def test_prepare_dpo_dataset_without_system_prompt() -> None:
    dataset = prepare_dpo_dataset([_pair(1)])
    assert dataset[0]["prompt"][0]["role"] == "user"


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #


def _make_checkpoint(root, step: int, *, complete: bool, with_adapter: bool = True) -> None:
    folder = root / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / "trainer_state.json", {"global_step": step})
    if with_adapter:
        (folder / "adapter_model.safetensors").write_bytes(b"weights")
        (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
    if complete:
        mark_checkpoint_complete(folder)


def test_checkpoint_root_per_objective(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    assert checkpoint_root(paths, TrainingObjective.SFT) == paths.checkpoints_dir / "sft"
    assert checkpoint_root(paths, "dpo") == paths.checkpoints_dir / "dpo"


def test_find_resume_checkpoint_ignores_incomplete(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    root = checkpoint_root(paths, TrainingObjective.SFT)
    _make_checkpoint(root, 3, complete=False)
    assert find_resume_checkpoint(paths, objective=TrainingObjective.SFT) is None

    (root / "checkpoint-3" / "adapter_model.safetensors").write_bytes(b"weights")
    assert not (root / "checkpoint-3" / CHECKPOINT_SENTINEL).exists()


def test_find_resume_checkpoint_picks_highest_complete(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    root = checkpoint_root(paths, TrainingObjective.SFT)
    _make_checkpoint(root, 5, complete=True)
    _make_checkpoint(root, 10, complete=False)
    _make_checkpoint(root, 20, complete=True)
    selected = find_resume_checkpoint(paths, objective=TrainingObjective.SFT)
    assert selected is not None
    assert selected.name == "checkpoint-20"


def test_find_resume_checkpoint_requires_weights(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    root = checkpoint_root(paths, TrainingObjective.SFT)
    _make_checkpoint(root, 1, complete=True, with_adapter=False)
    assert find_resume_checkpoint(paths, objective=TrainingObjective.SFT) is None


def test_find_resume_checkpoint_is_per_objective(tmp_path) -> None:
    paths = RunPaths(tmp_path / "run")
    _make_checkpoint(checkpoint_root(paths, TrainingObjective.DPO), 2, complete=True)
    assert find_resume_checkpoint(paths, objective=TrainingObjective.SFT) is None
    assert find_resume_checkpoint(paths, objective=TrainingObjective.DPO) is not None
