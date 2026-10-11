"""TrainingPlan: persistence, round-trip and display helpers."""

from __future__ import annotations

import json

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.types import HardwareInfo, ResourceEstimate
from autorlhf.training.plan import TrainingPlan

RUN_ID = "20260101-120000-plan"


def _plan() -> TrainingPlan:
    return TrainingPlan(
        run_id=RUN_ID,
        objective=TrainingObjective.DPO,
        method=TrainingMethod.LORA,
        hyperparameters=Hyperparameters(
            objective=TrainingObjective.DPO, lora_target_modules=["q_proj"], max_seq_length=512
        ),
        preset_id="single_gpu_safe_lora",
        rationale=["only LoRA fits the detected VRAM"],
        jev_ranking={
            "chosen": "single_gpu_safe_lora",
            "probabilities": {"single_gpu_safe_lora": 0.7, "cpu_tiny_lora": 0.3},
            "confidence": 0.62,
            "model_build": "typesafe/jev-1.13-20260917",
        },
        resource_estimate=ResourceEstimate(
            required_vram_mib=4096, required_ram_mib=8192, assumptions=["modelled"]
        ),
        hardware=HardwareInfo(device_type="cpu", device_count=0, cpu_count=8),
        sft_examples=[{"example_id": "s0001", "messages": []}],
        preference_pairs=[{"pair_id": "p0001-a-vs-b"}],
        train_count=4,
        validation_count=2,
        warnings=["VRAM estimate is modelled, not measured"],
    )


def test_plan_save_and_round_trip(tmp_path) -> None:
    plan = _plan()
    target = tmp_path / "training_plan.json"
    plan.save(target)

    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["run_id"] == RUN_ID
    assert raw["objective"] == "dpo"
    assert raw["method"] == "lora"
    assert raw["preset_id"] == "single_gpu_safe_lora"
    assert raw["jev_ranking"]["chosen"] == "single_gpu_safe_lora"
    assert raw["hardware"]["device_type"] == "cpu"
    assert raw["resource_estimate"]["required_vram_mib"] == 4096
    assert raw["sft_examples"][0]["example_id"] == "s0001"
    assert raw["preference_pairs"][0]["pair_id"] == "p0001-a-vs-b"

    reloaded = TrainingPlan.model_validate(raw)
    assert reloaded == plan
    assert json.loads(plan.to_json())["train_count"] == 4


def test_plan_defaults_are_empty_and_safe() -> None:
    plan = TrainingPlan(
        run_id=RUN_ID,
        objective=TrainingObjective.SFT,
        method=TrainingMethod.LORA,
        hyperparameters=Hyperparameters(),
        hardware=HardwareInfo(),
    )
    assert plan.rationale == []
    assert plan.jev_ranking is None
    assert plan.resource_estimate is None
    assert plan.sft_examples == []
    assert plan.preference_pairs == []
    assert plan.train_count == 0
    assert plan.validation_count == 0


def test_plan_summary_lines_describe_the_run() -> None:
    lines = _plan().summary_lines()
    joined = "\n".join(lines)
    assert "objective: dpo" in joined
    assert "method: lora" in joined
    assert "lora r=" in joined
    assert "effective" in joined
    assert "modelled, not measured" in joined
    assert "warning: VRAM estimate is modelled, not measured" in joined
