"""The training plan: what will be trained, with what, and why.

``TrainingPlan`` is what the TUI renders before a run starts and what
``TrainingJob`` consumes.  It is a pure data object (plus JSON persistence), so
it can be built and shown without importing torch.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from autorlhf.config.run_config import Hyperparameters
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.types import HardwareInfo, ResourceEstimate
from autorlhf.storage.runs import write_json


class TrainingPlan(BaseModel):
    """A frozen description of one training job.

    ``sft_examples`` and ``preference_pairs`` hold the raw JSONL records (see
    ``docs/DATA_FORMATS.md``), not a transformed dataset: the plan is evidence of
    exactly what the trainers will read.
    """

    run_id: str
    objective: TrainingObjective
    method: TrainingMethod
    hyperparameters: Hyperparameters
    preset_id: str | None = None
    rationale: list[str] = Field(default_factory=list)
    #: ``{probabilities, confidence, chosen, model_build}`` when Jev ranked presets.
    jev_ranking: dict | None = None
    resource_estimate: ResourceEstimate | None = None
    hardware: HardwareInfo
    sft_examples: list[dict] = Field(default_factory=list)
    preference_pairs: list[dict] = Field(default_factory=list)
    train_count: int = 0
    validation_count: int = 0
    warnings: list[str] = Field(default_factory=list)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    def save(self, path: Path) -> None:
        write_json(Path(path), self.model_dump(mode="json"))

    def summary_lines(self) -> list[str]:
        """Short human-readable lines for the TUI's plan panel."""

        hp = self.hyperparameters
        lines = [
            f"run: {self.run_id}",
            f"objective: {self.objective.value}   method: {self.method}",
            f"model preset: {self.preset_id or 'custom'}",
            f"sft examples: {len(self.sft_examples)}   preference pairs: "
            f"{len(self.preference_pairs)}",
            f"train/validation: {self.train_count}/{self.validation_count}",
            f"lora r={hp.lora_r} alpha={hp.lora_alpha} dropout={hp.lora_dropout}",
            f"lr={hp.learning_rate} epochs={hp.num_train_epochs} max_steps={hp.max_steps}",
            f"batch={hp.per_device_train_batch_size} x accum={hp.gradient_accumulation_steps} "
            f"(effective {hp.effective_batch_size()})",
            f"max_length={hp.max_seq_length} save_steps={hp.save_steps}",
        ]
        if self.resource_estimate is not None:
            estimate = self.resource_estimate
            lines.append(
                f"estimated VRAM {estimate.required_vram_mib} MiB / RAM "
                f"{estimate.required_ram_mib} MiB (modelled, not measured)"
            )
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        return lines


__all__ = ["TrainingPlan"]
