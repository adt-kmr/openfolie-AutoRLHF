"""The autorlhf fine-tuning engine (agent B).

Importing this package is cheap and does **not** import ``torch``, ``trl`` or
``peft``: the submodules listed here only need pydantic and the shared core, and
:mod:`autorlhf.training.trainers` is imported lazily by
:class:`~autorlhf.training.jobs.TrainingJob` when a run actually starts.

    from autorlhf.training import TrainingJob, detect_hardware

The public entry points are :class:`~autorlhf.training.jobs.TrainingJob`,
:class:`~autorlhf.training.plan.TrainingPlan`,
:func:`~autorlhf.training.hardware.detect_hardware` and
:func:`~autorlhf.training.models.load_model`.
"""

from __future__ import annotations

from autorlhf.training import hardware, jobs, lora, models, plan
from autorlhf.training.hardware import detect_hardware, estimate_resources
from autorlhf.training.jobs import TrainingJob, TrainingResult
from autorlhf.training.plan import TrainingPlan

__all__ = [
    "TrainingJob",
    "TrainingPlan",
    "TrainingResult",
    "detect_hardware",
    "estimate_resources",
    "hardware",
    "jobs",
    "lora",
    "models",
    "plan",
]
