"""Shared enums.

These are part of the on-disk contract: their values are persisted in SQLite and
JSONL artifacts, so treat string values as append-only.
"""

from __future__ import annotations

from enum import StrEnum


class RunStatus(StrEnum):
    """Lifecycle status of a run.

    ``PAUSED``/``INTERRUPTED``/``FAILED``/``CANCELLED`` are resumable or
    inspectable end states; ``COMPLETED`` is terminal.
    """

    CREATED = "created"
    PREPARING_DATA = "preparing_data"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    EVALUATING_BASE = "evaluating_base"
    TRAINING_SFT = "training_sft"
    TRAINING_DPO = "training_dpo"
    EVALUATING_FINAL = "evaluating_final"
    COMPLETED = "completed"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: Statuses that a fresh process can pick up again.
RESUMABLE_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.CREATED,
        RunStatus.PREPARING_DATA,
        RunStatus.GENERATING,
        RunStatus.REVIEWING,
        RunStatus.EVALUATING_BASE,
        RunStatus.TRAINING_SFT,
        RunStatus.TRAINING_DPO,
        RunStatus.EVALUATING_FINAL,
        RunStatus.PAUSED,
        RunStatus.INTERRUPTED,
        RunStatus.FAILED,
    }
)

#: Statuses that mean "no further work should be attempted".
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset({RunStatus.COMPLETED, RunStatus.CANCELLED})

#: Allowed forward transitions.  ``FAILED`` is reachable from any busy status,
#: and any resumable status may be re-entered from ``PAUSED``/``INTERRUPTED``.
_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset({RunStatus.PREPARING_DATA, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.PREPARING_DATA: frozenset(
        {
            RunStatus.GENERATING,
            RunStatus.REVIEWING,
            RunStatus.EVALUATING_BASE,
            RunStatus.TRAINING_SFT,
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.COMPLETED,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.GENERATING: frozenset(
        {
            RunStatus.REVIEWING,
            RunStatus.PREPARING_DATA,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.REVIEWING: frozenset(
        {
            RunStatus.EVALUATING_BASE,
            RunStatus.PREPARING_DATA,
            RunStatus.GENERATING,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.EVALUATING_BASE: frozenset(
        {
            RunStatus.TRAINING_SFT,
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.REVIEWING,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.TRAINING_SFT: frozenset(
        {
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.EVALUATING_BASE,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.TRAINING_DPO: frozenset(
        {
            RunStatus.EVALUATING_FINAL,
            RunStatus.EVALUATING_BASE,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.EVALUATING_FINAL: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.PAUSED,
            RunStatus.INTERRUPTED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.PAUSED: frozenset(
        {
            RunStatus.PREPARING_DATA,
            RunStatus.GENERATING,
            RunStatus.REVIEWING,
            RunStatus.EVALUATING_BASE,
            RunStatus.TRAINING_SFT,
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.INTERRUPTED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.INTERRUPTED: frozenset(
        {
            RunStatus.PREPARING_DATA,
            RunStatus.GENERATING,
            RunStatus.REVIEWING,
            RunStatus.EVALUATING_BASE,
            RunStatus.TRAINING_SFT,
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.PAUSED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.FAILED: frozenset(
        {
            RunStatus.PREPARING_DATA,
            RunStatus.GENERATING,
            RunStatus.REVIEWING,
            RunStatus.EVALUATING_BASE,
            RunStatus.TRAINING_SFT,
            RunStatus.TRAINING_DPO,
            RunStatus.EVALUATING_FINAL,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}


def can_transition(src: RunStatus | str, dst: RunStatus | str) -> bool:
    """Return True if ``src -> dst`` is a legal status transition."""

    src_e, dst_e = RunStatus(src), RunStatus(dst)
    if src_e == dst_e:
        return True
    return dst_e in _TRANSITIONS.get(src_e, frozenset())


def assert_transition(src: RunStatus | str, dst: RunStatus | str) -> None:
    """Raise :class:`ValueError` for an illegal status transition."""

    if not can_transition(src, dst):
        raise ValueError(f"illegal run status transition: {src} -> {dst}")


class RunPhase(StrEnum):
    """Coarse pipeline phase used for UI grouping (independent of status)."""

    SETUP = "setup"
    DATA = "data"
    GENERATE = "generate"
    REVIEW = "review"
    EVALUATE = "evaluate"
    TRAIN = "train"
    REPORT = "report"


class FeedbackMode(StrEnum):
    """Whether a human must approve each candidate comparison."""

    AUTOMATED = "automated"
    INTERACTIVE = "interactive"


class TrainingMethod(StrEnum):
    """Which weights are trained."""

    LORA = "lora"
    QLORA = "qlora"
    FFT = "fft"


class TrainingObjective(StrEnum):
    """What the loss is computed against."""

    SFT = "sft"
    DPO = "dpo"


class PresetId(StrEnum):
    """Stable identifiers for the small set of offered hyperparameter presets."""

    CPU_TINY_LORA = "cpu_tiny_lora"
    SINGLE_GPU_SAFE_LORA = "single_gpu_safe_lora"
    SINGLE_GPU_QLORA = "single_gpu_qlora"
    SINGLE_GPU_FAST_LORA = "single_gpu_fast_lora"
    MULTI_GPU_FFT = "multi_gpu_fft"


class SplitName(StrEnum):
    """Dataset splits.  Splitting happens at the prompt level, never at the
    response level, so near-duplicate answers cannot leak across splits."""

    TRAIN = "train"
    VALIDATION = "validation"
    EVAL = "eval"


class SampleStatus(StrEnum):
    """Fate of a single generated candidate/sample."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    EDITED = "edited"
    FAILED = "failed"


class CriterionId(StrEnum):
    """Default rubric criterion ids.  Custom criteria may use free-form ids."""

    CHARACTER_CONSISTENCY = "character_consistency"
    VOICE_STYLE = "voice_style"
    RESPONSIVENESS = "responsiveness"
    LORE_CONSISTENCY = "lore_consistency"
    BOUNDARY_ADHERENCE = "boundary_adherence"
    OVERALL_PREFERENCE = "overall_preference"


class DecisionKind(StrEnum):
    """Which Jev primitive produced a judgement."""

    SCORE = "score"
    CHOICE = "choice"
    NOUL = "noul"
