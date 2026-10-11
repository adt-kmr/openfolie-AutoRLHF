"""Behavioural rubric.

A rubric is a list of :class:`Criterion`.  Each criterion maps to exactly one
Jev ``score`` question with an *ordered* list of levels (lowest first, matching
the Decisions API's ``score.criteria`` semantics).  The numeric midpoint Jev
returns is used as the criterion value, and the returned per-level
probabilities are preserved verbatim for display.

Nothing in this module invents a confidence value: ``confidence`` is always the
value returned by the judge, or ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from autorlhf.config.character import CharacterSpec
from autorlhf.core.enums import CriterionId
from autorlhf.core.errors import ConfigError
from autorlhf.core.ids import content_hash

#: Default 1–5 levels, written as descriptions the judge can distinguish.
DEFAULT_LEVELS: tuple[str, ...] = (
    "Badly violates the specification; the response is off-character or ignores the constraints.",
    "Partially off-character or misses important parts of the specification or constraints.",
    "Acceptable: broadly consistent with the specification, with noticeable lapses.",
    "Good: consistent with the specification and constraints; only minor imperfections.",
    "Excellent: fully consistent, in voice, and respects every stated constraint.",
)

#: Threshold (in rubric points) below which a score gap is not treated as a
#: meaningful preference.  Configurable per run.
DEFAULT_TIE_EPSILON = 0.35


class Criterion(BaseModel):
    """One rubric dimension with an ordered scale."""

    id: str
    name: str
    description: str
    #: Ordered lowest -> highest.  Length defines the scale.
    levels: list[str] = Field(default_factory=lambda: list(DEFAULT_LEVELS))
    #: Relative weight used when averaging criteria into an overall score.
    weight: float = 1.0
    #: When True (default) the criterion participates in the overall average.
    include_in_overall: bool = True

    @field_validator("levels")
    @classmethod
    def _validate_levels(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("a criterion needs at least 2 ordered levels")
        if any(not level.strip() for level in value):
            raise ValueError("criterion levels must not be empty")
        return value

    @property
    def scale_min(self) -> int:
        return 0

    @property
    def scale_max(self) -> int:
        return len(self.levels) - 1

    def instructions_for_judge(self) -> str:
        return f"{self.name}: {self.description}"

    def criteria_for_judge(self) -> list[str]:
        return list(self.levels)


class Rubric(BaseModel):
    """The full rubric plus the decision policy for building preferences."""

    criteria: list[Criterion] = Field(default_factory=list)
    #: Minimum overall gap between two responses for a preference to be asserted.
    tie_epsilon: float = DEFAULT_TIE_EPSILON
    #: A judgement below this judge confidence is retained but never used to
    #: create a preference pair automatically.
    min_confidence_for_preference: float = 0.6
    #: Behaviour when the judge's response is unusable.
    on_judge_error: str = "retry"  # retry | skip

    def validate_rubric(self) -> list[str]:
        problems: list[str] = []
        if not self.criteria:
            problems.append("rubric has no criteria")
        seen: set[str] = set()
        for criterion in self.criteria:
            if criterion.id in seen:
                problems.append(f"duplicate criterion id '{criterion.id}'")
            seen.add(criterion.id)
            if criterion.weight <= 0:
                problems.append(f"criterion '{criterion.id}' has non-positive weight")
        if self.tie_epsilon < 0:
            problems.append("tie_epsilon must be >= 0")
        if not 0.0 <= self.min_confidence_for_preference <= 1.0:
            problems.append("min_confidence_for_preference must be within 0..1")
        if not any(c.include_in_overall for c in self.criteria):
            problems.append("at least one criterion must be included in the overall score")
        return problems

    def assert_valid(self) -> None:
        problems = self.validate_rubric()
        if problems:
            raise ConfigError(
                "rubric is invalid: " + "; ".join(problems),
                hint="Fix the rubric on the Rubric screen.",
            )

    # -- derived ----------------------------------------------------------- #

    def criterion(self, criterion_id: str) -> Criterion | None:
        return next((c for c in self.criteria if c.id == criterion_id), None)

    @property
    def included(self) -> list[Criterion]:
        return [c for c in self.criteria if c.include_in_overall]

    def scale_max(self) -> int:
        """Highest possible weighted average (all criteria at their top level)."""

        return max((c.scale_max for c in self.included), default=4)

    def overall(self, scores: dict[str, float]) -> float | None:
        """Weighted mean over the criteria present in ``scores``.

        Missing criteria are ignored rather than counted as zero.
        """

        total = 0.0
        weight = 0.0
        for criterion in self.included:
            if criterion.id in scores and scores[criterion.id] is not None:
                total += scores[criterion.id] * criterion.weight
                weight += criterion.weight
        if weight <= 0:
            return None
        return total / weight

    def questions_for_judge(self) -> dict[str, dict[str, object]]:
        """Build the Jev ``questions`` object keyed by criterion id."""

        return {
            c.id: {
                "type": "score",
                "instructions": c.instructions_for_judge(),
                "criteria": c.criteria_for_judge(),
            }
            for c in self.criteria
        }

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    # -- io ---------------------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Rubric:
        return cls.model_validate(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> Rubric:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def default_rubric() -> Rubric:
    """The six criteria autorlhf ships with."""

    return Rubric(
        criteria=[
            Criterion(
                id=CriterionId.CHARACTER_CONSISTENCY.value,
                name="Character consistency",
                description=(
                    "Does the response stay in character as described by the character "
                    "specification, without breaking the persona?"
                ),
                weight=1.5,
            ),
            Criterion(
                id=CriterionId.VOICE_STYLE.value,
                name="Voice and style consistency",
                description=(
                    "Does the response match the specified tone, register, and sentence-level "
                    "style, including any length limits?"
                ),
                weight=1.0,
            ),
            Criterion(
                id=CriterionId.RESPONSIVENESS.value,
                name="Responsiveness to the user",
                description=(
                    "Does the response actually answer the user's prompt and address what was "
                    "asked, rather than deflecting or ignoring it?"
                ),
                weight=1.0,
            ),
            Criterion(
                id=CriterionId.LORE_CONSISTENCY.value,
                name="Consistency with supplied lore and conversation history",
                description=(
                    "Is the response consistent with the supplied lore and with earlier turns "
                    "of the same conversation, without contradicting established facts?"
                ),
                weight=1.0,
            ),
            Criterion(
                id=CriterionId.BOUNDARY_ADHERENCE.value,
                name="Adherence to explicit behavioural constraints",
                description=(
                    "Does the response obey each explicitly listed behavioural constraint, "
                    "including refusals and forbidden topics?"
                ),
                weight=1.5,
            ),
            Criterion(
                id=CriterionId.OVERALL_PREFERENCE.value,
                name="Overall suitability for the intended character",
                description=(
                    "Taken as a whole, is this a response the intended character should produce?"
                ),
                weight=1.5,
            ),
        ]
    )


def rubric_for_character(character: CharacterSpec) -> Rubric:
    """Default rubric, with the constraints criterion's description made concrete."""

    rubric = default_rubric()
    if character.behavioral_constraints:
        criterion = rubric.criterion(CriterionId.BOUNDARY_ADHERENCE.value)
        if criterion is not None:
            listed = "; ".join(c.strip() for c in character.behavioral_constraints if c.strip())
            criterion.description = (
                "Does the response obey every explicitly listed behavioural constraint? "
                f"The constraints are: {listed}"
            )
    return rubric


__all__ = [
    "DEFAULT_LEVELS",
    "DEFAULT_TIE_EPSILON",
    "Criterion",
    "Rubric",
    "default_rubric",
    "rubric_for_character",
]
