"""Pipeline value objects: the dataset bundle, review items, human decisions.

These mirror the frozen shapes in ``docs/ARCHITECTURE.md``.  They are the only
things the pipeline, the review queue, and the TUI exchange, so they stay
JSON-serializable and carry their provenance (split, source, status).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from autorlhf.core.enums import SampleStatus
from autorlhf.llm.schemas import (
    CandidateResponse,
    Judgement,
    PairDecision,
    PreferencePair,
    SyntheticPrompt,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DatasetBundle(BaseModel):
    """Everything the dataset-preparation phase produced, in memory.

    ``judgements`` is keyed by candidate id so a candidate can be rendered next
    to the judge's opinion without a second lookup.
    """

    run_id: str
    version: str = "1"
    dataset_hash: str = ""
    split_hash: str = ""
    prompts: list[SyntheticPrompt] = Field(default_factory=list)
    candidates: list[CandidateResponse] = Field(default_factory=list)
    judgements: dict[str, Judgement] = Field(default_factory=dict)
    pair_decisions: list[PairDecision] = Field(default_factory=list)
    preference_pairs: list[PreferencePair] = Field(default_factory=list)
    train_prompt_ids: list[str] = Field(default_factory=list)
    validation_prompt_ids: list[str] = Field(default_factory=list)
    eval_prompt_ids: list[str] = Field(default_factory=list)
    counters: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    # -- lookups ----------------------------------------------------------- #

    def prompt_by_id(self, prompt_id: str) -> SyntheticPrompt | None:
        return next((p for p in self.prompts if p.prompt_id == prompt_id), None)

    def candidates_for(self, prompt_id: str) -> list[CandidateResponse]:
        return [c for c in self.candidates if c.prompt_id == prompt_id]

    def split_of(self, prompt_id: str) -> str:
        prompt = self.prompt_by_id(prompt_id)
        return prompt.split if prompt is not None else ""

    def usable_candidates(self, prompt_id: str) -> list[CandidateResponse]:
        """Candidates that may be judged or trained on (never failures, never empty)."""

        return [
            c
            for c in self.candidates_for(prompt_id)
            if c.status in {SampleStatus.ACCEPTED, SampleStatus.EDITED} and c.text.strip()
        ]

    def summary(self) -> dict[str, Any]:
        return self.to_summary()

    def to_summary(self) -> dict[str, Any]:
        """Compact, JSON-safe description for the dashboard and the manifest."""

        by_status: dict[str, int] = {}
        for candidate in self.candidates:
            key = (
                candidate.status.value
                if hasattr(candidate.status, "value")
                else str(candidate.status)
            )
            by_status[key] = by_status.get(key, 0) + 1
        judged = sum(1 for j in self.judgements.values() if j.error is None)
        return {
            "run_id": self.run_id,
            "version": self.version,
            "dataset_hash": self.dataset_hash,
            "split_hash": self.split_hash,
            "prompts": len(self.prompts),
            "candidates": len(self.candidates),
            "candidates_by_status": by_status,
            "judgements": len(self.judgements),
            "judgements_ok": judged,
            "judgements_failed": len(self.judgements) - judged,
            "pair_decisions": len(self.pair_decisions),
            "preference_pairs": len(self.preference_pairs),
            "train_prompts": len(self.train_prompt_ids),
            "validation_prompts": len(self.validation_prompt_ids),
            "eval_prompts": len(self.eval_prompt_ids),
            "counters": dict(self.counters),
            "warnings": list(self.warnings),
        }


class ReviewDecision(BaseModel):
    """A human reviewer's verdict on one :class:`ReviewItem`."""

    item_id: str
    reviewer: str = "human"
    action: Literal[
        "accept", "choose_a", "choose_b", "tie", "reject_both", "edit", "regenerate"
    ] = "accept"
    chosen_id: str | None = None
    rejected_id: str | None = None
    #: candidate_id -> criterion_id -> score
    scores: dict[str, dict[str, float]] = Field(default_factory=dict)
    #: candidate_id -> replacement text
    edited_texts: dict[str, str] = Field(default_factory=dict)
    comments: str = ""
    use_chosen_as_sft: bool = True
    created_at: str = Field(default_factory=_now)


class ReviewItem(BaseModel):
    """One prompt's candidates and judgements, as presented to a reviewer."""

    item_id: str
    prompt_id: str
    prompt: str
    context: list[dict[str, str]] = Field(default_factory=list)
    category: str = "ordinary"
    candidates: list[CandidateResponse] = Field(default_factory=list)
    judgements: dict[str, Judgement] = Field(default_factory=dict)
    comparison: PairDecision | None = None
    recommended_candidate_id: str | None = None
    recommendation_reasons: list[str] = Field(default_factory=list)
    human: ReviewDecision | None = None
    #: ``pending`` | ``completed`` | ``rejected`` | ``edited``
    status: str = "pending"

    def candidate_by_id(self, candidate_id: str | None) -> CandidateResponse | None:
        if candidate_id is None:
            return None
        return next((c for c in self.candidates if c.candidate_id == candidate_id), None)

    def apply_human(self, decision: ReviewDecision) -> None:
        """Fold a human decision into this item and its candidates.

        Human edits and human scores replace the machine's where they overlap;
        the machine's judgement is preserved under ``human_override`` so nothing
        is lost from the audit trail.
        """

        self.human = decision
        for candidate_id, text in decision.edited_texts.items():
            candidate = self.candidate_by_id(candidate_id)
            if candidate is None:
                continue
            candidate.edits.append(
                {
                    "source": "human",
                    "reviewer": decision.reviewer,
                    "text": text,
                    "created_at": decision.created_at,
                }
            )
            candidate.text = text
            candidate.status = SampleStatus.EDITED

        for candidate_id, criteria in decision.scores.items():
            candidate = self.candidate_by_id(candidate_id)
            judgement = self.judgements.get(candidate_id)
            if candidate is None or judgement is None:
                continue
            judgement.human_override = {
                "reviewer": decision.reviewer,
                "scores": dict(criteria),
                "created_at": decision.created_at,
            }
            judgement.source = "human"
            for criterion_id, value in criteria.items():
                score = judgement.criterion_scores.get(criterion_id)
                if score is None:
                    continue
                score.score = float(value)
                score.answer_type = "human"
                score.confidence = None

        if decision.chosen_id:
            self.recommended_candidate_id = decision.chosen_id
            self.recommendation_reasons.append(
                f"human ({decision.reviewer}) chose {decision.chosen_id}"
            )
        if decision.action == "reject_both":
            for candidate in self.candidates:
                if candidate.status is not SampleStatus.FAILED:
                    candidate.status = SampleStatus.REJECTED
            self.status = "rejected"
        elif decision.action == "regenerate":
            self.status = "pending"
        elif decision.edited_texts:
            self.status = "edited"
        else:
            self.status = "completed"


__all__ = ["DatasetBundle", "ReviewDecision", "ReviewItem"]
