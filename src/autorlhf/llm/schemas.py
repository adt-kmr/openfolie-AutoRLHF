"""Wire and domain schemas for the OpenRouter chat API and Jev decisions.

Two layers, deliberately separated:

* **Wire models** mirror the documented JSON of
  ``POST /api/alpha/decisions`` and ``POST /api/v1/chat/completions`` exactly.
  They are the only thing that ever talks to the network.
* **Domain models** (:class:`Judgement`, :class:`PairDecision`,
  :class:`PreferencePair`) are what the pipeline, database, and TUI use.  They
  carry provenance — judge model build, usage, timestamps, cache/error status —
  because the spec requires the audit trail, and ``confidence`` is only ever a
  value the judge returned.

Nothing here invents a confidence value or a probability.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from autorlhf.core.enums import DecisionKind, SampleStatus

# --------------------------------------------------------------------------- #
# Jev wire format — POST https://openrouter.ai/api/alpha/decisions
# --------------------------------------------------------------------------- #


class ScoreQuestion(BaseModel):
    """Ordered scale, lowest level first."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


class ChoiceQuestion(BaseModel):
    """One of several named options."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class NoulQuestion(BaseModel):
    """Yes/no, returned as P(true)."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: dict[str, str] | None = None


Question = Annotated[ScoreQuestion | ChoiceQuestion | NoulQuestion, Field(discriminator="type")]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    #: ``{"0": "lowest level text", ...}`` — echoes the criteria that were sent.
    legend: dict[str, str] | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]


class DecisionsRequest(BaseModel):
    """Exactly the documented request body."""

    model: str
    state: Any
    questions: dict[str, Question]
    #: Observability grouping; never sent to the provider.
    session_id: str | None = None
    user: str | None = None


class DecisionsUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


class DecisionsResponse(BaseModel):
    id: str = ""
    #: The exact build that answered, e.g. ``typesafe/jev-1.13-20260917``.
    model: str = ""
    provider: str = ""
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: DecisionsUsage = Field(default_factory=DecisionsUsage)


# --------------------------------------------------------------------------- #
# Chat wire format — OpenRouter chat completions
# --------------------------------------------------------------------------- #


class ChatChoice(BaseModel):
    index: int = 0
    message: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatUsageEnvelope(BaseModel):
    """OpenRouter returns ``cost`` inside ``usage`` on its own API."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0


class ChatCompletionResponse(BaseModel):
    id: str = ""
    model: str = ""
    provider: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: ChatUsageEnvelope = Field(default_factory=ChatUsageEnvelope)

    def first_text(self) -> str:
        if not self.choices:
            return ""
        content = self.choices[0].message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # some providers return content parts
            parts = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "".join(parts)
        return ""

    @property
    def finish_reason(self) -> str | None:
        return self.choices[0].finish_reason if self.choices else None


# --------------------------------------------------------------------------- #
# Domain models
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class CriterionScore(BaseModel):
    """One criterion's result for one response.

    ``score`` is ``None`` (never ``0``) when the judge failed — a failed remote
    judgement must not silently become a low score.
    """

    criterion_id: str
    criterion_name: str = ""
    score: float | None = None
    scale_min: int = 0
    scale_max: int = 4
    probabilities: dict[str, float] | None = None
    legend: dict[str, str] | None = None
    confidence: float | None = None
    answer_type: str = "score"
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.score is None

    def normalised(self) -> float | None:
        """``score`` mapped to 0..1 on its own scale, or ``None``."""

        if self.score is None:
            return None
        span = self.scale_max - self.scale_min
        if span <= 0:
            return None
        return max(0.0, min(1.0, (self.score - self.scale_min) / span))

    def display(self) -> str:
        if self.score is None:
            return f"failed ({self.error or 'no result'})"
        return f"{self.score:.1f}/{self.scale_max}"


class Judgement(BaseModel):
    """Jev's structured evaluation of one candidate response."""

    judgement_id: str
    run_id: str = ""
    prompt_id: str = ""
    candidate_id: str = ""
    kind: DecisionKind = DecisionKind.SCORE
    #: criterion id -> result
    criterion_scores: dict[str, CriterionScore] = Field(default_factory=dict)
    #: Weighted mean of the criteria that returned a score.
    overall: float | None = None
    #: Requested judge model id.
    judge_model: str = ""
    #: Exact build that answered (from the response's ``model`` field).
    judge_build: str = ""
    provider: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    cached: bool = False
    error: str | None = None
    retries: int = 0
    #: Verbatim response body for the audit trail.
    raw: dict[str, Any] = Field(default_factory=dict)
    #: Set when a human overrode the judgement.
    human_override: dict[str, Any] | None = None
    #: ``jev`` | ``human``
    source: str = "jev"

    @property
    def ok(self) -> bool:
        return self.error is None and not all(c.failed for c in self.criterion_scores.values())

    def scored_criteria(self) -> dict[str, float]:
        return {
            cid: cs.score  # type: ignore[misc]
            for cid, cs in self.criterion_scores.items()
            if cs.score is not None
        }

    def failed_criteria(self) -> list[str]:
        return [cid for cid, cs in self.criterion_scores.items() if cs.failed]

    def display_overall(self) -> str:
        return f"{self.overall:.2f}" if self.overall is not None else "n/a"


class PairDecision(BaseModel):
    """Jev's (or a human's) preference between two candidate responses."""

    pair_id: str
    run_id: str = ""
    prompt_id: str = ""
    option_a_id: str = ""
    option_b_id: str = ""
    #: ``a`` | ``b`` | ``tie``; ``None`` when no decision was obtained.
    winner: Literal["a", "b", "tie"] | None = None
    #: Raw option key returned by the judge.
    choice_raw: str = ""
    probabilities: dict[str, float] | None = None
    confidence: float | None = None
    #: ``overall::a - overall::b`` when both responses were scored.
    margin: float | None = None
    judge_model: str = ""
    judge_build: str = ""
    provider: str = ""
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_now)
    cached: bool = False
    error: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
    #: ``jev`` | ``human``
    source: str = "jev"
    reviewer: str = ""
    human_override: dict[str, Any] | None = None

    @property
    def usable(self) -> bool:
        return self.error is None and self.winner in {"a", "b"}

    def winner_id(self) -> str | None:
        if self.winner == "a":
            return self.option_a_id
        if self.winner == "b":
            return self.option_b_id
        return None

    def loser_id(self) -> str | None:
        if self.winner == "a":
            return self.option_b_id
        if self.winner == "b":
            return self.option_a_id
        return None


class PreferencePair(BaseModel):
    """A chosen/rejected pair accepted into DPO training data."""

    pair_id: str
    run_id: str = ""
    prompt_id: str = ""
    chosen_id: str
    rejected_id: str
    margin: float | None = None
    judge_confidence: float | None = None
    #: ``jev`` | ``human``
    source: str = "jev"
    decision: str = ""
    analyzer_notes: list[str] = Field(default_factory=list)
    reviewer: str = ""
    created_at: str = Field(default_factory=_now)
    payload: dict[str, Any] = Field(default_factory=dict)


class CandidateResponse(BaseModel):
    """One generated response to one synthetic prompt."""

    candidate_id: str
    prompt_id: str
    #: ``teacher`` (OpenRouter) or ``local`` (HF model).
    source: str = "teacher"
    model_id: str = ""
    text: str = ""
    messages: list[dict[str, str]] = Field(default_factory=list)
    status: SampleStatus = SampleStatus.PENDING
    error: str | None = None
    truncated: bool = False
    tokens: int | None = None
    latency_s: float | None = None
    seed: int | None = None
    temperature: float = 0.8
    created_at: str = Field(default_factory=_now)
    cached: bool = False
    #: Human or automatic edits applied to ``text``.
    edits: list[dict[str, Any]] = Field(default_factory=list)


class SyntheticPrompt(BaseModel):
    """A generated user prompt (plus the category it was generated for)."""

    prompt_id: str
    run_id: str = ""
    split: str = "train"
    #: ``ordinary`` | ``edge_case`` | ``boundary`` | ``context_following`` |
    #: ``varied_phrasing`` | ``multi_turn`` | ``seed``
    category: str = "ordinary"
    text: str = ""
    #: Prior turns, when the prompt continues a conversation.
    context: list[dict[str, str]] = Field(default_factory=list)
    #: Which seed conversation inspired it, if any.
    seed_label: str = ""
    avoids: str = ""
    created_at: str = Field(default_factory=_now)
    generator_model: str = ""
    cached: bool = False
    tags: list[str] = Field(default_factory=list)


class EvalPrompt(BaseModel):
    """A held-out prompt.  Never used to build training data."""

    prompt_id: str
    text: str
    context: list[dict[str, str]] = Field(default_factory=list)
    category: str = "ordinary"
    origin: str = "synthetic"


class EvalResponse(BaseModel):
    """One model's answer to one held-out prompt."""

    response_id: str
    prompt_id: str
    #: ``base`` | ``finetuned``
    variant: str
    model_id: str = ""
    adapter_path: str | None = None
    text: str = ""
    error: str | None = None
    truncated: bool = False
    latency_s: float | None = None
    tokens: int | None = None


class CriterionComparison(BaseModel):
    criterion_id: str
    criterion_name: str = ""
    base_mean: float | None = None
    finetuned_mean: float | None = None
    base_n: int = 0
    finetuned_n: int = 0
    delta: float | None = None
    scale_max: int = 4

    @property
    def improved(self) -> bool:
        return self.delta is not None and self.delta > 0

    def display_row(self) -> tuple[str, str, str, str]:
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.2f}"

        delta = "n/a" if self.delta is None else f"{self.delta:+.2f}"
        return (self.criterion_name, fmt(self.base_mean), fmt(self.finetuned_mean), delta)


class EvaluationReport(BaseModel):
    """Machine-readable base-vs-fine-tuned evaluation."""

    run_id: str
    variant: str  # ``base`` | ``finetuned``
    judge_model: str = ""
    judge_build: str = ""
    created_at: str = Field(default_factory=_now)
    rubric_fingerprint: str = ""
    eval_prompts_hash: str = ""
    responses: list[EvalResponse] = Field(default_factory=list)
    judgements: list[Judgement] = Field(default_factory=list)
    criterion_means: dict[str, float] = Field(default_factory=dict)
    overall_mean: float | None = None
    scored_responses: int = 0
    failed_judgements: int = 0
    requested_judgements: int = 0
    missing_prompt_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        if self.requested_judgements <= 0:
            return 0.0
        return self.scored_responses / self.requested_judgements


class ComparisonReport(BaseModel):
    """Base vs fine-tuned comparison, saved to ``comparison.json``."""

    run_id: str
    created_at: str = Field(default_factory=_now)
    judge_model: str = ""
    rubric_fingerprint: str = ""
    eval_prompts_hash: str = ""
    n_prompts: int = 0
    n_scored_base: int = 0
    n_scored_finetuned: int = 0
    criteria: list[CriterionComparison] = Field(default_factory=list)
    base_overall: float | None = None
    finetuned_overall: float | None = None
    overall_delta: float | None = None
    pairwise_base_wins: int = 0
    pairwise_finetuned_wins: int = 0
    pairwise_ties: int = 0
    pairwise_inconclusive: int = 0
    #: Example pairs for the UI: prompt, base text, tuned text, judgements.
    examples: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    inconclusive: bool = False
    notes: list[str] = Field(default_factory=list)


__all__ = [
    "Answer",
    "CandidateResponse",
    "ChatChoice",
    "ChatCompletionResponse",
    "ChatUsage",
    "ChatUsageEnvelope",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "ComparisonReport",
    "CriterionComparison",
    "CriterionScore",
    "DecisionsRequest",
    "DecisionsResponse",
    "DecisionsUsage",
    "EvalPrompt",
    "EvalResponse",
    "EvaluationReport",
    "Judgement",
    "NoulAnswer",
    "NoulQuestion",
    "PairDecision",
    "PreferencePair",
    "Question",
    "ScoreAnswer",
    "ScoreQuestion",
    "SyntheticPrompt",
]
