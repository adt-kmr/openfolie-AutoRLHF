"""Turning judgements into DPO preference pairs.

A pair is only written when the evidence actually supports it:

* the judge's winner must be decisive (``a`` or ``b``, never a tie),
* the overall margin must exceed ``rubric.tie_epsilon``,
* the judge's confidence, *when it returned one*, must be at least
  ``rubric.min_confidence_for_preference``.

Anything else is recorded as a rejection with the specific reason, so a run
never manufactures a preference label out of noise.  A human decision, when one
exists, always wins over the judge and is stored with ``source="human"`` while
the judge's recommendation is kept in the payload.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from autorlhf.config.rubric import Rubric
from autorlhf.core.enums import SampleStatus
from autorlhf.core.logging import get_logger
from autorlhf.llm.jev import JevEvaluator
from autorlhf.llm.schemas import CandidateResponse, Judgement, PairDecision, PreferencePair
from autorlhf.pipeline.dataset import preference_record
from autorlhf.pipeline.models import ReviewDecision
from autorlhf.pipeline.util import now_iso

logger = get_logger("autorlhf.pipeline.preferences")

#: Human actions that can never produce a preference pair.
_NON_PAIR_ACTIONS = {"tie", "reject_both", "regenerate"}


class PromptPreference(BaseModel):
    """The outcome of building one prompt's preference."""

    prompt_id: str
    #: ``accepted`` | ``rejected``
    status: str
    #: ``accepted`` or a rejection kind (``tie``, ``low_confidence``, ``invalid``,
    #: ``insufficient_candidates``).
    kind: str
    reasons: list[str] = Field(default_factory=list)
    pair: PreferencePair | None = None
    decision: PairDecision | None = None
    criterion_deltas: dict[str, float] = Field(default_factory=dict)
    #: Entry written into an accepted pair's ``audit.rejected_pairs``.
    rejection_entry: dict[str, Any] | None = None
    #: Rejections belonging to the same prompt (filled in by :func:`build_preferences`).
    audit_rejected: list[dict[str, Any]] = Field(default_factory=list)


class PreferenceBuild(BaseModel):
    """Everything :func:`build_preferences` produced for a dataset."""

    accepted: list[PromptPreference] = Field(default_factory=list)
    rejected: list[PromptPreference] = Field(default_factory=list)
    #: Raw rejection kind -> count.
    kinds: dict[str, int] = Field(default_factory=dict)

    def counters(self) -> dict[str, int]:
        """Counters in the names ``docs/DATA_FORMATS.md``/``dataset.py`` use."""

        return {
            "pairs_accepted": len(self.accepted),
            "pairs_rejected_tie": int(self.kinds.get("tie", 0)),
            "pairs_rejected_low_confidence": int(self.kinds.get("low_confidence", 0)),
            "pairs_rejected_invalid": int(self.kinds.get("invalid", 0))
            + int(self.kinds.get("insufficient_candidates", 0)),
        }


def _overall(judgements: dict[str, Judgement], candidate_id: str) -> float | None:
    judgement = judgements.get(candidate_id)
    return judgement.overall if judgement is not None else None


def _criterion_deltas(
    rubric: Rubric, judgements: dict[str, Judgement], chosen_id: str, rejected_id: str
) -> dict[str, float]:
    chosen = judgements.get(chosen_id)
    rejected = judgements.get(rejected_id)
    if chosen is None or rejected is None:
        return {}
    deltas: dict[str, float] = {}
    for criterion in rubric.criteria:
        a = chosen.criterion_scores.get(criterion.id)
        b = rejected.criterion_scores.get(criterion.id)
        if a is None or b is None or a.score is None or b.score is None:
            continue
        deltas[criterion.id] = round(float(a.score) - float(b.score), 4)
    return deltas


def _rejection_entry(
    *,
    pair_id: str,
    kind: str,
    reasons: list[str],
    decision: PairDecision | None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "kind": kind,
        "reasons": list(reasons),
        "margin": decision.margin if decision is not None else None,
        "confidence": decision.confidence if decision is not None else None,
        "winner": decision.winner if decision is not None else None,
    }


def _rejected(
    prompt_id: str,
    kind: str,
    reasons: list[str],
    *,
    pair_id: str = "",
    decision: PairDecision | None = None,
) -> PromptPreference:
    return PromptPreference(
        prompt_id=prompt_id,
        status="rejected",
        kind=kind,
        reasons=reasons,
        decision=decision,
        rejection_entry=_rejection_entry(
            pair_id=pair_id or f"{prompt_id}-pair", kind=kind, reasons=reasons, decision=decision
        ),
    )


def _accept(
    *,
    prompt_id: str,
    pair: PreferencePair,
    decision: PairDecision | None,
    reasons: list[str],
    criterion_deltas: dict[str, float],
) -> PromptPreference:
    return PromptPreference(
        prompt_id=prompt_id,
        status="accepted",
        kind="accepted",
        reasons=reasons,
        pair=pair,
        decision=decision,
        criterion_deltas=criterion_deltas,
    )


def _from_human(
    *,
    prompt_id: str,
    human: ReviewDecision,
    eligible: list[CandidateResponse],
    best: CandidateResponse,
    second: CandidateResponse,
    decision: PairDecision | None,
    rubric: Rubric,
    judgements: dict[str, Judgement],
    run_id: str,
) -> PromptPreference | None:
    """Build the outcome a human decision dictates.  Returns ``None`` to fall back to Jev."""

    pair_id = f"{prompt_id}-{best.candidate_id}-vs-{second.candidate_id}"
    if human.action in _NON_PAIR_ACTIONS:
        kind = "tie" if human.action == "tie" else "invalid"
        return _rejected(
            prompt_id,
            kind,
            [f"human decision '{human.action}'"],
            pair_id=pair_id,
            decision=decision,
        )

    if human.action == "choose_a":
        chosen, rejected = best, second
        reason = f"human chose A ({best.candidate_id})"
    elif human.action == "choose_b":
        chosen, rejected = second, best
        reason = f"human chose B ({second.candidate_id})"
    elif human.chosen_id:
        selected = next((c for c in eligible if c.candidate_id == human.chosen_id), None)
        if selected is None:
            return _rejected(
                prompt_id,
                "invalid",
                [f"human chose unknown candidate '{human.chosen_id}'"],
                pair_id=pair_id,
                decision=decision,
            )
        chosen = selected
        rejected = second if selected is best else best
        reason = f"human selected {chosen.candidate_id}"
    else:
        chosen, rejected = best, second
        reason = "human accepted the judge's recommendation"

    payload: dict[str, Any] = {
        "human": human.model_dump(mode="json"),
        "reasons": [reason],
    }
    if decision is not None:
        payload["jev_decision"] = decision.model_dump(mode="json")
        payload["jev_recommendation"] = decision.winner_id()
    pair = PreferencePair(
        pair_id=pair_id,
        run_id=run_id,
        prompt_id=prompt_id,
        chosen_id=chosen.candidate_id,
        rejected_id=rejected.candidate_id,
        margin=decision.margin if decision is not None else None,
        judge_confidence=decision.confidence if decision is not None else None,
        source="human",
        decision=human.action,
        reviewer=human.reviewer,
        created_at=now_iso(),
        payload=payload,
    )
    return _accept(
        prompt_id=prompt_id,
        pair=pair,
        decision=decision,
        reasons=[reason],
        criterion_deltas=_criterion_deltas(
            rubric, judgements, chosen.candidate_id, rejected.candidate_id
        ),
    )


def select_pair(
    *,
    prompt: Any,
    candidates: list[CandidateResponse],
    judgements: dict[str, Judgement],
    jev: JevEvaluator,
    rubric: Rubric,
    human: ReviewDecision | None = None,
) -> PromptPreference:
    """Score a prompt's candidates and decide whether a pair may be written."""

    prompt_id = prompt.prompt_id
    eligible = [
        c
        for c in candidates
        if c.prompt_id == prompt_id
        and c.status in {SampleStatus.ACCEPTED, SampleStatus.EDITED}
        and c.text.strip()
    ]
    if len(eligible) < 2:
        return _rejected(
            prompt_id,
            "insufficient_candidates",
            [f"only {len(eligible)} usable candidate response(s); need two"],
        )

    ranked = sorted(
        eligible,
        key=lambda c: (
            _overall(judgements, c.candidate_id) is None,
            -(_overall(judgements, c.candidate_id) or 0.0),
            c.candidate_id,
        ),
    )
    best, second = ranked[0], ranked[1]
    overall_a = _overall(judgements, best.candidate_id)
    overall_b = _overall(judgements, second.candidate_id)

    pair_id = f"{prompt_id}-{best.candidate_id}-vs-{second.candidate_id}"
    if overall_a is None and overall_b is None:
        return _rejected(
            prompt_id,
            "invalid",
            ["neither candidate has a usable judgement"],
            pair_id=pair_id,
        )

    decision = jev.compare_responses(
        prompt_id=prompt_id,
        option_a=best,
        option_b=second,
        prompt=prompt.text,
        context=prompt.context,
        overall_a=overall_a,
        overall_b=overall_b,
    )

    if human is not None:
        outcome = _from_human(
            prompt_id=prompt_id,
            human=human,
            eligible=eligible,
            best=best,
            second=second,
            decision=decision,
            rubric=rubric,
            judgements=judgements,
            run_id=jev.run_id,
        )
        if outcome is not None:
            return outcome

    if decision.error is not None:
        return _rejected(
            prompt_id,
            "invalid",
            [f"judge error: {decision.error}"],
            pair_id=pair_id,
            decision=decision,
        )
    if decision.winner is None:
        return _rejected(
            prompt_id,
            "invalid",
            ["the judge returned no winner"],
            pair_id=pair_id,
            decision=decision,
        )
    if decision.winner == "tie":
        return _rejected(
            prompt_id,
            "tie",
            ["the judge saw no meaningful preference"],
            pair_id=pair_id,
            decision=decision,
        )

    thresholds = (
        f"tie_epsilon {rubric.tie_epsilon:g}",
        f"min_confidence {rubric.min_confidence_for_preference:g}",
    )
    if decision.margin is None:
        return _rejected(
            prompt_id,
            "invalid",
            ["no comparable overall scores for both responses", *thresholds],
            pair_id=pair_id,
            decision=decision,
        )
    if abs(decision.margin) < rubric.tie_epsilon:
        return _rejected(
            prompt_id,
            "tie",
            [
                f"overall margin {abs(decision.margin):.3f} is below {thresholds[0]}",
                *thresholds[1:],
            ],
            pair_id=pair_id,
            decision=decision,
        )
    if (
        decision.confidence is not None
        and decision.confidence < rubric.min_confidence_for_preference
    ):
        return _rejected(
            prompt_id,
            "low_confidence",
            [
                f"judge confidence {decision.confidence:.3f} is below {thresholds[1]}",
                thresholds[0],
            ],
            pair_id=pair_id,
            decision=decision,
        )

    chosen_id = decision.winner_id()
    rejected_id = decision.loser_id()
    if chosen_id is None or rejected_id is None:  # pragma: no cover - guarded above
        return _rejected(
            prompt_id, "invalid", ["the judge's winner has no response id"], decision=decision
        )

    reasons = [
        f"overall margin {abs(decision.margin):.3f} exceeds {thresholds[0]}",
    ]
    if decision.confidence is not None:
        reasons.append(f"judge confidence {decision.confidence:.3f} meets {thresholds[1]}")
    else:
        reasons.append("the judge returned no confidence; the margin alone decided it")

    pair = PreferencePair(
        pair_id=pair_id,
        run_id=jev.run_id,
        prompt_id=prompt_id,
        chosen_id=chosen_id,
        rejected_id=rejected_id,
        margin=decision.margin,
        judge_confidence=decision.confidence,
        source="jev",
        decision=decision.choice_raw,
        created_at=now_iso(),
        payload={
            "probabilities": decision.probabilities,
            "judge_build": decision.judge_build,
            "option_a_id": decision.option_a_id,
            "option_b_id": decision.option_b_id,
        },
    )
    return _accept(
        prompt_id=prompt_id,
        pair=pair,
        decision=decision,
        reasons=reasons,
        criterion_deltas=_criterion_deltas(rubric, judgements, chosen_id, rejected_id),
    )


def build_preferences(
    *,
    prompts: list[Any],
    candidates: list[CandidateResponse],
    judgements: dict[str, Judgement],
    jev: JevEvaluator,
    rubric: Rubric,
    decisions: dict[str, ReviewDecision] | None = None,
    cancel_event: Any | None = None,
    on_progress: Any | None = None,
) -> PreferenceBuild:
    """Build a preference for every prompt, in order, honouring human decisions."""

    decisions = decisions or {}
    build = PreferenceBuild()
    kinds: dict[str, int] = defaultdict(int)
    total = len(prompts)
    for index, prompt in enumerate(prompts):
        if cancel_event is not None and cancel_event.is_set():
            break
        outcome = select_pair(
            prompt=prompt,
            candidates=candidates,
            judgements=judgements,
            jev=jev,
            rubric=rubric,
            human=decisions.get(prompt.prompt_id),
        )
        kinds[outcome.kind] += 1
        if outcome.status == "accepted":
            build.accepted.append(outcome)
        else:
            build.rejected.append(outcome)
        if on_progress is not None:
            on_progress(index + 1, total)

    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for outcome in build.rejected:
        if outcome.rejection_entry is not None:
            by_prompt[outcome.prompt_id].append(outcome.rejection_entry)
    for outcome in build.accepted:
        outcome.audit_rejected = list(by_prompt.get(outcome.prompt_id, []))
    build.kinds = dict(kinds)
    return build


def to_records(
    build: PreferenceBuild,
    *,
    run_id: str,
    prompts_by_id: dict[str, Any],
    candidates_by_id: dict[str, CandidateResponse],
    rubric: Rubric,
) -> list[dict[str, Any]]:
    """Serialise accepted pairs into ``preferences.jsonl`` records."""

    records: list[dict[str, Any]] = []
    for outcome in build.accepted:
        pair = outcome.pair
        prompt = prompts_by_id.get(outcome.prompt_id)
        if pair is None or prompt is None:
            continue
        chosen = candidates_by_id.get(pair.chosen_id)
        rejected = candidates_by_id.get(pair.rejected_id)
        if chosen is None or rejected is None:  # pragma: no cover - guarded upstream
            continue
        audit = {
            "thresholds": {
                "tie_epsilon": rubric.tie_epsilon,
                "min_confidence": rubric.min_confidence_for_preference,
            },
            "reasons": list(outcome.reasons),
            "rejected_pairs": list(outcome.audit_rejected),
        }
        records.append(
            preference_record(
                run_id=run_id,
                prompt=prompt,
                pair=pair,
                chosen=chosen,
                rejected=rejected,
                decision=outcome.decision,
                criterion_deltas=outcome.criterion_deltas,
                audit=audit,
            )
        )
    return records


__all__ = [
    "PreferenceBuild",
    "PromptPreference",
    "build_preferences",
    "select_pair",
    "to_records",
]
