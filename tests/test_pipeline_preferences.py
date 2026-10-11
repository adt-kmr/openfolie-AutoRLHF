"""Preference construction: thresholds, rejections, and human override."""

from __future__ import annotations

from autorlhf.core.enums import SampleStatus
from autorlhf.llm.schemas import (
    CandidateResponse,
    CriterionScore,
    Judgement,
    PairDecision,
    SyntheticPrompt,
)
from autorlhf.pipeline.models import ReviewDecision
from autorlhf.pipeline.preferences import build_preferences, select_pair, to_records


class StubJev:
    """A JevEvaluator stand-in that returns pre-canned pair decisions."""

    def __init__(self, decisions: list[PairDecision]) -> None:
        self.run_id = "run-1"
        self.decisions = decisions
        self.calls: list[dict] = []

    def compare_responses(self, **kwargs):
        self.calls.append(kwargs)
        return self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]


def prompt(prompt_id: str = "p0001", *, split: str = "train") -> SyntheticPrompt:
    return SyntheticPrompt(prompt_id=prompt_id, category="ordinary", text="Q?", split=split)


def candidate(candidate_id: str, *, text: str, prompt_id: str = "p0001") -> CandidateResponse:
    return CandidateResponse(
        candidate_id=candidate_id,
        prompt_id=prompt_id,
        text=text,
        status=SampleStatus.ACCEPTED,
        source="teacher",
        model_id="openai/gpt-4o-mini",
    )


def judgement(candidate_id: str, *, overall: float | None, prompt_id: str = "p0001") -> Judgement:
    return Judgement(
        judgement_id=f"j-{candidate_id}",
        prompt_id=prompt_id,
        candidate_id=candidate_id,
        criterion_scores={
            "character_consistency": CriterionScore(
                criterion_id="character_consistency", score=overall
            )
        },
        overall=overall,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
    )


def decision(
    *,
    winner: str | None = "a",
    margin: float | None = 2.0,
    confidence: float | None = 0.8,
    error: str | None = None,
    choice_raw: str = "a",
) -> PairDecision:
    return PairDecision(
        pair_id="p0001-c0-vs-c1",
        run_id="run-1",
        prompt_id="p0001",
        option_a_id="p0001-c0",
        option_b_id="p0001-c1",
        winner=winner,  # type: ignore[arg-type]
        choice_raw=choice_raw,
        probabilities={"a": 0.8, "b": 0.2},
        confidence=confidence,
        margin=margin,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
        error=error,
    )


def two_candidates():
    return [
        candidate("p0001-c0", text="better answer"),
        candidate("p0001-c1", text="worse answer"),
    ]


def judgements_for(best: float = 4.0, second: float = 2.0):
    return {
        "p0001-c0": judgement("p0001-c0", overall=best),
        "p0001-c1": judgement("p0001-c1", overall=second),
    }


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #


def test_pair_accepted_above_thresholds(rubric):
    jev = StubJev([decision(winner="a", margin=2.0, confidence=0.8)])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.status == "accepted"
    pair = outcome.pair
    assert pair is not None
    assert pair.chosen_id == "p0001-c0"
    assert pair.rejected_id == "p0001-c1"
    assert pair.margin == 2.0
    assert pair.judge_confidence == 0.8
    assert pair.source == "jev"
    assert pair.decision == "a"
    assert outcome.criterion_deltas == {"character_consistency": 2.0}
    assert any("margin" in reason for reason in outcome.reasons)
    assert any("confidence" in reason for reason in outcome.reasons)
    # the judge was asked with both overalls so the margin is comparable
    call = jev.calls[0]
    assert call["overall_a"] == 4.0
    assert call["overall_b"] == 2.0
    assert call["option_a"].candidate_id == "p0001-c0"


def test_tie_is_rejected_with_the_reason_recorded(rubric):
    jev = StubJev([decision(winner="tie", margin=2.0, confidence=0.9, choice_raw="tie")])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.status == "rejected"
    assert outcome.kind == "tie"
    assert outcome.pair is None
    assert outcome.rejection_entry is not None
    assert outcome.rejection_entry["kind"] == "tie"
    assert "no meaningful preference" in outcome.reasons[0]


def test_small_margin_is_a_tie(rubric):
    jev = StubJev([decision(winner="a", margin=0.1, confidence=0.9)])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(best=3.0, second=2.9),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.kind == "tie"
    assert "below" in outcome.reasons[0]


def test_low_confidence_is_rejected(rubric):
    jev = StubJev([decision(winner="a", margin=2.0, confidence=0.2)])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.status == "rejected"
    assert outcome.kind == "low_confidence"
    assert outcome.rejection_entry["kind"] == "low_confidence"
    assert outcome.rejection_entry["confidence"] == 0.2


def test_missing_confidence_still_accepts_on_margin(rubric):
    jev = StubJev([decision(winner="a", margin=2.0, confidence=None)])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.status == "accepted"
    assert outcome.pair is not None
    assert outcome.pair.judge_confidence is None
    assert any("no confidence" in reason for reason in outcome.reasons)


def test_judge_error_is_rejected(rubric):
    jev = StubJev([decision(winner=None, margin=None, error="judge down")])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.kind == "invalid"
    assert "judge down" in outcome.reasons[0]


def test_insufficient_candidates_makes_no_judge_call(rubric):
    jev = StubJev([decision()])
    outcome = select_pair(
        prompt=prompt(),
        candidates=[candidate("p0001-c0", text="only one")],
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    assert outcome.kind == "insufficient_candidates"
    assert jev.calls == []


def test_missing_overalls_are_rejected_as_invalid(rubric):
    jev = StubJev([decision(winner="a", margin=None)])
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements={},
        jev=jev,
        rubric=rubric,
    )
    assert outcome.kind == "invalid"
    assert "usable judgement" in outcome.reasons[0]


# --------------------------------------------------------------------------- #
# human override
# --------------------------------------------------------------------------- #


def test_human_decision_overrides_jev_and_is_recorded(rubric):
    jev = StubJev([decision(winner="a", margin=2.0, confidence=0.8)])
    human = ReviewDecision(
        item_id="p0001",
        action="choose_b",
        chosen_id="p0001-c1",
        reviewer="alice",
    )
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
        human=human,
    )
    assert outcome.status == "accepted"
    pair = outcome.pair
    assert pair is not None
    assert pair.source == "human"
    assert pair.chosen_id == "p0001-c1"
    assert pair.rejected_id == "p0001-c0"
    assert pair.reviewer == "alice"
    assert pair.decision == "choose_b"
    # Jev's recommendation is preserved in the payload
    assert pair.payload["jev_recommendation"] == "p0001-c0"
    assert pair.payload["jev_decision"]["winner"] == "a"
    assert pair.payload["human"]["reviewer"] == "alice"
    assert outcome.criterion_deltas == {"character_consistency": -2.0}


def test_human_tie_rejects_the_pair(rubric):
    jev = StubJev([decision(winner="a", margin=2.0)])
    human = ReviewDecision(item_id="p0001", action="tie")
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
        human=human,
    )
    assert outcome.status == "rejected"
    assert outcome.kind == "tie"
    assert "human decision 'tie'" in outcome.reasons[0]


def test_human_reject_both_rejects_the_pair(rubric):
    jev = StubJev([decision()])
    human = ReviewDecision(item_id="p0001", action="reject_both")
    outcome = select_pair(
        prompt=prompt(),
        candidates=two_candidates(),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
        human=human,
    )
    assert outcome.kind == "invalid"
    assert "reject_both" in outcome.reasons[0]


def test_human_can_select_a_third_candidate(rubric):
    jev = StubJev([decision(winner="a", margin=2.0)])
    third = candidate("p0001-c2", text="third answer")
    human = ReviewDecision(item_id="p0001", action="edit", chosen_id="p0001-c2")
    outcome = select_pair(
        prompt=prompt(),
        candidates=[*two_candidates(), third],
        judgements={**judgements_for(), "p0001-c2": judgement("p0001-c2", overall=1.0)},
        jev=jev,
        rubric=rubric,
        human=human,
    )
    assert outcome.status == "accepted"
    assert outcome.pair is not None
    assert outcome.pair.chosen_id == "p0001-c2"
    assert outcome.pair.rejected_id == "p0001-c0"


# --------------------------------------------------------------------------- #
# aggregation and records
# --------------------------------------------------------------------------- #


def test_build_preferences_aggregates_counters_and_audit(rubric):
    prompts = [prompt("p0001"), prompt("p0002")]
    candidates = [
        candidate("p0001-c0", text="a"),
        candidate("p0001-c1", text="b"),
        candidate("p0002-c0", text="c", prompt_id="p0002"),
        candidate("p0002-c1", text="d", prompt_id="p0002"),
    ]
    judgements = {
        "p0001-c0": judgement("p0001-c0", overall=4.0),
        "p0001-c1": judgement("p0001-c1", overall=2.0),
        "p0002-c0": judgement("p0002-c0", overall=3.0, prompt_id="p0002"),
        "p0002-c1": judgement("p0002-c1", overall=2.5, prompt_id="p0002"),
    }
    jev = StubJev(
        [
            decision(winner="a", margin=2.0),
            decision(winner="tie", margin=0.5, choice_raw="tie"),
        ]
    )
    build = build_preferences(
        prompts=prompts, candidates=candidates, judgements=judgements, jev=jev, rubric=rubric
    )
    assert len(build.accepted) == 1
    assert len(build.rejected) == 1
    assert build.kinds["accepted"] == 1
    assert build.kinds["tie"] == 1
    counters = build.counters()
    assert counters["pairs_accepted"] == 1
    assert counters["pairs_rejected_tie"] == 1
    # Rejections carry a machine-readable entry so they can be audited later.
    assert build.rejected[0].rejection_entry is not None
    assert build.rejected[0].rejection_entry["kind"] == "tie"
    assert build.rejected[0].rejection_entry["reasons"]
    # The accepted prompt had no rejected comparisons of its own.
    assert build.accepted[0].audit_rejected == []
    assert build.accepted[0].prompt_id == "p0001"
    assert build.rejected[0].prompt_id == "p0002"


def test_to_records_matches_the_frozen_preference_shape(rubric):
    prompts = {"p0001": prompt("p0001")}
    candidates = {c.candidate_id: c for c in two_candidates()}
    jev = StubJev([decision(winner="a", margin=2.0, confidence=0.8)])
    build = build_preferences(
        prompts=[prompts["p0001"]],
        candidates=list(candidates.values()),
        judgements=judgements_for(),
        jev=jev,
        rubric=rubric,
    )
    records = to_records(
        build, run_id="run-1", prompts_by_id=prompts, candidates_by_id=candidates, rubric=rubric
    )
    assert len(records) == 1
    record = records[0]
    for key in (
        "record_type",
        "dataset_version",
        "run_id",
        "pair_id",
        "prompt_id",
        "split",
        "prompt",
        "prompt_context",
        "chosen_id",
        "rejected_id",
        "chosen_text",
        "rejected_text",
        "margin",
        "judge_confidence",
        "source",
        "decision",
        "probabilities",
        "criterion_deltas",
        "audit",
        "pair_decision",
        "reviewer",
        "created_at",
    ):
        assert key in record
    assert record["record_type"] == "preference"
    assert record["source"] == "jev"
    assert record["chosen_text"] == "better answer"
    assert record["rejected_text"] == "worse answer"
    assert record["audit"]["thresholds"]["tie_epsilon"] == rubric.tie_epsilon
    assert record["audit"]["reasons"]
    assert record["pair_decision"]["winner"] == "a"
