"""Foundation tests: Jev wire-schema parsing and domain model contracts.

Covers spec item 10.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from autorlhf.llm.schemas import (
    ChatCompletionResponse,
    ChoiceAnswer,
    ChoiceQuestion,
    ComparisonReport,
    CriterionComparison,
    CriterionScore,
    DecisionsRequest,
    DecisionsResponse,
    Judgement,
    NoulAnswer,
    NoulQuestion,
    PairDecision,
    PreferencePair,
    ScoreAnswer,
    ScoreQuestion,
)


def test_decisions_response_parses_documented_shape(jev_score_response: dict) -> None:
    response = DecisionsResponse.model_validate(jev_score_response)

    assert response.model == "typesafe/jev-1.13-20260917"
    assert response.provider == "TypeSafe"
    assert response.usage.input_tokens == 476
    assert response.usage.output_tokens == 70
    assert response.usage.cost == pytest.approx(0.000019992)

    answer = response.answers["character_consistency"]
    assert isinstance(answer, ScoreAnswer)
    assert answer.type == "score"
    assert answer.score == pytest.approx(3.2)
    assert answer.legend is not None
    assert answer.legend["4"] == "Excellent: fully consistent."
    assert answer.probabilities is not None
    assert answer.probabilities["4"] == pytest.approx(0.35)
    assert answer.confidence == pytest.approx(0.72)


def test_choice_and_noul_answers_parse(jev_choice_response: dict, jev_noul_response: dict) -> None:
    choice = DecisionsResponse.model_validate(jev_choice_response).answers["preference"]
    assert isinstance(choice, ChoiceAnswer)
    assert choice.choice == "b"
    assert choice.probabilities is not None
    assert choice.probabilities["b"] == pytest.approx(0.81)
    assert choice.confidence == pytest.approx(0.67)

    noul = DecisionsResponse.model_validate(jev_noul_response).answers["condition"]
    assert isinstance(noul, NoulAnswer)
    assert noul.type == "noul"
    assert noul.noul == pytest.approx(0.96)


def test_response_with_all_answer_types_round_trips(
    jev_score_response: dict, jev_choice_response: dict, jev_noul_response: dict
) -> None:
    payload = {
        "id": "gen-dec-combined",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            **jev_score_response["answers"],
            **jev_choice_response["answers"],
            **jev_noul_response["answers"],
        },
        "usage": {"input_tokens": 1000, "output_tokens": 50, "cost": 0.01},
    }

    response = DecisionsResponse.model_validate(payload)
    assert set(response.answers) == {
        "character_consistency",
        "voice_style",
        "preference",
        "condition",
    }
    assert DecisionsResponse.model_validate(json.loads(response.model_dump_json())) == response


def test_optional_probabilities_confidence_and_legend_may_be_omitted() -> None:
    payload = {
        "id": "gen-dec-sparse",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "character_consistency": {"type": "score", "score": 2.0},
            "preference": {"type": "choice", "choice": "a"},
            "condition": {"type": "noul", "noul": 0.5},
        },
        "usage": {"input_tokens": 1, "output_tokens": 1, "cost": 0.0},
    }

    response = DecisionsResponse.model_validate(payload)

    score = response.answers["character_consistency"]
    assert score.probabilities is None
    assert score.confidence is None
    assert score.legend is None

    choice = response.answers["preference"]
    assert choice.probabilities is None
    assert choice.confidence is None


def test_unknown_answer_type_raises_validation_error() -> None:
    payload = {
        "id": "gen-dec-bad",
        "model": "typesafe/jev-1.13",
        "provider": "TypeSafe",
        "answers": {"weird": {"type": "banana", "value": 1}},
        "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0},
    }
    with pytest.raises(ValidationError):
        DecisionsResponse.model_validate(payload)


def test_decisions_request_round_trips_with_discriminated_questions() -> None:
    request = DecisionsRequest(
        model="typesafe/jev-1.13",
        state={"character": {"name": "Aria"}},
        questions={
            "character_consistency": ScoreQuestion(
                instructions="Rate consistency.", criteria=["low", "mid", "high"]
            ),
            "preference": ChoiceQuestion(
                instructions="Which is better?", criteria={"a": "first", "b": "second"}
            ),
            "condition": NoulQuestion(
                instructions="Is this in character?", criteria={"true": "yes", "false": "no"}
            ),
        },
    )

    restored = DecisionsRequest.model_validate(json.loads(request.model_dump_json()))
    assert isinstance(restored.questions["character_consistency"], ScoreQuestion)
    assert isinstance(restored.questions["preference"], ChoiceQuestion)
    assert isinstance(restored.questions["condition"], NoulQuestion)
    assert restored == request


def test_criterion_score_failure_is_none_never_zero() -> None:
    failed = CriterionScore(criterion_id="voice_style", error="judge returned no answer")
    assert failed.score is None
    assert failed.score != 0
    assert failed.failed is True
    assert failed.normalised() is None
    assert "failed" in failed.display()

    scored = CriterionScore(criterion_id="voice_style", score=3.0, scale_max=4)
    assert scored.failed is False
    assert scored.normalised() == pytest.approx(0.75)
    assert scored.display() == "3.0/4"


def test_failed_judgement_reports_failure_and_excludes_missing_scores() -> None:
    all_failed = Judgement(
        judgement_id="j1",
        criterion_scores={
            "voice_style": CriterionScore(criterion_id="voice_style", error="timeout"),
        },
        error="judge failed",
    )
    assert all_failed.ok is False
    assert all_failed.scored_criteria() == {}
    assert all_failed.failed_criteria() == ["voice_style"]
    assert all_failed.display_overall() == "n/a"

    partial = Judgement(
        judgement_id="j2",
        criterion_scores={
            "a": CriterionScore(criterion_id="a", score=4.0),
            "b": CriterionScore(criterion_id="b", error="timeout"),
        },
        overall=4.0,
    )
    assert partial.ok is True
    assert partial.scored_criteria() == {"a": 4.0}
    assert partial.failed_criteria() == ["b"]
    assert partial.display_overall() == "4.00"


def test_pair_decision_and_preference_pair_round_trip() -> None:
    decision = PairDecision(
        pair_id="p0001-a-vs-b",
        prompt_id="p0001",
        option_a_id="c0",
        option_b_id="c1",
        winner="b",
        choice_raw="b",
        probabilities={"a": 0.19, "b": 0.81},
        confidence=0.67,
        margin=0.9,
    )
    assert decision.usable is True
    assert decision.winner_id() == "c1"
    assert decision.loser_id() == "c0"
    assert PairDecision.model_validate(json.loads(decision.model_dump_json())) == decision

    tie = PairDecision(pair_id="p0001-a-vs-b", winner="tie")
    assert tie.usable is False
    assert tie.winner_id() is None

    pair = PreferencePair(
        pair_id="p0001-a-vs-b",
        prompt_id="p0001",
        chosen_id="c1",
        rejected_id="c0",
        margin=0.9,
        judge_confidence=0.67,
    )
    assert PreferencePair.model_validate(json.loads(pair.model_dump_json())) == pair


def test_comparison_report_round_trips() -> None:
    report = ComparisonReport(
        run_id="run-1",
        judge_model="typesafe/jev-1.13",
        n_prompts=5,
        n_scored_base=5,
        n_scored_finetuned=4,
        criteria=[
            CriterionComparison(
                criterion_id="voice_style",
                criterion_name="Voice",
                base_mean=3.0,
                finetuned_mean=3.8,
                base_n=5,
                finetuned_n=4,
                delta=0.8,
            )
        ],
        base_overall=3.0,
        finetuned_overall=3.8,
        overall_delta=0.8,
        pairwise_finetuned_wins=3,
        pairwise_base_wins=1,
        pairwise_ties=1,
    )

    restored = ComparisonReport.model_validate(json.loads(report.model_dump_json()))
    assert restored == report
    assert restored.criteria[0].improved is True
    assert restored.criteria[0].display_row() == ("Voice", "3.00", "3.80", "+0.80")


def test_chat_completion_response_first_text_handles_content_parts() -> None:
    plain = ChatCompletionResponse.model_validate({"choices": [{"message": {"content": "hi"}}]})
    assert plain.first_text() == "hi"

    parts = ChatCompletionResponse.model_validate(
        {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "a"},
                            {"type": "text", "text": "b"},
                        ]
                    }
                }
            ]
        }
    )
    assert parts.first_text() == "ab"
    assert ChatCompletionResponse().first_text() == ""
