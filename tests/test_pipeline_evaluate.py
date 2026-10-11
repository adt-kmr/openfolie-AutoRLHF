"""Evaluation aggregation and comparison maths."""

from __future__ import annotations

import pytest

from autorlhf.core.events import EventBus
from autorlhf.llm.schemas import (
    ComparisonReport,
    CriterionScore,
    EvalPrompt,
    EvalResponse,
    EvaluationReport,
    Judgement,
    PairDecision,
)
from autorlhf.pipeline import evaluate as evaluate_module
from autorlhf.pipeline.evaluate import (
    EvalModelHandle,
    compare_reports,
    eval_candidate_id,
    evaluate_model,
)


class StubJev:
    def __init__(self, judgements: list[Judgement]) -> None:
        self.judgements = judgements
        self.calls = 0

    def score_response(self, **kwargs):
        self.calls += 1
        return self.judgements.pop(0) if len(self.judgements) > 1 else self.judgements[0]


def judgement(prompt_id: str, variant: str, *, score: float | None, confidence: float | None = 0.7):
    return Judgement(
        judgement_id=f"j-{variant}-{prompt_id}",
        run_id="run-1",
        prompt_id=prompt_id,
        candidate_id=eval_candidate_id(variant, prompt_id),
        criterion_scores={
            "character_consistency": CriterionScore(
                criterion_id="character_consistency",
                criterion_name="Character consistency",
                score=score,
                confidence=confidence,
                probabilities={"0": 0.1, "4": 0.9} if score is not None else None,
            )
        },
        overall=score,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
        error=None if score is not None else "judge failed",
    )


def handle(variant: str = "base") -> EvalModelHandle:
    return EvalModelHandle(
        variant=variant, model=None, tokenizer=None, model_id="tiny", notes=["note"]
    )


def prompt(prompt_id: str) -> EvalPrompt:
    return EvalPrompt(prompt_id=prompt_id, text=f"question {prompt_id}", category="ordinary")


def test_evaluate_model_aggregates_scored_responses_only(monkeypatch, rubric):
    prompts = [prompt("p1"), prompt("p2")]
    responses = {
        "p1": EvalResponse(response_id="r1", prompt_id="p1", variant="base", text="answer one"),
        "p2": EvalResponse(response_id="r2", prompt_id="p2", variant="base", text=""),
    }
    responses["p2"].error = "generation failed"
    monkeypatch.setattr(
        evaluate_module,
        "generate_eval_response",
        lambda handle, p, **kwargs: responses[p.prompt_id],
    )
    jev = StubJev([judgement("p1", "base", score=4.0)])
    bus = EventBus()
    report = evaluate_model(
        handle=handle(),
        prompts=prompts,
        jev=jev,
        rubric=rubric,
        run_id="run-1",
        eval_prompts_hash="hash",
        variant="base",
        judge_model="typesafe/jev-1.13",
        min_responses_for_claims=1,
        bus=bus,
    )
    assert report.scored_responses == 1
    assert report.failed_judgements == 1
    assert report.requested_judgements == 2
    assert report.criterion_means["character_consistency"] == pytest.approx(4.0)
    assert report.overall_mean == pytest.approx(4.0)
    assert report.missing_prompt_ids == ["p2"]
    assert report.eval_prompts_hash == "hash"
    assert "note" in report.warnings
    assert jev.calls == 1  # the failed generation was never judged
    events = bus.history()
    assert any(event.verbose_only and "local inference start base 1/2" in event.message for event in events)
    assert any(event.verbose_only and "starting Jev evaluation model=typesafe/jev-1.13" in event.message for event in events)
    sample = next(event for event in events if event.kind == "sample")
    assert sample.data["prompt"] == "question p1"
    assert sample.data["response"] == "answer one"
    assert sample.data["judgement"]["overall"] == 4.0
    assert any(
        event.kind == "progress" and "evaluating base: 1/2" == event.message for event in events
    )


def test_evaluate_model_marks_low_coverage_inconclusive(monkeypatch, rubric):
    prompts = [prompt("p1")]
    monkeypatch.setattr(
        evaluate_module,
        "generate_eval_response",
        lambda handle, p, **kwargs: EvalResponse(
            response_id="r", prompt_id=p.prompt_id, variant="base", text="answer"
        ),
    )
    jev = StubJev([judgement("p1", "base", score=3.0)])
    report = evaluate_model(
        handle=handle(),
        prompts=prompts,
        jev=jev,
        rubric=rubric,
        run_id="run-1",
        eval_prompts_hash="hash",
        variant="base",
        judge_model="typesafe/jev-1.13",
        min_responses_for_claims=6,
    )
    assert report.scored_responses == 1
    assert any("min_responses_for_claims" in w for w in report.warnings)


def test_evaluate_model_averages_repeated_judge_passes(monkeypatch, rubric):
    prompts = [prompt("p1")]
    monkeypatch.setattr(
        evaluate_module,
        "generate_eval_response",
        lambda handle, p, **kwargs: EvalResponse(
            response_id="r", prompt_id=p.prompt_id, variant="base", text="answer"
        ),
    )
    jev = StubJev(
        [
            judgement("p1", "base", score=2.0, confidence=0.5),
            judgement("p1", "base", score=4.0, confidence=0.9),
        ]
    )
    report = evaluate_model(
        handle=handle(),
        prompts=prompts,
        jev=jev,
        rubric=rubric,
        run_id="run-1",
        eval_prompts_hash="hash",
        variant="base",
        judge_model="typesafe/jev-1.13",
        judge_repeats=2,
        min_responses_for_claims=1,
    )
    assert jev.calls == 2
    merged = report.judgements[0]
    assert merged.criterion_scores["character_consistency"].score == pytest.approx(3.0)
    assert merged.criterion_scores["character_consistency"].confidence == pytest.approx(0.7)
    assert merged.overall == pytest.approx(3.0)
    assert merged.error is None


def test_evaluate_model_records_a_failed_judgement_without_a_zero(monkeypatch, rubric):
    prompts = [prompt("p1")]
    monkeypatch.setattr(
        evaluate_module,
        "generate_eval_response",
        lambda handle, p, **kwargs: EvalResponse(
            response_id="r", prompt_id=p.prompt_id, variant="base", text="answer"
        ),
    )
    jev = StubJev([judgement("p1", "base", score=None)])
    report = evaluate_model(
        handle=handle(),
        prompts=prompts,
        jev=jev,
        rubric=rubric,
        run_id="run-1",
        eval_prompts_hash="hash",
        variant="base",
        judge_model="typesafe/jev-1.13",
        min_responses_for_claims=1,
    )
    assert report.scored_responses == 0
    assert report.failed_judgements == 1
    assert report.criterion_means == {}
    assert report.overall_mean is None


def _report(
    variant: str,
    *,
    scores: dict[str, float],
    texts: dict[str, str],
    overall: float | None,
) -> EvaluationReport:
    report = EvaluationReport(
        run_id="run-1",
        variant=variant,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
        eval_prompts_hash="hash",
    )
    for prompt_id, score in scores.items():
        report.responses.append(
            EvalResponse(
                response_id=f"{variant}-{prompt_id}",
                prompt_id=prompt_id,
                variant=variant,
                text=texts[prompt_id],
            )
        )
        report.judgements.append(judgement(prompt_id, variant, score=score))
    report.scored_responses = len(scores)
    report.requested_judgements = len(scores)
    report.criterion_means = (
        {"character_consistency": sum(scores.values()) / len(scores)} if scores else {}
    )
    report.overall_mean = overall
    return report


def test_compare_reports_computes_deltas_pairwise_and_inconclusive():
    base = _report(
        "base",
        scores={"p1": 2.0, "p2": 3.0},
        texts={"p1": "base one", "p2": "base two"},
        overall=2.5,
    )
    final = _report(
        "finetuned",
        scores={"p1": 3.0, "p2": 4.0},
        texts={"p1": "tuned one", "p2": "tuned two"},
        overall=3.5,
    )
    pairwise = [
        PairDecision(
            pair_id="p1-pair",
            prompt_id="p1",
            option_a_id=eval_candidate_id("base", "p1"),
            option_b_id=eval_candidate_id("finetuned", "p1"),
            winner="b",
        ),
        PairDecision(
            pair_id="p2-pair",
            prompt_id="p2",
            option_a_id=eval_candidate_id("base", "p2"),
            option_b_id=eval_candidate_id("finetuned", "p2"),
            winner="tie",
        ),
        PairDecision(
            pair_id="p3-pair",
            prompt_id="p3",
            winner=None,
            error="judge down",
        ),
    ]
    prompts_by_id = {pid: prompt(pid) for pid in ("p1", "p2")}
    comparison = compare_reports(
        base,
        final,
        pairwise_pairs=pairwise,
        prompts_by_id=prompts_by_id,
        min_responses_for_claims=6,
    )
    assert isinstance(comparison, ComparisonReport)
    assert comparison.base_overall == pytest.approx(2.5)
    assert comparison.finetuned_overall == pytest.approx(3.5)
    assert comparison.overall_delta == pytest.approx(1.0)
    criterion = comparison.criteria[0]
    assert criterion.criterion_id == "character_consistency"
    assert criterion.base_mean == pytest.approx(2.5)
    assert criterion.finetuned_mean == pytest.approx(3.5)
    assert criterion.delta == pytest.approx(1.0)
    assert criterion.base_n == 2
    assert criterion.finetuned_n == 2
    assert comparison.pairwise_finetuned_wins == 1
    assert comparison.pairwise_ties == 1
    assert comparison.pairwise_inconclusive == 1
    assert comparison.inconclusive is True
    assert any("min_responses_for_claims" in w for w in comparison.warnings)
    assert len(comparison.examples) == 2
    example = comparison.examples[0]
    assert example["prompt"] == "question p1"
    assert example["base_text"] == "base one"
    assert example["finetuned_text"] == "tuned one"
    assert example["pairwise_winner"] == "b"
    assert example["base_judgement"]["overall"] == 2.0


def test_compare_reports_flags_differences_within_noise():
    base = _report(
        "base",
        scores={f"p{i}": 3.0 for i in range(8)},
        texts={f"p{i}": f"base {i}" for i in range(8)},
        overall=3.0,
    )
    final = _report(
        "finetuned",
        scores={f"p{i}": 3.1 for i in range(8)},
        texts={f"p{i}": f"tuned {i}" for i in range(8)},
        overall=3.1,
    )
    comparison = compare_reports(
        base,
        final,
        prompts_by_id={f"p{i}": prompt(f"p{i}") for i in range(8)},
        min_responses_for_claims=6,
    )
    assert comparison.inconclusive is True
    assert any("noise" in note for note in comparison.notes)
    assert comparison.overall_delta == pytest.approx(0.1)


def test_compare_reports_is_conclusive_for_a_large_clear_effect():
    base = _report(
        "base",
        scores={f"p{i}": 2.0 for i in range(8)},
        texts={f"p{i}": f"base {i}" for i in range(8)},
        overall=2.0,
    )
    final = _report(
        "finetuned",
        scores={f"p{i}": 4.0 for i in range(8)},
        texts={f"p{i}": f"tuned {i}" for i in range(8)},
        overall=4.0,
    )
    comparison = compare_reports(
        base,
        final,
        prompts_by_id={f"p{i}": prompt(f"p{i}") for i in range(8)},
        min_responses_for_claims=6,
    )
    assert comparison.inconclusive is False
    assert comparison.warnings == []
    assert comparison.overall_delta == pytest.approx(2.0)
