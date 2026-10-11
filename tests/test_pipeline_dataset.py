"""Dataset record building, validation, counters, and reloading."""

from __future__ import annotations

from pathlib import Path

from autorlhf.config.character import DEFAULT_CHARACTER
from autorlhf.core.enums import SampleStatus, SplitName
from autorlhf.llm.schemas import (
    CandidateResponse,
    CriterionScore,
    Judgement,
    PairDecision,
    PreferencePair,
    SyntheticPrompt,
)
from autorlhf.pipeline.dataset import (
    build_eval_prompts,
    build_sft_records,
    candidate_record,
    compute_counters,
    load_dataset,
    preference_record,
    rewrite_records,
    sft_record,
    update_record,
    validate_dataset,
)
from autorlhf.pipeline.splits import plan_splits
from autorlhf.storage.runs import RunPaths

RUN = "20260101-120000-ab12"


def prompt(prompt_id: str = "p0001", *, split: str = "train") -> SyntheticPrompt:
    return SyntheticPrompt(
        prompt_id=prompt_id, category="ordinary", text=f"question for {prompt_id}", split=split
    )


def candidate(
    candidate_id: str = "p0001-c0",
    *,
    prompt_id: str = "p0001",
    text: str = "an answer",
    status: SampleStatus = SampleStatus.ACCEPTED,
    truncated: bool = False,
) -> CandidateResponse:
    return CandidateResponse(
        candidate_id=candidate_id,
        prompt_id=prompt_id,
        text=text,
        status=status,
        truncated=truncated,
        source="teacher",
        model_id="openai/gpt-4o-mini",
    )


def judgement(candidate_id: str = "p0001-c0", *, score: float | None = 3.0) -> Judgement:
    return Judgement(
        judgement_id=f"j-{candidate_id}",
        prompt_id="p0001",
        candidate_id=candidate_id,
        criterion_scores={
            "character_consistency": CriterionScore(
                criterion_id="character_consistency", criterion_name="Character", score=score
            )
        },
        overall=score,
        judge_model="typesafe/jev-1.13",
        judge_build="typesafe/jev-1.13-20260917",
    )


def test_candidate_record_matches_the_frozen_shape():
    record = candidate_record(
        run_id=RUN, prompt=prompt(), candidate=candidate(), judgement=judgement()
    )
    for key in (
        "record_type",
        "dataset_version",
        "run_id",
        "prompt_id",
        "candidate_id",
        "split",
        "category",
        "prompt",
        "prompt_context",
        "response",
        "source",
        "model_id",
        "status",
        "error",
        "truncated",
        "tokens",
        "latency_s",
        "cached",
        "seed",
        "temperature",
        "edits",
        "edited",
        "overall",
        "criterion_scores",
        "judgement",
        "reviewer",
        "created_at",
    ):
        assert key in record
    assert record["record_type"] == "candidate"
    assert record["split"] == "train"
    assert record["overall"] == 3.0
    assert record["criterion_scores"] == {"character_consistency": 3.0}
    assert record["judgement"]["judge_build"] == "typesafe/jev-1.13-20260917"


def test_failed_judgement_keeps_overall_null_and_scores_empty():
    failed = judgement(score=None)
    failed.error = "judge unreachable"
    record = candidate_record(run_id=RUN, prompt=prompt(), candidate=candidate(), judgement=failed)
    assert record["overall"] is None
    assert record["criterion_scores"] == {}
    assert record["judgement"]["error"] == "judge unreachable"


def test_validate_accepts_a_clean_dataset():
    records = [
        candidate_record(run_id=RUN, prompt=prompt(), candidate=candidate(), judgement=judgement())
    ]
    assert validate_dataset(records) == []


def test_validate_detects_empty_response():
    records = [
        candidate_record(
            run_id=RUN,
            prompt=prompt(),
            candidate=candidate(text=""),
            judgement=None,
        )
    ]
    assert any("empty response" in problem for problem in validate_dataset(records))


def test_validate_detects_duplicate_records():
    record = candidate_record(
        run_id=RUN, prompt=prompt(), candidate=candidate(), judgement=judgement()
    )
    problems = validate_dataset([record, dict(record)])
    assert any("duplicate record" in problem for problem in problems)


def test_validate_detects_broken_role_alternation():
    record = sft_record(
        run_id=RUN,
        example_id="s0001",
        prompt_id="p0001",
        split="train",
        messages=[
            {"role": "system", "content": "You are Aria."},
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "hello"},
        ],
        source="accepted",
    )
    problems = validate_dataset([record])
    assert any("both 'user'" in problem for problem in problems)


def test_validate_detects_sft_missing_system_turn():
    record = sft_record(
        run_id=RUN,
        example_id="s0001",
        prompt_id="p0001",
        split="train",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        source="accepted",
    )
    assert any("system turn" in problem for problem in validate_dataset([record]))


def test_validate_detects_invalid_preferences():
    chosen = candidate("p0001-c0", text="A")
    rejected = candidate("p0001-c1", text="B")
    candidates = [
        candidate_record(
            run_id=RUN, prompt=prompt(), candidate=chosen, judgement=judgement("p0001-c0")
        ),
        candidate_record(
            run_id=RUN, prompt=prompt(), candidate=rejected, judgement=judgement("p0001-c1")
        ),
    ]
    pair = PreferencePair(
        pair_id="p0001-c0-vs-c1",
        prompt_id="p0001",
        chosen_id="p0001-c0",
        rejected_id="p0001-c0",
    )
    record = preference_record(
        run_id=RUN,
        prompt=prompt(),
        pair=pair,
        chosen=chosen,
        rejected=rejected,
        decision=None,
        criterion_deltas={},
        audit={"thresholds": {}, "reasons": [], "rejected_pairs": []},
    )
    problems = validate_dataset([*candidates, record])
    assert any("same response" in problem for problem in problems)


def test_validate_detects_failed_side_of_a_preference():
    chosen = candidate("p0001-c0", text="A")
    failed = candidate("p0001-c1", text="", status=SampleStatus.FAILED)
    failed.error = "boom"
    candidates = [
        candidate_record(
            run_id=RUN, prompt=prompt(), candidate=chosen, judgement=judgement("p0001-c0")
        ),
        candidate_record(run_id=RUN, prompt=prompt(), candidate=failed, judgement=None),
    ]
    pair = PreferencePair(
        pair_id="p0001-c0-vs-c1",
        prompt_id="p0001",
        chosen_id="p0001-c0",
        rejected_id="p0001-c1",
    )
    record = preference_record(
        run_id=RUN,
        prompt=prompt(),
        pair=pair,
        chosen=chosen,
        rejected=failed,
        decision=None,
        criterion_deltas={},
        audit={"thresholds": {}, "reasons": [], "rejected_pairs": []},
    )
    problems = validate_dataset([*candidates, record])
    assert any("failed to generate" in problem for problem in problems)


def test_validate_detects_inconsistent_prompt_splits():
    first = candidate_record(
        run_id=RUN, prompt=prompt(split="train"), candidate=candidate(), judgement=judgement()
    )
    second = candidate_record(
        run_id=RUN,
        prompt=prompt(split="eval"),
        candidate=candidate("p0001-c1", text="other"),
        judgement=judgement("p0001-c1"),
    )
    problems = validate_dataset([first, second])
    assert any("appears with split" in problem for problem in problems)


def test_validate_detects_truncated_but_accepted():
    record = candidate_record(
        run_id=RUN,
        prompt=prompt(),
        candidate=candidate(truncated=True),
        judgement=judgement(),
    )
    assert any("truncated" in problem for problem in validate_dataset(records=[record]))


def test_compute_counters_covers_the_documented_names():
    prompts = [prompt("p0001"), prompt("p0002", split="eval")]
    candidates = [
        candidate("p0001-c0"),
        candidate("p0001-c1", text="", status=SampleStatus.FAILED),
        candidate("p0002-c0", prompt_id="p0002"),
    ]
    judgements = {"p0001-c0": judgement("p0001-c0"), "p0002-c0": judgement("p0002-c0")}
    failed = judgement("p0001-c1", score=None)
    failed.error = "nope"
    judgements["p0001-c1"] = failed
    counters = compute_counters(
        prompts=prompts,
        candidates=candidates,
        judgements=judgements,
        pairs=[
            PreferencePair(
                pair_id="x", prompt_id="p0001", chosen_id="p0001-c0", rejected_id="p0001-c1"
            )
        ],
        sft_records=[{"record_type": "sft"}],
        rejections={"tie": 2, "low_confidence": 1, "invalid": 1, "insufficient_candidates": 1},
    )
    assert counters["prompts_generated"] == 2
    assert counters["candidates_generated"] == 2
    assert counters["generation_failed"] == 1
    assert counters["judgements_ok"] == 2
    assert counters["judgements_failed"] == 1
    assert counters["pairs_accepted"] == 1
    assert counters["pairs_rejected_tie"] == 2
    assert counters["pairs_rejected_low_confidence"] == 1
    assert counters["pairs_rejected_invalid"] == 2  # invalid + insufficient_candidates bucket
    assert counters["accepted"] == 2
    assert counters["sft_examples"] == 1
    assert counters["split_train"] == 1
    assert counters["split_eval"] == 1


def test_build_sft_records_puts_the_system_prompt_first():
    prompts = [prompt("p0001")]
    candidates = [candidate("p0001-c0", text="an answer")]
    records = build_sft_records(
        run_id=RUN, prompts=prompts, candidates=candidates, character=DEFAULT_CHARACTER
    )
    assert records
    for record in records:
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][0]["content"] == DEFAULT_CHARACTER.system_prompt
        assert record["messages"][-1]["role"] == "assistant"
    sources = {record["source"] for record in records}
    assert "seed" in sources
    assert "accepted" in sources


def test_build_eval_prompts_uses_the_eval_split_and_seeds():
    prompts = [
        prompt("p0001", split="train"),
        prompt("p0002", split="eval"),
        prompt("p0003", split="eval"),
    ]
    assignment = plan_splits(
        prompts,
        eval_fraction=0.4,
        validation_fraction=0.0,
        min_eval_prompts=1,
        min_train_prompts=1,
        seed=0,
        dataset_hash="h",
    )
    # Force a deterministic assignment for the assertion.
    assignment.assignments = {"p0001": "train", "p0002": "eval", "p0003": "eval"}
    eval_prompts = build_eval_prompts(
        prompts, character=DEFAULT_CHARACTER, include_seed_prompts=True, assignment=assignment
    )
    ids = {p.prompt_id for p in eval_prompts}
    assert {"p0002", "p0003"} <= ids
    assert "p0001" not in ids
    assert any(p.origin == "seed" for p in eval_prompts)
    assert all(p.category for p in eval_prompts)


def test_update_record_rewrites_atomically(tmp_path: Path):
    path = tmp_path / "dataset.jsonl"
    records = [
        candidate_record(run_id=RUN, prompt=prompt(), candidate=candidate(), judgement=judgement())
    ]
    rewrite_records(path, records)
    changed = update_record(
        path, match={"candidate_id": "p0001-c0"}, changes={"response": "edited text"}
    )
    assert changed is True
    from autorlhf.storage.runs import read_jsonl

    reloaded = read_jsonl(path)
    assert reloaded[0]["response"] == "edited text"
    assert reloaded[0]["candidate_id"] == "p0001-c0"


def test_load_dataset_roundtrip(tmp_path: Path):
    paths = RunPaths(tmp_path / "run").ensure()
    prompts = [prompt("p0001", split="train"), prompt("p0002", split="eval")]
    candidates = [
        candidate("p0001-c0", text="A"),
        candidate("p0001-c1", text="B"),
        candidate("p0002-c0", prompt_id="p0002", text="C"),
    ]
    judgements = {
        "p0001-c0": judgement("p0001-c0"),
        "p0001-c1": judgement("p0001-c1"),
        "p0002-c0": judgement("p0002-c0"),
    }
    rewrite_records(
        paths.dataset_path,
        [
            candidate_record(
                run_id=RUN,
                prompt=prompts[0],
                candidate=candidates[0],
                judgement=judgements["p0001-c0"],
            ),
            candidate_record(
                run_id=RUN,
                prompt=prompts[0],
                candidate=candidates[1],
                judgement=judgements["p0001-c1"],
            ),
            candidate_record(
                run_id=RUN,
                prompt=prompts[1],
                candidate=candidates[2],
                judgement=judgements["p0002-c0"],
            ),
        ],
    )
    decision = PairDecision(
        pair_id="p0001-c0-vs-c1",
        prompt_id="p0001",
        option_a_id="p0001-c0",
        option_b_id="p0001-c1",
        winner="b",
        choice_raw="b",
        probabilities={"a": 0.2, "b": 0.8},
        confidence=0.7,
        margin=-1.0,
    )
    pair = PreferencePair(
        pair_id="p0001-c0-vs-c1",
        prompt_id="p0001",
        chosen_id="p0001-c1",
        rejected_id="p0001-c0",
        margin=-1.0,
        judge_confidence=0.7,
        decision="b",
    )
    rewrite_records(
        paths.preferences_path,
        [
            preference_record(
                run_id=RUN,
                prompt=prompts[0],
                pair=pair,
                chosen=candidates[1],
                rejected=candidates[0],
                decision=decision,
                criterion_deltas={"character_consistency": -1.0},
                audit={"thresholds": {}, "reasons": [], "rejected_pairs": []},
            )
        ],
    )
    rewrite_records(
        paths.sft_path,
        build_sft_records(
            run_id=RUN, prompts=prompts, candidates=candidates, character=DEFAULT_CHARACTER
        ),
    )
    assignment = plan_splits(
        prompts,
        eval_fraction=0.5,
        validation_fraction=0.0,
        min_eval_prompts=1,
        min_train_prompts=1,
        seed=0,
        dataset_hash="h",
    )
    assignment.assignments = {"p0001": "train", "p0002": "eval"}
    assignment.save(paths.split_manifest_path)

    bundle = load_dataset(paths)
    assert [p.prompt_id for p in bundle.prompts] == ["p0001", "p0002"]
    assert len(bundle.candidates) == 3
    assert set(bundle.judgements) == {"p0001-c0", "p0001-c1", "p0002-c0"}
    assert len(bundle.preference_pairs) == 1
    assert bundle.preference_pairs[0].chosen_id == "p0001-c1"
    assert len(bundle.pair_decisions) == 1
    assert bundle.pair_decisions[0].winner == "b"
    assert bundle.train_prompt_ids == ["p0001"]
    assert bundle.eval_prompt_ids == ["p0002"]
    assert bundle.counters["pairs_accepted"] == 1
    assert bundle.dataset_hash == "h"
    summary = bundle.to_summary()
    assert summary["candidates"] == 3
    assert summary["preference_pairs"] == 1
    assert summary["judgements_ok"] == 3


def test_split_name_enum_values_are_the_frozen_ones():
    assert {s.value for s in SplitName} == {"train", "validation", "eval"}
