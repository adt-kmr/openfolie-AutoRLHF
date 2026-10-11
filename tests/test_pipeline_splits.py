"""Prompt-level split tests: the anti-leakage invariant and split reuse."""

from __future__ import annotations

import random
from pathlib import Path

from autorlhf.config.run_config import GenerationConfig
from autorlhf.core.enums import SplitName
from autorlhf.llm.schemas import SyntheticPrompt
from autorlhf.pipeline.splits import (
    SplitAssignment,
    apply_splits,
    load_or_plan,
    plan_splits,
    validate_split,
)
from autorlhf.storage.runs import RunPaths

CATEGORIES = (
    "ordinary",
    "varied_phrasing",
    "context_following",
    "multi_turn",
    "edge_case",
    "boundary",
)


def make_prompts(count: int, *, seed: int = 0) -> list[SyntheticPrompt]:
    rng = random.Random(seed)
    return [
        SyntheticPrompt(
            prompt_id=f"p{i + 1:04d}",
            category=rng.choice(CATEGORIES),
            text=f"prompt {i}",
        )
        for i in range(count)
    ]


def test_no_prompt_id_appears_in_two_splits_for_many_seeds():
    for seed in range(25):
        prompts = make_prompts(40, seed=seed)
        assignment = plan_splits(
            prompts,
            eval_fraction=0.2,
            validation_fraction=0.15,
            min_eval_prompts=3,
            min_train_prompts=4,
            seed=seed,
            dataset_hash=f"hash-{seed}",
        )
        lists = assignment.lists()
        assert validate_split(assignment, prompts) == []
        assert len(lists["train"]) + len(lists["validation"]) + len(lists["eval"]) == len(prompts)
        assert set(lists["train"]).isdisjoint(lists["validation"])
        assert set(lists["train"]).isdisjoint(lists["eval"])
        assert set(lists["validation"]).isdisjoint(lists["eval"])
        assert all(pid.startswith("p") for pid in assignment.assignments)


def test_split_is_deterministic_for_the_same_seed_and_fingerprint():
    prompts = make_prompts(30, seed=3)
    first = plan_splits(
        prompts,
        eval_fraction=0.2,
        validation_fraction=0.1,
        min_eval_prompts=3,
        min_train_prompts=4,
        seed=7,
        dataset_hash="h",
    )
    second = plan_splits(
        prompts,
        eval_fraction=0.2,
        validation_fraction=0.1,
        min_eval_prompts=3,
        min_train_prompts=4,
        seed=7,
        dataset_hash="h",
    )
    assert first.assignments == second.assignments
    assert first.split_hash == second.split_hash


def test_minimums_are_respected():
    prompts = make_prompts(12)
    assignment = plan_splits(
        prompts,
        eval_fraction=0.5,
        validation_fraction=0.4,
        min_eval_prompts=3,
        min_train_prompts=4,
        seed=1,
        dataset_hash="h",
    )
    assert assignment.counts["eval"] >= 3
    assert assignment.counts["train"] >= 4
    assert sum(assignment.counts.values()) == 12


def test_every_category_is_stratified_when_it_has_room():
    prompts = make_prompts(60, seed=11)
    assignment = plan_splits(
        prompts,
        eval_fraction=0.2,
        validation_fraction=0.15,
        min_eval_prompts=3,
        min_train_prompts=4,
        seed=11,
        dataset_hash="h",
    )
    by_category: dict[str, set[str]] = {}
    for prompt in prompts:
        by_category.setdefault(prompt.category, set()).add(assignment.split_of(prompt.prompt_id))
    # A category with at least 8 prompts should land in all three splits.
    for category, splits in by_category.items():
        size = sum(1 for p in prompts if p.category == category)
        if size >= 8:
            assert len(splits) >= 2, category


def test_splits_json_is_reused_when_the_fingerprint_matches(tmp_path: Path):
    paths = RunPaths(tmp_path / "run").ensure()
    prompts = make_prompts(24, seed=5)
    generation = GenerationConfig(num_prompts=24)
    first = load_or_plan(
        paths, prompts, generation=generation, dataset_hash="fingerprint-a", run_id="r1"
    )
    second = load_or_plan(
        paths, prompts, generation=generation, dataset_hash="fingerprint-a", run_id="r1"
    )
    assert first.split_hash == second.split_hash
    assert first.created_at == second.created_at  # loaded, not re-planned


def test_splits_json_is_regenerated_when_the_fingerprint_changes(tmp_path: Path):
    paths = RunPaths(tmp_path / "run").ensure()
    prompts = make_prompts(24, seed=5)
    generation = GenerationConfig(num_prompts=24)
    load_or_plan(paths, prompts, generation=generation, dataset_hash="fingerprint-a", run_id="r1")
    changed = load_or_plan(
        paths, prompts, generation=generation, dataset_hash="fingerprint-b", run_id="r1"
    )
    assert changed.dataset_hash == "fingerprint-b"
    reloaded = SplitAssignment.load(paths.split_manifest_path)
    assert reloaded is not None
    assert reloaded.dataset_hash == "fingerprint-b"


def test_validate_split_reports_missing_and_miscounted_prompts():
    prompts = make_prompts(6)
    assignment = plan_splits(
        prompts,
        eval_fraction=0.2,
        validation_fraction=0.0,
        min_eval_prompts=1,
        min_train_prompts=1,
        seed=0,
        dataset_hash="h",
    )
    broken = assignment.model_copy(deep=True)
    broken.assignments.pop(prompts[0].prompt_id)
    broken.counts["train"] = 999
    problems = validate_split(broken, prompts)
    assert any("no split assignment" in p for p in problems)
    assert any("counts" in p for p in problems)


def test_validate_split_detects_a_corrupt_split_hash():
    prompts = make_prompts(8)
    assignment = plan_splits(
        prompts,
        eval_fraction=0.25,
        validation_fraction=0.0,
        min_eval_prompts=1,
        min_train_prompts=1,
        seed=0,
        dataset_hash="h",
    )
    assert validate_split(assignment, prompts) == []
    assignment.split_hash = "deadbeef"
    problems = validate_split(assignment, prompts)
    assert any("split_hash" in p for p in problems)


def test_apply_splits_stamps_each_prompt():
    prompts = make_prompts(10)
    assignment = plan_splits(
        prompts,
        eval_fraction=0.3,
        validation_fraction=0.1,
        min_eval_prompts=1,
        min_train_prompts=1,
        seed=2,
        dataset_hash="h",
    )
    apply_splits(prompts, assignment)
    for prompt in prompts:
        assert prompt.split == assignment.split_of(prompt.prompt_id)
        assert prompt.split in {s.value for s in SplitName}
