"""Synthetic prompt generation: categories, validation, determinism, caching."""

from __future__ import annotations

import re
from types import SimpleNamespace

from autorlhf.config.character import DEFAULT_CHARACTER
from autorlhf.config.run_config import GenerationConfig
from autorlhf.core.events import EventBus
from autorlhf.pipeline.prompts import (
    PROMPT_CATEGORIES,
    generate_prompts,
    plan_categories,
    seed_eval_prompts,
)
from autorlhf.storage.cache import ResponseCache

CATEGORY_RE = re.compile(r"Category: (\w+)")
COUNT_RE = re.compile(r"Write (\d+) distinct")


class FakePromptClient:
    """Answers a prompt batch with exactly the number of prompts requested."""

    def __init__(self, *, duplicate: bool = False, malformed: bool = False) -> None:
        self.calls: list[dict] = []
        self.duplicate = duplicate
        self.malformed = malformed

    def chat_json(self, *, model, messages, schema_hint, **kwargs):
        self.calls.append(
            {"model": model, "messages": messages, "schema_hint": schema_hint, **kwargs}
        )
        content = messages[0].content
        category = CATEGORY_RE.search(content).group(1)
        count = int(COUNT_RE.search(content).group(1))
        items = [
            {
                "text": f"{category} prompt {index}",
                "category": category,
                "context": [],
                "avoids": "",
            }
            for index in range(count)
        ]
        if self.duplicate:
            # The same text for every requested prompt, so de-duplication can
            # never satisfy the requested count.
            items = [
                {"text": "duplicate prompt", "category": category, "context": [], "avoids": ""}
                for _ in range(count)
            ]
        if self.malformed:
            items = [{"category": category}, "not-an-object", {"text": "   "}]
        return {"prompts": items}, SimpleNamespace(text="")


def test_plan_categories_sums_to_the_target():
    for target in (1, 7, 10, 24, 100):
        for edge in (0.0, 0.2, 0.35, 0.5, 0.9):
            plan = plan_categories(target, edge)
            assert sum(plan.values()) == target, (target, edge, plan)
            assert set(plan) == set(PROMPT_CATEGORIES)


def test_plan_categories_respects_the_edge_case_fraction():
    plan = plan_categories(40, 0.5)
    edge = plan["edge_case"] + plan["boundary"]
    assert edge == 20


def test_generate_prompts_assigns_deterministic_ids_and_covers_categories(store):
    client = FakePromptClient()
    generation = GenerationConfig(num_prompts=10, edge_case_fraction=0.4)
    prompts = generate_prompts(
        client=client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=ResponseCache(store),
    )
    assert len(prompts) == 10
    assert [p.prompt_id for p in prompts] == [f"p{i:04d}" for i in range(1, 11)]
    assert {p.category for p in prompts} == set(PROMPT_CATEGORIES)
    assert all(p.generator_model == generation.teacher_model for p in prompts)
    assert all(p.text.strip() for p in prompts)


def test_generate_prompts_reuses_the_cache_on_resume(store):
    cache = ResponseCache(store)
    generation = GenerationConfig(num_prompts=8, edge_case_fraction=0.4)
    first_client = FakePromptClient()
    first = generate_prompts(
        client=first_client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=cache,
    )
    second_client = FakePromptClient()
    second = generate_prompts(
        client=second_client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=cache,
    )
    assert second_client.calls == []
    assert [p.prompt_id for p in first] == [p.prompt_id for p in second]
    assert [p.text for p in first] == [p.text for p in second]


def test_generate_prompts_rejects_duplicates_and_malformed_entries(store, bus):
    client = FakePromptClient(duplicate=True)
    generation = GenerationConfig(num_prompts=6, edge_case_fraction=0.3)
    prompts = generate_prompts(
        client=client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=ResponseCache(store),
        bus=bus,
    )
    # Duplicates collapse to one prompt per category, so the target cannot be met.
    assert len(prompts) < 6
    assert len({p.text for p in prompts}) == len(prompts)
    assert any(event.kind == "warning" for event in bus.history())


def test_generate_prompts_drops_malformed_items(store):
    client = FakePromptClient(malformed=True)
    generation = GenerationConfig(num_prompts=4, edge_case_fraction=0.5)
    prompts = generate_prompts(
        client=client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=ResponseCache(store),
    )
    assert prompts == []


def test_generate_prompts_honours_cancellation(store):
    import threading

    client = FakePromptClient()
    generation = GenerationConfig(num_prompts=24, edge_case_fraction=0.5)
    cancel = threading.Event()
    cancel.set()
    prompts = generate_prompts(
        client=client,
        character=DEFAULT_CHARACTER,
        generation=generation,
        run_id="run-1",
        cache=ResponseCache(store),
        cancel_event=cancel,
    )
    assert prompts == []
    assert client.calls == []


def test_generate_prompts_emits_progress(store):
    bus = EventBus()
    client = FakePromptClient()
    generate_prompts(
        client=client,
        character=DEFAULT_CHARACTER,
        generation=GenerationConfig(num_prompts=6, edge_case_fraction=0.3),
        run_id="run-1",
        cache=ResponseCache(store),
        bus=bus,
    )
    assert any(event.kind == "progress" for event in bus.history())


def test_seed_eval_prompts_derives_held_out_prompts():
    prompts = seed_eval_prompts(DEFAULT_CHARACTER)
    assert prompts
    assert all(p.origin == "seed" for p in prompts)
    assert all(p.prompt_id.startswith("e") for p in prompts)
    assert prompts[0].text == "Will the fog lift today?"


def test_seed_eval_prompts_limit():
    assert len(seed_eval_prompts(DEFAULT_CHARACTER, limit=1)) == 1
