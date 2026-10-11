"""Candidate generation: teacher/local sources, caching, failures, cancellation."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import SampleStatus
from autorlhf.core.events import EventBus
from autorlhf.core.types import GenerationResult
from autorlhf.llm.schemas import SyntheticPrompt
from autorlhf.pipeline.candidates import (
    CandidateGenerator,
    build_messages,
    candidate_id,
    candidate_temperature,
)
from autorlhf.storage.cache import ResponseCache


class StubClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return GenerationResult(
            model_id=kwargs["model"],
            text=f"teacher answer {len(self.calls)}",
            source=kwargs["model"],
            prompt_tokens=11,
            completion_tokens=7,
            latency_s=0.01,
            truncated=False,
        )


@pytest.fixture()
def config(minimal_character, rubric) -> RunConfig:
    config = RunConfig(slug="candidates", character=minimal_character, rubric=rubric)
    config.generation.num_prompts = 2
    config.generation.candidates_per_prompt = 2
    config.generation.teacher_model = "openai/gpt-4o-mini"
    return config


def prompts() -> list[SyntheticPrompt]:
    return [
        SyntheticPrompt(prompt_id="p0001", category="ordinary", text="Will the fog lift?"),
        SyntheticPrompt(
            prompt_id="p0002",
            category="multi_turn",
            text="And tomorrow?",
            context=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        ),
    ]


@pytest.fixture()
def fake_training(monkeypatch):
    """Fake the two training entry points the local path uses."""

    from autorlhf.training import models as training_models

    state = {"loaded": 0, "generated": 0, "raise": False, "meta": None}

    def _load_model(cfg, hp, *, hf_token, for_training=False, device=None):
        state["loaded"] += 1
        return SimpleNamespace(model=object(), tokenizer=object(), warnings=[])

    def _generate_text(model, tokenizer, messages, **kwargs):
        state["generated"] += 1
        if state["raise"]:
            raise RuntimeError("no local model")
        return "local answer", state["meta"] or {"completion_tokens": 5, "truncated": False}

    monkeypatch.setattr(training_models, "load_model", _load_model)
    monkeypatch.setattr(training_models, "generate_text", _generate_text)
    return state


def make_generator(config, client=None, cache=None, bus=None, **kwargs) -> CandidateGenerator:
    return CandidateGenerator(
        client=client,
        config=config,
        run_id="run-1",
        cache=cache,
        bus=bus,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def test_candidate_id_and_temperature_are_deterministic():
    assert candidate_id("p0001", 1) == "p0001-c1"
    assert candidate_temperature(0.8, 0, 2) == pytest.approx(0.725)
    assert candidate_temperature(0.8, 1, 2) == pytest.approx(0.875)
    assert candidate_temperature(2.0, 1, 2) == 2.0  # clamped


def test_build_messages_puts_system_first_then_context_then_user():
    messages = build_messages(prompts()[1], "You are Aria.")
    assert messages[0].role == "system"
    assert messages[0].content == "You are Aria."
    assert [m.role for m in messages[1:]] == ["user", "assistant", "user"]
    assert messages[-1].content == "And tomorrow?"


# --------------------------------------------------------------------------- #
# teacher
# --------------------------------------------------------------------------- #


def test_teacher_source_generates_candidates_with_varied_temperature(config):
    config.generation.candidate_source = "teacher"
    client = StubClient()
    generator = make_generator(config, client=client)
    candidates = generator.generate(prompts())

    assert [c.candidate_id for c in candidates] == [
        "p0001-c0",
        "p0001-c1",
        "p0002-c0",
        "p0002-c1",
    ]
    assert all(c.source == "teacher" for c in candidates)
    assert all(c.status is SampleStatus.PENDING for c in candidates)
    assert all(c.text.startswith("teacher answer") for c in candidates)
    assert all(c.tokens == 7 for c in candidates)
    assert candidates[0].temperature != candidates[1].temperature
    assert candidates[0].seed != candidates[1].seed
    assert len(client.calls) == 4


def test_progress_events_are_emitted(config):
    config.generation.candidate_source = "teacher"
    bus = EventBus()
    make_generator(config, client=StubClient(), bus=bus).generate(prompts())
    assert any(event.kind == "progress" for event in bus.history())


def test_cancellation_stops_generation(config):
    config.generation.candidate_source = "teacher"
    cancel = threading.Event()
    cancel.set()
    generator = make_generator(config, client=StubClient(), cancel_event=cancel)
    assert generator.generate(prompts()) == []


# --------------------------------------------------------------------------- #
# local
# --------------------------------------------------------------------------- #


def test_local_source_loads_the_model_once_and_keeps_it_alive(config, fake_training):
    config.generation.candidate_source = "local"
    generator = make_generator(config)
    candidates = generator.generate(prompts())
    assert all(c.source == "local" for c in candidates)
    assert all(c.text == "local answer" for c in candidates)
    assert fake_training["loaded"] == 1
    assert fake_training["generated"] == 4
    generator.close()
    assert generator._local is None


def test_local_generation_is_cached_across_runs(config, fake_training, store):
    config.generation.candidate_source = "local"
    cache = ResponseCache(store)
    first = make_generator(config, cache=cache).generate(prompts())
    generated_after_first = fake_training["generated"]
    second = make_generator(config, cache=cache).generate(prompts())
    assert fake_training["generated"] == generated_after_first
    assert all(c.cached for c in second)
    assert [c.text for c in second] == [c.text for c in first]


def test_local_failure_is_recorded_not_raised(config, fake_training):
    config.generation.candidate_source = "local"
    fake_training["raise"] = True
    bus = EventBus()
    candidates = make_generator(config, bus=bus).generate(prompts())
    assert len(candidates) == 4
    assert all(c.status is SampleStatus.FAILED for c in candidates)
    assert all(c.text == "" for c in candidates)
    assert all(c.error for c in candidates)
    assert any(event.kind == "warning" for event in bus.history())


def test_local_metadata_is_recorded(config, fake_training):
    config.generation.candidate_source = "local"
    fake_training["meta"] = {"completion_tokens": 9, "truncated": True}
    candidates = make_generator(config).generate(prompts())
    assert candidates[0].tokens == 9
    assert candidates[0].truncated is True
    assert candidates[0].latency_s is not None


# --------------------------------------------------------------------------- #
# mixed / resume
# --------------------------------------------------------------------------- #


def test_mixed_source_alternates_teacher_and_local(config, fake_training):
    config.generation.candidate_source = "mixed"
    candidates = make_generator(config, client=StubClient()).generate(prompts())
    by_id = {c.candidate_id: c for c in candidates}
    assert by_id["p0001-c0"].source == "teacher"
    assert by_id["p0001-c1"].source == "local"
    assert by_id["p0002-c0"].source == "teacher"
    assert by_id["p0002-c1"].source == "local"


def test_successful_existing_candidates_are_reused_on_resume(config):
    config.generation.candidate_source = "teacher"
    client = StubClient()
    generator = make_generator(config, client=client)
    first = generator.generate(prompts())
    assert len(client.calls) == 4

    second = generator.generate(prompts(), existing=first)
    assert len(client.calls) == 4  # nothing regenerated
    assert [c.candidate_id for c in second] == [c.candidate_id for c in first]

    failed = [c.model_copy(update={"status": SampleStatus.FAILED, "text": ""}) for c in first[:1]]
    third = generator.generate(prompts(), existing=failed)
    assert len(client.calls) == 8  # the failed one was retried
    assert len(third) == 4


def test_failed_candidates_never_become_training_data(config):
    from autorlhf.config.character import DEFAULT_CHARACTER
    from autorlhf.llm.schemas import CandidateResponse
    from autorlhf.pipeline.dataset import build_sft_records

    failed = CandidateResponse(
        candidate_id="p0001-c0",
        prompt_id="p0001",
        text="",
        status=SampleStatus.FAILED,
        error="boom",
    )
    records = build_sft_records(
        run_id="run-1",
        prompts=prompts(),
        candidates=[failed],
        character=DEFAULT_CHARACTER,
    )
    assert records  # the seed conversations are still supervised examples
    assert all(record["source"] == "seed" for record in records)
    assert all(record["prompt_id"] != "p0001" for record in records)
