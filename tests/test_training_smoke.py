"""End-to-end training smoke tests on CPU, fully offline.

The acceptance criterion for the whole project lives here: a real one-step LoRA
SFT run against a tiny local model must complete and write a loadable adapter.
Everything runs on CPU with models built in a temp directory by the shared
fixtures, so no network access and no GPU are needed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from autorlhf.config.paths import AppPaths
from autorlhf.config.run_config import Hyperparameters, RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.errors import InsufficientDataError
from autorlhf.core.events import EventBus
from autorlhf.core.types import HardwareInfo
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import (
    RunPaths,
    is_adapter_complete,
    is_checkpoint_complete,
    list_checkpoint_dirs,
)
from autorlhf.training.jobs import TrainingJob, TrainingResult
from autorlhf.training.plan import TrainingPlan
from autorlhf.training.trainers import (
    TokenMetricsCallback,
    checkpoint_root,
    find_resume_checkpoint,
    run_sft,
)

RUN_ID = "20260101-120000-smoke"

SYSTEM_PROMPT = "You are a terse test assistant."


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _load_tiny(model_dir) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype="float32", attn_implementation="eager"
    )
    return model, tokenizer


def _hp(**overrides: Any) -> Hyperparameters:
    hp = Hyperparameters(
        method=TrainingMethod.LORA,
        objective=TrainingObjective.SFT,
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_target_modules=["c_attn", "c_proj"],
        learning_rate=1e-2,
        sft_learning_rate=1e-2,
        num_train_epochs=1.0,
        sft_epochs=1.0,
        max_steps=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_seq_length=128,  # the tiny model's context; the char-level tokenizer needs headroom
        logging_steps=1,
        save_steps=2,
        eval_steps=2,
        save_total_limit=2,
        gradient_checkpointing=False,
        seed=0,
        sft_completion_only_loss=True,
    )
    for key, value in overrides.items():
        setattr(hp, key, value)
    return hp


def _sft_records(count: int = 4, *, split: str = "train") -> list[dict]:
    return [
        {
            "record_type": "sft",
            "dataset_version": "1",
            "run_id": RUN_ID,
            "example_id": f"s{index:04d}",
            "prompt_id": f"p{index:04d}",
            "split": split,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question number {index}."},
                {"role": "assistant", "content": f"Answer {index}."},
            ],
            "source": "seed",
            "status": "accepted",
            "created_at": "2026-01-01T12:00:00",
        }
        for index in range(count)
    ]


def _pair_records(count: int = 3, *, split: str = "train") -> list[dict]:
    return [
        {
            "record_type": "preference",
            "dataset_version": "1",
            "run_id": RUN_ID,
            "pair_id": f"p{index:04d}-a-vs-b",
            "prompt_id": f"p{index:04d}",
            "split": split,
            "prompt": f"Question number {index}?",
            "prompt_context": [],
            "chosen_id": f"p{index:04d}-c1",
            "rejected_id": f"p{index:04d}-c0",
            "chosen_text": f"Short answer {index}.",
            "rejected_text": f"A much longer, rambling answer number {index}.",
            "margin": 0.9,
            "judge_confidence": 0.81,
            "source": "jev",
            "decision": "b",
            "created_at": "2026-01-01T12:00:00",
        }
        for index in range(count)
    ]


def _plan(
    *,
    objective: TrainingObjective,
    hp: Hyperparameters,
    sft: list[dict] | None = None,
    pairs: list[dict] | None = None,
) -> TrainingPlan:
    return TrainingPlan(
        run_id=RUN_ID,
        objective=objective,
        method=hp.method,
        hyperparameters=hp,
        preset_id="cpu_tiny_lora",
        rationale=["smoke test"],
        hardware=HardwareInfo(device_type="cpu", device_count=0, cpu_count=4),
        sft_examples=sft or [],
        preference_pairs=pairs or [],
        train_count=len(sft or []),
    )


def _job(
    *,
    tmp_path,
    plan: TrainingPlan,
    config: RunConfig,
    bus: EventBus,
    store: RunStore | None = None,
    cancel: threading.Event | None = None,
    pause: threading.Event | None = None,
) -> TrainingJob:
    paths = RunPaths(tmp_path / "run").ensure()
    return TrainingJob(
        run_id=RUN_ID,
        paths=paths,
        config=config,
        plan=plan,
        bus=bus,
        cancel_event=cancel or threading.Event(),
        pause_event=pause or threading.Event(),
        env=Environment(AppPaths.for_tests(tmp_path)),
        store=store,
    )


# --------------------------------------------------------------------------- #
# the acceptance criterion
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_run_sft_one_step_writes_loadable_adapter(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore
) -> None:
    hp = _hp()
    paths = RunPaths(tmp_path / "run").ensure()
    model, tokenizer = _load_tiny(tiny_chat_model_dir)

    result = run_sft(
        run_id=RUN_ID,
        paths=paths,
        hp=hp,
        model=model,
        tokenizer=tokenizer,
        train_records=_sft_records(4),
        bus=bus,
        store=store,
        report_to=(),
        run_name="smoke-sft",
        fan_in_fan_out=True,
        extra_warnings=[],
    )

    assert isinstance(result, TrainingResult)
    assert result.status == "completed", result.error
    assert result.method is TrainingMethod.LORA
    assert result.objective is TrainingObjective.SFT
    assert result.steps_completed == 2
    assert result.adapter_path == str(paths.adapter_dir)
    assert result.error is None

    # The adapter on disk is complete and loadable.
    assert is_adapter_complete(paths.adapter_dir)
    assert (paths.adapter_dir / "adapter_model.safetensors").is_file()
    assert (paths.adapter_dir / "adapter_config.json").is_file()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(
        str(tiny_chat_model_dir), dtype="float32", attn_implementation="eager"
    )
    reloaded = PeftModel.from_pretrained(base, str(paths.adapter_dir))
    assert reloaded is not None

    # Checkpoints are complete and registered.
    checkpoints = list_checkpoint_dirs(checkpoint_root(paths, TrainingObjective.SFT))
    assert checkpoints, "save_steps=2 must have written a checkpoint"
    assert all(is_checkpoint_complete(path) for path in checkpoints)
    assert result.checkpoints
    assert store.last_complete_checkpoint(RUN_ID) is not None
    assert "adapter-sft" in {artifact["name"] for artifact in store.list_artifacts(RUN_ID)}

    # Metrics reached both the result and SQLite, and the bus saw them.
    assert result.metrics
    assert "loss" in store.metric_names(RUN_ID)
    kinds = {event.kind for event in bus.history()}
    assert {"status", "metric", "progress", "artifact"} <= kinds


@pytest.mark.slow
def test_run_sft_resumes_from_complete_checkpoint(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore
) -> None:
    paths = RunPaths(tmp_path / "run").ensure()
    records = _sft_records(3)

    model, tokenizer = _load_tiny(tiny_chat_model_dir)
    first = run_sft(
        run_id=RUN_ID,
        paths=paths,
        hp=_hp(max_steps=1, save_steps=1),
        model=model,
        tokenizer=tokenizer,
        train_records=records,
        bus=bus,
        store=store,
        report_to=(),
        run_name="smoke-resume",
        fan_in_fan_out=True,
    )
    assert first.status == "completed", first.error
    assert first.steps_completed == 1
    resumable = find_resume_checkpoint(paths, objective=TrainingObjective.SFT)
    assert resumable is not None

    model2, tokenizer2 = _load_tiny(tiny_chat_model_dir)
    second = run_sft(
        run_id=RUN_ID,
        paths=paths,
        hp=_hp(max_steps=2, save_steps=1),
        model=model2,
        tokenizer=tokenizer2,
        train_records=records,
        bus=bus,
        store=store,
        resume_from=resumable,
        report_to=(),
        run_name="smoke-resume",
        fan_in_fan_out=True,
    )
    assert second.status == "completed", second.error
    assert second.resumed_from == str(resumable)
    assert second.steps_completed == 2


# --------------------------------------------------------------------------- #
# TrainingJob: warm-up + DPO, stop, gating, data errors
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_training_job_sft_warmup_then_dpo(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    run_config.model.model_id = str(tiny_chat_model_dir)
    run_config.model.torch_dtype = "float32"
    run_config.model.attn_implementation = "eager"
    hp = _hp(
        objective=TrainingObjective.DPO,
        method=TrainingMethod.LORA,
        lora_target_modules=[],  # discovery must fill these in
        sft_warmup=True,
        max_steps=1,
        save_steps=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
    )
    plan = _plan(
        objective=TrainingObjective.DPO,
        hp=hp,
        sft=_sft_records(3),
        pairs=_pair_records(3) + _pair_records(1, split="validation"),
    )
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    result = job.run()

    assert result.status == "completed", result.error
    assert result.objective is TrainingObjective.DPO
    assert result.adapter_path == str(job.paths.adapter_dir)
    assert is_adapter_complete(job.paths.adapter_dir)
    # Both stages ran and both checkpoint series exist.
    assert list_checkpoint_dirs(checkpoint_root(job.paths, TrainingObjective.SFT))
    assert list_checkpoint_dirs(checkpoint_root(job.paths, TrainingObjective.DPO))
    assert any(key.startswith("sft_") for key in result.metrics)
    assert any(not key.startswith("sft_") for key in result.metrics)
    # Target modules were discovered from the real model, not guessed: they show up
    # in the saved adapter config, with the GPT-2 fan_in_fan_out flag.
    adapter_config = json.loads(
        (job.paths.adapter_dir / "adapter_config.json").read_text(encoding="utf-8")
    )
    assert set(adapter_config["target_modules"]) == {"c_attn", "c_proj"}
    assert adapter_config["fan_in_fan_out"] is True
    assert adapter_config["base_model_name_or_path"] == str(tiny_chat_model_dir)
    assert any(
        "detected architecture family 'gpt2'" in event.message
        for event in bus.history()
        if event.kind == "log"
    )
    phases = [event.message for event in bus.history() if event.kind == "phase"]
    assert any("SFT warm-up" in message for message in phases)
    assert any("DPO" in message for message in phases)
    assert any(event.kind == "sample" for event in bus.history())
    assert job.status["result_status"] == "completed"


@pytest.mark.slow
def test_training_job_stop_returns_interrupted_result(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    run_config.model.model_id = str(tiny_chat_model_dir)
    run_config.model.torch_dtype = "float32"
    cancel = threading.Event()
    cancel.set()  # a stop that arrives before the first step must still be graceful
    hp = _hp(max_steps=5, save_steps=1)
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(3))
    job = _job(
        tmp_path=tmp_path,
        plan=plan,
        config=run_config,
        bus=bus,
        store=store,
        cancel=cancel,
    )
    result = job.run()

    assert result.status == "interrupted"
    assert result.error is None
    assert 0 < result.steps_completed < 5
    assert result.adapter_path is not None
    assert is_adapter_complete(Path(result.adapter_path))
    assert any(event.kind == "warning" for event in bus.history())


def test_training_job_raises_insufficient_data(
    tmp_path, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    hp = _hp()
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=[], pairs=[])
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    with pytest.raises(InsufficientDataError) as info:
        job.run()
    assert "no training data at all" in str(info.value)
    assert info.value.hint


def test_training_job_raises_when_objective_has_no_data(
    tmp_path, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    hp = _hp(objective=TrainingObjective.DPO)
    plan = _plan(objective=TrainingObjective.DPO, hp=hp, sft=_sft_records(2), pairs=[])
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    with pytest.raises(InsufficientDataError) as info:
        job.run()
    assert "preference pair" in str(info.value)


def test_training_job_gates_qlora_on_cpu(
    tmp_path, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    hp = _hp(method=TrainingMethod.QLORA, load_in_4bit=True)
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(2))
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    result = job.run()

    assert result.status == "failed"
    assert result.adapter_path is None
    assert result.error is not None
    assert "qlora" in result.error.lower()
    assert "GPU" in result.error
    assert not job.paths.adapter_dir.exists()
    errors = [event for event in bus.history() if event.kind == "error"]
    assert errors and errors[0].level == "error"


def test_training_job_gates_fft_on_cpu(
    tmp_path, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    hp = _hp(method=TrainingMethod.FFT)
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(2))
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    result = job.run()
    assert result.status == "failed"
    assert "fft" in (result.error or "").lower()
    assert "GPU" in (result.error or "")


def test_training_job_control_surface(tmp_path, bus: EventBus, run_config: RunConfig) -> None:
    hp = _hp()
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(1))
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus)
    assert job.status["state"] == "idle"
    assert job.status["objective"] == "sft"

    job.request_stop()
    assert job.cancel_event.is_set()
    assert job.status["stop_requested"] is True
    job.request_resume()
    assert not job.pause_event.is_set()
    job.request_pause()
    assert job.pause_event.is_set()


# --------------------------------------------------------------------------- #
# callback behaviour (fast, no training)
# --------------------------------------------------------------------------- #


def test_callback_publishes_and_persists_metrics(bus: EventBus, store: RunStore) -> None:
    callback = TokenMetricsCallback(
        bus=bus, run_id=RUN_ID, objective=TrainingObjective.SFT, store=store
    )
    control = SimpleNamespace(should_training_stop=False)
    callback.on_log(
        None,
        SimpleNamespace(global_step=3, max_steps=6),
        control,
        {"loss": 1.25, "learning_rate": 0.001, "note": "not a number"},
    )
    events = [event for event in bus.history() if event.kind == "metric"]
    assert {event.data["name"] for event in events} == {"loss", "learning_rate"}
    assert store.latest_metric(RUN_ID, "loss") == pytest.approx(1.25)
    progress = [event for event in bus.history() if event.kind == "progress"]
    assert progress and progress[-1].fraction == pytest.approx(0.5)


def test_callback_pauses_at_step_boundary(bus: EventBus) -> None:
    pause = threading.Event()
    pause.set()
    callback = TokenMetricsCallback(
        bus=bus,
        run_id=RUN_ID,
        objective=TrainingObjective.SFT,
        pause_event=pause,
        cancel_event=threading.Event(),
    )
    control = SimpleNamespace(should_training_stop=False)
    state = SimpleNamespace(global_step=1, max_steps=10)
    finished = threading.Event()

    def step() -> None:
        callback.on_step_end(None, state, control)
        finished.set()

    worker = threading.Thread(target=step, daemon=True)
    worker.start()
    time.sleep(0.25)
    assert not finished.is_set(), "the callback must block while paused"
    assert control.should_training_stop is False

    pause.clear()
    assert finished.wait(5.0), "clearing the pause flag must let training continue"
    assert control.should_training_stop is False


def test_callback_stop_requests_training_stop(bus: EventBus) -> None:
    cancel = threading.Event()
    cancel.set()
    callback = TokenMetricsCallback(
        bus=bus, run_id=RUN_ID, objective=TrainingObjective.DPO, cancel_event=cancel
    )
    control = SimpleNamespace(should_training_stop=False)
    callback.on_step_end(None, SimpleNamespace(global_step=2, max_steps=10), control)
    assert control.should_training_stop is True
    assert callback.stop_requested is True


def test_callback_marks_checkpoint_complete(tmp_path, bus: EventBus) -> None:
    output_dir = tmp_path / "out"
    folder = output_dir / "checkpoint-4"
    folder.mkdir(parents=True)
    (folder / "trainer_state.json").write_text("{}", encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(b"weights")
    assert not is_checkpoint_complete(folder)

    callback = TokenMetricsCallback(bus=bus, run_id=RUN_ID, objective=TrainingObjective.SFT)
    callback.on_save(
        SimpleNamespace(output_dir=str(output_dir)),
        SimpleNamespace(global_step=4),
        SimpleNamespace(should_save=True),
    )
    assert is_checkpoint_complete(folder)
    assert any(event.kind == "artifact" for event in bus.history())


def test_jobs_reexports_find_resume_checkpoint(tmp_path) -> None:
    from autorlhf.training.jobs import find_resume_checkpoint as from_jobs

    paths = RunPaths(tmp_path / "run").ensure()
    assert from_jobs(paths, objective=TrainingObjective.SFT) is None
    assert from_jobs(paths, objective=TrainingObjective.DPO) is None


def test_ensure_trl_logprob_compat_makes_top1_boolean() -> None:
    """The DPO metric path bitwise-ANDs `is_top1`, so it must be boolean."""

    import torch
    import trl.trainer.utils as trl_utils

    from autorlhf.training.trainers import ensure_trl_logprob_compat, run_sft  # noqa: F401

    ensure_trl_logprob_compat()
    substitute = trl_utils._ChunkedLogProbFunction
    probe = substitute.apply(
        torch.zeros(2, 4),
        torch.zeros(8, 4),
        None,
        torch.tensor([1, 2]),
        1.0,
        1024,
        None,
        1.0,
        ("log_probs", "entropy", "mean_logits", "is_top1"),
    )
    assert probe[4] is not None
    assert probe[4].dtype == torch.bool

    # A float-returning substitute is wrapped, so the guard is not a no-op by luck.
    class _FloatTop1:
        @staticmethod
        def apply(*args: Any, **kwargs: Any) -> tuple:
            return (
                torch.zeros(2),
                torch.zeros(2),
                None,
                torch.zeros(2),
                torch.ones(2, dtype=torch.float32),
            )

    original = trl_utils._ChunkedLogProbFunction
    _FloatTop1.__module__ = "autorlhf.training.test_double"
    try:
        trl_utils._ChunkedLogProbFunction = _FloatTop1
        assert ensure_trl_logprob_compat() is True
        wrapped = trl_utils._ChunkedLogProbFunction
        assert wrapped is not _FloatTop1
        assert (
            wrapped.apply(
                torch.zeros(2, 4), torch.zeros(8, 4), None, torch.tensor([1, 2]), 1.0, 1024
            )[4].dtype
            == torch.bool
        )
    finally:
        trl_utils._ChunkedLogProbFunction = original


@pytest.mark.slow
def test_training_job_dpo_without_warmup(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    """DPO from a fresh LoRA adapter, with no supervised data at all."""

    run_config.model.model_id = str(tiny_chat_model_dir)
    run_config.model.torch_dtype = "float32"
    hp = _hp(
        objective=TrainingObjective.DPO,
        sft_warmup=False,
        max_steps=1,
        save_steps=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
    )
    plan = _plan(objective=TrainingObjective.DPO, hp=hp, pairs=_pair_records(3))
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    result = job.run()

    assert result.status == "completed", result.error
    assert result.objective is TrainingObjective.DPO
    assert is_adapter_complete(job.paths.adapter_dir)
    assert not list_checkpoint_dirs(checkpoint_root(job.paths, TrainingObjective.SFT))
    assert "loss" in store.metric_names(RUN_ID)
    # W&B is switched off in the run configuration, so there is no run url.
    assert result.wandb_url is None


@pytest.mark.slow
def test_training_job_reuses_warmup_adapter_and_resumes_dpo(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    run_config.model.model_id = str(tiny_chat_model_dir)
    run_config.model.torch_dtype = "float32"
    hp = _hp(
        objective=TrainingObjective.DPO,
        sft_warmup=True,
        max_steps=1,
        save_steps=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
    )
    plan = _plan(
        objective=TrainingObjective.DPO,
        hp=hp,
        sft=_sft_records(2),
        pairs=_pair_records(2),
    )
    first = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    assert first.run().status == "completed"
    assert is_adapter_complete(first.paths.root / "adapter-sft")

    second_bus = EventBus()
    hp2 = hp.model_copy(update={"max_steps": 2})
    plan2 = _plan(
        objective=TrainingObjective.DPO,
        hp=hp2,
        sft=_sft_records(2),
        pairs=_pair_records(2),
    )
    second = _job(tmp_path=tmp_path, plan=plan2, config=run_config, bus=second_bus, store=store)
    result = second.run()

    assert result.status == "completed", result.error
    assert result.resumed_from is not None
    assert any(
        "reusing the existing SFT warm-up adapter" in event.message
        for event in second_bus.history()
    )
    # The second run resumed the DPO checkpoint series instead of restarting it.
    assert second.status["steps_completed"] == 2


# --------------------------------------------------------------------------- #
# tracking wiring (offline: the real backends are never initialised)
# --------------------------------------------------------------------------- #


def test_tracking_is_not_touched_when_disabled(
    tmp_path, bus: EventBus, run_config: RunConfig
) -> None:
    hp = _hp()
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(1))
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus)
    assert job._tracker() is None
    assert job._report_to() == []
    assert job._wandb_url() is None
    job._start_tracking()
    # (lazy-import inertness is asserted in a fresh subprocess by
    # tests/test_integration_tracking.py; an in-process sys.modules check would
    # depend on test ordering once the tracking module is imported elsewhere.)


def test_tracking_wrapper_is_used_when_enabled(
    tmp_path, bus: EventBus, run_config: RunConfig, monkeypatch
) -> None:
    from types import SimpleNamespace

    hp = _hp()
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(1))
    run_config.wandb_enabled = True
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus)

    tracker = job._tracker()
    assert tracker is not None
    assert tracker.__class__.__name__ == "TrainingTracker"

    # Substitute a stub so that no network call is made.
    calls: list[str] = []

    class _Stub:
        report_to = ["wandb"]

        def __init__(self) -> None:
            self.state = SimpleNamespace(wandb_url="https://wandb.example/run/1", warnings=["w1"])

        def start(self):
            calls.append("start")
            return self.state

    stub = _Stub()
    monkeypatch.setattr(job, "_tracker", lambda: stub)
    job._start_tracking()

    assert calls == ["start"]
    assert job._report_to() == ["wandb"]
    assert job._wandb_url() == "https://wandb.example/run/1"
    assert "w1" in job._warnings


def test_tracking_failure_degrades_to_a_warning(
    tmp_path, bus: EventBus, run_config: RunConfig, monkeypatch
) -> None:
    hp = _hp()
    plan = _plan(objective=TrainingObjective.SFT, hp=hp, sft=_sft_records(1))
    run_config.wandb_enabled = True
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus)

    class _Broken:
        report_to: list[str] = []
        state = SimpleNamespace(wandb_url=None, warnings=[])

        def start(self):
            raise RuntimeError("no credentials")

    monkeypatch.setattr(job, "_tracker", _Broken)
    job._start_tracking()  # must not raise

    assert any("experiment tracking failed" in warning for warning in job._warnings)
    assert job._report_to() == []
    assert job._wandb_url() is None


@pytest.mark.slow
def test_training_job_honours_an_explicit_resume_checkpoint(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore, run_config: RunConfig
) -> None:
    run_config.model.model_id = str(tiny_chat_model_dir)
    run_config.model.torch_dtype = "float32"
    plan = _plan(
        objective=TrainingObjective.SFT, hp=_hp(max_steps=1, save_steps=1), sft=_sft_records(3)
    )
    job = _job(tmp_path=tmp_path, plan=plan, config=run_config, bus=bus, store=store)
    assert job.run().status == "completed"
    checkpoint = find_resume_checkpoint(job.paths, objective=TrainingObjective.SFT)
    assert checkpoint is not None

    plan2 = _plan(
        objective=TrainingObjective.SFT, hp=_hp(max_steps=2, save_steps=1), sft=_sft_records(3)
    )
    second = _job(tmp_path=tmp_path, plan=plan2, config=run_config, bus=bus, store=store)
    result = second.run(resume_from=checkpoint)

    assert result.status == "completed", result.error
    assert result.resumed_from == str(checkpoint)
    assert result.steps_completed == 2


def _marker_free_model_dir(tmp_path, tiny_chat_model_dir) -> Path:
    """A copy of the tiny model whose chat template has no {% generation %} markers."""

    target = tmp_path / "marker-free-model"
    target.mkdir(exist_ok=True)
    for item in tiny_chat_model_dir.iterdir():
        if item.is_file():
            (target / item.name).write_bytes(item.read_bytes())
    for name in ("chat_template.jinja", "chat_template.json"):
        (target / name).unlink(missing_ok=True)
    tokenizer_config = target / "tokenizer_config.json"
    payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
    payload.pop("chat_template", None)
    tokenizer_config.write_text(json.dumps(payload), encoding="utf-8")
    return target


@pytest.mark.slow
def test_run_sft_refuses_a_template_that_cannot_mask_the_assistant(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore
) -> None:
    """TRL cannot tokenize conversational data without {% generation %} markers."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from autorlhf.core.errors import ChatTemplateError

    marker_free = _marker_free_model_dir(tmp_path, tiny_chat_model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(marker_free))
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(marker_free), dtype="float32", attn_implementation="eager"
    )

    with pytest.raises(ChatTemplateError) as info:
        run_sft(
            run_id=RUN_ID,
            paths=RunPaths(tmp_path / "run").ensure(),
            hp=_hp(),
            model=model,
            tokenizer=tokenizer,
            train_records=_sft_records(2),
            bus=bus,
            store=store,
            report_to=(),
            fan_in_fan_out=True,
        )
    assert "generation" in info.value.hint
    assert not store.metric_names(RUN_ID)  # nothing ran, and nothing was recorded


@pytest.mark.slow
def test_run_sft_works_with_the_generic_fallback_template(
    tmp_path, tiny_chat_model_dir, bus: EventBus, store: RunStore
) -> None:
    """The fallback template carries {% generation %} markers, so training works."""

    from autorlhf.config.run_config import HFModelConfig
    from autorlhf.training.models import load_model

    marker_free = _marker_free_model_dir(tmp_path, tiny_chat_model_dir)
    cfg = HFModelConfig(
        model_id=str(marker_free),
        torch_dtype="float32",
        attn_implementation="eager",
        allow_generic_chat_template=True,
    )
    hp = _hp(max_steps=1, save_steps=1)
    loaded = load_model(cfg, hp, hf_token=None)
    assert loaded.chat_template_source == "generic-fallback"
    assert any("generic template" in warning for warning in loaded.warnings)

    result = run_sft(
        run_id=RUN_ID,
        paths=RunPaths(tmp_path / "run").ensure(),
        hp=hp,
        model=loaded.model,
        tokenizer=loaded.tokenizer,
        train_records=_sft_records(3),
        bus=bus,
        store=store,
        report_to=(),
        run_name="smoke-fallback-template",
        fan_in_fan_out=True,
        extra_warnings=loaded.warnings,
    )

    assert result.status == "completed", result.error
    assert is_adapter_complete(Path(result.adapter_path))
