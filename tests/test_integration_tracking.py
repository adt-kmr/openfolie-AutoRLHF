"""Integration: optional tracking (W&B / Weave) is inert unless enabled."""

from __future__ import annotations

import logging
import subprocess
import sys
import types

from autorlhf.config.settings import Environment
from autorlhf.integrations.tracking import (
    WEAVE_EGRESS_NOTICE,
    TrainingTracker,
    weave_client,
)


def _clean_modules() -> None:
    sys.modules.pop("wandb", None)
    sys.modules.pop("weave", None)


def test_import_never_pulls_optional_libraries():
    code = (
        "import sys, autorlhf.integrations, autorlhf.doctor, autorlhf.setup_env; "
        "print('wandb' in sys.modules, 'weave' in sys.modules, 'textual' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False False False"


def test_both_disabled_is_inert(run_config, app_paths, bus):
    config = run_config.model_copy(deep=True)
    assert config.wandb_enabled is False
    assert config.weave_enabled is False
    env = Environment(app_paths)
    _clean_modules()

    received = []
    bus.subscribe(lambda event: received.append(event))

    tracker = TrainingTracker(config=config, env=env, run_id="run-inert", bus=bus)
    state = tracker.start()

    assert tracker.report_to == []
    assert tracker.enabled is False
    assert state.wandb_enabled is False
    assert state.weave_enabled is False
    assert state.warnings == []

    # No lazy import happened, and no event was emitted by tracking itself.
    assert "wandb" not in sys.modules
    assert "weave" not in sys.modules
    assert received == []

    tracker.log({"loss": 1.0, "lr": 1e-5}, step=1)
    tracker.finish(summary={"final_loss": 0.5})
    assert received == []


def test_wandb_init_failure_warns_without_raising(run_config, app_paths, bus, monkeypatch):
    config = run_config.model_copy(deep=True)
    config.wandb_enabled = True
    config.wandb_project = "demo"
    env = Environment(app_paths)

    fake = types.ModuleType("wandb")

    def _boom(*args, **kwargs):
        raise RuntimeError("authentication failed (simulated)")

    fake.init = _boom
    fake.log = lambda *a, **k: None
    fake.finish = lambda *a, **k: None
    fake.Settings = type("Settings", (), {"__init__": lambda self, *a, **k: None})
    monkeypatch.setitem(sys.modules, "wandb", fake)

    warnings = []
    bus.subscribe(lambda event: warnings.append(event) if event.kind == "warning" else None)

    tracker = TrainingTracker(config=config, env=env, run_id="run-wandb", bus=bus)
    state = tracker.start()

    assert state.wandb_enabled is False
    assert tracker.report_to == []
    assert state.wandb_url is None
    assert state.warnings and any("W&B init failed" in w for w in state.warnings)
    assert any("W&B init failed" in event.message for event in warnings)
    # Finishing after a failed start must also be safe.
    tracker.finish()


def test_wandb_init_success_forwards_logs(run_config, app_paths, bus, monkeypatch):
    config = run_config.model_copy(deep=True)
    config.wandb_enabled = True
    config.wandb_project = "demo"
    env = Environment(app_paths)

    logged: list[tuple[dict, int]] = []
    finished: list[bool] = []

    class _FakeRun:
        url = "https://wandb.example/demo/run"
        summary: dict = {}

    fake = types.ModuleType("wandb")
    fake.init = lambda *a, **k: _FakeRun()
    fake.log = lambda metrics, step: logged.append((dict(metrics), step))
    fake.finish = lambda *a, **k: finished.append(True)
    fake.Settings = type("Settings", (), {"__init__": lambda self, *a, **k: None})
    monkeypatch.setitem(sys.modules, "wandb", fake)

    tracker = TrainingTracker(config=config, env=env, run_id="run-ok", bus=bus)
    state = tracker.start()
    assert state.wandb_enabled is True
    assert tracker.report_to == ["wandb"]
    assert state.wandb_url == "https://wandb.example/demo/run"

    tracker.log({"loss": 0.25}, step=7)
    assert logged == [({"loss": 0.25}, 7)]

    tracker.finish(summary={"final_loss": 0.25})
    assert finished == [True]


def test_weave_without_consent_returns_none_and_warns(run_config, app_paths):
    config = run_config.model_copy(deep=True)
    config.weave_enabled = True
    config.weave_consent = False
    env = Environment(app_paths)
    _clean_modules()

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    logger = logging.getLogger("autorlhf.integrations.tracking")
    logger.addHandler(handler)
    try:
        client = weave_client(env, config=config)
    finally:
        logger.removeHandler(handler)

    assert client is None
    assert any(record == WEAVE_EGRESS_NOTICE for record in records)
    joined = " ".join(records)
    assert "external service" in joined
    assert "prompts" in joined and "judgements" in joined
    # Consent being absent means weave must not even be imported.
    assert "weave" not in sys.modules


def test_weave_disabled_returns_none(app_paths):
    env = Environment(app_paths)
    config = None
    assert weave_client(env, config=config) is None


def test_weave_with_consent_initialises(run_config, app_paths, monkeypatch):
    config = run_config.model_copy(deep=True)
    config.weave_enabled = True
    config.weave_consent = True
    config.weave_project = "traces"
    env = Environment(app_paths)

    inited: list[str] = []
    fake = types.ModuleType("weave")
    fake.init = lambda project: inited.append(project) or object()
    fake.op = lambda *a, **k: lambda fn: fn
    fake.finish = lambda: None
    fake.get_current_call = lambda: None
    fake.attributes = lambda attrs: _NullContext()
    monkeypatch.setitem(sys.modules, "weave", fake)

    client = weave_client(env, config=config)
    assert client is not None
    assert inited == ["traces"]


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def test_weave_tracker_without_consent_records_warning(run_config, app_paths, bus):
    config = run_config.model_copy(deep=True)
    config.weave_enabled = True
    config.weave_consent = False
    env = Environment(app_paths)
    tracker = TrainingTracker(config=config, env=env, run_id="run-weave", bus=bus)
    state = tracker.start()
    assert state.weave_enabled is False
    assert any("external service" in warning for warning in state.warnings)
