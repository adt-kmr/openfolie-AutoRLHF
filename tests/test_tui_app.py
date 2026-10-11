"""Application flow: starting a run, worker plumbing, and graceful quitting."""

from __future__ import annotations

import threading
import time

from test_tui_support import StubWorker, make_app, running

from autorlhf.tui.screens.review import ReviewScreen
from autorlhf.tui.screens.training import TrainingScreen
from autorlhf.tui.widgets import MetricTile


def _valid(run_config):
    config = run_config.model_copy(deep=True)
    config.generation.num_prompts = 25
    return config


async def test_start_run_pushes_training_and_review_in_interactive_mode(
    app_paths, store, bus, run_config
):
    created: list[StubWorker] = []

    def factory(**kwargs):
        worker = StubWorker(**kwargs)
        created.append(worker)
        return worker

    app = make_app(
        app_paths, store=store, bus=bus, config=_valid(run_config), worker_factory=factory
    )
    async with running(app) as pilot:
        await pilot.pause()
        assert app.start_run(interactive=True) is True
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        names = [type(screen).__name__ for screen in app.screen_stack]
        assert "TrainingScreen" in names
        assert app.state.interactive is True
        assert app.state.run_id
        assert created and created[0].kwargs["interactive"] is True
        app._worker_thread.join(timeout=5)
        await pilot.pause()


async def test_start_run_refuses_an_invalid_configuration(app_paths, store, bus, run_config):
    bad = _valid(run_config)
    bad.generation.num_prompts = 2
    app = make_app(app_paths, store=store, bus=bus, config=bad)
    async with running(app) as pilot:
        await pilot.pause()
        assert app.start_run() is False
        assert app.worker is None
        assert app.state.run_id is None


async def test_worker_bus_events_reach_the_training_screen(app_paths, store, bus, run_config):
    def factory(**kwargs):
        worker = StubWorker(**kwargs)
        run_id = kwargs["run_id"]

        def emit() -> None:
            bus.emit("phase", "training", run_id=run_id, phase="train")
            bus.emit("metric", "loss", run_id=run_id, name_key="loss", value=0.25, step=7)

        worker.on_run = emit
        return worker

    app = make_app(
        app_paths, store=store, bus=bus, config=_valid(run_config), worker_factory=factory
    )
    async with running(app) as pilot:
        await pilot.pause()
        assert app.start_run() is True
        await pilot.pause()
        app._worker_thread.join(timeout=5)
        await pilot.pause()
        await pilot.pause()
        # the app re-renders the run list from the events the worker published
        screen = next(s for s in app.screen_stack if isinstance(s, TrainingScreen))
        assert screen.query_one("#tile-loss", MetricTile).value == "0.250"
        assert screen.query_one("#tile-phase", MetricTile).value == "train"


async def test_request_quit_waits_for_a_live_worker(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.worker = StubWorker(run_id="manual", bus=bus)
        app._worker_thread = threading.Thread(target=time.sleep, args=(0.2,))
        app._worker_thread.start()
        app.action_request_quit()
        assert app._exit_when_idle is True
        assert app.worker.stopped is True
        assert app.is_running is True  # the UI stays up until the worker stops
        app._worker_thread.join(timeout=5)


async def test_request_quit_exits_immediately_without_a_worker(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.action_request_quit()
        await pilot.pause()
        assert app._exit_when_idle is False


async def test_toggle_pause_uses_the_worker_flags(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.worker = StubWorker(run_id="manual", bus=bus)
        app.toggle_pause()
        assert app.worker.paused is True
        app.toggle_pause()
        assert app.worker.paused is False
