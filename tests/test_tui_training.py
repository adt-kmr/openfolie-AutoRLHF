"""Training view (live event stream) and the evaluation comparison table."""

from __future__ import annotations

import threading
import time

from test_tui_support import make_app, make_run, running, static_text
from textual.widgets import DataTable, ProgressBar

from autorlhf.core.enums import RunStatus
from autorlhf.llm.schemas import (
    ComparisonReport,
    CriterionComparison,
    EvaluationReport,
)
from autorlhf.tui.screens.evaluation import EvaluationScreen
from autorlhf.tui.screens.training import TrainingScreen
from autorlhf.tui.widgets import KeyValuePanel, LogPane, MetricTile


def _comparison(run_id: str) -> ComparisonReport:
    return ComparisonReport(
        run_id=run_id,
        judge_model="typesafe/jev-1.13",
        n_prompts=8,
        n_scored_base=7,
        n_scored_finetuned=8,
        criteria=[
            CriterionComparison(
                criterion_id="character_consistency",
                criterion_name="Character consistency",
                base_mean=2.9,
                finetuned_mean=3.6,
                base_n=7,
                finetuned_n=8,
                delta=0.7,
            ),
            CriterionComparison(
                criterion_id="voice_style",
                criterion_name="Voice and style",
                base_mean=3.1,
                finetuned_mean=3.0,
                base_n=7,
                finetuned_n=8,
                delta=-0.1,
            ),
        ],
        base_overall=3.0,
        finetuned_overall=3.3,
        overall_delta=0.3,
        pairwise_base_wins=1,
        pairwise_finetuned_wins=5,
        pairwise_ties=1,
        pairwise_inconclusive=1,
        examples=[
            {
                "prompt": "Will the fog lift today?",
                "base_text": "It is hard to say.",
                "tuned_text": "Not before the afternoon tide.",
                "base_judgement": {"overall": 2.4},
                "tuned_judgement": {"overall": 3.7},
            }
        ],
        warnings=["coverage is below min_responses_for_claims (6)"],
        inconclusive=False,
    )


async def test_training_screen_renders_an_event_stream(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingScreen)

        bus.emit("phase", "training dpo", run_id=run_id, phase="train")
        bus.emit("progress", "step 3/10", run_id=run_id, step=3, fraction=0.3)
        bus.emit("metric", "loss", run_id=run_id, name_key="loss", value=0.42, step=3)
        bus.emit("metric", "eval_loss", run_id=run_id, name_key="eval_loss", value=0.51, step=3)
        bus.emit("metric", "learning_rate", run_id=run_id, name_key="learning_rate", value=1e-5)
        bus.emit("warning", "retrying a failed judge call", run_id=run_id)
        bus.emit("error", "one judgement failed", run_id=run_id)
        bus.emit(
            "sample",
            "candidate ready",
            run_id=run_id,
            prompt="Will the fog lift today?",
            response="Not before the afternoon tide.",
        )
        await pilot.pause()
        await app._event_queue.join()
        await pilot.pause(0.5)
        assert screen.query_one("#tile-step", MetricTile).value == "3"
        assert screen.query_one("#tile-loss", MetricTile).value == "0.420"
        assert screen.query_one("#tile-eval_loss", MetricTile).value == "0.510"
        assert screen.query_one("#tile-phase", MetricTile).value == "train"
        assert screen.query_one("#training-progress", ProgressBar).progress == 30.0

        warnings = screen.query_one("#training-warnings", LogPane).text()
        assert "retrying a failed judge call" in warnings
        assert "one judgement failed" in warnings

        # non-verbose hides prompts/responses, verbose shows them
        assert (
            "Not before the afternoon tide."
            not in screen.query_one("#training-samples", LogPane).text()
        )
        screen.verbose = True
        bus.emit(
            "sample",
            "candidate ready",
            run_id=run_id,
            prompt="And the tide?",
            response="It turns at dusk.",
        )
        await pilot.pause()
        assert "It turns at dusk." in screen.query_one("#training-samples", LogPane).text()


async def test_metric_burst_batches_state_panel_refreshes(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        original_render = screen._render_kv
        refresh_count = 0

        async def count_refreshes():
            nonlocal refresh_count
            refresh_count += 1
            await original_render()

        screen._render_kv = count_refreshes
        for step in range(100):
            bus.emit(
                "metric", "loss", run_id=run_id, name_key="loss", value=step / 1000, step=step
            )
        await pilot.pause(0.1)
        await app._event_queue.join()
        await pilot.pause(0.1)

        assert screen.query_one("#tile-step", MetricTile).value == "99"
        assert screen.query_one("#tile-loss", MetricTile).value == "0.099"
        assert 0 < refresh_count < 10


async def test_activity_logs_are_dispatched_asynchronously(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        pane = screen.query_one("#training-log", LogPane)

        for index in range(50):
            bus.emit("log", f"queued activity entry {index:02}", run_id=run_id)
        assert "queued activity entry 00" not in pane.text()
        await pilot.pause()
        activity = pane.text()
        positions = [activity.index(f"queued activity entry {index:02}") for index in range(50)]
        assert positions == sorted(positions)


async def test_elapsed_timer_ticks_while_worker_blocks(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        screen.started_at = time.time() - 5

        blocker = threading.Event()
        worker = threading.Thread(target=blocker.wait)
        worker.start()
        try:
            await pilot.pause(1.2)
            elapsed = screen.query_one("#tile-elapsed", MetricTile).value
            assert elapsed != "-"
            assert elapsed != "00:05"
        finally:
            blocker.set()
            worker.join(timeout=1)

async def test_training_screen_replays_history_on_mount(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True
    async with running(app) as pilot:
        await pilot.pause()
        bus.emit("metric", "loss", run_id=run_id, name_key="loss", value=0.9, step=2)
        bus.emit("phase", "training", run_id=run_id, phase="train")
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingScreen)
        assert screen.query_one("#tile-loss", MetricTile).value == "0.900"


async def test_training_screen_controls_call_the_app(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True
    calls: list[str] = []
    app.request_pause = lambda: calls.append("pause")  # type: ignore[method-assign]
    app.request_resume = lambda: calls.append("resume")  # type: ignore[method-assign]
    app.request_stop = lambda: calls.append("stop")  # type: ignore[method-assign]

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingScreen)
        screen.action_pause()
        screen.action_resume()
        screen.action_stop()
        assert calls == ["pause", "resume", "stop"]


async def test_evaluation_screen_renders_a_comparison_report(app_paths, store, bus, run_config):
    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.COMPLETED)
    report = _comparison(run_id)
    run_paths.comparison_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("evaluation")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EvaluationScreen)

        table = screen.query_one("#eval-criteria", DataTable)
        assert table.row_count == 2
        assert str(table.get_row_at(0)[0]) == "Character consistency"
        assert str(table.get_row_at(0)[3]) == "+0.70"

        summary = screen.query_one("#eval-summary", KeyValuePanel).text()
        assert "3.30" in summary  # fine-tuned overall
        assert "1 / 5 / 1" in summary  # pairwise base / tuned / tie
        warnings = static_text(screen.query_one("#eval-warnings"))
        assert "coverage is below" in warnings
        detail = static_text(screen.query_one("#eval-example-detail"))
        assert "Not before the afternoon tide." in detail
        assert "overall=3.7" in detail


async def test_evaluation_screen_accepts_a_report_object_and_resets(
    app_paths, store, bus, run_config
):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.COMPLETED)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    assert app.adopt_run(run_id) is True
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("evaluation")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, EvaluationScreen)
        assert screen.report == {}
        screen.show_report(_comparison(run_id))
        await pilot.pause()
        assert screen.query_one("#eval-criteria", DataTable).row_count == 2
        assert len(screen.examples) == 1

        screen.show_report(EvaluationReport(run_id=run_id, variant="base"))
        await pilot.pause()
        assert screen.query_one("#eval-criteria", DataTable).row_count == 0
