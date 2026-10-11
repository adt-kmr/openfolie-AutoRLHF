"""Interactive review queue: the human decision always wins and is persisted."""

from __future__ import annotations

from test_tui_support import make_app, make_run, review_item, running, static_text
from textual.containers import Vertical

from autorlhf.core.enums import RunStatus
from autorlhf.tui.screens.review import ReviewScreen


def _open(app, run_id: str):
    assert app.adopt_run(run_id) is True


async def test_review_records_a_human_override(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    submitted: list[dict] = []
    app.submit_review_decision = lambda decision: submitted.append(decision)  # type: ignore[method-assign]

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)

        assert screen.add_item(review_item()) is True
        await pilot.pause()
        assert len(screen.items) == 1
        assert screen.current is not None

        decision = screen.submit_action("choose_b")
        assert decision["action"] == "choose_b"
        assert decision["chosen_id"] == "p0001-c1"
        assert decision["rejected_id"] == "p0001-c0"
        assert decision["reviewer"] == "human"
        assert decision["created_at"]
        assert submitted and submitted[-1] is decision
        assert screen.items[0]["status"] == "completed"
        assert screen.pending() == []


async def test_review_keeps_the_judge_recommendation_visible_but_not_applied(
    app_paths, store, bus, run_config
):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        screen.add_item(review_item())
        await pilot.pause()
        recommendation = static_text(screen.query_one("#review-recommendation"))
        assert "p0001-c0" in recommendation
        assert "never applied automatically" in recommendation
        # nothing was decided just by looking at it
        assert screen.decisions == {}


async def test_review_accept_uses_the_recommended_candidate(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    submitted: list[dict] = []
    app.submit_review_decision = lambda decision: submitted.append(decision)  # type: ignore[method-assign]
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        screen.add_item(review_item())
        await pilot.pause()
        screen.set_score("p0001-c1", "character_consistency", 4.0)
        screen.set_edited_text("p0001-c1", "edited by hand")
        decision = screen.submit_action("accept")
        assert decision["chosen_id"] == "p0001-c0"
        assert decision["scores"] == {"p0001-c1": {"character_consistency": 4.0}}
        assert decision["edited_texts"] == {"p0001-c1": "edited by hand"}
        assert decision["use_chosen_as_sft"] is True


async def test_review_tie_and_regenerate_choose_nobody(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    submitted: list[dict] = []
    app.submit_review_decision = lambda decision: submitted.append(decision)  # type: ignore[method-assign]
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        screen.add_item(review_item("p0002"))
        await pilot.pause()
        tie = screen.submit_action("tie")
        assert tie["chosen_id"] is None and tie["rejected_id"] is None
        screen.add_item(review_item("p0003"))
        await pilot.pause()
        screen.load_index(0)
        regenerate = screen.build_decision("regenerate")
        assert regenerate["action"] == "regenerate"
        assert regenerate["rejected_id"] == "p0002-c0"


async def test_review_adds_items_from_bus_events_and_persists_without_a_queue(
    app_paths, store, bus, run_config
):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)

        bus.emit("sample", "review item", run_id=run_id, review_item=review_item("p0009"))
        await pilot.pause()
        assert [item["item_id"] for item in screen.items] == ["p0009"]
        # a duplicate event must not double up
        bus.emit("sample", "review item", run_id=run_id, review_item=review_item("p0009"))
        await pilot.pause()
        assert len(screen.items) == 1

        # the real submission path persists to SQLite when no queue is attached
        app.submit_review_decision(screen.submit_action("choose_a"))
        stored = store.get_review(run_id, "p0009")
        assert stored is not None
        assert stored["status"] == "completed"
        assert stored["payload"]["chosen_id"] == "p0009-c0"


async def test_review_shows_the_character_and_judge_distributions(
    app_paths, store, bus, run_config
):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        screen.add_item(review_item())
        await pilot.pause()

        summary = static_text(screen.query_one("#review-character"))
        assert "terse test assistant" in summary
        assert "Never mention being an AI." in summary

        bars = list(screen.query_one("#review-scorebars", Vertical).children)
        assert len(bars) == 2  # two candidates, one criterion each
        text = "\n".join(str(bar.content) for bar in bars)
        assert "60.0%" in text  # the judge's own probabilities, drawn not invented
        assert "confidence 0.70" in text


async def test_review_restores_pending_items_from_sqlite(app_paths, store, bus, run_config):
    """A fresh process shows the review items an interrupted session left behind."""

    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    store.save_review(
        run_id,
        "p0007",
        prompt_id="p0007",
        status="pending",
        payload=review_item("p0007"),
    )
    store.save_review(
        run_id,
        "p0008",
        prompt_id="p0008",
        status="completed",
        payload=review_item("p0008"),
    )
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    app.submit_review_decision = lambda decision: None  # type: ignore[method-assign]

    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        assert [item["item_id"] for item in screen.items] == ["p0007"]
        assert "0/1 decided" in static_text(screen.query_one("#review-stats"))


async def test_review_stats_report_progress(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.REVIEWING)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    _open(app, run_id)
    app.submit_review_decision = lambda decision: None  # type: ignore[method-assign]
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("review")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ReviewScreen)
        screen.load_items([review_item("p0001"), review_item("p0002")])
        await pilot.pause()
        assert "0/2 decided" in static_text(screen.query_one("#review-stats"))
        screen.submit_action("tie")
        await pilot.pause()
        assert "1/2 decided" in static_text(screen.query_one("#review-stats"))
