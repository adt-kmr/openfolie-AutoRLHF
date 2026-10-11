"""Dashboard screen: run table, empty-state routing, recovery actions."""

from __future__ import annotations

from test_tui_support import make_app, make_run, running, static_text
from textual.widgets import DataTable, ListView

from autorlhf.core.enums import RunStatus
from autorlhf.tui.screens.dashboard import DashboardScreen
from autorlhf.tui.screens.settings import SettingsScreen
from autorlhf.tui.screens.setup import SetupScreen


async def test_dashboard_routes_to_setup_when_there_are_no_runs(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)


async def test_dashboard_lists_a_stored_run(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, DashboardScreen)
        table = app.screen.query_one("#runs-table", DataTable)
        assert table.row_count == 1
        assert str(table.get_row_at(0)[0]) == run_id
        # a brand-new run is not offered for resume
        assert len(app.screen.query_one("#incomplete-list", ListView).children) == 0


async def test_dashboard_lists_incomplete_runs_and_details(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.TRAINING_DPO)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DashboardScreen)
        assert screen._incomplete and screen._incomplete[0].id == run_id
        screen._show_inspect(run_id)
        detail = static_text(screen.query_one("#run-detail"))
        assert run_id in detail
        assert "training_dpo" in detail


async def test_dashboard_abandons_a_run_and_deletes_its_data(app_paths, store, bus, run_config):
    run_id, run_paths = make_run(store, app_paths, run_config, status=RunStatus.PREPARING_DATA)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, DashboardScreen)
        screen._confirm_abandon(run_id)
        await pilot.pause()
        await pilot.click("#confirm-no")
        await pilot.pause()
        assert store.get_run(run_id) is not None
        screen._confirm_abandon(run_id)
        await pilot.pause()
        await pilot.click("#confirm-yes")
        await pilot.pause()
        await pilot.pause()
        assert store.get_run(run_id) is None
        assert not run_paths.root.exists()
        assert screen.query_one("#runs-table", DataTable).row_count == 0
        assert screen._incomplete == []


async def test_dashboard_resume_delegates_to_start_run(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.INTERRUPTED)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    captured: list[dict] = []
    app.start_run = lambda **kwargs: captured.append(kwargs) or True  # type: ignore[method-assign]
    assert app.resume_run(run_id) is True
    assert captured and captured[0]["resume"] is True
    assert captured[0]["run_id"] == run_id


async def test_dashboard_restart_cancels_then_starts_fresh(app_paths, store, bus, run_config):
    run_id, _paths = make_run(store, app_paths, run_config, status=RunStatus.FAILED)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    captured: list[dict] = []
    app.start_run = lambda **kwargs: captured.append(kwargs) or True  # type: ignore[method-assign]
    assert app.restart_run(run_id) is True
    assert store.require_run(run_id).status is RunStatus.CANCELLED
    assert captured and captured[0].get("run_id") is None
    assert captured[0]["config"].slug == run_config.slug


async def test_dashboard_actions_open_other_screens(app_paths, store, bus, run_config):
    make_run(store, app_paths, run_config, status=RunStatus.COMPLETED)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.screen.action_open_settings()
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        app.pop_screen()
        await pilot.pause()
        app.screen.action_new_run()
        await pilot.pause()
        assert isinstance(app.screen, SetupScreen)


async def test_adopt_run_refuses_an_unknown_id(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        assert app.adopt_run("does-not-exist") is False
