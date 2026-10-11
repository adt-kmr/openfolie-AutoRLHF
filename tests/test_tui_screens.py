"""Every screen in the spec exists and renders without a traceback."""

from __future__ import annotations

import pytest
from test_tui_support import make_app, running

from autorlhf.tui.screens import SCREEN_CLASSES

EXPECTED_SCREENS = {
    "dashboard",
    "setup",
    "model_config",
    "character",
    "rubric",
    "generation",
    "training_config",
    "review",
    "dataset",
    "training",
    "evaluation",
    "settings",
}


def test_every_spec_screen_is_registered():
    assert set(SCREEN_CLASSES) == EXPECTED_SCREENS


@pytest.mark.parametrize("name", sorted(EXPECTED_SCREENS))
async def test_screen_renders(name, app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app, size=(140, 45)) as pilot:
        await pilot.pause()
        await app.push_screen(name)
        await pilot.pause()
        assert type(app.screen).__name__ == SCREEN_CLASSES[name].__name__
        # a second refresh pass catches layout-time failures too
        app.refresh(layout=True)
        await pilot.pause()


async def test_screens_navigate_as_a_stack(app_paths, store, bus, run_config):
    from test_tui_support import make_run

    from autorlhf.core.enums import RunStatus

    make_run(store, app_paths, run_config, status=RunStatus.COMPLETED)
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        start = app.screen
        await app.push_screen("settings")
        await pilot.pause()
        assert len(app.screen_stack) == 2
        app.pop_screen()
        await pilot.pause()
        assert app.screen is start
        # escape must not pop the last screen off the stack
        app.screen.action_go_back()
        await pilot.pause()
        assert app.screen is start
