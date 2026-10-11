"""First-run setup flow."""

from __future__ import annotations

from test_tui_support import make_app, running
from textual.widgets import Button, Checkbox, Input, TabbedContent

from autorlhf.config.character import DEFAULT_CHARACTER
from autorlhf.core.enums import FeedbackMode
from autorlhf.tui.forms import (
    CharacterForm,
    GenerationForm,
    ModelConfigForm,
    RubricForm,
    TrackingForm,
)
from autorlhf.tui.screens.dashboard import DashboardScreen
from autorlhf.tui.screens.setup import SetupScreen
from autorlhf.tui.widgets import MaskedSecretField


async def _open_setup(app):
    await app.push_screen("setup")
    return app.screen


async def test_setup_renders_every_tab(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        tabs = screen.query_one(TabbedContent)
        assert len(list(tabs.query("TabPane"))) == 6
        # every form is loaded from the draft
        assert screen.query_one(ModelConfigForm).query_one("#model-id", Input).value
        assert screen.query_one(CharacterForm).query_one("#char-name", Input).value
        assert len(screen.query_one(RubricForm).criteria) == 6
        assert screen.query_one(GenerationForm).query_one("#gen-prompts", Input).value == "24"
        assert isinstance(screen.query_one(TrackingForm), TrackingForm)


async def test_setup_collect_is_valid_and_finish_persists(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        assert screen.collect() == []
        # credentials typed here are written to the secret file, never echoed
        secret = screen.query_one("#setup-openrouter-field", MaskedSecretField)
        secret.input.value = "sk-or-v1-not-a-real-key-0123456789"
        assert screen.save_credentials() == 1
        assert app.env.secrets.get("OPENROUTER_API_KEY").startswith("sk-or-v1-")
        assert secret.value == ""  # cleared after saving

        screen.action_finish()
        await pilot.pause()
        assert app.env.settings.first_run_complete is True
        assert app.env.settings.default_hf_model == app.state.draft.model.model_id
        assert app.env.paths.settings_path.is_file()
        assert isinstance(app.screen, DashboardScreen)


async def test_setup_demo_character_button_loads_spec(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        form = screen.query_one(CharacterForm)
        form.remove_seed(0)
        assert len(form.seeds) == len(DEFAULT_CHARACTER.seed_conversations) - 1
        screen.apply_demo_character()
        assert len(form.seeds) == len(DEFAULT_CHARACTER.seed_conversations)
        assert screen.query_one(RubricForm).criteria[0].name == "Character consistency"


async def test_setup_collect_reports_invalid_generation(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        screen.query_one(GenerationForm).query_one("#gen-prompts", Input).value = "2"
        problems = screen.collect()
        assert any("min_eval_prompts" in problem for problem in problems)


async def test_setup_tracking_requires_consent(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus)
    async with running(app) as pilot:
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupScreen)
        tracking = screen.query_one(TrackingForm)
        # enabling Weave without consent must be refused, not silently allowed
        tracking.set_consent(False)
        tracking.query_one("#track-weave", Checkbox).value = True
        problems = tracking.apply(app.state.draft)
        assert any("consent" in problem for problem in problems)
        assert app.state.draft.weave_enabled is False

        tracking.set_consent(True)
        tracking.query_one("#track-weave", Checkbox).value = True
        tracking.query_one("#track-weave-project", Input).value = "autorlhf-demo"
        assert tracking.apply(app.state.draft) == []
        assert app.state.draft.weave_enabled is True
        assert app.state.draft.weave_consent is True


async def test_setup_interactive_mode_sets_feedback_mode(app_paths, store, bus):
    app = make_app(app_paths, store=store, bus=bus, interactive=True)
    async with running(app) as pilot:
        await pilot.pause()
        assert app.state.draft.feedback_mode is FeedbackMode.INTERACTIVE
        assert app.screen.query_one("#setup-finish", Button).label == "Finish setup"
