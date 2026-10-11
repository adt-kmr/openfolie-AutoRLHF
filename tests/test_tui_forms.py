"""Model, character, rubric and generation forms."""

from __future__ import annotations

from test_tui_support import make_app, running, static_text
from textual.widgets import Checkbox, Input, ListView, Select, TextArea

from autorlhf.tui.forms import CharacterForm, GenerationForm, ModelConfigForm, RubricForm
from autorlhf.tui.forms.character import parse_seed_text
from autorlhf.tui.screens.character import CharacterScreen
from autorlhf.tui.screens.generation import GenerationScreen
from autorlhf.tui.screens.model_config import ModelConfigScreen
from autorlhf.tui.screens.rubric import RubricScreen


def test_parse_seed_text_accepts_alternating_turns():
    seed, reason = parse_seed_text("user: hello\nassistant: hi there")
    assert reason == ""
    assert seed is not None
    assert seed.is_valid()
    assert seed.user_turns() == ["hello"]


def test_parse_seed_text_rejects_broken_input():
    assert parse_seed_text("just some prose")[0] is None
    assert parse_seed_text("user: hi")[0] is None  # never ends on an assistant turn
    assert parse_seed_text("user: a\nsystem: b\nuser: c")[0] is None


async def test_model_form_applies_to_the_draft(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("model_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ModelConfigScreen)
        form = screen.query_one(ModelConfigForm)
        form.query_one("#model-id", Input).value = "tiny-local-model"
        form.query_one("#model-revision", Input).value = "abc123"
        form.query_one("#model-dtype", Select).value = "float32"
        form.query_one("#model-attn", Select).value = "eager"
        form.query_one("#model-generic-template", Checkbox).value = True
        assert form.apply(app.state.draft) == []
        model = app.state.draft.model
        assert model.model_id == "tiny-local-model"
        assert model.revision == "abc123"
        assert model.torch_dtype == "float32"
        assert model.attn_implementation == "eager"
        assert model.allow_generic_chat_template is True

        form.query_one("#model-id", Input).value = "   "
        assert form.apply(app.state.draft) == ["model id must not be empty"]


async def test_model_form_reports_token_presence_never_the_value(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    app.env.secrets.update({"HF_TOKEN": "hf_not-a-real-token-abcdefghijklmnop"})
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("model_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, ModelConfigScreen)
        state = static_text(screen.query_one("#model-token-state"))
        assert "present" in state
        assert "hf_not-a-real-token" not in state


async def test_character_form_adds_and_deletes_seed_conversations(
    app_paths, store, bus, run_config
):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("character")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, CharacterScreen)
        form = screen.query_one(CharacterForm)
        before = len(form.seeds)
        assert form.add_seed_from_text("user: who are you?\nassistant: I am the test one.") == ""
        assert len(form.seeds) == before + 1
        assert form.seeds[-1].is_valid()

        assert form.add_seed_from_text("assistant: no user turn") != ""
        assert len(form.seeds) == before + 1

        assert form.remove_seed(before) is True
        assert len(form.seeds) == before
        assert form.remove_seed(99) is False

        assert form.add_constraint("Always be terse.") is True
        assert form.constraints == ["Never mention being an AI.", "Always be terse."]
        assert form.add_constraint("   ") is False


async def test_character_form_updates_the_selected_seed(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("character")
        await pilot.pause()
        form = app.screen.query_one(CharacterForm)
        assert form.add_seed_from_text("user: first\nassistant: ok") == ""
        await pilot.pause()
        form.query_one("#char-seeds", ListView).index = 0
        assert form.update_selected_seed("user: replaced\nassistant: done", label="renamed") == ""
        assert form.seeds[0].label == "renamed"
        assert form.seeds[0].user_turns() == ["replaced"]
        assert form.update_selected_seed("not a conversation") != ""
        assert form.seeds[0].user_turns() == ["replaced"]


async def test_character_form_apply_roundtrip(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("character")
        await pilot.pause()
        form = app.screen.query_one(CharacterForm)
        form.add_seed_from_text("user: hi\nassistant: hello", label="greeting")
        assert form.apply(app.state.draft) == []
        character = app.state.draft.character
        assert character.seed_conversations[-1].label == "greeting"
        assert "Never mention being an AI." in character.behavioral_constraints

        form.query_one("#char-system", TextArea).text = ""
        problems = form.apply(app.state.draft)
        assert any("system prompt is empty" in problem for problem in problems)


async def test_rubric_form_edits_criteria_and_thresholds(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("rubric")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, RubricScreen)
        form = screen.query_one(RubricForm)

        assert form.select_criterion(1) is True
        form.query_one("#rubric-criterion-name", Input).value = "Voice quality"
        form.query_one("#rubric-criterion-levels", TextArea).text = "bad\nok\ngreat"
        form.query_one("#rubric-criterion-weight", Input).value = "2.0"
        assert form.apply_criterion() is True
        assert form.criteria[1].name == "Voice quality"
        assert form.criteria[1].levels == ["bad", "ok", "great"]
        assert form.criteria[1].weight == 2.0

        form.set_thresholds(tie_epsilon=0.5, min_confidence=0.7)
        assert form.apply(app.state.draft) == []
        assert app.state.draft.rubric.tie_epsilon == 0.5
        assert app.state.draft.rubric.min_confidence_for_preference == 0.7

        form.query_one("#rubric-tie-epsilon", Input).value = "not-a-number"
        assert any("tie_epsilon" in problem for problem in form.apply(app.state.draft))


async def test_rubric_form_defaults_button_restores_six_criteria(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("rubric")
        await pilot.pause()
        form = app.screen.query_one(RubricForm)
        form.rubric.criteria.pop()
        form.reload_defaults()
        assert len(form.criteria) == 6


async def test_generation_form_validates_numbers(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=run_config)
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("generation")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, GenerationScreen)
        form = screen.query_one(GenerationForm)
        form.query_one("#gen-teacher", Input).value = "teacher/model"
        form.query_one("#gen-judge", Input).value = "typesafe/jev-1.13"
        form.query_one("#gen-source", Select).value = "mixed"
        form.query_one("#gen-temperature", Input).value = "warm"
        problems = form.apply(app.state.draft)
        assert any("temperature" in problem for problem in problems)

        form.query_one("#gen-temperature", Input).value = "0.7"
        form.query_one("#gen-prompts", Input).value = "20"
        assert form.apply(app.state.draft) == []
        assert app.state.draft.generation.teacher_model == "teacher/model"
        assert app.state.draft.generation.candidate_source == "mixed"
        assert app.state.draft.generation.temperature == 0.7

        # the split guard still fires when the counts cannot produce a held-out set
        form.query_one("#gen-prompts", Input).value = "12"
        problems = form.apply(app.state.draft)
        assert any("min_eval_prompts" in problem for problem in problems)
