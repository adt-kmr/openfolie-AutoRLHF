"""Training configuration: hardware panel, presets, manual overrides, start."""

from __future__ import annotations

from test_tui_support import make_app, running, static_text
from textual.widgets import DataTable, Input, Select

from autorlhf.config.presets import PresetId
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.tui.screens.training_config import RANKING_DISCLAIMER, TrainingConfigScreen
from autorlhf.tui.widgets import KeyValuePanel


def _valid_config(run_config):
    config = run_config.model_copy(deep=True)
    config.generation.num_prompts = 25
    return config


async def test_training_config_shows_measured_hardware_and_presets(
    app_paths, store, bus, run_config
):
    app = make_app(app_paths, store=store, bus=bus, config=_valid_config(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingConfigScreen)

        hardware = screen.query_one("#hw-summary", KeyValuePanel).text()
        assert "test-cpu" in hardware
        assert "no accelerator" in hardware

        table = screen.query_one("#preset-table", DataTable)
        assert table.row_count == 5
        first = [str(cell) for cell in table.get_row_at(0)]
        assert first[2] == "yes"  # the CPU preset is the only feasible one here
        assert first[3] == "yes"  # ...and it is the hardware-only recommendation

        infeasible = [
            [str(cell) for cell in table.get_row_at(index)] for index in range(table.row_count)
        ]
        assert any(row[2] == "no" for row in infeasible)
        assert all(row[3] == "" for row in infeasible if row[2] == "no")
        reasons = " ".join(row[5] for row in infeasible if row[2] == "no")
        assert "accelerator" in reasons or "VRAM" in reasons

        ranking = static_text(screen.query_one("#jev-ranking"))
        assert RANKING_DISCLAIMER in ranking
        assert "hardware-only default" in ranking


async def test_training_config_applies_a_feasible_preset_and_refuses_an_infeasible_one(
    app_paths, store, bus, run_config
):
    app = make_app(app_paths, store=store, bus=bus, config=_valid_config(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingConfigScreen)

        assert screen.select_preset(PresetId.CPU_TINY_LORA.value) is True
        training = app.state.draft.training
        assert training.preset_id == PresetId.CPU_TINY_LORA.value
        assert training.lora_r == 8
        assert training.method is TrainingMethod.LORA
        assert training.max_seq_length == 512

        assert screen.select_preset(PresetId.MULTI_GPU_FFT.value) is False
        detail = static_text(screen.query_one("#preset-detail"))
        assert "cannot run here" in detail
        assert training.preset_id == PresetId.CPU_TINY_LORA.value


async def test_training_config_manual_overrides_and_validation(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid_config(run_config))
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingConfigScreen)

        screen.query_one("#hp-objective", Select).value = TrainingObjective.SFT.value
        screen.query_one("#hp-method", Select).value = TrainingMethod.LORA.value
        screen.query_one("#hp-learning_rate", Input).value = "3e-4"
        screen.query_one("#hp-gradient_accumulation_steps", Input).value = "2"
        screen.query_one("#hp-target-modules", Input).value = "c_attn, c_proj"
        assert screen.apply_manual_values() == []

        training = app.state.draft.training
        assert training.objective is TrainingObjective.SFT
        assert training.learning_rate == 3e-4
        assert training.gradient_accumulation_steps == 2
        assert training.lora_target_modules == ["c_attn", "c_proj"]
        assert training.preset_id is None  # a manual edit is no longer a preset

        screen.query_one("#hp-learning_rate", Input).value = "fast"
        problems = screen.apply_manual_values()
        assert any("Learning rate" in problem for problem in problems)


async def test_training_config_starts_a_validated_run(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid_config(run_config))
    started: list[dict] = []
    app.start_run = lambda **kwargs: started.append(kwargs) or True  # type: ignore[method-assign]
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training_config")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, TrainingConfigScreen)
        assert screen.start_run() is True
        assert started and started[0]["config"] is app.state.draft

        # an invalid value blocks the start
        screen.query_one("#hp-max_seq_length", Input).value = "8"
        assert screen.start_run() is False
        assert len(started) == 1


async def test_training_config_reloads_the_hardware_probe(app_paths, store, bus, run_config):
    app = make_app(app_paths, store=store, bus=bus, config=_valid_config(run_config))
    calls = {"n": 0}

    def fake_hardware():
        calls["n"] += 1
        from test_tui_support import CPU_HARDWARE

        return CPU_HARDWARE

    app.detect_hardware = fake_hardware  # type: ignore[method-assign]
    async with running(app) as pilot:
        await pilot.pause()
        app.push_screen("training_config")
        await pilot.pause()
        assert calls["n"] == 1
        app.screen.action_reload_hardware()
        await pilot.pause()
        assert calls["n"] == 2
