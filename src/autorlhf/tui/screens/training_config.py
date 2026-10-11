"""Training configuration: hardware, presets, the judge's ranking, manual overrides."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, Select, Static

from autorlhf.config.presets import PresetFeasibility, feasible_presets, recommend_preset
from autorlhf.core.enums import TrainingMethod, TrainingObjective
from autorlhf.core.types import HardwareInfo
from autorlhf.tui.format import integer, probability_bar
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import KeyValuePanel

PRESET_COLUMNS = ("Preset", "Method", "Feasible", "Recommended", "Est. VRAM", "Notes")

#: ``(attribute, label, cast)`` — every numeric hyperparameter the screen edits.
HYPERPARAMETER_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("learning_rate", "Learning rate", "float"),
    ("num_train_epochs", "Epochs", "float"),
    ("max_steps", "Max steps (-1 uses epochs)", "int"),
    ("per_device_train_batch_size", "Per-device batch", "int"),
    ("gradient_accumulation_steps", "Gradient accumulation", "int"),
    ("max_seq_length", "Max sequence length", "int"),
    ("lora_r", "LoRA rank", "int"),
    ("lora_alpha", "LoRA alpha", "int"),
    ("lora_dropout", "LoRA dropout", "float"),
    ("save_steps", "Save every N steps", "int"),
    ("eval_steps", "Eval every N steps", "int"),
    ("logging_steps", "Log every N steps", "int"),
    ("dpo_beta", "DPO beta", "float"),
)

METHOD_OPTIONS = [
    ("LoRA", TrainingMethod.LORA.value),
    ("QLoRA (4-bit)", TrainingMethod.QLORA.value),
    ("Full fine-tuning", TrainingMethod.FFT.value),
]
OBJECTIVE_OPTIONS = [
    ("DPO (preference pairs)", TrainingObjective.DPO.value),
    ("SFT (supervised only)", TrainingObjective.SFT.value),
]

RANKING_DISCLAIMER = (
    "The judge's numbers are its preference among the listed options. They are not a "
    "probability that training will succeed."
)


class TrainingConfigScreen(AutorlhfScreen):
    """Chooses hyperparameters against the measured hardware, then starts the run."""

    BINDINGS = [
        Binding("ctrl+s", "manual_apply", "Apply values"),
        Binding("ctrl+r", "reload_hardware", "Re-detect hardware"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._hardware: HardwareInfo | None = None
        self._feasibilities: list[PresetFeasibility] = []
        self._selected_preset_id: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Training configuration", id="training-config-title", classes="screen-title")
        with Horizontal(id="training-config-body"):
            with VerticalScroll(id="training-left"):
                yield Label("Hardware (measured)")
                yield KeyValuePanel(id="hw-summary")
                yield Label("Presets, feasible first")
                yield DataTable(id="preset-table", cursor_type="row")
                yield Button("Apply selected preset", id="preset-apply", variant="primary")
                yield Static("", id="preset-detail", classes="meta")
                yield Label("Judge ranking")
                yield Static("", id="jev-ranking", classes="meta")
            with VerticalScroll(id="training-right"):
                yield Label("Objective and method")
                with Horizontal(classes="row"):
                    yield Select(OBJECTIVE_OPTIONS, id="hp-objective", allow_blank=False)
                    yield Select(METHOD_OPTIONS, id="hp-method", allow_blank=False)
                yield Label("Resolved hyperparameters")
                yield KeyValuePanel(id="hp-resolved")
                yield Label("Manual overrides")
                for attribute, label, _cast in HYPERPARAMETER_FIELDS:
                    with Horizontal(classes="row"):
                        yield Label(label, classes="field-label")
                        yield Input(id=f"hp-{attribute}")
                yield Label("LoRA target modules (comma separated)")
                yield Input(id="hp-target-modules")
                with Horizontal(classes="row"):
                    yield Button("Apply manual values", id="hp-apply")
                    yield Button("Validate", id="hp-validate")
                    yield Button("Start run", id="hp-start", variant="success")
                yield Static("", id="training-problems", classes="problems")
        yield Footer()

    # -- lifecycle --------------------------------------------------------- #

    def on_mount(self) -> None:
        table = self.query_one("#preset-table", DataTable)
        table.add_columns(*PRESET_COLUMNS)
        self.action_reload_hardware()

    def action_reload_hardware(self) -> None:
        hardware = self.autorlhf.detect_hardware()
        self._hardware = hardware
        training = self.state.draft.training
        param_count = self.autorlhf.draft_param_count()
        self._feasibilities = feasible_presets(
            hardware, objective=training.objective, param_count=param_count
        )
        self._render_hardware(hardware)
        self._render_presets(hardware)
        self.refresh_values()
        self._render_ranking(hardware)

    # -- rendering --------------------------------------------------------- #

    def _render_hardware(self, hardware: HardwareInfo) -> None:
        vram = (
            "n/a (no accelerator)"
            if hardware.total_vram_mib is None
            else (f"{hardware.total_vram_mib / 1024:.1f} GiB (measured)")
        )
        ram = (
            "unknown"
            if hardware.total_ram_mib is None
            else (f"{hardware.total_ram_mib / 1024:.1f} GiB (measured)")
        )
        rows = [
            (
                "device",
                f"{hardware.device_type} x{hardware.device_count} {hardware.device_name}".strip(),
            ),
            ("VRAM total", vram),
            ("RAM total", ram),
            ("CPUs", integer(hardware.cpu_count)),
            ("bf16 / fp16", f"{hardware.bf16_supported} / {hardware.fp16_supported}"),
            ("bitsandbytes", str(hardware.bitsandbytes_available)),
            ("torch", hardware.torch_version or "not installed"),
        ]
        if hardware.estimate_notes:
            rows.append(("notes", "; ".join(hardware.estimate_notes)))
        for warning in hardware.warnings:
            rows.append(("warning", warning))
        self.query_one("#hw-summary", KeyValuePanel).set_rows(rows)

    def _render_presets(self, hardware: HardwareInfo) -> None:
        table = self.query_one("#preset-table", DataTable)
        table.clear()
        training = self.state.draft.training
        recommended = recommend_preset(
            hardware,
            objective=training.objective,
            param_count=self.autorlhf.draft_param_count(),
            preferred=training.preset_id,
        )
        recommended_id = recommended.preset.id.value if recommended else None
        for feasibility in self._feasibilities:
            self._add_preset_row(table, feasibility, recommendation_id=recommended_id)
        if self._feasibilities:
            first = self._feasibilities[0]
            self._selected_preset_id = first.preset.id.value
            if table.row_count:
                table.move_cursor(row=0)

    def _add_preset_row(
        self,
        table: DataTable,
        feasibility: PresetFeasibility,
        *,
        recommendation_id: str | None = None,
    ) -> None:
        preset = feasibility.preset
        estimate = feasibility.estimate
        vram = "-" if estimate is None else f"{estimate.required_vram_mib / 1024:.1f} GiB"
        if feasibility.feasible:
            note = "tight fit" if feasibility.tight else "ok"
        else:
            note = feasibility.reason
        table.add_row(
            preset.label,
            preset.method.value,
            "yes" if feasibility.feasible else "no",
            "yes" if preset.id.value == recommendation_id else "",
            vram,
            note,
            key=preset.id.value,
        )

    def refresh_values(self) -> None:
        training = self.state.draft.training
        self.query_one("#hp-objective", Select).value = training.objective.value
        self.query_one("#hp-method", Select).value = training.method.value
        resolved = [
            ("objective/method", f"{training.objective.value}/{training.method.value}"),
            ("preset", training.preset_id or "(manual)"),
            ("learning rate", f"{training.learning_rate:g}"),
            ("epochs", f"{training.num_train_epochs:g}"),
            ("effective batch", str(training.effective_batch_size())),
            ("max seq length", str(training.max_seq_length)),
            (
                "LoRA r/alpha/dropout",
                f"{training.lora_r}/{training.lora_alpha}/{training.lora_dropout:g}",
            ),
            ("target modules", ", ".join(training.lora_target_modules) or "(discovered at load)"),
            ("optim", training.optim),
            (
                "save/eval/log steps",
                f"{training.save_steps}/{training.eval_steps}/{training.logging_steps}",
            ),
            ("SFT warm-up", "yes" if training.sft_warmup else "no"),
        ]
        if training.objective is TrainingObjective.DPO:
            resolved.append(
                ("DPO beta / loss", f"{training.dpo_beta:g} / {training.dpo_loss_type}")
            )
        if training.method is TrainingMethod.QLORA:
            resolved.append(
                (
                    "4-bit",
                    f"{training.bnb_4bit_quant_type}, double quant={training.bnb_4bit_use_double_quant}",
                )
            )
        self.query_one("#hp-resolved", KeyValuePanel).set_rows(resolved)
        self._fill_manual_inputs()

    def _fill_manual_inputs(self) -> None:
        training = self.state.draft.training
        for attribute, _label, _cast in HYPERPARAMETER_FIELDS:
            current = getattr(training, attribute)
            self.query_one(f"#hp-{attribute}", Input).value = (
                f"{current:g}" if isinstance(current, float) else str(current)
            )
        self.query_one("#hp-target-modules", Input).value = ", ".join(training.lora_target_modules)

    def _render_ranking(self, hardware: HardwareInfo) -> None:
        target = self.query_one("#jev-ranking", Static)
        ranking = self.autorlhf.plan_ranking()
        lines: list[str] = []
        if ranking:
            lines.append("Judge ranking from the last plan:")
            probabilities = ranking.get("probabilities") or {}
            lines.append(probability_bar(probabilities))
            confidence = ranking.get("confidence")
            if confidence is not None:
                lines.append(f"confidence {confidence:.2f}")
            if ranking.get("chosen"):
                lines.append(f"chosen: {ranking['chosen']}")
            if ranking.get("model_build"):
                lines.append(f"judge build: {ranking['model_build']}")
            lines.append(RANKING_DISCLAIMER)
        else:
            recommended = recommend_preset(
                hardware,
                objective=self.state.draft.training.objective,
                param_count=self.autorlhf.draft_param_count(),
                preferred=self.state.draft.training.preset_id,
            )
            lines.append("No judge ranking yet: it is produced while the run plans training.")
            lines.extend(self._rationale(hardware, recommended))
            lines.append(RANKING_DISCLAIMER)
        target.update("\n".join(lines))

    def _rationale(
        self, hardware: HardwareInfo, recommended: PresetFeasibility | None
    ) -> list[str]:
        """The app's own, honest explanation for the default (never the judge's)."""

        if recommended is None:
            return ["No preset is runnable on the detected hardware."]
        preset = recommended.preset
        reasons = [f"hardware-only default: {preset.label} ({preset.method.value})."]
        if not hardware.has_accelerator:
            reasons.append("No accelerator was detected, so only CPU-capable presets are offered.")
        if recommended.tight:
            reasons.append(
                "The modelled memory use is close to the measured VRAM; expect slow steps."
            )
        if preset.experimental:
            reasons.append("This preset is marked experimental.")
        if preset.warning:
            reasons.append(preset.warning)
        return reasons

    # -- operations -------------------------------------------------------- #

    def select_preset(self, preset_id: str) -> bool:
        self._selected_preset_id = preset_id
        feasibility = next((f for f in self._feasibilities if f.preset.id.value == preset_id), None)
        if feasibility is None:
            self.query_one("#preset-detail", Static).update(f"unknown preset '{preset_id}'")
            return False
        if not feasibility.feasible:
            self.query_one("#preset-detail", Static).update(
                f"{feasibility.preset.label} cannot run here: {feasibility.reason}"
            )
            return False
        self.state.draft.training = feasibility.preset.to_hyperparameters(self.state.draft.training)
        self.refresh_values()
        self.query_one("#preset-detail", Static).update(
            f"Applied {feasibility.preset.label}: {feasibility.preset.summary}"
        )
        return True

    def apply_manual_values(self) -> list[str]:
        training = self.state.draft.training
        problems: list[str] = []
        objective = self.query_one("#hp-objective", Select).value
        method = self.query_one("#hp-method", Select).value
        if isinstance(objective, str) and objective:
            training.objective = TrainingObjective(objective)
        if isinstance(method, str) and method:
            training.method = TrainingMethod(method)
        for attribute, label, cast in HYPERPARAMETER_FIELDS:
            raw = self.query_one(f"#hp-{attribute}", Input).value.strip()
            try:
                value = float(raw) if cast == "float" else int(raw)
            except ValueError:
                problems.append(f"{label}: '{raw}' is not a number")
                continue
            setattr(training, attribute, value)
        modules = [
            item.strip()
            for item in self.query_one("#hp-target-modules", Input).value.split(",")
            if item.strip()
        ]
        if modules:
            training.lora_target_modules = modules
        if training.method is TrainingMethod.QLORA:
            training.load_in_4bit = True
        problems.extend(training.validate_hyperparameters())
        self._show_problems(problems)
        if not problems:
            training.preset_id = None
            self.refresh_values()
        return problems

    def validate_all(self) -> list[str]:
        problems = self.apply_manual_values()
        problems = list(problems) + list(self.state.draft.validate_config())
        self._show_problems(problems)
        return problems

    def start_run(self) -> bool:
        problems = self.validate_all()
        if problems:
            self.autorlhf.notify(
                f"{len(problems)} problem(s) must be fixed before starting",
                severity="error",
                title="Training configuration",
            )
            return False
        return self.autorlhf.start_run(config=self.state.draft)

    def _show_problems(self, problems: list[str]) -> None:
        self.query_one("#training-problems", Static).update(
            "\n".join(f"- {problem}" for problem in problems)
        )

    # -- events ------------------------------------------------------------ #

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self._selected_preset_id = str(event.row_key.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "preset-apply" and self._selected_preset_id:
            self.select_preset(self._selected_preset_id)
        elif button == "hp-apply":
            self.apply_manual_values()
        elif button == "hp-validate":
            self.validate_all()
        elif button == "hp-start":
            self.start_run()

    def action_manual_apply(self) -> None:
        self.apply_manual_values()


__all__ = [
    "HYPERPARAMETER_FIELDS",
    "METHOD_OPTIONS",
    "OBJECTIVE_OPTIONS",
    "PRESET_COLUMNS",
    "RANKING_DISCLAIMER",
    "TrainingConfigScreen",
]
