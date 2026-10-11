"""Live training view driven by the event bus."""

from __future__ import annotations

import time
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Footer, Header, Label, ProgressBar, Static

from autorlhf.tui.format import human_duration, num, pct
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import KeyValuePanel, LogPane, MetricTile

TILE_KEYS = (
    ("phase", "Phase"),
    ("step", "Step"),
    ("loss", "Train loss"),
    ("eval_loss", "Eval loss"),
    ("lr", "Learning rate"),
    ("elapsed", "Elapsed"),
    ("throughput", "Throughput"),
    ("epoch", "Epoch"),
)


class TrainingScreen(AutorlhfScreen):
    """Shows every event the run emits, with pause/resume/stop controls."""

    BINDINGS = [
        Binding("p", "pause", "Pause"),
        Binding("u", "resume", "Resume"),
        Binding("s", "stop", "Stop"),
        Binding("v", "toggle_verbose", "Verbose"),
        Binding("r", "open_review", "Review queue"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.verbose = False
        self.started_at: float | None = None
        self.finished = False
        self.last_step = 0
        self._tiles: dict[str, MetricTile] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Training", id="training-title", classes="screen-title")
        with Horizontal(id="training-tiles"):
            for key, label in TILE_KEYS:
                yield MetricTile(label, "-", id=f"tile-{key}")
        yield ProgressBar(total=100.0, show_eta=False, id="training-progress")
        with Horizontal(id="training-controls"):
            yield Button("Pause", id="training-pause")
            yield Button("Resume", id="training-resume")
            yield Button("Stop gracefully", id="training-stop", variant="error")
            yield Checkbox("Verbose (full prompts, responses, judgments)", id="training-verbose")
            yield Button("Evaluation", id="training-evaluation")
        with Horizontal(id="training-body"):
            with Vertical(id="training-state"):
                yield Label("Run state")
                yield KeyValuePanel(id="training-kv")
                yield Label("Checkpoints")
                yield Static("", id="training-checkpoints")
            with Vertical(id="training-events"):
                yield Label("Samples and judgements")
                yield LogPane(id="training-samples", max_lines=500, markup=False)
                yield Label("Warnings, retries and errors")
                yield LogPane(id="training-warnings", max_lines=300, markup=False)
                yield Label("Activity")
                yield LogPane(id="training-log", max_lines=600, markup=False)
        yield Footer()

    def on_mount(self) -> None:
        self.verbose = self.state.verbose
        self.query_one("#training-verbose", Checkbox).value = self.verbose
        self._render_kv()
        self.autorlhf.replay_events(self.handle_event)

    # -- helpers ----------------------------------------------------------- #

    @property
    def _tile_map(self) -> dict[str, MetricTile]:
        if not self._tiles:
            for key, _label in TILE_KEYS:
                self._tiles[key] = self.query_one(f"#tile-{key}", MetricTile)
        return self._tiles

    def _set_tile(self, key: str, value: str) -> None:
        tile = self._tile_map.get(key)
        if tile is not None:
            tile.set_value(value)

    def _log(
        self, pane_id: str, message: str, *, level: str = "info", verbose_only: bool = False
    ) -> None:
        if verbose_only and not self.verbose:
            return
        self.query_one(f"#{pane_id}", LogPane).append(
            message, level=level, verbose_only=verbose_only
        )

    def _render_kv(self) -> None:
        state = self.state
        worker = self.autorlhf.worker
        rows: list[tuple[str, Any]] = [
            ("run id", state.run_id or "(none)"),
            ("paths", str(state.paths.root) if state.paths else "(none)"),
            ("interactive", str(state.interactive)),
            ("verbose", str(self.verbose)),
        ]
        if state.run_id:
            try:
                counters = state.store.counters(state.run_id)
                if counters:
                    rows.append(("counters", counters))
                record = state.store.get_run(state.run_id)
                if record is not None:
                    rows.append(("status", record.status.value))
                    rows.append(("step", str(record.step)))
                    rows.append(("last checkpoint", record.last_checkpoint or "-"))
            except Exception as exc:  # noqa: BLE001 - state panel is advisory
                rows.append(("state error", str(exc)))
        if worker is not None:
            rows.append(("calls", ", ".join(worker.calls) or "-"))
            rows.append(("paused", str(worker.paused)))
        self.query_one("#training-kv", KeyValuePanel).set_rows(rows)
        self._render_checkpoints()

    def _render_checkpoints(self) -> None:
        target = self.query_one("#training-checkpoints", Static)
        if not self.state.run_id:
            target.update("-")
            return
        try:
            checkpoints = self.state.store.list_checkpoints(self.state.run_id)
        except Exception as exc:  # noqa: BLE001
            target.update(f"could not list checkpoints: {exc}")
            return
        if not checkpoints:
            target.update("None saved yet.")
            return
        target.update(
            "\n".join(
                f"{item['path']} step={item['step']} {'complete' if item['complete'] else 'incomplete'}"
                for item in checkpoints[:5]
            )
        )

    # -- events ------------------------------------------------------------ #

    def handle_event(self, event) -> None:
        if self.started_at is None:
            self.started_at = event.ts
        kind = event.kind
        if kind == "metric":
            self._on_metric(event)
        elif kind == "progress":
            self._on_progress(event)
        elif kind == "phase":
            self._on_phase(event)
        elif kind == "status":
            self._on_status(event)
        elif kind == "sample":
            self._on_sample(event)
        elif kind in {"warning", "error"}:
            self._log(
                "training-warnings",
                event.message or event.kind,
                level="error" if kind == "error" else "warning",
            )
        elif kind == "artifact":
            self._log("training-log", f"artifact: {event.message}", verbose_only=event.verbose_only)
        elif kind == "log":
            self._log("training-log", event.message, verbose_only=event.verbose_only)
        elif kind == "counters":
            self._render_kv()

    def _on_metric(self, event) -> None:
        name = str(event.data.get("name_key") or event.data.get("name") or event.message)
        value = event.data.get("value")
        if value is None:
            value = event.data.get(name)
        step = event.data.get("step")
        if step is not None:
            self._set_tile("step", str(step))
        if name in {"loss", "train_loss"}:
            self._set_tile("loss", num(float(value) if value is not None else None))
        elif name in {"eval_loss", "validation_loss"}:
            self._set_tile("eval_loss", num(float(value) if value is not None else None))
        elif name in {"learning_rate", "lr"}:
            self._set_tile("lr", num(float(value) if value is not None else None, digits=6))
        elif name == "epoch":
            self._set_tile("epoch", num(float(value) if value is not None else None, digits=2))
        elif name in {"tokens_per_second", "samples_per_second", "throughput"}:
            self._set_tile("throughput", num(float(value) if value is not None else None, digits=2))
        elif "memory" in name or "vram" in name:
            self._log("training-log", f"{name}: {value}")
        else:
            self._log("training-log", f"{name}: {value}")
        self._update_elapsed(event.ts)
        self._render_kv()

    def _on_progress(self, event) -> None:
        if event.step is not None:
            self.last_step = int(event.step)
            self._set_tile("step", str(event.step))
        fraction = event.fraction
        if fraction is not None:
            self.query_one("#training-progress", ProgressBar).update(
                progress=max(0.0, min(1.0, fraction)) * 100.0
            )
            self._set_tile("throughput", pct(max(0.0, min(1.0, fraction))))
        if event.message:
            self._log("training-log", event.message)
        self._update_elapsed(event.ts)

    def _on_phase(self, event) -> None:
        phase = event.phase or event.data.get("phase") or event.message
        self._set_tile("phase", str(phase))
        self._log("training-log", event.message or f"phase: {phase}")
        self._render_kv()

    def _on_status(self, event) -> None:
        status = event.data.get("status")
        if status:
            self._set_tile("phase", str(status))
            if status in {"completed", "failed", "cancelled", "interrupted"}:
                self.finished = True
        self._log("training-log", f"status: {event.message}")
        self._render_kv()

    def _on_sample(self, event) -> None:
        item = event.data.get("review_item")
        prompt = event.data.get("prompt") or event.message
        response = event.data.get("response") or event.data.get("text") or ""
        if item:
            self._log("training-samples", f"review item: {item.get('prompt_id', '?')}")
        if prompt:
            self._log("training-samples", f"prompt: {prompt}", verbose_only=True)
        if response:
            self._log("training-samples", f"response: {response}", verbose_only=True)
        if not prompt and not response and not item:
            self._log("training-samples", event.message, verbose_only=event.verbose_only)
        judgement = event.data.get("judgement")
        if judgement:
            self._log(
                "training-samples",
                f"judgement: overall={judgement.get('overall')}",
                verbose_only=True,
            )

    def _update_elapsed(self, ts: float | None) -> None:
        if self.started_at is None:
            return
        elapsed = max(0.0, (ts or time.time()) - self.started_at)
        self._set_tile("elapsed", human_duration(elapsed))

    # -- controls ---------------------------------------------------------- #

    def action_pause(self) -> None:
        self.autorlhf.request_pause()
        self._log("training-log", "pause requested (applies at the next step boundary)")

    def action_resume(self) -> None:
        self.autorlhf.request_resume()
        self._log("training-log", "resume requested")

    def action_stop(self) -> None:
        self.autorlhf.request_stop()
        self._log(
            "training-warnings", "stop requested; finishing the current step", level="warning"
        )

    def action_toggle_verbose(self) -> None:
        checkbox = self.query_one("#training-verbose", Checkbox)
        checkbox.value = not checkbox.value

    def action_open_review(self) -> None:
        """Open the interactive review queue without losing this view."""

        self.autorlhf.push_screen("review")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id == "training-verbose":
            self.verbose = bool(event.value)
            self.autorlhf.set_verbose(self.verbose)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "training-pause":
            self.action_pause()
        elif button == "training-resume":
            self.action_resume()
        elif button == "training-stop":
            self.action_stop()
        elif button == "training-evaluation":
            self.autorlhf.push_screen("evaluation")


__all__ = ["TILE_KEYS", "TrainingScreen"]
