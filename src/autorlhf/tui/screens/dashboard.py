"""Run dashboard: the run table plus the incomplete-run recovery actions."""

from __future__ import annotations

import asyncio
import shutil

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Label, ListItem, ListView, Static

from autorlhf.core.enums import RunStatus, can_transition
from autorlhf.storage.db import RunRecord
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets.modals import ConfirmModal

RUN_COLUMNS = ("Run", "Created", "Model", "Objective/Method", "Status", "Step", "Checkpoint")


class DashboardScreen(AutorlhfScreen):
    """Lists every run and offers inspect / resume / restart / abandon."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("n", "new_run", "New run"),
        Binding("i", "inspect", "Inspect"),
        Binding("s", "resume", "Resume"),
        Binding("x", "abandon", "Abandon"),
        Binding("c", "open_settings", "Settings"),
        Binding("g", "start_run", "Start run"),
        Binding("t", "open_training", "Training"),
        Binding("e", "open_evaluation", "Evaluation"),
        Binding("a", "open_dataset", "Dataset"),
        Binding("d", "run_doctor", "Doctor"),
        Binding("q", "app.request_quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.selected_run_id: str | None = None
        self._records: list[RunRecord] = []
        self._incomplete: list[RunRecord] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("autorlhf - runs", id="dashboard-title")
        with Horizontal(id="dashboard-body"):
            with Vertical(id="runs-panel"):
                yield Label("Runs")
                yield DataTable(id="runs-table", cursor_type="row")
            with Vertical(id="incomplete-panel"):
                yield Label("Incomplete runs")
                yield ListView(id="incomplete-list")
                with Horizontal(id="incomplete-buttons"):
                    yield Button("Inspect", id="btn-inspect")
                    yield Button("Resume", id="btn-resume", variant="primary")
                    yield Button("Restart", id="btn-restart")
                    yield Button("Abandon", id="btn-abandon", variant="error")
                yield Static("", id="run-detail")
        yield Static("", id="dashboard-status")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#runs-table", DataTable)
        table.add_columns(*RUN_COLUMNS)
        count = self.refresh_runs()
        if count == 0 and not self.autorlhf.setup_shown:
            self.autorlhf.setup_shown = True
            self.autorlhf.open_setup()

    # -- data -------------------------------------------------------------- #

    def refresh_runs(self) -> int:
        """Reload the run table and the incomplete-run list.  Returns the count."""

        store = self.state.store
        self._records = store.list_runs(limit=200)
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        for record in self._records:
            table.add_row(
                record.id,
                record.created_at,
                record.model_id,
                f"{record.objective.value.upper()}/{record.method.value}",
                record.status.value,
                str(record.step),
                record.last_checkpoint or "-",
                key=record.id,
            )
        if self._records:
            self.selected_run_id = self._records[0].id

        self._incomplete = [
            record
            for record in self._records
            if record.is_resumable and record.status is not RunStatus.CREATED
        ]
        view = self.query_one("#incomplete-list", ListView)
        view.clear()
        for record in self._incomplete:
            view.append(ListItem(Label(f"{record.id}  {record.status.value}  step={record.step}")))

        status = self.query_one("#dashboard-status", Static)
        if not self._records:
            status.update("No runs yet - opening first-run setup.")
        else:
            resumable = len(self._incomplete)
            status.update(
                f"{len(self._records)} run(s); {resumable} incomplete. "
                "n new run  i inspect  s resume  x abandon  c settings  d doctor"
            )
        return len(self._records)

    def _selected_record(self) -> RunRecord | None:
        if not self.selected_run_id:
            return None
        return next((r for r in self._records if r.id == self.selected_run_id), None)

    # -- actions ----------------------------------------------------------- #

    def action_refresh(self) -> None:
        self.refresh_runs()

    def action_new_run(self) -> None:
        self.autorlhf.open_setup()

    def action_open_settings(self) -> None:
        self.autorlhf.push_screen("settings")

    def action_start_run(self) -> None:
        """Configure presets/hyperparameters, then start the run."""

        self.autorlhf.push_screen("training_config")

    def action_open_training(self) -> None:
        record = self._selected_record()
        if record is not None and self.autorlhf.adopt_run(record.id):
            self.autorlhf.push_screen("training")

    def action_open_evaluation(self) -> None:
        record = self._selected_record()
        if record is not None and self.autorlhf.adopt_run(record.id):
            self.autorlhf.push_screen("evaluation")

    def action_open_dataset(self) -> None:
        record = self._selected_record()
        if record is not None and self.autorlhf.adopt_run(record.id):
            self.autorlhf.push_screen("dataset")

    def action_inspect(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        self._show_inspect(record.id)

    def action_resume(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        self._resume(record.id)

    def action_abandon(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        self._confirm_abandon(record.id)

    def action_run_doctor(self) -> None:
        self._show_doctor()

    # -- button/event handlers -------------------------------------------- #

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is not None and event.row_key.value:
            self.selected_run_id = str(event.row_key.value)
            self._show_inspect(self.selected_run_id)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "incomplete-list" or event.item is None:
            return
        index = event.list_view.index
        if index is None or not 0 <= index < len(self._incomplete):
            return
        self.selected_run_id = self._incomplete[index].id
        self._show_inspect(self.selected_run_id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        record = self._selected_record()
        if record is None:
            self.query_one("#run-detail", Static).update("Select a run first.")
            return
        if event.button.id == "btn-inspect":
            self._show_inspect(record.id)
        elif event.button.id == "btn-resume":
            self._resume(record.id)
        elif event.button.id == "btn-restart":
            self._restart(record.id)
        elif event.button.id == "btn-abandon":
            self._confirm_abandon(record.id)

    # -- operations -------------------------------------------------------- #

    def _show_inspect(self, run_id: str) -> None:
        detail = self.query_one("#run-detail", Static)
        try:
            summary = self.state.store.run_summary(run_id)
        except KeyError as exc:
            detail.update(f"run {run_id} is not in the store: {exc}")
            return
        record = summary["run"]
        lines = [
            f"{record['id']} [{record['status']}]",
            f"model: {record['model_id']} @ {record['revision'] or 'main'}",
            f"objective/method: {record['objective']}/{record['method']}",
            f"phase: {record['phase'] or '-'} step: {record['step']}",
            f"last checkpoint: {record['last_checkpoint'] or '-'}",
            f"counters: {summary['counters'] or '{}'}",
            f"reviews: {summary['reviews'] or '{}'} preferences: {summary['preferences']}",
            f"complete checkpoints: {summary['checkpoints']}",
        ]
        if record.get("error"):
            lines.append(f"error: {record['error']}")
        detail.update("\n".join(lines))

    def _resume(self, run_id: str) -> None:
        detail = self.query_one("#run-detail", Static)
        try:
            started = self.autorlhf.resume_run(run_id)
        except Exception as exc:  # noqa: BLE001 - surface, never wedge the UI
            detail.update(f"could not resume {run_id}: {exc}")
            return
        if started:
            self.autorlhf.push_screen("training")
        else:
            detail.update(f"could not start {run_id}; see the notification")

    def _restart(self, run_id: str) -> None:
        """Start a fresh run reusing the abandoned run's pinned configuration."""

        detail = self.query_one("#run-detail", Static)
        try:
            self.autorlhf.restart_run(run_id)
        except Exception as exc:  # noqa: BLE001
            detail.update(f"could not restart {run_id}: {exc}")
            return
        self.refresh_runs()

    def _confirm_abandon(self, run_id: str) -> None:
        try:
            record = self.state.store.require_run(run_id)
        except KeyError as exc:
            self.query_one("#run-detail", Static).update(str(exc))
            return
        if (
            not can_transition(record.status, RunStatus.CANCELLED)
            and record.status is not RunStatus.CANCELLED
        ):
            self.query_one("#run-detail", Static).update(
                f"{run_id} is {record.status.value}; cannot abandon it"
            )
            return
        modal = ConfirmModal(
            f"Permanently delete run {run_id}, including its files and history? This cannot be undone.",
            title="Delete run",
            confirm="Delete run",
            cancel="Keep run",
        )
        self.autorlhf.push_screen(
            modal, lambda confirmed: self._abandon(run_id) if confirmed else None
        )

    def _abandon(self, run_id: str) -> None:
        detail = self.query_one("#run-detail", Static)
        try:
            record = self.state.store.require_run(run_id)
        except KeyError as exc:
            detail.update(str(exc))
            return
        if (
            not can_transition(record.status, RunStatus.CANCELLED)
            and record.status is not RunStatus.CANCELLED
        ):
            detail.update(f"{run_id} is {record.status.value}; cannot abandon it")
            return

        app = self.autorlhf
        worker = app.worker
        thread = app._worker_thread
        active_thread = (
            thread
            if worker is not None
            and worker.run_id == run_id
            and thread is not None
            and thread.is_alive()
            else None
        )
        if active_thread is not None:
            app.request_stop()
        detail.update(f"Deleting run {run_id}…")
        asyncio.create_task(self._finish_abandon(run_id, active_thread))

    async def _finish_abandon(self, run_id: str, thread) -> None:
        detail = self.query_one("#run-detail", Static)
        try:
            if thread is not None:
                await asyncio.to_thread(thread.join)
            record = self.state.store.require_run(run_id)
            run_dir = self.state.env.paths.run_dir(record.folder)
            runs_dir = self.state.env.paths.runs_dir.resolve()
            if run_dir.is_symlink() or run_dir.resolve().parent != runs_dir:
                raise ValueError(f"refusing to delete run directory outside {runs_dir}: {run_dir}")
            if run_dir.exists():
                await asyncio.to_thread(shutil.rmtree, run_dir)
            self.state.store.delete_run(run_id)
        except Exception as exc:  # noqa: BLE001 - keep the UI alive and report failed deletion
            detail.update(f"could not delete {run_id}: {exc}")
            return
        if self.state.run_id == run_id:
            self.state.run_id = None
            self.state.paths = None
            self.state.config = None
        self.refresh_runs()
        detail.update(f"Run {run_id} deleted")

    def _show_doctor(self) -> None:
        detail = self.query_one("#run-detail", Static)
        try:
            from autorlhf.doctor import run_doctor
        except ImportError as exc:
            detail.update(f"the doctor module is unavailable: {exc}")
            return
        try:
            report = run_doctor(env=self.state.env)
        except Exception as exc:  # noqa: BLE001 - run_doctor should never raise, but UI must not die
            detail.update(f"doctor failed to run: {exc}")
            return
        detail.update(report.render())


__all__ = ["DashboardScreen", "RUN_COLUMNS"]
