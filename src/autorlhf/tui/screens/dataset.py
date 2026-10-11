"""Dataset inspector: table of candidates, validation warnings, and inline edits."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Label, Select, Static, TextArea

from autorlhf.storage.runs import read_jsonl, write_text_atomic
from autorlhf.tui.format import num
from autorlhf.tui.screens.base import AutorlhfScreen

DATASET_COLUMNS = ("Prompt", "Candidate", "Split", "Category", "Status", "Overall", "Response")
ALL = "all"


def dataset_warnings(
    records: list[dict[str, Any]], preferences: list[dict[str, Any]] | None = None
) -> list[str]:
    """Local, always-available validation of the frozen record format.

    The pipeline exposes its own validator; this one is a second pair of eyes and
    runs even before the pipeline package exists.
    """

    problems: list[str] = []
    by_candidate: Counter[tuple[str, str]] = Counter()
    prompt_texts: dict[str, str] = {}
    for record in records:
        candidate_id = str(record.get("candidate_id", ""))
        prompt_id = str(record.get("prompt_id", ""))
        by_candidate[(prompt_id, candidate_id)] += 1
        text = record.get("response") or ""
        if record.get("status") != "failed" and not str(text).strip():
            problems.append(f"{candidate_id}: empty response")
        if record.get("truncated"):
            problems.append(f"{candidate_id}: response was truncated at max_new_tokens")
        if prompt_id in prompt_texts and prompt_texts[prompt_id] != record.get("prompt", ""):
            problems.append(f"{prompt_id}: prompt text differs between candidates")
        prompt_texts[prompt_id] = record.get("prompt", "")
        context = record.get("prompt_context") or []
        roles = [turn.get("role") for turn in context if isinstance(turn, dict)]
        if any(role not in {"system", "user", "assistant"} for role in roles):
            problems.append(f"{candidate_id}: prompt_context has an unknown role")
        for index in range(1, len(roles)):
            if roles[index] == roles[index - 1] and roles[index] != "system":
                problems.append(
                    f"{candidate_id}: prompt_context does not alternate at turn {index}"
                )
                break
        judgement = record.get("judgement")
        if judgement is None and record.get("status") == "accepted":
            problems.append(f"{candidate_id}: accepted but never judged")
    for (prompt_id, candidate_id), count in by_candidate.items():
        if count > 1:
            problems.append(f"{prompt_id}/{candidate_id}: duplicate records ({count})")

    seen_texts: Counter[tuple[str, str]] = Counter()
    for record in records:
        text = (record.get("response") or "").strip()
        if text:
            seen_texts[(str(record.get("prompt_id")), text)] += 1
    for (prompt_id, _text), count in seen_texts.items():
        if count > 1:
            problems.append(f"{prompt_id}: {count} identical candidate responses")

    for preference in preferences or []:
        pair_id = preference.get("pair_id", "?")
        chosen = preference.get("chosen_id")
        rejected = preference.get("rejected_id")
        if not chosen or not rejected:
            problems.append(f"{pair_id}: preference is missing a side")
        elif chosen == rejected:
            problems.append(f"{pair_id}: chosen and rejected are the same candidate")
    return problems


class DatasetScreen(AutorlhfScreen):
    """Inspect, filter and edit the generated dataset."""

    BINDINGS = [
        Binding("r", "reload", "Reload"),
        Binding("v", "validate", "Validate"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, Any]] = []
        self.visible_records: list[dict[str, Any]] = []
        self.selected_index: int | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Dataset", id="dataset-title", classes="screen-title")
        with Horizontal(id="dataset-body"):
            with Vertical(id="dataset-side"):
                yield Label("Filters")
                yield Select(
                    [
                        (ALL, ALL),
                        ("train", "train"),
                        ("validation", "validation"),
                        ("eval", "eval"),
                    ],
                    id="dataset-split-filter",
                    allow_blank=False,
                )
                yield Select(
                    [
                        (ALL, ALL),
                        ("pending", "pending"),
                        ("accepted", "accepted"),
                        ("rejected", "rejected"),
                        ("edited", "edited"),
                        ("failed", "failed"),
                    ],
                    id="dataset-status-filter",
                    allow_blank=False,
                )
                yield Static("", id="dataset-counts", classes="meta")
                yield Label("Validation warnings")
                yield Static("", id="dataset-warnings", classes="problems")
            with Vertical(id="dataset-main"):
                yield DataTable(id="dataset-table", cursor_type="row")
        with VerticalScroll(id="dataset-editor"):
            yield Label("Selected record")
            yield Static("", id="dataset-record")
            yield Label("Response text")
            yield TextArea("", id="dataset-edit-text")
            with Horizontal(classes="row"):
                yield Button("Apply edit", id="dataset-edit", variant="primary")
                yield Button("Accept", id="dataset-accept")
                yield Button("Reject", id="dataset-reject")
                yield Button("Reload", id="dataset-reload")
                yield Button("Validate", id="dataset-validate")
            yield Static("", id="dataset-status", classes="problems")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#dataset-table", DataTable).add_columns(*DATASET_COLUMNS)
        self.action_reload()

    # -- loading ----------------------------------------------------------- #

    @property
    def dataset_path(self):
        paths = self.state.paths
        return paths.dataset_path if paths is not None else None

    def action_reload(self) -> None:
        self.reload()

    def reload(self) -> int:
        path = self.dataset_path
        self.records = read_jsonl(path) if path is not None else []
        self._apply_filters()
        self._render_warnings()
        status = self.query_one("#dataset-status", Static)
        if path is None:
            status.update("No run is open. Select a run on the dashboard first.")
        elif not self.records:
            status.update(f"{path} does not exist yet.")
        else:
            status.update(f"Loaded {len(self.records)} record(s) from {path}")
        return len(self.records)

    def _apply_filters(self) -> None:
        split = str(self.query_one("#dataset-split-filter", Select).value)
        status = str(self.query_one("#dataset-status-filter", Select).value)
        self.visible_records = [
            record
            for record in self.records
            if (split == ALL or record.get("split") == split)
            and (status == ALL or record.get("status") == status)
        ]
        table = self.query_one("#dataset-table", DataTable)
        table.clear()
        for index, record in enumerate(self.visible_records):
            table.add_row(
                str(record.get("prompt_id", "")),
                str(record.get("candidate_id", "")),
                str(record.get("split", "")),
                str(record.get("category", "")),
                str(record.get("status", "")),
                num(record.get("overall"), digits=2),
                str(record.get("response") or "")[:80].replace("\n", " "),
                key=str(index),
            )
        counts = Counter(str(r.get("split", "?")) for r in self.records)
        by_status = Counter(str(r.get("status", "?")) for r in self.records)
        self.query_one("#dataset-counts", Static).update(
            f"visible {len(self.visible_records)}/{len(self.records)} | splits {dict(counts)} | "
            f"status {dict(by_status)}"
        )

    def _render_warnings(self) -> None:
        preferences = []
        paths = self.state.paths
        if paths is not None:
            preferences = read_jsonl(paths.preferences_path)
        problems = dataset_warnings(self.records, preferences)
        try:
            from autorlhf.pipeline.dataset import validate_dataset

            extra = validate_dataset(self.records)
            problems = list(problems) + [str(item) for item in extra]
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 - the local checker already ran
            problems.append(f"pipeline validator failed: {exc}")
        target = self.query_one("#dataset-warnings", Static)
        target.update("\n".join(f"- {p}" for p in problems) if problems else "No warnings.")

    # -- selection --------------------------------------------------------- #

    def select_index(self, index: int) -> bool:
        if not 0 <= index < len(self.visible_records):
            return False
        self.selected_index = index
        record = self.visible_records[index]
        lines = [
            f"{record.get('prompt_id')}/{record.get('candidate_id')} "
            f"[{record.get('status')}] split={record.get('split')}",
            f"prompt: {record.get('prompt')}",
            f"overall: {record.get('overall')} truncated={record.get('truncated')}",
            f"source: {record.get('source')} model: {record.get('model_id')}",
        ]
        self.query_one("#dataset-record", Static).update("\n".join(lines))
        self.query_one("#dataset-edit-text", TextArea).text = str(record.get("response") or "")
        return True

    @property
    def selected_record(self) -> dict[str, Any] | None:
        if self.selected_index is None or not 0 <= self.selected_index < len(self.visible_records):
            return None
        return self.visible_records[self.selected_index]

    # -- edits ------------------------------------------------------------- #

    def apply_edit(self, text: str | None = None) -> bool:
        record = self.selected_record
        if record is None:
            self.query_one("#dataset-status", Static).update("Select a record first.")
            return False
        new_text = self.query_one("#dataset-edit-text", TextArea).text if text is None else text
        original = str(record.get("response") or "")
        if new_text == original:
            self.query_one("#dataset-status", Static).update("Nothing to change.")
            return False
        record["response"] = new_text
        record["status"] = "edited"
        record["edited"] = True
        edits = list(record.get("edits") or [])
        edits.append({"field": "response", "from": original, "to": new_text, "by": "human"})
        record["edits"] = edits
        self._write_back()
        self.query_one("#dataset-status", Static).update("Edit saved to dataset.jsonl")
        self.select_index(self.selected_index or 0)
        return True

    def set_status(self, status: str) -> bool:
        record = self.selected_record
        if record is None:
            self.query_one("#dataset-status", Static).update("Select a record first.")
            return False
        record["status"] = status
        self._write_back()
        self.query_one("#dataset-status", Static).update(f"Marked {status}")
        self.select_index(self.selected_index or 0)
        return True

    def _write_back(self) -> None:
        path = self.dataset_path
        if path is None:
            return
        payload = "".join(
            json.dumps(record, default=str, sort_keys=True) + "\n" for record in self.records
        )
        write_text_atomic(path, payload)
        self._apply_filters()
        self._render_warnings()

    # -- events ------------------------------------------------------------ #

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        try:
            index = int(str(event.row_key.value))
        except ValueError:
            return
        self.select_index(index)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id in {"dataset-split-filter", "dataset-status-filter"}:
            self._apply_filters()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "dataset-edit":
            self.apply_edit()
        elif button == "dataset-accept":
            self.set_status("accepted")
        elif button == "dataset-reject":
            self.set_status("rejected")
        elif button == "dataset-reload":
            self.reload()
        elif button == "dataset-validate":
            self._render_warnings()


__all__ = ["DATASET_COLUMNS", "DatasetScreen", "dataset_warnings"]
