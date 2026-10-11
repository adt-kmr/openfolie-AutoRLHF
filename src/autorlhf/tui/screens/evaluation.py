"""Base-vs-fine-tuned comparison screen."""

from __future__ import annotations

import json
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, DataTable, Footer, Header, Label, ListItem, ListView, Static

from autorlhf.tui.format import num
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import KeyValuePanel

CRITERION_COLUMNS = ("Criterion", "Base", "Fine-tuned", "Change", "Base n", "Tuned n")


def as_dict(report: Any) -> dict[str, Any]:
    """Accept a ``ComparisonReport``, a plain dict, or JSON text."""

    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    if isinstance(report, str):
        try:
            loaded = json.loads(report)
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}
    dump = getattr(report, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return {}


def example_rows(example: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for label, key in (
        ("prompt", "prompt"),
        ("base", "base_text"),
        ("fine-tuned", "tuned_text"),
    ):
        if example.get(key):
            rows.append((label, str(example[key])))
    base_judgement = example.get("base_judgement") or example.get("base_score")
    tuned_judgement = example.get("tuned_judgement") or example.get("tuned_score")
    if base_judgement is not None:
        rows.append(("base judgement", _describe_judgement(base_judgement)))
    if tuned_judgement is not None:
        rows.append(("tuned judgement", _describe_judgement(tuned_judgement)))
    if not rows:
        rows.append(("example", json.dumps(example, default=str)[:400]))
    return rows


def _describe_judgement(judgement: Any) -> str:
    if isinstance(judgement, dict):
        overall = judgement.get("overall")
        error = judgement.get("error")
        parts = [f"overall={overall}"]
        if error:
            parts.append(f"error={error}")
        return " ".join(parts)
    return str(judgement)


class EvaluationScreen(AutorlhfScreen):
    """Renders ``comparison.json`` / a ``ComparisonReport``."""

    BINDINGS = [
        Binding("r", "reload", "Reload"),
        Binding("t", "open_training", "Training"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.report: dict[str, Any] = {}
        self.examples: list[dict[str, Any]] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Evaluation", id="evaluation-title", classes="screen-title")
        with Horizontal(id="evaluation-body"):
            with Vertical(id="evaluation-main"):
                yield Label("Per-criterion means (scored responses only)")
                yield DataTable(id="eval-criteria", cursor_type="row")
                yield Label("Summary")
                yield KeyValuePanel(id="eval-summary")
                yield Label("Warnings")
                yield Static("", id="eval-warnings", classes="problems")
            with Vertical(id="evaluation-examples"):
                yield Label("Representative pairs")
                yield ListView(id="eval-example-list")
                yield VerticalScroll(Static("", id="eval-example-detail"))
                yield Button("Reload comparison.json", id="eval-reload")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#eval-criteria", DataTable).add_columns(*CRITERION_COLUMNS)
        self.action_reload()

    # -- loading ----------------------------------------------------------- #

    def action_reload(self) -> None:
        paths = self.state.paths
        if paths is None:
            self.show_report({})
            self.query_one("#eval-warnings", Static).update(
                "No run is open. Select a run on the dashboard first."
            )
            return
        if not paths.comparison_path.is_file():
            self.show_report({})
            self.query_one("#eval-warnings", Static).update(
                f"{paths.comparison_path} does not exist yet."
            )
            return
        self.show_report(json.loads(paths.comparison_path.read_text(encoding="utf-8")))

    def action_open_training(self) -> None:
        self.autorlhf.push_screen("training")

    def show_report(self, report: Any) -> dict[str, Any]:
        self.report = as_dict(report)
        table = self.query_one("#eval-criteria", DataTable)
        table.clear()
        for criterion in self.report.get("criteria") or []:
            delta = criterion.get("delta")
            table.add_row(
                str(criterion.get("criterion_name") or criterion.get("criterion_id")),
                num(criterion.get("base_mean"), digits=2),
                num(criterion.get("finetuned_mean"), digits=2),
                "n/a" if delta is None else f"{delta:+.2f}",
                str(criterion.get("base_n", 0)),
                str(criterion.get("finetuned_n", 0)),
                key=str(criterion.get("criterion_id")),
            )

        summary = [
            ("prompts", str(self.report.get("n_prompts", 0))),
            ("scored base", str(self.report.get("n_scored_base", 0))),
            ("scored fine-tuned", str(self.report.get("n_scored_finetuned", 0))),
            ("base overall", num(self.report.get("base_overall"), digits=2)),
            ("fine-tuned overall", num(self.report.get("finetuned_overall"), digits=2)),
            ("overall change", num(self.report.get("overall_delta"), digits=2)),
            (
                "pairwise base / tuned / tie",
                f"{self.report.get('pairwise_base_wins', 0)} / "
                f"{self.report.get('pairwise_finetuned_wins', 0)} / "
                f"{self.report.get('pairwise_ties', 0)}",
            ),
            ("inconclusive", str(self.report.get("inconclusive", False))),
        ]
        self.query_one("#eval-summary", KeyValuePanel).set_rows(summary)

        warnings = list(self.report.get("warnings") or []) + list(self.report.get("notes") or [])
        target = self.query_one("#eval-warnings", Static)
        target.update(
            "\n".join(f"- {warning}" for warning in warnings) if warnings else "No warnings."
        )

        self.examples = [dict(example) for example in self.report.get("examples") or []]
        view = self.query_one("#eval-example-list", ListView)
        view.clear()
        for index, example in enumerate(self.examples):
            prompt = str(example.get("prompt", f"example {index + 1}"))[:70]
            view.append(ListItem(Label(prompt)))
        if self.examples:
            self.select_example(0)
        else:
            self.query_one("#eval-example-detail", Static).update("No examples in the report.")
        return self.report

    def select_example(self, index: int) -> bool:
        if not 0 <= index < len(self.examples):
            return False
        rows = example_rows(self.examples[index])
        self.query_one("#eval-example-detail", Static).update(
            "\n\n".join(f"{label}:\n{value}" for label, value in rows)
        )
        return True

    # -- events ------------------------------------------------------------ #

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "eval-example-list" or event.item is None:
            return
        if event.list_view.index is not None:
            self.select_example(event.list_view.index)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "eval-reload":
            self.action_reload()


__all__ = ["CRITERION_COLUMNS", "EvaluationScreen", "as_dict", "example_rows"]
