"""Rubric screen: criteria, levels, weights and the preference thresholds."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from autorlhf.tui.forms import RubricForm
from autorlhf.tui.screens.base import AutorlhfScreen


class RubricScreen(AutorlhfScreen):
    """Edits the draft run's rubric."""

    BINDINGS = [Binding("ctrl+s", "apply", "Apply")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Rubric", id="rubric-title", classes="screen-title")
        yield VerticalScroll(RubricForm(self.state.draft))
        yield Footer()

    def action_apply(self) -> None:
        problems = self.query_one(RubricForm).apply(self.state.draft)
        if not problems:
            self.autorlhf.notify("Rubric applied to the draft run", title="Rubric")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rubric-apply":
            self.action_apply()


__all__ = ["RubricScreen"]
