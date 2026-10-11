"""Generation screen: models, counts, splits, concurrency and verbosity."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from autorlhf.tui.forms import GenerationForm
from autorlhf.tui.screens.base import AutorlhfScreen


class GenerationScreen(AutorlhfScreen):
    """Edits the draft run's generation and evaluation settings."""

    BINDINGS = [Binding("ctrl+s", "apply", "Apply")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Generation settings", id="generation-title", classes="screen-title")
        yield VerticalScroll(GenerationForm(self.state.draft))
        yield Footer()

    def action_apply(self) -> None:
        form = self.query_one(GenerationForm)
        problems = form.apply(self.state.draft)
        if not problems:
            self.autorlhf.set_verbose(self.state.draft.verbose)
            self.autorlhf.notify("Generation settings applied to the draft run", title="Generation")
        else:
            self.autorlhf.notify(
                f"{len(problems)} problem(s) in the generation settings",
                severity="error",
                title="Generation",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gen-apply":
            self.action_apply()


__all__ = ["GenerationScreen"]
