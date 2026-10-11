"""Character screen: prompt, constraints, lore and seed conversations."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from autorlhf.tui.forms import CharacterForm
from autorlhf.tui.screens.base import AutorlhfScreen


class CharacterScreen(AutorlhfScreen):
    """Edits the draft run's character specification."""

    BINDINGS = [Binding("ctrl+s", "apply", "Apply")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Character specification", id="character-title", classes="screen-title")
        yield VerticalScroll(CharacterForm(self.state.draft))
        yield Footer()

    def action_apply(self) -> None:
        problems = self.query_one(CharacterForm).apply(self.state.draft)
        if not problems:
            self.autorlhf.notify("Character applied to the draft run", title="Character")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "char-apply":
            self.action_apply()


__all__ = ["CharacterScreen"]
