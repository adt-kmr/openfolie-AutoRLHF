"""Common behaviour for every autorlhf screen."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual.binding import Binding
from textual.screen import Screen

from autorlhf.tui.state import AppState

if TYPE_CHECKING:  # pragma: no cover - typing only
    from autorlhf.tui.app import AutorlhfApp


class AutorlhfScreen(Screen):
    """Adds typed access to the app and the shared state, plus safe navigation."""

    BINDINGS = [
        Binding("escape", "go_back", "Back", show=False),
    ]

    @property
    def autorlhf(self) -> AutorlhfApp:
        return cast("AutorlhfApp", self.app)

    @property
    def state(self) -> AppState:
        return self.autorlhf.state

    def action_go_back(self) -> None:
        stack = getattr(self.app, "screen_stack", ())
        if len(stack) > 1:
            self.app.pop_screen()

    def handle_event(self, event) -> None:
        """Hook for bus events; screens that care override it."""

        return None


__all__ = ["AutorlhfScreen"]
