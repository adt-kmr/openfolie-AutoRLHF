"""Modal dialogs used by the setup and settings flows."""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static

#: The explicit data-egress notice; Weave is never enabled before this is shown.
WEAVE_NOTICE = (
    "Weave tracing sends prompts, generated responses and Jev judgements to an "
    "external service. Nothing is uploaded unless you enable it here, and the "
    "notice is recorded so it is not silently re-enabled later."
)


class ConfirmModal(ModalScreen[bool]):
    """A yes/no dialog that dismisses with the chosen answer."""

    def __init__(
        self, message: str, *, title: str = "Confirm", confirm: str = "Yes", cancel: str = "No"
    ) -> None:
        super().__init__()
        self.message = message
        self.heading = title
        self.confirm_label = confirm
        self.cancel_label = cancel

    def compose(self):
        with Vertical(id="confirm-modal"):
            yield Static(self.heading, id="confirm-title")
            yield Static(self.message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button(self.confirm_label, id="confirm-yes", variant="primary")
                yield Button(self.cancel_label, id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")


class NoticeModal(ModalScreen[bool]):
    """A blocking notice with a single acknowledgement button."""

    def __init__(self, message: str, *, title: str = "Notice") -> None:
        super().__init__()
        self.message = message
        self.heading = title

    def compose(self):
        with Vertical(id="notice-modal"):
            yield Static(self.heading, id="notice-title")
            yield Static(self.message, id="notice-message")
            yield Button("Acknowledge", id="notice-ok", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(True)


__all__ = ["ConfirmModal", "NoticeModal", "WEAVE_NOTICE"]
