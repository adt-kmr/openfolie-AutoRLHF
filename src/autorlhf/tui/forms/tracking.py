"""Optional experiment-tracking opt-in (W&B metrics, Weave traces)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Checkbox, Input, Label, Static

from autorlhf.config.run_config import RunConfig
from autorlhf.tui.widgets.modals import WEAVE_NOTICE, ConfirmModal


class TrackingForm(Vertical):
    """Both integrations are off by default and never required.

    Weave is gated behind :data:`WEAVE_NOTICE` and a recorded consent flag, so
    enabling it is always an explicit, informed action.
    """

    def __init__(self, config: RunConfig | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="form")
        self._config = config
        self._weave_consent = False

    def compose(self) -> ComposeResult:
        yield Label("Experiment tracking (optional)", classes="form-heading")
        yield Checkbox("Log training metrics to Weights & Biases", id="track-wandb")
        yield Label("W&B project")
        yield Input(placeholder="autorlhf", id="track-wandb-project")
        yield Checkbox("Trace OpenRouter and Jev calls with Weave", id="track-weave")
        yield Label("Weave project")
        yield Input(placeholder="autorlhf", id="track-weave-project")
        yield Static(WEAVE_NOTICE, id="track-weave-notice", classes="note")
        yield Static("Consent not recorded.", id="track-consent-state")
        yield Button("Apply tracking settings", id="track-apply", variant="primary")
        yield Static("", id="track-problems", classes="problems")

    def on_mount(self) -> None:
        if self._config is not None:
            self.load(self._config)

    @property
    def weave_consent(self) -> bool:
        return self._weave_consent

    def load(self, config: RunConfig) -> None:
        self._weave_consent = bool(config.weave_consent)
        self.query_one("#track-wandb", Checkbox).value = config.wandb_enabled
        self.query_one("#track-wandb-project", Input).value = config.wandb_project
        self.query_one("#track-weave", Checkbox).value = config.weave_enabled
        self.query_one("#track-weave-project", Input).value = config.weave_project
        self._refresh_consent()

    def apply(self, config: RunConfig) -> list[str]:
        problems: list[str] = []
        wandb_on = self.query_one("#track-wandb", Checkbox).value
        weave_on = self.query_one("#track-weave", Checkbox).value
        config.wandb_enabled = wandb_on
        config.wandb_project = self.query_one("#track-wandb-project", Input).value.strip()
        config.weave_enabled = weave_on and self._weave_consent
        config.weave_project = self.query_one("#track-weave-project", Input).value.strip()
        if wandb_on and not config.wandb_project:
            problems.append("W&B is enabled but no project is set")
        if weave_on and not self._weave_consent:
            problems.append("Weave tracing needs explicit consent before it can be enabled")
        config.weave_consent = self._weave_consent
        self._show_problems(problems)
        return problems

    def set_consent(self, consent: bool) -> None:
        self._weave_consent = consent
        if not consent:
            self.query_one("#track-weave", Checkbox).value = False
        self._refresh_consent()

    def _refresh_consent(self) -> None:
        self.query_one("#track-consent-state", Static).update(
            "Data-egress consent recorded." if self._weave_consent else "Consent not recorded."
        )

    def _show_problems(self, problems: list[str]) -> None:
        self.query_one("#track-problems", Static).update(
            "\n".join(f"- {problem}" for problem in problems)
        )

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if event.checkbox.id != "track-weave":
            return
        if not event.value or self._weave_consent:
            return

        def decided(approved: bool | None) -> None:
            self.set_consent(bool(approved))

        self.app.push_screen(ConfirmModal(WEAVE_NOTICE, title="Enable Weave tracing?"), decided)


__all__ = ["TrackingForm"]
