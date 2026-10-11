"""Settings screen: credentials and the optional tracking toggles, after setup."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Label, Static

from autorlhf.tui.forms import TrackingForm
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import MaskedSecretField

SECRET_FIELDS = (
    ("settings-openrouter", "OPENROUTER_API_KEY", "OpenRouter API key"),
    ("settings-hf", "HF_TOKEN", "Hugging Face token (optional)"),
    ("settings-wandb", "WANDB_API_KEY", "W&B API key (optional)"),
)


class SettingsScreen(AutorlhfScreen):
    """Edits credentials and tracking without redoing the whole first-run flow."""

    BINDINGS = [Binding("ctrl+s", "save_credentials", "Save credentials")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Settings", id="settings-title", classes="screen-title")
        with VerticalScroll(id="settings-body"):
            yield Label("Credentials")
            for field_id, _key, label in SECRET_FIELDS:
                yield MaskedSecretField(label, field_id=field_id)
            yield Horizontal(
                Button("Save credentials", id="settings-save", variant="primary"),
                Button("Verify OpenRouter key", id="settings-verify"),
                Button("Show credential status", id="settings-describe"),
                id="settings-buttons",
            )
            yield Static("", id="settings-secret-state", classes="meta")
            yield TrackingForm(self.state.draft)
            yield Horizontal(
                Button("Apply tracking", id="settings-tracking"),
                Button("Persist as defaults", id="settings-persist"),
                Button("Run doctor", id="settings-doctor"),
                id="settings-tracking-buttons",
            )
            yield Static("", id="settings-status", classes="problems")
        yield Footer()

    def on_mount(self) -> None:
        env = self.state.env
        for field_id, key, _label in SECRET_FIELDS:
            present = bool(env.secrets.get(key))
            self.query_one(f"#{field_id}-field", MaskedSecretField).set_present(
                present, display="stored (value hidden)"
            )

    # -- credentials ------------------------------------------------------- #

    def save_credentials(self) -> int:
        values: dict[str, str] = {}
        for field_id, key, _label in SECRET_FIELDS:
            typed = self.query_one(f"#{field_id}-field", MaskedSecretField).value
            if typed:
                values[key] = typed
        if values:
            self.state.env.secrets.update(values)
            self.state.env.refresh()
            for field_id, _key, _label in SECRET_FIELDS:
                self.query_one(f"#{field_id}-field", MaskedSecretField).clear()
            self._status("Stored: " + ", ".join(values) + " (values are never displayed)")
        else:
            self._status("Nothing typed; existing credentials left untouched.")
        return len(values)

    def action_save_credentials(self) -> None:
        self.save_credentials()

    def describe_credentials(self) -> str:
        rows = self.state.env.secrets.describe()
        text = "\n".join(
            f"{row['key']}: {row['display']}"
            for row in rows
            if "API_KEY" in row["key"] or "TOKEN" in row["key"]
        )
        self.query_one("#settings-secret-state", Static).update(text or "(none)")
        return text

    def _verify_key(self) -> None:
        key = self.query_one("#settings-openrouter-field", MaskedSecretField).value
        if not key and not self.state.env.openrouter_key():
            self._status("Enter an OpenRouter key (or store one) before verifying.")
            return
        try:
            from autorlhf.llm.client import OpenRouterClient
        except ImportError as exc:
            self._status(f"could not import the OpenRouter client: {exc}")
            return
        client = OpenRouterClient(str(key or self.state.env.openrouter_key()))
        try:
            ok, message = client.verify_credentials()
        finally:
            client.close()
        self._status(("Verified: " if ok else "Rejected: ") + message)

    # -- tracking ---------------------------------------------------------- #

    def apply_tracking(self) -> list[str]:
        problems = self.query_one(TrackingForm).apply(self.state.draft)
        settings = self.state.env.settings
        settings.wandb_enabled = self.state.draft.wandb_enabled
        settings.wandb_project = self.state.draft.wandb_project
        settings.weave_enabled = self.state.draft.weave_enabled
        settings.weave_project = self.state.draft.weave_project
        settings.weave_consent = self.state.draft.weave_consent
        if not problems:
            self.state.env.save_settings()
            self._status("Tracking settings saved.")
        else:
            self._status("\n".join(f"- {problem}" for problem in problems))
        return problems

    def persist_defaults(self) -> None:
        env = self.state.env
        settings = env.settings
        draft = self.state.draft
        settings.default_hf_model = draft.model.model_id
        settings.default_teacher_model = draft.generation.teacher_model
        settings.default_judge_model = draft.generation.judge_model
        settings.max_concurrency = draft.generation.max_concurrency
        settings.request_timeout_s = draft.generation.request_timeout_s
        settings.max_retries = draft.generation.max_retries
        settings.verbose = draft.verbose
        settings.wandb_enabled = draft.wandb_enabled
        settings.wandb_project = draft.wandb_project
        settings.weave_enabled = draft.weave_enabled
        settings.weave_project = draft.weave_project
        settings.weave_consent = draft.weave_consent
        settings.default_feedback_mode = draft.feedback_mode.value
        env.save_settings()
        self._status("Saved as the defaults for new runs.")

    def run_doctor(self) -> str:
        try:
            from autorlhf.doctor import run_doctor
        except ImportError as exc:
            self._status(f"the doctor module is unavailable: {exc}")
            return ""
        try:
            report = run_doctor(env=self.state.env)
        except Exception as exc:  # noqa: BLE001 - run_doctor should not raise
            self._status(f"doctor failed: {exc}")
            return ""
        text = report.render()
        self._status(text)
        return text

    def _status(self, message: str) -> None:
        self.query_one("#settings-status", Static).update(message)

    # -- events ------------------------------------------------------------ #

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "settings-save":
            self.save_credentials()
        elif button == "settings-verify":
            self._verify_key()
        elif button == "settings-describe":
            self.describe_credentials()
        elif button == "settings-tracking":
            self.apply_tracking()
        elif button == "settings-persist":
            self.persist_defaults()
        elif button == "settings-doctor":
            self.run_doctor()


__all__ = ["SECRET_FIELDS", "SettingsScreen"]
