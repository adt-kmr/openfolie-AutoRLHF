"""First-run flow: credentials, model, character, rubric, generation, tracking."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Footer, Header, Static, TabbedContent, TabPane

from autorlhf.config.character import DEFAULT_CHARACTER, CharacterSpec
from autorlhf.tui.forms import (
    CharacterForm,
    GenerationForm,
    ModelConfigForm,
    RubricForm,
    TrackingForm,
)
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import MaskedSecretField

SETUP_INTRO = (
    "Welcome to autorlhf. Nothing is sent anywhere until you start a run: fill these "
    "tabs in, then finish setup. Secrets are stored in your config.env with 0600 "
    "permissions and are never written to a run directory."
)


class SetupScreen(AutorlhfScreen):
    """Guided configuration.  Reuses the same forms as the edit screens."""

    BINDINGS = [
        Binding("ctrl+s", "finish", "Finish"),
        Binding("ctrl+v", "validate", "Validate"),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("First-run setup", id="setup-title")
        yield Static(SETUP_INTRO, id="setup-intro", classes="note")
        with TabbedContent(initial="tab-credentials"):
            with TabPane("1. Credentials", id="tab-credentials"):
                yield VerticalScroll(
                    MaskedSecretField(
                        "OpenRouter API key (required for generation and judging)",
                        field_id="setup-openrouter",
                        placeholder="sk-or-...",
                    ),
                    MaskedSecretField(
                        "Hugging Face token (optional, for gated checkpoints)",
                        field_id="setup-hf",
                    ),
                    Horizontal(
                        Button("Save credentials", id="setup-save-credentials", variant="primary"),
                        Button("Verify OpenRouter key", id="setup-verify"),
                        id="setup-credential-buttons",
                    ),
                    Static("", id="setup-credential-state"),
                    Static(
                        "A project-local .env is read if present but never written; the "
                        "canonical file is config.env under autorlhf's data directory.",
                        classes="note",
                    ),
                )
            with TabPane("2. Model", id="tab-model"):
                yield VerticalScroll(ModelConfigForm(self.state.draft))
            with TabPane("3. Character", id="tab-character"):
                yield VerticalScroll(CharacterForm(self.state.draft))
            with TabPane("4. Rubric", id="tab-rubric"):
                yield VerticalScroll(RubricForm(self.state.draft))
            with TabPane("5. Generation", id="tab-generation"):
                yield VerticalScroll(GenerationForm(self.state.draft))
            with TabPane("6. Tracking", id="tab-tracking"):
                yield VerticalScroll(TrackingForm(self.state.draft))
        with Horizontal(id="setup-actions"):
            yield Button("Validate", id="setup-validate")
            yield Button("Load demo character", id="setup-demo")
            yield Button("Finish setup", id="setup-finish", variant="primary")
        yield Static("", id="setup-status")
        yield Footer()

    def on_mount(self) -> None:
        env = self.state.env
        self.query_one("#setup-openrouter-field", MaskedSecretField).set_present(
            bool(env.openrouter_key()),
            display=f"stored ({len(env.openrouter_key() or '')} chars)",
        )
        self.query_one("#setup-hf-field", MaskedSecretField).set_present(
            bool(env.hf_token()),
            display=f"stored ({len(env.hf_token() or '')} chars)",
        )

    # -- collecting -------------------------------------------------------- #

    def collect(self) -> list[str]:
        """Apply every form into the draft and return all validation problems."""

        draft = self.state.draft
        problems: list[str] = []
        model_form = self.query_one(ModelConfigForm)
        problems.extend(model_form.apply(draft))
        problems.extend(self.query_one(CharacterForm).apply(draft))
        problems.extend(self.query_one(RubricForm).apply(draft))
        problems.extend(self.query_one(GenerationForm).apply(draft))
        problems.extend(self.query_one(TrackingForm).apply(draft))
        problems.extend(draft.validate_config())
        self.query_one("#setup-status", Static).update(
            "\n".join(f"- {problem}" for problem in problems) or "Configuration is valid."
        )
        return problems

    def save_credentials(self) -> int:
        """Persist any newly typed credentials; returns how many were written."""

        values: dict[str, str] = {}
        key = self.query_one("#setup-openrouter-field", MaskedSecretField).value
        token = self.query_one("#setup-hf-field", MaskedSecretField).value
        if key:
            values["OPENROUTER_API_KEY"] = key
        if token:
            values["HF_TOKEN"] = token
        if values:
            self.state.env.secrets.update(values)
            self.state.env.refresh()
            for field in ("setup-openrouter", "setup-hf"):
                self.query_one(f"#{field}-field", MaskedSecretField).clear()
            self.query_one("#setup-credential-state", Static).update(
                "Stored: " + ", ".join(values) + " (values are not displayed again)"
            )
        return len(values)

    def apply_demo_character(self) -> None:
        clone = _config_with_character(self.state.draft, DEFAULT_CHARACTER)
        self.query_one(CharacterForm).load(clone)
        self.query_one(RubricForm).load(clone)
        self.query_one("#setup-status", Static).update("Loaded the demo character and rubric.")

    # -- actions ----------------------------------------------------------- #

    def action_validate(self) -> None:
        self.collect()

    def action_finish(self) -> None:
        problem_count = len(self.collect())
        if problem_count:
            self.autorlhf.notify(
                f"{problem_count} configuration problem(s) must be fixed first",
                severity="error",
                title="Setup",
            )
            return
        self.save_credentials()
        self.autorlhf.complete_setup()
        self.autorlhf.switch_screen("dashboard")

    # -- events ------------------------------------------------------------ #

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "setup-save-credentials":
            self.save_credentials()
        elif button == "setup-verify":
            self._verify_key()
        elif button == "setup-validate":
            self.collect()
        elif button == "setup-demo":
            self.apply_demo_character()
        elif button == "setup-finish":
            self.action_finish()

    def _verify_key(self) -> None:
        target = self.query_one("#setup-credential-state", Static)
        candidate = self.query_one("#setup-openrouter-field", MaskedSecretField).value
        if not candidate and not self.state.env.openrouter_key():
            target.update("Enter an OpenRouter key (or store one) before verifying.")
            return
        try:
            from autorlhf.llm.client import OpenRouterClient
        except ImportError as exc:
            target.update(f"could not import the OpenRouter client: {exc}")
            return
        key = candidate or self.state.env.openrouter_key()
        client = OpenRouterClient(str(key))
        try:
            ok, message = client.verify_credentials()
        finally:
            client.close()
        target.update(("Key verified: " if ok else "Key rejected: ") + message)


def _config_with_character(config, character: CharacterSpec):
    """Return a copy of ``config`` whose character is ``character``."""

    from autorlhf.config.rubric import rubric_for_character

    clone = config.model_copy(deep=True)
    clone.character = character.model_copy(deep=True)
    clone.rubric = rubric_for_character(clone.character)
    return clone


__all__ = ["SETUP_INTRO", "SetupScreen"]
