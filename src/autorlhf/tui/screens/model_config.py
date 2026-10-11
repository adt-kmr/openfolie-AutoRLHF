"""Hugging Face model screen: identity, dtype, attention, token presence, metadata."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Button, Footer, Header, Static

from autorlhf.tui.format import integer
from autorlhf.tui.forms import ModelConfigForm
from autorlhf.tui.screens.base import AutorlhfScreen

METADATA_NOTE = (
    "Metadata is read from the checkpoint's config.json only; weights are never downloaded here."
)


class ModelConfigScreen(AutorlhfScreen):
    """Edits the draft run's model configuration."""

    BINDINGS = [Binding("ctrl+s", "apply", "Apply")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Model configuration", id="model-title", classes="screen-title")
        yield VerticalScroll(ModelConfigForm(self.state.draft))
        yield Footer()

    def on_mount(self) -> None:
        self.query_one(ModelConfigForm).set_token_present(bool(self.state.env.hf_token()))

    def action_apply(self) -> None:
        problems = self.query_one(ModelConfigForm).apply(self.state.draft)
        if not problems:
            self.autorlhf.notify("Model settings applied to the draft run", title="Model")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "model-apply":
            self.action_apply()
        elif event.button.id == "model-probe":
            self.probe_metadata()

    def probe_metadata(self) -> None:
        """Report config metadata and a parameter count without fetching weights."""

        form = self.query_one(ModelConfigForm)
        form.apply(self.state.draft)
        model = self.state.draft.model
        lines = [METADATA_NOTE]
        token = self.state.env.hf_token()
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model.model_id,
                revision=model.revision,
                token=token,
                trust_remote_code=model.trust_remote_code,
            )
            lines.append(
                "AutoConfig: "
                f"model_type={getattr(config, 'model_type', 'unknown')} "
                f"layers={getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', '?'))} "
                f"hidden={getattr(config, 'hidden_size', getattr(config, 'n_embd', '?'))} "
                f"vocab={getattr(config, 'vocab_size', '?')}"
            )
        except Exception as exc:  # noqa: BLE001 - a bad model id must be reported, not raised
            lines.append(f"AutoConfig: could not be read ({exc})")

        try:
            from autorlhf.training.models import model_param_count_from_metadata

            count = model_param_count_from_metadata(model, hf_token=token)
        except ImportError:
            lines.append("parameter count: unavailable (training module not installed)")
        except Exception as exc:  # noqa: BLE001
            lines.append(f"parameter count: could not be derived ({exc})")
        else:
            lines.append(
                "parameter count (from metadata): "
                + (integer(count) if count is not None else "unknown")
            )
        form.show_metadata("\n".join(lines))


__all__ = ["METADATA_NOTE", "ModelConfigScreen"]
