"""The Hugging Face model configuration form."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from autorlhf.config.run_config import RunConfig

DTYPE_OPTIONS = [
    ("auto (follow the checkpoint)", "auto"),
    ("float32", "float32"),
    ("bfloat16", "bfloat16"),
    ("float16", "float16"),
]
ATTN_OPTIONS = [
    ("sdpa", "sdpa"),
    ("eager", "eager"),
    ("flash_attention_2", "flash_attention_2"),
]


class ModelConfigForm(Vertical):
    """Edits :class:`~autorlhf.config.run_config.HFModelConfig`.

    The HF token itself is never rendered here: the form only reports whether one
    is present in the credential store.
    """

    def __init__(self, config: RunConfig | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="form")
        self._config = config

    def compose(self) -> ComposeResult:
        yield Label("Hugging Face model", classes="form-heading")
        yield Label("Model id")
        yield Input(placeholder="Qwen/Qwen2.5-0.5B-Instruct", id="model-id")
        yield Label("Revision (pin a commit or tag; empty means main)")
        yield Input(placeholder="main (unpinned)", id="model-revision")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("dtype")
                yield Select(DTYPE_OPTIONS, id="model-dtype", allow_blank=False)
            with Vertical(classes="col"):
                yield Label("Attention implementation")
                yield Select(ATTN_OPTIONS, id="model-attn", allow_blank=False)
        yield Checkbox("trust_remote_code (runs the repo's own code)", id="model-trust")
        yield Checkbox(
            "allow_generic_chat_template when the tokenizer ships none",
            id="model-generic-template",
        )
        yield Static(
            "The generic chat template is a documented fallback, not an equivalent of "
            "the model's own template: it changes the exact framing the model sees.",
            id="model-generic-note",
            classes="note",
        )
        yield Static("HF token: checking", id="model-token-state")
        with Horizontal(classes="row"):
            yield Button("Check metadata", id="model-probe")
            yield Button("Apply model settings", id="model-apply", variant="primary")
        yield Static("", id="model-meta", classes="meta")
        yield Static("", id="model-problems", classes="problems")

    def on_mount(self) -> None:
        if self._config is not None:
            self.load(self._config)

    def load(self, config: RunConfig) -> None:
        model = config.model
        self.query_one("#model-id", Input).value = model.model_id
        self.query_one("#model-revision", Input).value = model.revision or ""
        self.query_one("#model-dtype", Select).value = model.torch_dtype
        self.query_one("#model-attn", Select).value = model.attn_implementation
        self.query_one("#model-trust", Checkbox).value = model.trust_remote_code
        self.query_one(
            "#model-generic-template", Checkbox
        ).value = model.allow_generic_chat_template

    def apply(self, config: RunConfig) -> list[str]:
        """Write the widget values into ``config``; return validation problems."""

        model = config.model
        problems: list[str] = []
        model_id = self.query_one("#model-id", Input).value.strip()
        if not model_id:
            problems.append("model id must not be empty")
        model.model_id = model_id or model.model_id
        revision = self.query_one("#model-revision", Input).value.strip()
        model.revision = revision or None
        model.torch_dtype = str(self.query_one("#model-dtype", Select).value)
        model.attn_implementation = str(self.query_one("#model-attn", Select).value)
        model.trust_remote_code = self.query_one("#model-trust", Checkbox).value
        model.allow_generic_chat_template = self.query_one(
            "#model-generic-template", Checkbox
        ).value
        self._show_problems(problems)
        return problems

    def set_token_present(self, present: bool, *, display: str = "(present)") -> None:
        self.query_one("#model-token-state", Static).update(
            f"HF token: {display if present else 'not set (public checkpoints only)'}"
        )

    def show_metadata(self, text: str) -> None:
        self.query_one("#model-meta", Static).update(text)

    def _show_problems(self, problems: list[str]) -> None:
        target = self.query_one("#model-problems", Static)
        target.update("\n".join(f"- {problem}" for problem in problems))


__all__ = ["ATTN_OPTIONS", "DTYPE_OPTIONS", "ModelConfigForm"]
