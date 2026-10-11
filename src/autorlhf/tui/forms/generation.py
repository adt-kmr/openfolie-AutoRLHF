"""Generation and judging settings form."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, Select, Static

from autorlhf.config.run_config import RunConfig

SOURCE_OPTIONS = [
    ("teacher (OpenRouter)", "teacher"),
    ("local (the Hugging Face model)", "local"),
    ("mixed (teacher + local)", "mixed"),
]


class GenerationForm(Vertical):
    """Edits :class:`~autorlhf.config.run_config.GenerationConfig`."""

    def __init__(self, config: RunConfig | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="form")
        self._config = config

    def compose(self) -> ComposeResult:
        yield Label("Generation and judging", classes="form-heading")
        with Horizontal(classes="row"):
            with Vertical(classes="col grow"):
                yield Label("Teacher model (writes prompts and responses)")
                yield Input(id="gen-teacher")
            with Vertical(classes="col grow"):
                yield Label("Judge (Jev) model")
                yield Input(id="gen-judge")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Prompts")
                yield Input(id="gen-prompts", type="integer")
            with Vertical(classes="col"):
                yield Label("Candidates per prompt")
                yield Input(id="gen-candidates", type="integer")
            with Vertical(classes="col"):
                yield Label("Candidate source")
                yield Select(SOURCE_OPTIONS, id="gen-source", allow_blank=False)
            with Vertical(classes="col"):
                yield Label("Seed")
                yield Input(id="gen-seed", type="integer")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Eval fraction")
                yield Input(id="gen-eval-fraction")
            with Vertical(classes="col"):
                yield Label("Validation fraction")
                yield Input(id="gen-validation-fraction")
            with Vertical(classes="col"):
                yield Label("Edge-case fraction")
                yield Input(id="gen-edge-fraction")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Max new tokens")
                yield Input(id="gen-max-new-tokens", type="integer")
            with Vertical(classes="col"):
                yield Label("Temperature")
                yield Input(id="gen-temperature")
            with Vertical(classes="col"):
                yield Label("Top-p")
                yield Input(id="gen-top-p")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Max concurrency")
                yield Input(id="gen-concurrency", type="integer")
            with Vertical(classes="col"):
                yield Label("Request timeout (s)")
                yield Input(id="gen-timeout")
            with Vertical(classes="col"):
                yield Label("Max retries")
                yield Input(id="gen-retries", type="integer")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Max eval prompts")
                yield Input(id="gen-max-eval", type="integer")
            with Vertical(classes="col"):
                yield Label("Judge repeats per response")
                yield Input(id="gen-judge-repeats", type="integer")
            yield Checkbox("Pairwise base-vs-tuned judgements", id="gen-pairwise")
            yield Checkbox("Include seed prompts in the held-out set", id="gen-seed-prompts")
        yield Checkbox("Verbose UI (full prompts, responses and judgements)", id="gen-verbose")
        yield Button("Apply generation settings", id="gen-apply", variant="primary")
        yield Static("", id="gen-problems", classes="problems")

    def on_mount(self) -> None:
        if self._config is not None:
            self.load(self._config)

    def load(self, config: RunConfig) -> None:
        generation = config.generation
        evaluation = config.evaluation
        self.query_one("#gen-teacher", Input).value = generation.teacher_model
        self.query_one("#gen-judge", Input).value = generation.judge_model
        self.query_one("#gen-prompts", Input).value = str(generation.num_prompts)
        self.query_one("#gen-candidates", Input).value = str(generation.candidates_per_prompt)
        self.query_one("#gen-source", Select).value = generation.candidate_source
        self.query_one("#gen-seed", Input).value = str(generation.seed)
        self.query_one("#gen-eval-fraction", Input).value = f"{generation.eval_fraction:g}"
        self.query_one(
            "#gen-validation-fraction", Input
        ).value = f"{generation.validation_fraction:g}"
        self.query_one("#gen-edge-fraction", Input).value = f"{generation.edge_case_fraction:g}"
        self.query_one("#gen-max-new-tokens", Input).value = str(generation.max_new_tokens)
        self.query_one("#gen-temperature", Input).value = f"{generation.temperature:g}"
        self.query_one("#gen-top-p", Input).value = f"{generation.top_p:g}"
        self.query_one("#gen-concurrency", Input).value = str(generation.max_concurrency)
        self.query_one("#gen-timeout", Input).value = f"{generation.request_timeout_s:g}"
        self.query_one("#gen-retries", Input).value = str(generation.max_retries)
        self.query_one("#gen-max-eval", Input).value = str(evaluation.max_eval_prompts)
        self.query_one("#gen-judge-repeats", Input).value = str(evaluation.judge_repeats)
        self.query_one("#gen-pairwise", Checkbox).value = evaluation.pairwise
        self.query_one("#gen-seed-prompts", Checkbox).value = evaluation.include_seed_prompts
        self.query_one("#gen-verbose", Checkbox).value = config.verbose

    def apply(self, config: RunConfig) -> list[str]:
        generation = config.generation
        evaluation = config.evaluation
        problems: list[str] = []

        def text(field_id: str) -> str:
            return self.query_one(field_id, Input).value.strip()

        def number(field_id: str, name: str, cast):
            raw = text(field_id)
            if not raw:
                problems.append(f"{name} must not be empty")
                return None
            try:
                return cast(raw)
            except ValueError:
                problems.append(f"{name} '{raw}' is not a number")
                return None

        generation.teacher_model = text("#gen-teacher") or generation.teacher_model
        generation.judge_model = text("#gen-judge") or generation.judge_model
        prompts = number("#gen-prompts", "prompts", int)
        candidates = number("#gen-candidates", "candidates per prompt", int)
        if prompts is not None:
            generation.num_prompts = prompts
        if candidates is not None:
            generation.candidates_per_prompt = candidates
        source = self.query_one("#gen-source", Select).value
        if isinstance(source, str) and source:
            generation.candidate_source = source
        seed = number("#gen-seed", "seed", int)
        if seed is not None:
            generation.seed = seed
        for field_id, name, attribute in (
            ("#gen-eval-fraction", "eval fraction", "eval_fraction"),
            ("#gen-validation-fraction", "validation fraction", "validation_fraction"),
            ("#gen-edge-fraction", "edge-case fraction", "edge_case_fraction"),
            ("#gen-temperature", "temperature", "temperature"),
            ("#gen-top-p", "top-p", "top_p"),
        ):
            value = number(field_id, name, float)
            if value is not None:
                setattr(generation, attribute, value)
        for field_id, name, attribute, cast in (
            ("#gen-max-new-tokens", "max new tokens", "max_new_tokens", int),
            ("#gen-concurrency", "max concurrency", "max_concurrency", int),
            ("#gen-timeout", "request timeout", "request_timeout_s", float),
            ("#gen-retries", "max retries", "max_retries", int),
        ):
            value = number(field_id, name, cast)
            if value is not None:
                setattr(generation, attribute, value)
        max_eval = number("#gen-max-eval", "max eval prompts", int)
        if max_eval is not None:
            evaluation.max_eval_prompts = max_eval
        repeats = number("#gen-judge-repeats", "judge repeats", int)
        if repeats is not None:
            evaluation.judge_repeats = repeats
        evaluation.pairwise = self.query_one("#gen-pairwise", Checkbox).value
        evaluation.include_seed_prompts = self.query_one("#gen-seed-prompts", Checkbox).value
        config.verbose = self.query_one("#gen-verbose", Checkbox).value

        if prompts is not None and candidates is not None and not problems:
            problems.extend(generation.validate_generation())
        self._show_problems(problems)
        return problems

    def _show_problems(self, problems: list[str]) -> None:
        self.query_one("#gen-problems", Static).update(
            "\n".join(f"- {problem}" for problem in problems)
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gen-apply" and self._config is not None:
            self.apply(self._config)


__all__ = ["GenerationForm", "SOURCE_OPTIONS"]
