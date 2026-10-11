"""Rubric form: the criteria the judge scores, their weights, and the thresholds."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Checkbox, Input, Label, ListItem, ListView, Static, TextArea

from autorlhf.config.rubric import Criterion, Rubric, default_rubric
from autorlhf.config.run_config import RunConfig


class RubricForm(Vertical):
    """Edits :class:`~autorlhf.config.rubric.Rubric` in place.

    The edited criterion is written back into the in-memory rubric when
    "Apply criterion" is pressed, so the levels and weights survive moving
    between criteria.
    """

    def __init__(self, config: RunConfig | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="form")
        self._config = config
        self._rubric: Rubric = default_rubric()
        self._selected: int = 0

    def compose(self) -> ComposeResult:
        yield Label("Rubric", classes="form-heading")
        with Horizontal(id="rubric-body"):
            with Vertical(classes="col"):
                yield Label("Criteria")
                yield ListView(id="rubric-criteria")
                with Horizontal(classes="row"):
                    yield Button("Reload defaults", id="rubric-defaults")
            with Vertical(classes="col grow"):
                yield Label("Criterion name")
                yield Input(id="rubric-criterion-name")
                yield Label("Description handed to the judge")
                yield TextArea("", id="rubric-criterion-description")
                yield Label("Levels, lowest first (one per line)")
                yield TextArea("", id="rubric-criterion-levels")
                with Horizontal(classes="row"):
                    with Vertical(classes="col"):
                        yield Label("Weight")
                        yield Input(id="rubric-criterion-weight")
                    yield Checkbox(
                        "Counts toward the overall score", id="rubric-criterion-included"
                    )
                    yield Button("Apply criterion", id="rubric-criterion-apply", variant="primary")
        with Horizontal(classes="row"):
            with Vertical(classes="col"):
                yield Label("Tie epsilon (score gap below which it is a tie)")
                yield Input(id="rubric-tie-epsilon")
            with Vertical(classes="col"):
                yield Label("Minimum judge confidence for a preference")
                yield Input(id="rubric-min-confidence")
        yield Button("Apply rubric", id="rubric-apply", variant="primary")
        yield Static("", id="rubric-problems", classes="problems")

    def on_mount(self) -> None:
        if self._config is not None:
            self.load(self._config)

    # -- loading ----------------------------------------------------------- #

    def load(self, config: RunConfig) -> None:
        self._rubric = config.rubric.model_copy(deep=True)
        self._refresh_criteria()
        self.query_one("#rubric-tie-epsilon", Input).value = f"{self._rubric.tie_epsilon:g}"
        self.query_one(
            "#rubric-min-confidence", Input
        ).value = f"{self._rubric.min_confidence_for_preference:g}"
        self.select_criterion(0)

    @property
    def rubric(self) -> Rubric:
        return self._rubric

    @property
    def criteria(self) -> list[Criterion]:
        return list(self._rubric.criteria)

    # -- editing ----------------------------------------------------------- #

    def select_criterion(self, index: int) -> bool:
        """Show criterion ``index`` in the editors (writing back the previous one)."""

        if not 0 <= index < len(self._rubric.criteria):
            return False
        self.apply_criterion()
        self._selected = index
        criterion = self._rubric.criteria[index]
        self.query_one("#rubric-criterion-name", Input).value = criterion.name
        self.query_one("#rubric-criterion-description", TextArea).text = criterion.description
        self.query_one("#rubric-criterion-levels", TextArea).text = "\n".join(criterion.levels)
        self.query_one("#rubric-criterion-weight", Input).value = f"{criterion.weight:g}"
        self.query_one("#rubric-criterion-included", Checkbox).value = criterion.include_in_overall
        view = self.query_one("#rubric-criteria", ListView)
        view.index = index
        return True

    def apply_criterion(self) -> bool:
        """Write the editors back into the selected criterion."""

        if not 0 <= self._selected < len(self._rubric.criteria):
            return False
        criterion = self._rubric.criteria[self._selected]
        name = self.query_one("#rubric-criterion-name", Input).value.strip()
        description = self.query_one("#rubric-criterion-description", TextArea).text.strip()
        levels = [
            line.strip()
            for line in self.query_one("#rubric-criterion-levels", TextArea).text.splitlines()
            if line.strip()
        ]
        weight_text = self.query_one("#rubric-criterion-weight", Input).value.strip()
        if not name:
            self._show_problems(["criterion name must not be empty"])
            return False
        if len(levels) < 2:
            self._show_problems(["a criterion needs at least two ordered levels"])
            return False
        try:
            weight = float(weight_text)
        except ValueError:
            self._show_problems([f"weight '{weight_text}' is not a number"])
            return False
        if weight <= 0:
            self._show_problems(["weight must be greater than zero"])
            return False
        self._rubric.criteria[self._selected] = criterion.model_copy(
            update={
                "name": name,
                "description": description,
                "levels": levels,
                "weight": weight,
                "include_in_overall": self.query_one("#rubric-criterion-included", Checkbox).value,
            }
        )
        self._refresh_criteria()
        self._show_problems([])
        return True

    def set_thresholds(self, *, tie_epsilon: float, min_confidence: float) -> None:
        self.query_one("#rubric-tie-epsilon", Input).value = f"{tie_epsilon:g}"
        self.query_one("#rubric-min-confidence", Input).value = f"{min_confidence:g}"

    def reload_defaults(self) -> None:
        self._rubric = default_rubric()
        self._refresh_criteria()
        self.select_criterion(0)

    # -- apply ------------------------------------------------------------- #

    def apply(self, config: RunConfig) -> list[str]:
        self.apply_criterion()
        problems: list[str] = []
        for field_id, attribute in (
            ("#rubric-tie-epsilon", "tie_epsilon"),
            ("#rubric-min-confidence", "min_confidence_for_preference"),
        ):
            raw = self.query_one(field_id, Input).value.strip()
            try:
                setattr(self._rubric, attribute, float(raw))
            except ValueError:
                problems.append(f"{attribute} '{raw}' is not a number")
        problems.extend(self._rubric.validate_rubric())
        if not problems:
            config.rubric = self._rubric.model_copy(deep=True)
        self._show_problems(problems)
        return problems

    # -- rendering --------------------------------------------------------- #

    def _refresh_criteria(self) -> None:
        view = self.query_one("#rubric-criteria", ListView)
        view.clear()
        for criterion in self._rubric.criteria:
            view.append(ListItem(Label(f"{criterion.name}  [weight {criterion.weight:g}]")))

    def _show_problems(self, problems: list[str]) -> None:
        self.query_one("#rubric-problems", Static).update(
            "\n".join(f"- {problem}" for problem in problems)
        )

    # -- events ------------------------------------------------------------ #

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "rubric-criteria":
            return
        index = event.list_view.index
        if index is None or index == self._selected:
            return
        self.apply_criterion()
        self._selected = index
        criterion = self._rubric.criteria[index]
        self.query_one("#rubric-criterion-name", Input).value = criterion.name
        self.query_one("#rubric-criterion-description", TextArea).text = criterion.description
        self.query_one("#rubric-criterion-levels", TextArea).text = "\n".join(criterion.levels)
        self.query_one("#rubric-criterion-weight", Input).value = f"{criterion.weight:g}"
        self.query_one("#rubric-criterion-included", Checkbox).value = criterion.include_in_overall

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "rubric-defaults":
            self.reload_defaults()
        elif event.button.id == "rubric-criterion-apply":
            self.apply_criterion()


__all__ = ["RubricForm"]
