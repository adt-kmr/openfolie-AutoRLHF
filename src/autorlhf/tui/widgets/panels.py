"""Reusable presentation widgets.

Everything renders through :class:`rich.text.Text` / :class:`rich.table.Table`
rather than markup strings, so a model id or a response containing square
brackets can never crash the UI by being parsed as markup.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rich.text import Text
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, Label, RichLog, Static

from autorlhf.tui.format import probability_bar

LEVEL_STYLES = {
    "info": "white",
    "debug": "dim",
    "warning": "yellow",
    "error": "bold red",
    "ok": "green",
}


class MetricTile(Static):
    """A labelled number that screens update in place."""

    def __init__(self, label: str, value: str = "-", *, id: str | None = None) -> None:
        super().__init__(id=id, classes="metric-tile")
        self._label = label
        self._value = value
        self._refresh_display()

    def set_value(self, value: str) -> None:
        self._value = value
        self._refresh_display()

    @property
    def value(self) -> str:
        return self._value

    def _refresh_display(self) -> None:
        text = Text()
        text.append(f"{self._label}\n", style="dim")
        text.append(self._value, style="bold")
        self.update(text)


class KeyValuePanel(Static):
    """A borderless two-column table of ``(label, value)`` rows."""

    def __init__(self, *, title: str = "", id: str | None = None) -> None:
        super().__init__(id=id, classes="kv-panel")
        self._title = title
        #: Plain-text mirror of the rendered rows.
        self.rows: list[tuple[str, str]] = []

    def set_rows(self, rows: Iterable[tuple[str, Any]]) -> None:
        self.rows = [(str(label), str(value)) for label, value in rows]
        table = _grid_table()
        for label, value in self.rows:
            table.add_row(Text(label, style="dim"), Text(value))
        self.update(table)

    def set_text(self, text: str) -> None:
        self.rows = []
        self.update(Text(text))

    def text(self) -> str:
        return "\n".join(f"{label}: {value}" for label, value in self.rows)


class ScoreBar(Static):
    """One criterion's judge score plus its probability distribution."""

    def show(
        self,
        name: str,
        *,
        score: float | None,
        scale_max: int | None = None,
        probabilities: Mapping[str, float] | None = None,
        confidence: float | None = None,
        error: str | None = None,
    ) -> None:
        text = Text()
        text.append(f"{name}: ", style="bold")
        if error:
            text.append(f"failed ({error})", style="red")
        elif score is None:
            text.append("n/a", style="dim")
        else:
            shown = f"{score:.2f}"
            if scale_max is not None:
                shown += f"/{scale_max}"
            text.append(shown)
        if confidence is not None:
            text.append(f"  confidence {confidence:.2f}", style="dim")
        text.append("\n")
        text.append(probability_bar(probabilities), style="cyan")
        self.update(text)


class MaskedSecretField(Vertical):
    """A never-echoed secret input with a presence indicator.

    The value is only ever read through :meth:`value`; it is never rendered and
    never written to a log.  ``field_id`` is required so the child input ids stay
    stable for tests and handlers.
    """

    def __init__(self, label: str, *, field_id: str, placeholder: str = "") -> None:
        super().__init__(id=f"{field_id}-field", classes="secret-field")
        self.label_text = label
        self.field_id = field_id
        self.placeholder = placeholder

    def compose(self):
        yield Label(self.label_text)
        yield Input(password=True, placeholder=self.placeholder, id=f"{self.field_id}-input")
        yield Static("(unset)", id=f"{self.field_id}-state")

    @property
    def input(self) -> Input:
        return self.query_one(f"#{self.field_id}-input", Input)

    @property
    def value(self) -> str:
        return self.input.value.strip()

    def clear(self) -> None:
        self.input.value = ""

    def set_present(self, present: bool, *, display: str = "(unset)") -> None:
        indicator = self.query_one(f"#{self.field_id}-state", Static)
        indicator.update(
            Text(display if present else "(unset)", style="green" if present else "dim")
        )


class LogPane(RichLog):
    """A bounded log view; errors and warnings get a colour, nothing gets lost."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: Plain-text mirror of what was rendered, for tests and for `--export`
        #: style summaries.  Lines hidden by the verbosity filter are not kept,
        #: exactly like the rendering.
        self.entries: list[str] = []

    def append(self, message: str, *, level: str = "info", verbose_only: bool = False) -> None:
        text = Text()
        if verbose_only:
            text.append("[debug] ", style="dim")
        text.append(message, style=LEVEL_STYLES.get(level, "white"))
        self.entries.append(message)
        self.write(text)

    def text(self) -> str:
        return "\n".join(self.entries)


class StatStrip(Horizontal):
    """A row of metric tiles that screens can address by key."""

    def add_tile(self, key: str, label: str) -> MetricTile:
        tile = MetricTile(label, "-", id=f"tile-{key}")
        self.mount(tile)
        return tile


def _grid_table():
    from rich.table import Table

    return Table.grid(padding=(0, 2), expand=False)


__all__ = [
    "KeyValuePanel",
    "LogPane",
    "MetricTile",
    "MaskedSecretField",
    "ScoreBar",
    "StatStrip",
]
