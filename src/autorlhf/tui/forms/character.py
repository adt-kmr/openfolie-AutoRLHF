"""Character specification form: prompt, constraints, lore, seed conversations."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, ListItem, ListView, Static, TextArea

from autorlhf.config.character import MAX_SEED_CONVERSATIONS, SeedConversation
from autorlhf.config.run_config import RunConfig
from autorlhf.core.types import ChatMessage

SEED_PLACEHOLDER = "user: Will the fog lift today?\nassistant: Not before the afternoon tide."
ROLES = ("system", "user", "assistant")


def parse_seed_text(text: str) -> tuple[SeedConversation | None, str]:
    """Parse ``role: content`` lines into a seed conversation.

    Returns ``(conversation, "")`` or ``(None, reason)``.  The syntax is
    deliberately literal: a line whose prefix is not a role is an error rather
    than being silently dropped, because a silently dropped turn corrupts the
    training data.
    """

    messages: list[ChatMessage] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            return None, f"line {number} is not 'role: content'"
        role, _, content = line.partition(":")
        role = role.strip().lower()
        content = content.strip()
        if role not in ROLES:
            return None, f"line {number} uses unknown role '{role}'"
        if role == "system":
            continue
        if not content:
            return None, f"line {number} ({role}) is empty"
        messages.append(ChatMessage(role=role, content=content))
    if not messages:
        return None, "no user/assistant turns found"
    seed = SeedConversation(messages=messages)
    problems = seed.validate_conversation()
    if problems:
        return None, "; ".join(problems)
    return seed, ""


class CharacterForm(Vertical):
    """Edits :class:`~autorlhf.config.character.CharacterSpec`."""

    def __init__(self, config: RunConfig | None = None, *, id: str | None = None) -> None:
        super().__init__(id=id, classes="form")
        self._config = config
        self._constraints: list[str] = []
        self._seeds: list[SeedConversation] = []

    def compose(self) -> ComposeResult:
        yield Label("Character", classes="form-heading")
        yield Label("Name")
        yield Input(placeholder="Aria the Lighthouse Keeper", id="char-name")
        yield Label("System prompt")
        yield TextArea("", id="char-system")
        yield Label("Lore notes (world facts the model must not contradict)")
        yield TextArea("", id="char-lore")
        yield Label("Behavioural constraints")
        with Horizontal(classes="row"):
            yield Input(placeholder="Never mention being an AI.", id="char-constraint-input")
            yield Button("Add", id="char-constraint-add")
            yield Button("Remove selected", id="char-constraint-remove")
        yield ListView(id="char-constraints")
        yield Label("Seed conversations ('role: content' per line)")
        yield TextArea(SEED_PLACEHOLDER, id="char-seed-text")
        with Horizontal(classes="row"):
            yield Input(placeholder="label (optional)", id="char-seed-label")
            yield Button("Add conversation", id="char-seed-add")
            yield Button("Update selected", id="char-seed-update")
            yield Button("Remove selected", id="char-seed-remove")
        yield ListView(id="char-seeds")
        yield Button("Apply character", id="char-apply", variant="primary")
        yield Static("", id="char-problems", classes="problems")

    def on_mount(self) -> None:
        if self._config is not None:
            self.load(self._config)

    # -- loading ----------------------------------------------------------- #

    def load(self, config: RunConfig) -> None:
        character = config.character
        self.query_one("#char-name", Input).value = character.name
        self.query_one("#char-system", TextArea).text = character.system_prompt
        self.query_one("#char-lore", TextArea).text = character.lore
        self._constraints = list(character.behavioral_constraints)
        self._seeds = [seed.model_copy(deep=True) for seed in character.seed_conversations]
        self._refresh_constraints()
        self._refresh_seeds()

    @property
    def constraints(self) -> list[str]:
        return list(self._constraints)

    @property
    def seeds(self) -> list[SeedConversation]:
        return list(self._seeds)

    # -- constraints ------------------------------------------------------- #

    def add_constraint(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return False
        self._constraints.append(text)
        self._refresh_constraints()
        return True

    def remove_constraint(self, index: int) -> bool:
        if not 0 <= index < len(self._constraints):
            return False
        del self._constraints[index]
        self._refresh_constraints()
        return True

    def remove_selected_constraint(self) -> bool:
        return self.remove_constraint(self.query_one("#char-constraints", ListView).index or 0)

    # -- seeds ------------------------------------------------------------- #

    def add_seed(self, seed: SeedConversation, *, label: str = "") -> bool:
        if len(self._seeds) >= MAX_SEED_CONVERSATIONS:
            return False
        if not seed.is_valid():
            return False
        if label.strip():
            seed = seed.model_copy(deep=True)
            seed.label = label.strip()
        self._seeds.append(seed)
        self._refresh_seeds()
        return True

    def add_seed_from_text(self, text: str, *, label: str = "") -> str:
        """Add a seed conversation; returns ``""`` on success or the reason."""

        seed, reason = parse_seed_text(text)
        if seed is None:
            return reason
        if not self.add_seed(seed, label=label):
            return f"could not add conversation: {reason or 'invalid conversation'}"
        return ""

    def remove_seed(self, index: int) -> bool:
        if not 0 <= index < len(self._seeds):
            return False
        del self._seeds[index]
        self._refresh_seeds()
        return True

    def remove_selected_seed(self) -> bool:
        return self.remove_seed(self.query_one("#char-seeds", ListView).index or 0)

    def edit_seed(self, index: int, seed: SeedConversation, *, label: str = "") -> bool:
        if not 0 <= index < len(self._seeds) or not seed.is_valid():
            return False
        replacement = seed.model_copy(deep=True)
        if label.strip():
            replacement.label = label.strip()
        self._seeds[index] = replacement
        self._refresh_seeds()
        return True

    def update_selected_seed(self, text: str, *, label: str = "") -> str:
        """Replace the highlighted conversation; returns ``""`` on success."""

        index = self.query_one("#char-seeds", ListView).index
        if index is None or not 0 <= index < len(self._seeds):
            return "select a conversation to update first"
        seed, reason = parse_seed_text(text)
        if seed is None:
            return reason
        if not self.edit_seed(index, seed, label=label):
            return "could not update the selected conversation"
        return ""

    # -- apply ------------------------------------------------------------- #

    def apply(self, config: RunConfig) -> list[str]:
        character = config.character
        character.name = self.query_one("#char-name", Input).value.strip() or character.name
        character.system_prompt = self.query_one("#char-system", TextArea).text
        character.lore = self.query_one("#char-lore", TextArea).text
        character.behavioral_constraints = list(self._constraints)
        character.seed_conversations = [seed.model_copy(deep=True) for seed in self._seeds]
        problems = character.validate_spec()
        self._show_problems(problems)
        return problems

    # -- rendering --------------------------------------------------------- #

    def _refresh_constraints(self) -> None:
        view = self.query_one("#char-constraints", ListView)
        view.clear()
        for text in self._constraints:
            view.append(ListItem(Label(text)))

    def _refresh_seeds(self) -> None:
        view = self.query_one("#char-seeds", ListView)
        view.clear()
        for index, seed in enumerate(self._seeds):
            name = seed.label or f"conversation {index + 1}"
            turns = len(seed.to_training_messages())
            preview = seed.user_turns()[0] if seed.user_turns() else ""
            view.append(ListItem(Label(f"{name} ({turns} turns): {preview[:60]}")))

    def _show_problems(self, problems: list[str]) -> None:
        target = self.query_one("#char-problems", Static)
        target.update("\n".join(f"- {problem}" for problem in problems))

    # -- events ------------------------------------------------------------ #

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        if button == "char-constraint-add":
            field = self.query_one("#char-constraint-input", Input)
            if self.add_constraint(field.value):
                field.value = ""
        elif button == "char-constraint-remove":
            self.remove_selected_constraint()
        elif button == "char-seed-add":
            text = self.query_one("#char-seed-text", TextArea).text
            label = self.query_one("#char-seed-label", Input).value
            reason = self.add_seed_from_text(text, label=label)
            if reason:
                self._show_problems([reason])
            else:
                self._show_problems([])
        elif button == "char-seed-update":
            text = self.query_one("#char-seed-text", TextArea).text
            label = self.query_one("#char-seed-label", Input).value
            reason = self.update_selected_seed(text, label=label)
            self._show_problems([reason] if reason else [])
        elif button == "char-seed-remove":
            self.remove_selected_seed()


__all__ = ["CharacterForm", "ROLES", "SEED_PLACEHOLDER", "parse_seed_text"]
