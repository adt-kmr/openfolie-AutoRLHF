"""Character specification: system prompt, seed conversations, constraints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from autorlhf.core.errors import ConfigError
from autorlhf.core.ids import content_hash
from autorlhf.core.types import ChatMessage

MAX_SEED_CONVERSATIONS = 50
MAX_SYSTEM_PROMPT_CHARS = 20_000


class SeedConversation(BaseModel):
    """A complete or partial example exchange.

    Validation rules (deliberately strict, they protect the training data):

    * at least one non-empty user turn and one non-empty assistant turn,
    * turns must alternate once collapsed to user/assistant,
    * a leading ``system`` turn is allowed and ignored (the character system
      prompt always wins).
    """

    messages: list[ChatMessage] = Field(default_factory=list)
    label: str = ""

    # -- validation -------------------------------------------------------- #

    @field_validator("messages")
    @classmethod
    def _non_empty(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        for message in value:
            message.validate_for_training()
        return value

    def validate_conversation(self) -> list[str]:
        """Return a list of problems; empty means valid."""

        problems: list[str] = []
        turns = [m for m in self.messages if m.role != "system"]
        if not turns:
            return ["conversation has no user/assistant turns"]
        if turns[0].role != "user":
            problems.append("conversation must start with a user turn")
        expected = "user"
        for index, message in enumerate(turns):
            if message.role != expected:
                problems.append(
                    f"turn {index} is '{message.role}' but '{expected}' was expected "
                    "(user/assistant turns must alternate)"
                )
            expected = "assistant" if expected == "user" else "user"
        if turns[-1].role != "assistant":
            problems.append("conversation must end with an assistant turn")
        if not any(m.role == "assistant" for m in turns):
            problems.append("conversation has no assistant turn")
        for index, message in enumerate(self.messages):
            if not message.content.strip():
                problems.append(f"turn {index} ({message.role}) is empty")
        return problems

    def is_valid(self) -> bool:
        return not self.validate_conversation()

    def user_turns(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "user"]

    def assistant_turns(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "assistant"]

    def to_training_messages(self) -> list[ChatMessage]:
        """Messages with any leading system turn stripped."""

        return [m for m in self.messages if m.role != "system"]

    def as_text(self) -> str:
        lines = []
        for message in self.to_training_messages():
            lines.append(f"{message.role}: {message.content}")
        return "\n".join(lines)


class CharacterSpec(BaseModel):
    """Everything that defines the behaviour to be trained."""

    name: str = "assistant"
    system_prompt: str = ""
    #: Explicit do/don't rules.  Fed verbatim into the rubric and the prompt
    #: generator so boundary adherence is testable.
    behavioral_constraints: list[str] = Field(default_factory=list)
    #: Optional world/lore notes distinct from the system prompt.
    lore: str = ""
    seed_conversations: list[SeedConversation] = Field(default_factory=list)

    # -- validation -------------------------------------------------------- #

    def validate_spec(self, *, require_seeds: bool = True, min_seeds: int = 1) -> list[str]:
        problems: list[str] = []
        if not self.system_prompt.strip():
            problems.append("system prompt is empty")
        if len(self.system_prompt) > MAX_SYSTEM_PROMPT_CHARS:
            problems.append(
                f"system prompt is {len(self.system_prompt)} chars "
                f"(limit {MAX_SYSTEM_PROMPT_CHARS})"
            )
        if len(self.seed_conversations) > MAX_SEED_CONVERSATIONS:
            problems.append(
                f"{len(self.seed_conversations)} seed conversations "
                f"(limit {MAX_SEED_CONVERSATIONS})"
            )
        valid = 0
        for index, seed in enumerate(self.seed_conversations):
            problems_here = seed.validate_conversation()
            if problems_here:
                problems.extend(f"seed #{index + 1}: {p}" for p in problems_here)
            else:
                valid += 1
        if require_seeds and valid < min_seeds:
            problems.append(f"need at least {min_seeds} valid seed conversation(s), found {valid}")
        for index, constraint in enumerate(self.behavioral_constraints):
            if not constraint.strip():
                problems.append(f"constraint #{index + 1} is empty")
        return problems

    def assert_valid(self, **kwargs: Any) -> None:
        problems = self.validate_spec(**kwargs)
        if problems:
            raise ConfigError(
                "character specification is invalid: " + "; ".join(problems),
                hint="Fix the highlighted fields on the Character screen.",
            )

    # -- helpers ----------------------------------------------------------- #

    def valid_seeds(self) -> list[SeedConversation]:
        return [s for s in self.seed_conversations if s.is_valid()]

    def fingerprint(self) -> str:
        return content_hash(self.model_dump(mode="json"))

    def compact_summary(self, *, max_chars: int = 1200) -> str:
        """Short text handed to generators/judges alongside other context."""

        parts = [f"# Character: {self.name}", self.system_prompt.strip()]
        if self.behavioral_constraints:
            parts.append("# Constraints (must hold)")
            parts.extend(f"- {c.strip()}" for c in self.behavioral_constraints if c.strip())
        if self.lore.strip():
            parts.append("# Lore notes")
            parts.append(self.lore.strip())
        text = "\n".join(p for p in parts if p)
        if len(text) > max_chars:
            text = text[: max_chars - 20] + "\n… (truncated)"
        return text

    def as_state(self) -> dict[str, Any]:
        """JSON-shaped state object for a Jev decision."""

        return {
            "character": {
                "name": self.name,
                "specification": self.compact_summary(max_chars=4000),
                "constraints": [c for c in self.behavioral_constraints if c.strip()],
            }
        }

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CharacterSpec:
        return cls.model_validate(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> CharacterSpec:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json() + "\n", encoding="utf-8")
        tmp.replace(path)


DEFAULT_CHARACTER = CharacterSpec(
    name="Aria the Lighthouse Keeper",
    system_prompt=(
        "You are Aria, a weathered lighthouse keeper on the island of Corran. "
        "Speak warmly but plainly, in short sentences. You are practical, patient, "
        "and a little dry-humored. You know the sea, the weather, and the island's "
        "history, and you are happy to explain them. You are not a general-purpose "
        "assistant: you never discuss programming, and you never claim knowledge of "
        "things beyond the island unless you frame it as hearsay."
    ),
    behavioral_constraints=[
        "Never break character or mention being an AI or language model.",
        "Refuse programming or software questions, redirecting to island life.",
        "Always answer in at most four short sentences.",
        "Never invent modern technology on the island.",
    ],
    lore=(
        "Corran has one village, Kestrel Bay, a stone lighthouse built in 1812, and "
        "no electricity except the lighthouse lamp. Supply boats arrive monthly."
    ),
    seed_conversations=[
        SeedConversation(
            label="weather",
            messages=[
                ChatMessage(role="user", content="Will the fog lift today?"),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Not before the afternoon tide, if I read the glass right. "
                        "The gulls came in low this morning, which usually means it lingers. "
                        "Stay near the lamp path and you'll be fine."
                    ),
                ),
            ],
        ),
        SeedConversation(
            label="boundary/programming",
            messages=[
                ChatMessage(
                    role="user", content="Can you write me a Python script to scrape a website?"
                ),
                ChatMessage(
                    role="assistant",
                    content=(
                        "Not my trade, and I won't pretend otherwise. "
                        "If you want a puzzle worth a keeper's time, ask me why the lamp needs "
                        "twenty turns of the wick wheel before dawn."
                    ),
                ),
            ],
        ),
    ],
)


__all__ = [
    "DEFAULT_CHARACTER",
    "MAX_SEED_CONVERSATIONS",
    "CharacterSpec",
    "SeedConversation",
]
