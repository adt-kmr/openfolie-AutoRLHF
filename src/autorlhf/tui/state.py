"""Mutable application state shared by every screen.

Holds the user's *draft* run configuration (the one the setup/settings screens
edit), the active run identity, and the handles the pipeline worker needs.
Keeping it in one object means a screen never has to know how the app was
constructed and tests can drive the app with a prebuilt state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from autorlhf.config.character import DEFAULT_CHARACTER, CharacterSpec
from autorlhf.config.rubric import rubric_for_character
from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import FeedbackMode
from autorlhf.core.events import EventBus
from autorlhf.core.ids import new_run_id, run_folder_name
from autorlhf.storage.db import RunRecord, RunStore
from autorlhf.storage.runs import RunPaths, load_config, save_config


@dataclass
class AppState:
    """Everything the screens and the worker read and mutate."""

    env: Environment
    store: RunStore
    bus: EventBus
    #: The configuration the setup screens edit before a run is created.
    draft: RunConfig
    interactive: bool = False
    verbose: bool = False
    #: Identity of the run currently on screen (None before anything is started).
    run_id: str | None = None
    paths: RunPaths | None = None
    config: RunConfig | None = None
    #: Free-form notes the dashboard shows under the run table.
    notes: list[str] = field(default_factory=list)

    @property
    def settings(self):
        return self.env.settings

    def relabel(self) -> None:
        """Keep the terminal title in sync with the active run."""

        slug = (self.config or self.draft).slug
        self.draft.slug = slug


def default_draft(
    env: Environment,
    *,
    interactive: bool = False,
    character: CharacterSpec | None = None,
) -> RunConfig:
    """A valid starting configuration seeded from saved settings.

    The shipped demo character is used only as a scaffold: the character, rubric
    and every number are editable before a run is created.
    """

    settings = env.settings
    spec = (character or DEFAULT_CHARACTER).model_copy(deep=True)
    config = RunConfig(
        slug="run",
        feedback_mode=FeedbackMode.INTERACTIVE if interactive else FeedbackMode.AUTOMATED,
        character=spec,
        rubric=rubric_for_character(spec),
        verbose=settings.verbose,
    )
    config.model.model_id = settings.default_hf_model or config.model.model_id
    config.generation.teacher_model = (
        settings.default_teacher_model or config.generation.teacher_model
    )
    config.generation.judge_model = settings.default_judge_model or config.generation.judge_model
    config.generation.max_concurrency = settings.max_concurrency
    config.generation.request_timeout_s = settings.request_timeout_s
    config.generation.max_retries = settings.max_retries
    config.wandb_enabled = settings.wandb_enabled
    config.wandb_project = settings.wandb_project
    config.weave_enabled = settings.weave_enabled
    config.weave_project = settings.weave_project
    config.weave_consent = settings.weave_consent
    return config


def create_run(state: AppState, config: RunConfig) -> tuple[str, RunPaths]:
    """Create a brand new run: SQLite row, run directory, pinned config."""

    run_id = new_run_id()
    folder = run_folder_name(run_id, config.slug or config.character.name)
    paths = RunPaths(state.env.paths.run_dir(folder)).ensure()
    record = state.store.create_run(
        config,
        run_id=run_id,
        folder=folder,
        config_path=paths.config_path,
        notes=f"created from the {'interactive' if state.interactive else 'automated'} UI",
    )
    save_config(paths, config)
    return record.id, paths


def load_run(state: AppState, run_id: str) -> tuple[RunRecord, RunPaths, RunConfig]:
    """Resolve a stored run's record, artifact paths and pinned configuration."""

    record = state.store.require_run(run_id)
    paths = RunPaths(state.env.paths.run_dir(record.folder))
    config = load_config(paths)
    return record, paths, config


__all__ = ["AppState", "create_run", "default_draft", "load_run"]
