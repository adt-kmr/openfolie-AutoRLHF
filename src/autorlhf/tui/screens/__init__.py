"""The twelve autorlhf screens."""

from __future__ import annotations

from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.screens.character import CharacterScreen
from autorlhf.tui.screens.dashboard import DashboardScreen
from autorlhf.tui.screens.dataset import DatasetScreen
from autorlhf.tui.screens.evaluation import EvaluationScreen
from autorlhf.tui.screens.generation import GenerationScreen
from autorlhf.tui.screens.model_config import ModelConfigScreen
from autorlhf.tui.screens.review import ReviewScreen
from autorlhf.tui.screens.rubric import RubricScreen
from autorlhf.tui.screens.settings import SettingsScreen
from autorlhf.tui.screens.setup import SetupScreen
from autorlhf.tui.screens.training import TrainingScreen
from autorlhf.tui.screens.training_config import TrainingConfigScreen

#: Screen name -> class, registered on the app so ``push_screen(name)`` works.
SCREEN_CLASSES = {
    "dashboard": DashboardScreen,
    "setup": SetupScreen,
    "model_config": ModelConfigScreen,
    "character": CharacterScreen,
    "rubric": RubricScreen,
    "generation": GenerationScreen,
    "training_config": TrainingConfigScreen,
    "review": ReviewScreen,
    "dataset": DatasetScreen,
    "training": TrainingScreen,
    "evaluation": EvaluationScreen,
    "settings": SettingsScreen,
}

__all__ = [
    "SCREEN_CLASSES",
    "AutorlhfScreen",
    "CharacterScreen",
    "DashboardScreen",
    "DatasetScreen",
    "EvaluationScreen",
    "GenerationScreen",
    "ModelConfigScreen",
    "ReviewScreen",
    "RubricScreen",
    "SettingsScreen",
    "SetupScreen",
    "TrainingConfigScreen",
    "TrainingScreen",
]
