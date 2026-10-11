"""The pipeline must be importable without textual or the training stack."""

from __future__ import annotations

import subprocess
import sys

HEAVY = ("textual", "torch", "transformers", "trl", "peft", "datasets", "accelerate")

CHECK = """
import sys
import autorlhf.pipeline.orchestrate  # noqa: F401
import autorlhf.pipeline.context  # noqa: F401
import autorlhf.pipeline.candidates  # noqa: F401
import autorlhf.pipeline.evaluate  # noqa: F401
import autorlhf.llm.client  # noqa: F401
import autorlhf.llm.jev  # noqa: F401
heavy = {heavy!r}
print(",".join(name for name in heavy if name in sys.modules))
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )


def test_importing_the_pipeline_does_not_import_textual_or_torch():
    result = _run(CHECK.format(heavy=HEAVY))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout


def test_importing_the_pipeline_is_fast():
    result = _run(
        "import time; start=time.perf_counter(); import autorlhf.pipeline.orchestrate;"
        "print(round(time.perf_counter()-start, 2))"
    )
    assert result.returncode == 0, result.stderr
    assert float(result.stdout.strip()) < 5.0


CONTEXT_CHECK = """
import pathlib, sys
from autorlhf.config.character import CharacterSpec
from autorlhf.config.paths import AppPaths
from autorlhf.config.rubric import default_rubric
from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.events import EventBus
from autorlhf.core.ids import run_folder_name
from autorlhf.pipeline.context import PipelineContext
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths

paths = AppPaths.for_tests(pathlib.Path({root!r})).ensure()
env = Environment(paths)
store = RunStore(paths.db_path)
config = RunConfig(
    character=CharacterSpec(name="x", system_prompt="You are x."), rubric=default_rubric()
)
folder = run_folder_name("20260101-120000-ab12", config.slug)
store.create_run(config, folder=folder, run_id="20260101-120000-ab12")
ctx = PipelineContext(
    env=env,
    store=store,
    paths=RunPaths(paths.run_dir(folder)).ensure(),
    config=config,
    bus=EventBus(),
)
print(ctx.run_id, ctx._openrouter is None, ctx._jev is None, "torch" in sys.modules)
"""


def test_context_construction_opens_no_clients_and_imports_no_torch(tmp_path):
    result = _run(CONTEXT_CHECK.format(root=str(tmp_path)))
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "20260101-120000-ab12 True True False"
