"""The shared context every pipeline stage receives.

It bundles the environment, the store, the run's paths, the config, the event
bus, and the lazily constructed OpenRouter/Jev clients.  Building a context
performs no network IO and imports neither ``textual`` nor ``torch``: the paid
clients are created the first time a stage actually needs them, so a resumed run
that is still in the dataset phase never even constructs an SDK client.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import RunStatus
from autorlhf.core.events import Event, EventBus, EventKind
from autorlhf.core.logging import get_logger
from autorlhf.llm.client import OpenRouterClient
from autorlhf.llm.jev import JevEvaluator
from autorlhf.pipeline.review import ReviewQueue
from autorlhf.storage.cache import ResponseCache
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths

logger = get_logger("autorlhf.pipeline.context")


class PipelineContext:
    """Everything a pipeline stage needs, with lazy expensive pieces."""

    def __init__(
        self,
        *,
        env: Environment,
        store: RunStore,
        paths: RunPaths,
        config: RunConfig,
        bus: EventBus,
        interactive: bool = False,
        verbose: bool = False,
    ) -> None:
        self.env = env
        self.store = store
        self.paths = paths
        self.config = config
        self.bus = bus
        self.interactive = interactive
        self.verbose = verbose
        self.phase = "setup"

        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()

        self.run_id = self._resolve_run_id()
        self.hf_token = env.hf_token()
        self.cache = ResponseCache(store, enabled=env.settings.cache_enabled)
        self.review_queue = ReviewQueue(store, bus, self.run_id)

        self._openrouter: OpenRouterClient | None = None
        self._jev: JevEvaluator | None = None
        self._lock = threading.RLock()  # reentrant: jev builds openrouter

    # -- run id ------------------------------------------------------------ #

    def _resolve_run_id(self) -> str:
        folder = self.paths.root.name
        for record in self.store.list_runs(limit=500):
            if record.folder == folder:
                return record.id
        logger.warning("no run row found for folder %s", folder)
        return ""

    # -- lazy paid clients ------------------------------------------------- #

    @property
    def openrouter(self) -> OpenRouterClient:
        """The OpenRouter client, built on first use (raises without a key)."""

        with self._lock:
            if self._openrouter is None:
                key = self.env.require_openrouter()
                self._openrouter = OpenRouterClient(
                    key,
                    timeout_s=self.config.generation.request_timeout_s,
                    max_retries=self.config.generation.max_retries,
                    max_concurrency=self.config.generation.max_concurrency,
                    cache=self.cache,
                    run_id=self.run_id,
                    app_title="autorlhf",
                )
            return self._openrouter

    @property
    def jev(self) -> JevEvaluator:
        with self._lock:
            if self._jev is None:
                self._jev = JevEvaluator(
                    self.openrouter,
                    self.config.rubric,
                    self.config.character,
                    judge_model=self.config.generation.judge_model,
                    cache=self.cache,
                    run_id=self.run_id,
                )
            return self._jev

    def has_credentials(self) -> bool:
        return bool(self.env.openrouter_key())

    # -- events ------------------------------------------------------------ #

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def emit(self, kind: EventKind, message: str = "", **kw: Any) -> Event:
        kw.setdefault("phase", self.phase)
        kw.setdefault("verbose_only", not self.verbose and kind == "log")
        return self.bus.emit(kind, message, run_id=self.run_id, **kw)

    def progress(self, message: str, *, fraction: float | None = None, **data: Any) -> Event:
        return self.emit("progress", message, fraction=fraction, **data)

    # -- cooperative control ----------------------------------------------- #

    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def paused(self) -> bool:
        return self.pause_event.is_set()

    def wait_if_paused(self) -> None:
        while self.pause_event.is_set():
            if self.cancel_event.is_set():
                return
            time.sleep(0.1)

    # -- status ------------------------------------------------------------ #

    def set_status(self, status: RunStatus, *, phase: str | None = None, note: str = "") -> None:
        """Move the run to ``status``, routing through ``PAUSED`` when needed.

        ``PAUSED`` can legally re-enter any resumable status, so an illegal
        direct transition (which happens when a step is re-run on resume) is
        routed through it rather than raising.
        """

        from autorlhf.core.enums import can_transition

        current = self.store.get_run(self.run_id)
        if current is None:
            return
        if current.status != status and not can_transition(current.status, status):
            try:
                self.store.update_status(self.run_id, RunStatus.PAUSED, note="rerouting")
            except ValueError:  # pragma: no cover - defensive
                pass
        try:
            self.store.update_status(self.run_id, status, phase=phase or self.phase, note=note)
        except ValueError as exc:  # pragma: no cover - defensive
            logger.warning("could not move run %s to %s: %s", self.run_id, status.value, exc)

    def bump(self, name: str, delta: int = 1) -> int:
        return self.store.bump_counter(self.run_id, name, delta)

    def set_counter(self, name: str, value: int) -> None:
        self.store.set_counter(self.run_id, name, value)

    # -- lifecycle --------------------------------------------------------- #

    def close(self) -> None:
        """Release the paid clients.  Never raises."""

        with self._lock:
            client, self._openrouter = self._openrouter, None
            self._jev = None
        if client is not None:
            client.close()


__all__ = ["PipelineContext"]
