"""The autorlhf Textual application.

Owns the environment, the store, the event bus, the current run, and the worker
thread.  Screens never touch the worker thread directly: they ask the app, which
keeps the pause/stop plumbing in one place and makes the UI testable without a
network or a GPU.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from typing import Any

from textual.app import App
from textual.binding import Binding
from textual.screen import Screen

from autorlhf.config.paths import AppPaths
from autorlhf.config.run_config import RunConfig
from autorlhf.config.settings import Environment
from autorlhf.core.enums import FeedbackMode, RunStatus, can_transition
from autorlhf.core.errors import AutorlhfError
from autorlhf.core.events import Event, EventBus
from autorlhf.core.types import HardwareInfo
from autorlhf.storage.db import RunStore
from autorlhf.tui.screens import SCREEN_CLASSES, DashboardScreen
from autorlhf.tui.state import AppState, create_run, default_draft, load_run
from autorlhf.tui.worker import PipelineHooks, PipelineWorker

WorkerFactory = Callable[..., PipelineWorker]


class AutorlhfApp(App[None]):
    """Keyboard-first workbench shell."""

    TITLE = "autorlhf"
    SUB_TITLE = "local-first Jev-guided fine-tuning"
    CSS_PATH = "styles.tcss"
    SCREENS = dict(SCREEN_CLASSES)
    BINDINGS = [
        Binding("ctrl+c", "request_quit", "Quit", priority=True, show=False),
        Binding("q", "request_quit", "Quit"),
        Binding("ctrl+p", "toggle_pause", "Pause/Resume", show=False),
        Binding("ctrl+v", "toggle_verbose", "Verbose", show=False),
        Binding("ctrl+n", "new_run", "New run", show=False),
        Binding("f1", "open_dashboard", "Runs", show=False),
    ]

    def __init__(
        self,
        *,
        paths: AppPaths | None = None,
        env: Environment | None = None,
        store: RunStore | None = None,
        bus: EventBus | None = None,
        config: RunConfig | None = None,
        interactive: bool = False,
        verbose: bool = False,
        hooks: PipelineHooks | None = None,
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        super().__init__()
        self.env = env or Environment(paths)
        self.bus = bus or EventBus()
        self.store = store or RunStore(self.env.paths.db_path, event_bus=self.bus)
        draft = (
            config.model_copy(deep=True)
            if config is not None
            else default_draft(self.env, interactive=interactive)
        )
        self.state = AppState(
            env=self.env,
            store=self.store,
            bus=self.bus,
            draft=draft,
            interactive=interactive,
            verbose=verbose or draft.verbose,
        )
        self.hooks = hooks
        self.worker_factory = worker_factory
        self.worker: PipelineWorker | None = None
        self._worker_thread: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._event_queue: asyncio.Queue[tuple[Callable[[Event], Any] | None, Event]] | None = None
        self._event_consumer: asyncio.Task[None] | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._exit_when_idle = False
        #: True once the first-run wizard has been presented, so the dashboard
        #: never bounces the user back into it repeatedly.
        self.setup_shown = False

    # -- lifecycle --------------------------------------------------------- #

    def get_default_screen(self) -> Screen:
        return DashboardScreen()

    def on_mount(self) -> None:
        from autorlhf.core.logging import configure_logging

        configure_logging(verbose=self.state.verbose)
        self.title = self.TITLE
        self._event_loop = asyncio.get_running_loop()
        self._event_queue = asyncio.Queue()
        self._event_consumer = asyncio.create_task(self._consume_events())
        self._subscribe_bus()

    def on_unmount(self) -> None:
        self._unsubscribe_bus()
        if self._event_consumer is not None:
            self._event_consumer.cancel()


    # -- bus plumbing ------------------------------------------------------ #

    def _subscribe_bus(self) -> None:
        if self._unsubscribe is None:
            self._unsubscribe = self.bus.subscribe(self._on_bus_event)

    def _unsubscribe_bus(self) -> None:
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def _on_bus_event(self, event: Event) -> None:
        """Schedule event handling without waiting for UI rendering."""

        loop = self._event_loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._queue_event, None, event)

    def _queue_event(self, handler: Callable[[Event], Any] | None, event: Event) -> None:
        if self._event_queue is not None:
            self._event_queue.put_nowait((handler, event))

    async def _consume_events(self) -> None:
        queue = self._event_queue
        if queue is None:
            return
        while True:
            handler, event = await queue.get()
            try:
                if handler is None:
                    await self._handle_event(event)
                else:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
            except Exception as exc:  # noqa: BLE001 - one bad update must not stop dispatch
                self.log(f"could not dispatch event {event.kind}: {exc}")
            finally:
                queue.task_done()
            await asyncio.sleep(0)

    async def _handle_event(self, event: Event) -> None:
        current = self.screen
        handler = getattr(current, "handle_event", None)
        if callable(handler):
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result
        if event.kind == "phase" and event.phase:
            self.sub_title = f"{event.phase} - {self.state.run_id or ''}".strip(" -")

    def replay_events(self, handler: Callable[[Event], Any]) -> int:
        """Queue the current history for asynchronous handling on the UI loop."""

        events = self.bus.history()
        loop = self._event_loop
        if loop is not None and not loop.is_closed():
            for event in events:
                loop.call_soon_threadsafe(self._queue_event, handler, event)
        return len(events)
    # -- run identity ------------------------------------------------------ #

    def adopt_run(self, run_id: str) -> bool:
        """Point the UI at an existing run without starting any work."""

        try:
            record, paths, config = load_run(self.state, run_id)
        except Exception as exc:  # noqa: BLE001 - report, never crash the dashboard
            self.notify(f"could not open run {run_id}: {exc}", severity="error", title="Run")
            return False
        self.state.run_id = record.id
        self.state.paths = paths
        self.state.config = config
        self.state.interactive = config.feedback_mode is FeedbackMode.INTERACTIVE
        self.sub_title = f"{record.id} [{record.status.value}]"
        return True

    def start_run(
        self,
        *,
        config: RunConfig | None = None,
        run_id: str | None = None,
        resume: bool = False,
        interactive: bool | None = None,
        worker: PipelineWorker | None = None,
        push_training: bool = True,
    ) -> bool:
        """Create or resume a run and start the pipeline worker."""

        if self._worker_alive():
            self.notify(
                "a run is already in progress; stop it first", severity="warning", title="Run"
            )
            return False
        working = (config or self.state.draft).model_copy(deep=True)
        interactive = self.state.interactive if interactive is None else interactive
        working.feedback_mode = FeedbackMode.INTERACTIVE if interactive else FeedbackMode.AUTOMATED
        try:
            working.assert_valid()
        except AutorlhfError as exc:
            self.notify(str(exc), severity="error", title="Configuration")
            return False
        except Exception as exc:  # noqa: BLE001
            self.notify(f"invalid configuration: {exc}", severity="error", title="Configuration")
            return False

        try:
            if run_id:
                record, paths, _stored = load_run(self.state, run_id)
                run_id = record.id
            else:
                run_id, paths = create_run(self.state, working)
        except AutorlhfError as exc:
            self.notify(str(exc), severity="error", title="Run")
            return False
        except Exception as exc:  # noqa: BLE001
            self.notify(f"could not create the run: {exc}", severity="error", title="Run")
            return False

        self.state.run_id = run_id
        self.state.paths = paths
        self.state.config = working
        self.state.interactive = interactive
        self.sub_title = f"{run_id} starting"

        self.worker = worker or self._make_worker(run_id, paths, working, interactive, resume)
        self._subscribe_bus()
        self._worker_thread = threading.Thread(
            target=self._worker_body, name=f"autorlhf-{run_id}", daemon=True
        )
        self._worker_thread.start()
        if push_training:
            self.push_screen("training")
            if interactive:
                # Interactive runs block in the review queue, so open it straight
                # away; items stream in over the bus as the pipeline submits them.
                self.push_screen("review")
        return True

    def resume_run(self, run_id: str) -> bool:
        """Continue an interrupted run from its highest complete checkpoint."""

        try:
            record, paths, config = load_run(self.state, run_id)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"could not read run {run_id}: {exc}", severity="error", title="Resume")
            return False
        return self.start_run(config=config, run_id=record.id, resume=True)

    def restart_run(self, run_id: str) -> bool:
        """Abandon ``run_id`` and start a fresh run from its pinned configuration."""

        try:
            record, _paths, config = load_run(self.state, run_id)
        except Exception as exc:  # noqa: BLE001
            self.notify(f"could not read run {run_id}: {exc}", severity="error", title="Restart")
            return False
        current = self.store.get_run(record.id)
        if current is not None and can_transition(current.status, RunStatus.CANCELLED):
            self.store.update_status(
                record.id, RunStatus.CANCELLED, note="restarted from the dashboard"
            )
        return self.start_run(config=config)

    def _make_worker(
        self, run_id: str, paths, config: RunConfig, interactive: bool, resume: bool
    ) -> PipelineWorker:
        if self.worker_factory is not None:
            return self.worker_factory(
                run_id=run_id,
                env=self.state.env,
                store=self.state.store,
                paths=paths,
                config=config,
                bus=self.state.bus,
                interactive=interactive,
                verbose=self.state.verbose,
                hooks=self.hooks,
                resume=resume,
            )
        return PipelineWorker(
            run_id=run_id,
            env=self.state.env,
            store=self.state.store,
            paths=paths,
            config=config,
            bus=self.state.bus,
            interactive=interactive,
            verbose=self.state.verbose,
            hooks=self.hooks,
            resume=resume,
        )

    def _worker_alive(self) -> bool:
        return self._worker_thread is not None and self._worker_thread.is_alive()

    # -- worker thread ----------------------------------------------------- #

    def _worker_body(self) -> None:
        worker = self.worker
        if worker is None:  # pragma: no cover - defensive
            return
        try:
            worker.run()
        except BaseException as exc:  # noqa: BLE001 - the UI must never be left wedged
            worker.error = exc
            self._mark_failed(worker.run_id, exc)
        finally:
            if threading.current_thread() is threading.main_thread():
                self._on_pipeline_done(worker)
            else:
                try:
                    self.call_from_thread(self._on_pipeline_done, worker)
                except Exception as exc:  # noqa: BLE001 - the app may already be shutting down
                    self.log(f"could not report pipeline completion: {exc}")

    def _mark_failed(self, run_id: str, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        record = self.store.get_run(run_id)
        if record is not None and can_transition(record.status, RunStatus.FAILED):
            try:
                self.store.update_status(run_id, RunStatus.FAILED, error=message)
            except Exception as update_exc:  # noqa: BLE001 - keep the original failure visible
                self.bus.emit(
                    "warning",
                    f"could not record the failure status: {update_exc}",
                    run_id=run_id,
                )
        self.bus.emit("error", message, run_id=run_id, level="error")

    def _on_pipeline_done(self, worker: PipelineWorker) -> None:
        if worker.error is not None:
            self.notify(
                f"run failed: {worker.error}", severity="error", title="Pipeline", timeout=12
            )
        else:
            self.notify("run finished", title="Pipeline", timeout=6)
        self.sub_title = f"{worker.run_id} {'failed' if worker.error else 'finished'}"
        if self._exit_when_idle:
            self.exit()

    # -- controls ---------------------------------------------------------- #

    def request_stop(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        self.notify("stop requested; the run will end at a safe point", title="Run")

    def request_pause(self) -> None:
        if self.worker is not None:
            self.worker.request_pause()

    def request_resume(self) -> None:
        if self.worker is not None:
            self.worker.request_resume()

    def toggle_pause(self) -> None:
        if self.worker is None:
            return
        if self.worker.paused:
            self.worker.request_resume()
        else:
            self.worker.request_pause()

    def action_toggle_pause(self) -> None:
        self.toggle_pause()

    def set_verbose(self, verbose: bool) -> None:
        self.state.verbose = verbose
        self.state.draft.verbose = verbose

    def action_toggle_verbose(self) -> None:
        self.set_verbose(not self.state.verbose)
        self.notify("verbose" if self.state.verbose else "normal", title="Verbosity")

    def action_request_quit(self) -> None:
        if self._worker_alive():
            self._exit_when_idle = True
            self.request_stop()
            return
        self.exit()

    def action_new_run(self) -> None:
        self.open_setup()

    def action_open_dashboard(self) -> None:
        self.switch_screen("dashboard")

    def open_setup(self) -> None:
        """Open the first-run/new-run wizard."""

        self.push_screen("setup")

    # -- review ------------------------------------------------------------ #

    def submit_review_decision(self, decision: dict[str, Any]) -> None:
        """Hand a human decision to the review queue and persist it.

        Overridden in tests, and by the pipeline-less review screen, so this is
        the single write path for human feedback.
        """

        queue = None
        if self.worker is not None and self.worker.ctx is not None:
            queue = getattr(self.worker.ctx, "review_queue", None)
        if queue is not None:
            from autorlhf.pipeline.models import ReviewDecision

            queue.decide(ReviewDecision.model_validate(decision))
            return
        if self.state.run_id:
            self.store.save_review(
                self.state.run_id,
                str(decision.get("item_id", "")),
                prompt_id=str(decision.get("item_id", "")),
                status="completed",
                payload=dict(decision),
            )

    # -- setup ------------------------------------------------------------- #

    def complete_setup(self) -> None:
        """Persist the wizard's choices as the new defaults."""

        draft = self.state.draft
        settings = self.env.settings
        settings.first_run_complete = True
        settings.default_hf_model = draft.model.model_id
        settings.default_teacher_model = draft.generation.teacher_model
        settings.default_judge_model = draft.generation.judge_model
        settings.max_concurrency = draft.generation.max_concurrency
        settings.request_timeout_s = draft.generation.request_timeout_s
        settings.max_retries = draft.generation.max_retries
        settings.verbose = draft.verbose
        settings.default_feedback_mode = draft.feedback_mode.value
        settings.wandb_enabled = draft.wandb_enabled
        settings.wandb_project = draft.wandb_project
        settings.weave_enabled = draft.weave_enabled
        settings.weave_project = draft.weave_project
        settings.weave_consent = draft.weave_consent
        self.env.save_settings()
        self.set_verbose(draft.verbose)

    # -- host facts -------------------------------------------------------- #

    def detect_hardware(self) -> HardwareInfo:
        """Measure the machine; degrade honestly when the probe is unavailable."""

        fallback = HardwareInfo(
            device_type="cpu",
            device_name="unavailable",
            device_count=0,
            cpu_count=os.cpu_count() or 1,
            estimate_notes=["hardware was not measured: the training probe is unavailable"],
        )
        try:
            from autorlhf.training.hardware import detect_hardware
        except Exception as exc:  # noqa: BLE001 - the probe is optional at this stage
            fallback.warnings.append(f"could not import the hardware probe: {exc}")
            return fallback
        try:
            return detect_hardware()
        except Exception as exc:  # noqa: BLE001
            fallback.warnings.append(f"hardware probe failed: {exc}")
            return fallback

    def draft_param_count(self) -> int | None:
        """Parameter count from metadata only; ``None`` when it cannot be derived."""

        try:
            from autorlhf.training.models import model_param_count_from_metadata
        except Exception:  # noqa: BLE001 - optional until the training package lands
            return None
        try:
            return model_param_count_from_metadata(
                self.state.draft.model, hf_token=self.env.hf_token()
            )
        except Exception:  # noqa: BLE001 - a missing model must not break the screen
            return None

    def plan_ranking(self) -> dict[str, Any] | None:
        """The judge's preset ranking from the most recent plan, if any."""

        plan = getattr(self.worker, "plan", None) if self.worker is not None else None
        ranking = getattr(plan, "jev_ranking", None)
        return ranking if isinstance(ranking, dict) else None


__all__ = ["AutorlhfApp", "WorkerFactory"]
