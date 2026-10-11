"""Small shared helpers for the pipeline: pause/cancel, text normalisation, ids."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TypeVar

T = TypeVar("T")

#: How long a pause poll sleeps before checking the flags again.
PAUSE_POLL_S = 0.1


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def normalise_text(text: str) -> str:
    """Case- and whitespace-insensitive form used for de-duplication."""

    return " ".join(text.lower().split())


def wait_if_paused(
    pause_event: threading.Event | None, cancel_event: threading.Event | None
) -> bool:
    """Block while paused.  Returns False when cancellation was requested."""

    while pause_event is not None and pause_event.is_set():
        if cancel_event is not None and cancel_event.is_set():
            return False
        time.sleep(PAUSE_POLL_S)
    return not (cancel_event is not None and cancel_event.is_set())


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def run_bounded(
    jobs: Iterable[Callable[[], T]],
    *,
    max_workers: int,
    cancel_event: threading.Event | None = None,
    on_done: Callable[[int], None] | None = None,
) -> list[T | None]:
    """Run ``jobs`` with bounded concurrency, returning results in submit order.

    Jobs that are never submitted (because cancellation was requested) come back
    as ``None``.  An exception inside a job propagates after every running job
    has been awaited, so no worker is left behind.
    """

    job_list = list(jobs)
    if not job_list:
        return []
    results: list[T | None] = [None] * len(job_list)
    workers = max(1, min(int(max_workers), len(job_list)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="autorlhf") as pool:
        futures: dict[Future[T], int] = {}
        for index, job in enumerate(job_list):
            if cancel_event is not None and cancel_event.is_set():
                break
            futures[pool.submit(job)] = index
        done = 0
        errors: list[BaseException] = []
        for future in futures:
            try:
                results[futures[future]] = future.result()
            except BaseException as exc:  # noqa: BLE001 - re-raised after the pool drains
                errors.append(exc)
            done += 1
            if on_done is not None:
                on_done(done)
        if errors:
            raise errors[0]
    return results


__all__ = [
    "PAUSE_POLL_S",
    "clamp",
    "normalise_text",
    "now_iso",
    "run_bounded",
    "wait_if_paused",
]
