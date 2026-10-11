"""Human review queue, backed by the ``reviews`` table.

In interactive mode the pipeline submits one :class:`ReviewItem` per prompt and
blocks on a per-item :class:`threading.Event` until the TUI calls
:meth:`ReviewQueue.decide`.  Because every item and decision is persisted
immediately, a fresh process can resume an interrupted review session with
``pending()`` / ``load()``.

Human edits are written through to the stored item *and* to the dataset records,
so the training data reflects what the reviewer actually approved.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from autorlhf.core.events import EventBus
from autorlhf.core.logging import get_logger
from autorlhf.pipeline.models import ReviewDecision, ReviewItem
from autorlhf.pipeline.util import now_iso
from autorlhf.storage.db import RunStore
from autorlhf.storage.runs import RunPaths, read_jsonl

logger = get_logger("autorlhf.pipeline.review")

#: How long a single ``wait_for_decision`` poll sleeps.
POLL_S = 0.2


class ReviewQueue:
    """Submit, observe, and resolve human review items."""

    def __init__(self, store: RunStore, bus: EventBus, run_id: str) -> None:
        self.store = store
        self.bus = bus
        self.run_id = run_id
        self._lock = threading.Lock()
        self._events: dict[str, threading.Event] = {}

    # -- helpers ----------------------------------------------------------- #

    def _event_for(self, item_id: str) -> threading.Event:
        with self._lock:
            event = self._events.get(item_id)
            if event is None:
                event = threading.Event()
                self._events[item_id] = event
            return event

    def _emit(self, message: str, **data: Any) -> None:
        if self.bus is not None:
            self.bus.emit("sample", message, run_id=self.run_id, phase="review", **data)

    # -- submitting -------------------------------------------------------- #

    def submit_items(self, items: list[ReviewItem]) -> None:
        """Persist every item as ``pending`` and announce it on the bus."""

        for item in items:
            self.store.save_review(
                self.run_id,
                item.item_id,
                prompt_id=item.prompt_id,
                status="pending",
                payload=item.model_dump(mode="json"),
            )
            self._emit(
                f"review item {item.item_id}",
                review_item=item.model_dump(mode="json"),
            )

    # -- reading ----------------------------------------------------------- #

    def _from_record(self, record: dict[str, Any] | None) -> ReviewItem | None:
        if not record:
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict) or not payload:
            return None
        try:
            item = ReviewItem.model_validate(payload)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not read review item %s: %s", record.get("item_id"), exc)
            return None
        return item

    def load(self, item_id: str) -> ReviewItem | None:
        return self._from_record(self.store.get_review(self.run_id, item_id))

    def pending(self) -> list[ReviewItem]:
        items: list[ReviewItem] = []
        for record in self.store.list_reviews(self.run_id, status="pending"):
            item = self._from_record(record)
            if item is not None:
                items.append(item)
        return items

    def stats(self) -> dict[str, int]:
        stats = dict(self.store.review_stats(self.run_id))
        stats["total"] = sum(stats.values())
        return stats

    # -- deciding ---------------------------------------------------------- #

    def stored_decision(self, item_id: str) -> ReviewDecision | None:
        record = self.store.get_review(self.run_id, item_id)
        if not record:
            return None
        payload = record.get("payload")
        if not isinstance(payload, dict):
            return None
        raw = payload.get("decision")
        if not isinstance(raw, dict):
            return None
        try:
            return ReviewDecision.model_validate(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("could not read decision for %s: %s", item_id, exc)
            return None

    def decide(self, decision: ReviewDecision) -> None:
        """Apply a human decision (called from the TUI thread)."""

        item = self.load(decision.item_id)
        if item is None:
            raise KeyError(f"unknown review item '{decision.item_id}'")
        item.apply_human(decision)
        payload = item.model_dump(mode="json")
        payload["decision"] = decision.model_dump(mode="json")
        payload["decided_at"] = now_iso()
        self.store.save_review(
            self.run_id,
            item.item_id,
            prompt_id=item.prompt_id,
            status=item.status,
            payload=payload,
        )
        self._emit(
            f"review decision {decision.action} for {item.item_id}",
            review_item=payload,
            review_decision=decision.model_dump(mode="json"),
        )
        self._event_for(decision.item_id).set()

    def wait_for_decision(
        self,
        item_id: str,
        *,
        cancel_event: threading.Event,
        timeout: float | None = None,
    ) -> ReviewDecision | None:
        """Block until the item is decided, the timeout expires, or cancellation."""

        deadline = None if timeout is None else time.monotonic() + float(timeout)
        event = self._event_for(item_id)
        while True:
            decision = self.stored_decision(item_id)
            if decision is not None:
                return decision
            if cancel_event is not None and cancel_event.is_set():
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            wait_for = POLL_S if remaining is None else max(0.01, min(POLL_S, remaining))
            if event.wait(wait_for):
                event.clear()
                decision = self.stored_decision(item_id)
                if decision is not None:
                    return decision

    # -- dataset write-through --------------------------------------------- #

    def apply_to_dataset(self, paths: RunPaths, decision: ReviewDecision) -> bool:
        """Rewrite the dataset records for ``decision``'s prompt (atomic)."""

        from autorlhf.pipeline.dataset import rewrite_records

        path: Path = paths.dataset_path
        records = read_jsonl(path)
        if not records:
            return False
        changed = False
        for record in records:
            if record.get("record_type") != "candidate":
                continue
            if record.get("prompt_id") != decision.item_id:
                continue
            candidate_id = str(record.get("candidate_id", ""))
            if candidate_id in decision.edited_texts:
                record["response"] = decision.edited_texts[candidate_id]
                record["status"] = "edited"
                record["edited"] = True
                edits = list(record.get("edits") or [])
                edits.append(
                    {
                        "source": "human",
                        "reviewer": decision.reviewer,
                        "text": decision.edited_texts[candidate_id],
                        "created_at": decision.created_at,
                    }
                )
                record["edits"] = edits
                record["reviewer"] = decision.reviewer
                changed = True
            if decision.action == "reject_both":
                record["status"] = "rejected"
                record["reviewer"] = decision.reviewer
                changed = True
            elif decision.chosen_id and decision.rejected_id:
                # The human's preference curation is persisted on the dataset
                # records too, so the dataset screen and the accepted/rejected
                # counters reflect it.
                if candidate_id == decision.chosen_id:
                    record["status"] = "accepted"
                    record["reviewer"] = decision.reviewer
                    changed = True
                elif candidate_id == decision.rejected_id:
                    record["status"] = "rejected"
                    record["reviewer"] = decision.reviewer
                    changed = True
            if candidate_id in decision.scores and isinstance(record.get("judgement"), dict):
                judgement = record["judgement"]
                override = judgement.get("human_override") or {}
                override["reviewer"] = decision.reviewer
                override["scores"] = dict(decision.scores[candidate_id])
                judgement["human_override"] = override
                judgement["source"] = "human"
                record["judgement"] = judgement
                changed = True
        if changed:
            rewrite_records(path, records)
        return changed


__all__ = ["POLL_S", "ReviewQueue"]
