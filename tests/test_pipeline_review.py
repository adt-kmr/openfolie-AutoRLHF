"""ReviewQueue: submit, block, decide, persist, and write through to the dataset."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from autorlhf.core.enums import SampleStatus
from autorlhf.core.events import EventBus
from autorlhf.llm.schemas import CandidateResponse, Judgement, PairDecision
from autorlhf.pipeline.dataset import candidate_record, rewrite_records
from autorlhf.pipeline.models import ReviewDecision, ReviewItem
from autorlhf.pipeline.review import ReviewQueue
from autorlhf.storage.runs import RunPaths, read_jsonl

RUN = "20260101-120000-ab12"


def make_item(item_id: str = "p0001") -> ReviewItem:
    candidates = [
        CandidateResponse(
            candidate_id=f"{item_id}-c0",
            prompt_id=item_id,
            text="first answer",
            status=SampleStatus.ACCEPTED,
            source="teacher",
        ),
        CandidateResponse(
            candidate_id=f"{item_id}-c1",
            prompt_id=item_id,
            text="second answer",
            status=SampleStatus.ACCEPTED,
            source="teacher",
        ),
    ]
    return ReviewItem(
        item_id=item_id,
        prompt_id=item_id,
        prompt="question?",
        category="ordinary",
        candidates=candidates,
        judgements={
            candidates[0].candidate_id: Judgement(
                judgement_id="j0", prompt_id=item_id, candidate_id=candidates[0].candidate_id
            )
        },
        comparison=PairDecision(
            pair_id=f"{item_id}-pair",
            prompt_id=item_id,
            option_a_id=candidates[0].candidate_id,
            option_b_id=candidates[1].candidate_id,
            winner="a",
        ),
        recommended_candidate_id=candidates[0].candidate_id,
        recommendation_reasons=["judge preferred A"],
    )


def test_submit_then_decide_then_wait_returns_the_decision(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item()])
    assert [item.item_id for item in queue.pending()] == ["p0001"]

    decision = ReviewDecision(item_id="p0001", action="choose_b", chosen_id="p0001-c1")
    queue.decide(decision)

    returned = queue.wait_for_decision("p0001", cancel_event=threading.Event(), timeout=1.0)
    assert returned is not None
    assert returned.action == "choose_b"
    assert returned.chosen_id == "p0001-c1"
    assert queue.pending() == []
    assert queue.stats()["total"] == 1


def test_wait_for_decision_blocks_until_the_tui_decides(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0002")])

    def _decide_later() -> None:
        time.sleep(0.15)
        queue.decide(ReviewDecision(item_id="p0002", action="accept"))

    worker = threading.Thread(target=_decide_later)
    worker.start()
    started = time.monotonic()
    returned = queue.wait_for_decision("p0002", cancel_event=threading.Event(), timeout=5.0)
    worker.join()
    assert returned is not None
    assert returned.action == "accept"
    assert time.monotonic() - started >= 0.1


def test_wait_for_decision_times_out_and_honours_cancellation(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0003")])
    assert queue.wait_for_decision("p0003", cancel_event=threading.Event(), timeout=0.05) is None

    cancel = threading.Event()
    cancel.set()
    assert queue.wait_for_decision("p0003", cancel_event=cancel, timeout=5.0) is None


def test_persisted_decision_survives_a_restart_of_the_queue(store, bus):
    first = ReviewQueue(store, bus, RUN)
    first.submit_items([make_item("p0004")])
    first.decide(ReviewDecision(item_id="p0004", action="choose_a", chosen_id="p0004-c0"))

    restarted = ReviewQueue(store, bus, RUN)
    decision = restarted.stored_decision("p0004")
    assert decision is not None
    assert decision.action == "choose_a"
    item = restarted.load("p0004")
    assert item is not None
    assert item.human is not None
    assert item.status == "completed"


def test_decide_unknown_item_raises(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    try:
        queue.decide(ReviewDecision(item_id="nope", action="accept"))
    except KeyError as exc:
        assert "nope" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected KeyError")


def test_edits_are_applied_to_the_item_and_to_the_dataset(store, bus, tmp_path: Path):
    paths = RunPaths(tmp_path / "run").ensure()
    from autorlhf.llm.schemas import SyntheticPrompt

    prompt = SyntheticPrompt(prompt_id="p0005", category="ordinary", text="q?", split="train")
    candidate = CandidateResponse(
        candidate_id="p0005-c0",
        prompt_id="p0005",
        text="original answer",
        status=SampleStatus.ACCEPTED,
        source="teacher",
    )
    rewrite_records(
        paths.dataset_path,
        [candidate_record(run_id=RUN, prompt=prompt, candidate=candidate, judgement=None)],
    )

    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0005")])
    decision = ReviewDecision(
        item_id="p0005",
        action="edit",
        chosen_id="p0005-c0",
        edited_texts={"p0005-c0": "reviewed answer"},
        scores={"p0005-c0": {"character_consistency": 4.0}},
    )
    queue.decide(decision)
    assert queue.apply_to_dataset(paths, decision) is True

    records = read_jsonl(paths.dataset_path)
    assert records[0]["response"] == "reviewed answer"
    assert records[0]["status"] == "edited"
    assert records[0]["edited"] is True
    assert records[0]["reviewer"] == "human"
    assert records[0]["edits"][0]["source"] == "human"

    item = queue.load("p0005")
    assert item is not None
    assert item.candidates[0].text == "reviewed answer"
    assert item.candidates[0].status is SampleStatus.EDITED
    assert item.status == "edited"


def test_reject_both_marks_the_candidates_rejected(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0006")])
    queue.decide(ReviewDecision(item_id="p0006", action="reject_both"))
    item = queue.load("p0006")
    assert item is not None
    assert item.status == "rejected"
    assert all(c.status is SampleStatus.REJECTED for c in item.candidates)


def test_bus_receives_sample_events(store, bus):
    seen: list[str] = []
    bus.subscribe(lambda event: seen.append(event.kind))
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0007")])
    queue.decide(ReviewDecision(item_id="p0007", action="accept"))
    assert seen.count("sample") == 2
    assert "sample" in seen


def test_stats_reports_pending_and_completed(store, bus):
    queue = ReviewQueue(store, bus, RUN)
    queue.submit_items([make_item("p0008"), make_item("p0009")])
    queue.decide(ReviewDecision(item_id="p0008", action="accept"))
    stats = queue.stats()
    assert stats["pending"] == 1
    assert stats["completed"] == 1
    assert stats["total"] == 2


def test_bus_can_be_none(store):
    queue = ReviewQueue(store, EventBus(), RUN)
    queue.submit_items([make_item("p0010")])
    assert len(queue.pending()) == 1
