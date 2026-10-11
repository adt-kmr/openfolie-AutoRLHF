"""Interactive review queue.

The screen works on the JSON shape of ``ReviewItem`` (what the event bus carries)
so it stays usable while the pipeline package is being built, and every decision
it submits uses the frozen ``ReviewDecision`` fields.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    Static,
    TextArea,
)

from autorlhf.tui.format import probability_bar
from autorlhf.tui.screens.base import AutorlhfScreen
from autorlhf.tui.widgets import ScoreBar

ACTIONS = ("accept", "choose_a", "choose_b", "tie", "reject_both", "edit", "regenerate")

DECIDED = {"completed", "skipped"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def render_prompt(item: dict[str, Any]) -> str:
    lines = [f"[{item.get('category', 'ordinary')}] {item.get('prompt', '')}"]
    for turn in item.get("context") or []:
        lines.append(f"  {turn.get('role', '?')}: {turn.get('content', '')}")
    return "\n".join(lines)


def render_candidates(item: dict[str, Any]) -> str:
    blocks: list[str] = []
    for candidate in item.get("candidates") or []:
        header = (
            f"{candidate.get('candidate_id', '?')} "
            f"[{candidate.get('source', 'unknown')} {candidate.get('model_id', '')}]"
        )
        status = candidate.get("status", "")
        if status and status != "accepted":
            header += f" ({status})"
        body = candidate.get("text") or candidate.get("error") or "(empty response)"
        blocks.append(f"{header}\n{body}")
    return "\n\n".join(blocks) if blocks else "(no candidates for this item)"


def render_judgements(item: dict[str, Any]) -> str:
    judgements = item.get("judgements") or {}
    if not judgements:
        return "(no judgements recorded yet)"
    blocks: list[str] = []
    for candidate_id, judgement in judgements.items():
        lines = [f"{candidate_id}: overall {judgement.get('overall')}"]
        if judgement.get("error"):
            lines.append(f"  judge failed: {judgement['error']}")
        for criterion_id, score in (judgement.get("criterion_scores") or {}).items():
            name = score.get("criterion_name") or criterion_id
            if score.get("score") is None:
                lines.append(f"  {name}: failed ({score.get('error') or 'no result'})")
                continue
            lines.append(
                f"  {name}: {score['score']:.2f}/{score.get('scale_max', 4)}"
                + (
                    f"  confidence {score['confidence']:.2f}"
                    if score.get("confidence") is not None
                    else ""
                )
            )
            bar = probability_bar(score.get("probabilities"))
            if not bar.startswith("("):
                lines.append("    " + bar.replace("\n", "\n    "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_recommendation(item: dict[str, Any]) -> str:
    lines: list[str] = []
    chosen = item.get("recommended_candidate_id")
    lines.append(f"Judge recommendation: {chosen or '(none)'}")
    for reason in item.get("recommendation_reasons") or []:
        lines.append(f"  - {reason}")
    comparison = item.get("comparison")
    if comparison:
        lines.append(
            f"Comparison: winner={comparison.get('winner')} "
            f"margin={comparison.get('margin')} error={comparison.get('error')}"
        )
        confidence = comparison.get("confidence")
        if confidence is not None:
            lines.append(f"  confidence {confidence:.2f}")
        bar = probability_bar(comparison.get("probabilities"))
        if not bar.startswith("("):
            lines.append("  " + bar.replace("\n", "\n  "))
    lines.append(
        "The recommendation is shown for reference only; it is never applied automatically."
    )
    return "\n".join(lines)


class ReviewScreen(AutorlhfScreen):
    """Review queue with per-criterion human scoring and explicit overrides."""

    BINDINGS = [
        Binding("a", "choose_a", "Choose A"),
        Binding("b", "choose_b", "Choose B"),
        Binding("t", "tie", "Tie"),
        Binding("x", "reject_both", "Reject both"),
        Binding("e", "apply_edit", "Apply edit"),
        Binding("g", "regenerate", "Regenerate"),
        Binding("n", "next_item", "Next item"),
        Binding("s", "apply_scores", "Apply scores"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.items: list[dict[str, Any]] = []
        self.current_index: int = -1
        self.current: dict[str, Any] | None = None
        self.scores: dict[str, dict[str, float]] = {}
        self.edited_texts: dict[str, str] = {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.comments: str = ""
        self._score_widgets: dict[tuple[str, str], Input] = {}
        self._score_epoch: int = 0

    # -- composition ------------------------------------------------------- #

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Static("Interactive review", id="review-title", classes="screen-title")
        with Horizontal(id="review-body"):
            with Vertical(id="review-queue"):
                yield Label("Queue")
                yield ListView(id="review-items")
                yield Static("", id="review-stats", classes="meta")
            with VerticalScroll(id="review-detail"):
                yield Static("", id="review-character")
                yield Static("", id="review-prompt")
                yield Label("Candidates")
                yield Static("", id="review-candidates")
                yield Label("Judge scores")
                yield Static("", id="review-judgements")
                yield Label("Judge distributions")
                yield Vertical(id="review-scorebars")
                yield Label("Recommendation")
                yield Static("", id="review-recommendation")
                yield Label("Your scores (blank keeps the judge's value)")
                yield Vertical(id="review-score-inputs")
                yield Label("Edit a response")
                with Horizontal(classes="row"):
                    yield Input(placeholder="candidate id", id="review-edit-target")
                    yield TextArea("", id="review-edit-text")
                yield Input(placeholder="comments", id="review-comments")
                yield Checkbox("Use the chosen response as an SFT example", id="review-use-sft")
                with Horizontal(classes="row"):
                    yield Button("Apply scores", id="review-scores")
                    yield Button("Apply edit", id="review-edit")
                    yield Button("Choose A", id="review-choose-a", variant="primary")
                    yield Button("Choose B", id="review-choose-b", variant="primary")
                    yield Button("Tie", id="review-tie")
                    yield Button("Reject both", id="review-reject", variant="error")
                    yield Button("Regenerate", id="review-regenerate")
                yield Static("", id="review-status", classes="problems")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#review-use-sft", Checkbox).value = True
        self.load_pending_from_store()
        self.render_stats()
        if not self.items:
            self.query_one("#review-status", Static).update(
                "Waiting for review items from the pipeline."
            )

    def load_pending_from_store(self) -> int:
        """Restore an interrupted review session from the ``reviews`` table."""

        run_id = self.state.run_id
        if not run_id:
            return 0
        try:
            rows = self.state.store.list_reviews(run_id, status="pending")
        except Exception as exc:  # noqa: BLE001 - the queue is still usable without them
            self.query_one("#review-status", Static).update(
                f"could not read stored review items: {exc}"
            )
            return 0
        for row in rows:
            payload = row.get("payload") or {}
            if payload.get("item_id"):
                self.add_item(payload)
        return len(self.items)

    # -- queue management -------------------------------------------------- #

    def load_items(self, items: list[dict[str, Any]]) -> int:
        self.items = []
        self.current_index = -1
        for item in items:
            self.add_item(item)
        self.refresh_queue()
        if self.items:
            self.load_index(0)
        return len(self.items)

    def add_item(self, item: dict[str, Any]) -> bool:
        item_id = item.get("item_id") or item.get("prompt_id")
        if not item_id:
            return False
        if any(existing.get("item_id") == item_id for existing in self.items):
            return False
        item = dict(item)
        item.setdefault("item_id", item_id)
        self.items.append(item)
        self.refresh_queue()
        if self.current is None:
            self.load_index(0)
        return True

    def pending(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item.get("status", "pending") not in DECIDED]

    def refresh_queue(self) -> None:
        view = self.query_one("#review-items", ListView)
        view.clear()
        for item in self.items:
            status = item.get("status", "pending")
            prompt = (item.get("prompt") or "")[:60]
            view.append(ListItem(Label(f"[{status}] {prompt}")))
        self.render_stats()

    def render_stats(self) -> None:
        decided = len([i for i in self.items if i.get("status") in DECIDED])
        stats = f"{decided}/{len(self.items)} decided, {len(self.pending())} pending"
        try:
            store = self.state.store
            run_id = self.state.run_id
            if run_id:
                counts = store.review_stats(run_id)
                if counts:
                    stats += f" | stored: {counts}"
        except Exception:  # noqa: BLE001 - stats are display only
            pass
        self.query_one("#review-stats", Static).update(stats)

    def load_index(self, index: int) -> bool:
        if not 0 <= index < len(self.items):
            return False
        self.current_index = index
        self.current = self.items[index]
        self.scores = {}
        self.edited_texts = {}
        self.render_current()
        view = self.query_one("#review-items", ListView)
        view.index = index
        return True

    def advance(self) -> bool:
        pending = self.pending()
        if not pending:
            self.query_one("#review-status", Static).update(
                "Every item has a decision. The pipeline continues automatically."
            )
            self.render_stats()
            return False
        return self.load_index(self.items.index(pending[0]))

    def action_next_item(self) -> None:
        self.advance()

    # -- rendering --------------------------------------------------------- #

    def render_current(self) -> None:
        item = self.current
        if item is None:
            return
        self.query_one("#review-character", Static).update(self.character_summary())
        self.query_one("#review-prompt", Static).update(render_prompt(item))
        self.query_one("#review-candidates", Static).update(render_candidates(item))
        self.query_one("#review-judgements", Static).update(render_judgements(item))
        self.query_one("#review-recommendation", Static).update(render_recommendation(item))
        self.query_one("#review-edit-target", Input).value = (
            (item.get("recommended_candidate_id") or "") or self._first_candidate_id(item) or ""
        )
        self.query_one("#review-edit-text", TextArea).text = ""
        # One serial per render: `remove_children` is applied by the message pump,
        # so freshly built widgets must not reuse the previous ids.
        self._score_epoch += 1
        self.build_score_inputs(item)
        self.build_score_bars(item)

    def character_summary(self, *, max_chars: int = 700) -> str:
        config = self.state.config
        if config is None:
            return "(no character specification is loaded for this run)"
        return config.character.compact_summary(max_chars=max_chars)

    def build_score_inputs(self, item: dict[str, Any]) -> None:
        container = self.query_one("#review-score-inputs", Vertical)
        container.remove_children()
        self._score_widgets = {}
        weights = self._criterion_names(item)
        widgets: list[Any] = []
        for candidate in item.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id"))
            for criterion_id, criterion_name in weights.items():
                field = Input(
                    placeholder="blank = keep",
                    id=f"score-{self._score_epoch}-{candidate_id}-{criterion_id}",
                )
                self._score_widgets[(candidate_id, criterion_id)] = field
                widgets.append(Label(f"{candidate_id} / {criterion_name}", classes="field-label"))
                widgets.append(field)
        if widgets:
            container.mount_all(widgets)

    def build_score_bars(self, item: dict[str, Any]) -> None:
        """One score bar per candidate/criterion, drawn from the judge's numbers."""

        container = self.query_one("#review-scorebars", Vertical)
        container.remove_children()
        widgets: list[Any] = []
        for candidate in item.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id"))
            judgement = (item.get("judgements") or {}).get(candidate_id) or {}
            for criterion_id, score in (judgement.get("criterion_scores") or {}).items():
                bar = ScoreBar(id=f"bars-{self._score_epoch}-{candidate_id}-{criterion_id}")
                bar.show(
                    f"{candidate_id} / {score.get('criterion_name') or criterion_id}",
                    score=score.get("score"),
                    scale_max=score.get("scale_max"),
                    probabilities=score.get("probabilities"),
                    confidence=score.get("confidence"),
                    error=score.get("error"),
                )
                widgets.append(bar)
        if not widgets:
            widgets.append(Static("(no judgements recorded yet)"))
        container.mount_all(widgets)

    @staticmethod
    def _criterion_names(item: dict[str, Any]) -> dict[str, str]:
        names: dict[str, str] = {}
        for judgement in (item.get("judgements") or {}).values():
            for criterion_id, score in (judgement.get("criterion_scores") or {}).items():
                names.setdefault(criterion_id, score.get("criterion_name") or criterion_id)
        return names

    @staticmethod
    def _first_candidate_id(item: dict[str, Any]) -> str | None:
        candidates = item.get("candidates") or []
        return str(candidates[0].get("candidate_id")) if candidates else None

    @staticmethod
    def _candidate_ids(item: dict[str, Any]) -> list[str]:
        return [str(c.get("candidate_id")) for c in (item.get("candidates") or [])]

    # -- scoring ----------------------------------------------------------- #

    def set_score(self, candidate_id: str, criterion_id: str, value: float) -> None:
        self.scores.setdefault(candidate_id, {})[criterion_id] = float(value)

    def set_edited_text(self, candidate_id: str, text: str) -> None:
        self.edited_texts[candidate_id] = text

    def action_apply_scores(self) -> None:
        self.apply_scores()

    def apply_scores(self) -> list[str]:
        problems: list[str] = []
        for (candidate_id, criterion_id), field in self._score_widgets.items():
            raw = field.value.strip()
            if not raw:
                continue
            try:
                self.set_score(candidate_id, criterion_id, float(raw))
            except ValueError:
                problems.append(f"{candidate_id}/{criterion_id}: '{raw}' is not a number")
        self.comments = self.query_one("#review-comments", Input).value
        self.query_one("#review-status", Static).update(
            "\n".join(f"- {problem}" for problem in problems) or self._summary()
        )
        return problems

    def _summary(self) -> str:
        return f"Human scores: {len(self.scores)} candidate(s); edits: {len(self.edited_texts)}"

    # -- decisions --------------------------------------------------------- #

    def build_decision(self, action: str) -> dict[str, Any]:
        item = self.current or {}
        candidate_ids = self._candidate_ids(item)
        first = candidate_ids[0] if candidate_ids else None
        second = candidate_ids[1] if len(candidate_ids) > 1 else None
        chosen: str | None = None
        rejected: str | None = None
        if action == "choose_a":
            chosen, rejected = first, second
        elif action == "choose_b":
            chosen, rejected = second, first
        elif action == "accept":
            chosen = item.get("recommended_candidate_id") or first
            rejected = next((c for c in candidate_ids if c != chosen), None)
        elif action == "edit":
            target = self.query_one("#review-edit-target", Input).value.strip() or first
            chosen = target
            rejected = next((c for c in candidate_ids if c != target), None)
        elif action == "regenerate":
            rejected = self.query_one("#review-edit-target", Input).value.strip() or first
        use_sft = self.query_one("#review-use-sft", Checkbox).value
        return {
            "item_id": str(item.get("item_id") or item.get("prompt_id") or ""),
            "reviewer": "human",
            "action": action,
            "chosen_id": chosen,
            "rejected_id": rejected,
            "scores": {k: dict(v) for k, v in self.scores.items()},
            "edited_texts": dict(self.edited_texts),
            "comments": self.comments,
            "use_chosen_as_sft": bool(use_sft),
            "created_at": _now(),
        }

    def submit_action(self, action: str) -> dict[str, Any]:
        """Record a human decision, persist it, and move to the next item."""

        item = self.current
        if item is None:
            return {}
        decision = self.build_decision(action)
        self.decisions[decision["item_id"]] = decision
        try:
            self.autorlhf.submit_review_decision(decision)
        except Exception as exc:  # noqa: BLE001 - show the failure, keep the queue usable
            self.query_one("#review-status", Static).update(f"could not record the decision: {exc}")
            return decision
        item["status"] = "completed"
        self.refresh_queue()
        self.query_one("#review-status", Static).update(
            f"Recorded '{action}' for {decision['item_id']}."
        )
        self.advance()
        return decision

    def action_choose_a(self) -> None:
        self.submit_action("choose_a")

    def action_choose_b(self) -> None:
        self.submit_action("choose_b")

    def action_tie(self) -> None:
        self.submit_action("tie")

    def action_reject_both(self) -> None:
        self.submit_action("reject_both")

    def action_regenerate(self) -> None:
        self.submit_action("regenerate")

    def action_apply_edit(self) -> None:
        target = self.query_one("#review-edit-target", Input).value.strip()
        text = self.query_one("#review-edit-text", TextArea).text
        if not target:
            self.query_one("#review-status", Static).update("Set a candidate id to edit.")
            return
        self.set_edited_text(target, text)
        self.submit_action("edit")

    # -- events ------------------------------------------------------------ #

    def handle_event(self, event) -> None:
        if event.kind == "sample" and event.data.get("review_item"):
            self.add_item(dict(event.data["review_item"]))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "review-items":
            return
        if event.list_view.index is not None:
            self.load_index(event.list_view.index)

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "review-items" or event.item is None:
            return
        index = event.list_view.index
        if index is not None and index != self.current_index:
            self.load_index(index)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button = event.button.id
        handlers = {
            "review-scores": self.action_apply_scores,
            "review-edit": self.action_apply_edit,
            "review-choose-a": self.action_choose_a,
            "review-choose-b": self.action_choose_b,
            "review-tie": self.action_tie,
            "review-reject": self.action_reject_both,
            "review-regenerate": self.action_regenerate,
        }
        handler = handlers.get(button or "")
        if handler is not None:
            handler()


__all__ = [
    "ACTIONS",
    "DECIDED",
    "ReviewScreen",
    "render_candidates",
    "render_judgements",
    "render_prompt",
    "render_recommendation",
]
