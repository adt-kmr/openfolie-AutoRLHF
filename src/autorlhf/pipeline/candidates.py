"""Candidate response generation.

Every prompt gets ``candidates_per_prompt`` responses, from the OpenRouter teacher,
from the local Hugging Face model, or from both (``mixed``).  The local model is
loaded once and kept alive for the whole phase.

Rules that keep the dataset honest:

* candidate ids are deterministic (``{prompt_id}-c{n}``) and request hashes are
  stable, so resuming reuses exactly what was already generated;
* a failed generation is recorded with ``status="failed"`` and the error, never as
  an empty response, and never becomes training data;
* each candidate uses a slightly different temperature for diversity, recorded on
  the candidate so the dataset is reproducible.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import SampleStatus
from autorlhf.core.events import EventBus
from autorlhf.core.ids import content_hash
from autorlhf.core.logging import get_logger
from autorlhf.core.types import ChatMessage
from autorlhf.llm.client import OpenRouterClient
from autorlhf.llm.schemas import CandidateResponse, SyntheticPrompt
from autorlhf.pipeline.util import clamp, run_bounded, wait_if_paused
from autorlhf.storage.cache import NS_LOCAL_GEN, ResponseCache

logger = get_logger("autorlhf.pipeline.candidates")

#: Temperature spread applied across a prompt's candidates.
TEMPERATURE_STEP = 0.15


def candidate_id(prompt_id: str, index: int) -> str:
    return f"{prompt_id}-c{index}"


def candidate_temperature(base: float, index: int, count: int) -> float:
    """A small, deterministic spread around ``base`` so candidates differ."""

    offset = TEMPERATURE_STEP * (index - (count - 1) / 2.0)
    return round(clamp(base + offset, 0.05, 2.0), 3)


def build_messages(prompt: SyntheticPrompt, system_prompt: str) -> list[ChatMessage]:
    """The exact messages handed to a generator for one prompt."""

    messages = [ChatMessage(role="system", content=system_prompt)]
    for turn in prompt.context:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant", "system"} and isinstance(content, str) and content:
            messages.append(ChatMessage(role=role, content=content))  # type: ignore[arg-type]
    messages.append(ChatMessage(role="user", content=prompt.text))
    return messages


class CandidateGenerator:
    """Generates and caches candidate responses for a prompt set."""

    def __init__(
        self,
        *,
        client: OpenRouterClient | None,
        config: RunConfig,
        run_id: str = "",
        cache: ResponseCache | None = None,
        bus: EventBus | None = None,
        hf_token: str | None = None,
        cancel_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.run_id = run_id
        self.cache = cache
        self.bus = bus
        self.hf_token = hf_token
        self.cancel_event = cancel_event
        self.pause_event = pause_event
        self.generation = config.generation
        self._local: Any | None = None
        self._local_tokenizer: Any | None = None
        self._local_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # local model
    # ------------------------------------------------------------------ #

    def _ensure_local(self) -> Any:
        """Load the local model once and keep it alive (lazy: training is optional)."""

        with self._local_lock:
            if self._local is not None:
                return self._local
            from autorlhf.training import models as training_models

            loaded = training_models.load_model(
                self.config.model,
                self.config.training,
                hf_token=self.hf_token,
                for_training=False,
            )
            tokenizer = getattr(loaded, "tokenizer", None)
            if tokenizer is None:
                tokenizer, _warnings = training_models.load_tokenizer(
                    self.config.model, hf_token=self.hf_token
                )
            self._local = loaded
            self._local_tokenizer = tokenizer
            logger.info("local candidate generator loaded %s", self.config.model.model_id)
            return loaded

    def close(self) -> None:
        """Release the local model so the process can exit cleanly."""

        with self._local_lock:
            self._local = None
            self._local_tokenizer = None

    # ------------------------------------------------------------------ #
    # generation
    # ------------------------------------------------------------------ #

    def _source_for(self, index: int) -> str:
        mode = self.generation.candidate_source
        if mode == "mixed":
            return "teacher" if index % 2 == 0 else "local"
        return mode

    def _teacher_candidate(self, prompt: SyntheticPrompt, index: int) -> CandidateResponse:
        if self.client is None:
            raise RuntimeError("the teacher source requires an OpenRouter client")
        temperature = candidate_temperature(
            self.generation.temperature, index, self.generation.candidates_per_prompt
        )
        seed = self.generation.seed + index
        messages = build_messages(prompt, self.config.character.system_prompt)
        result = self.client.chat(
            model=self.generation.teacher_model,
            messages=messages,
            temperature=temperature,
            top_p=self.generation.top_p,
            max_tokens=self.generation.max_new_tokens,
            seed=seed,
        )
        return CandidateResponse(
            candidate_id=candidate_id(prompt.prompt_id, index),
            prompt_id=prompt.prompt_id,
            source="teacher",
            model_id=self.generation.teacher_model,
            text=result.text,
            messages=[m.model_dump(mode="json") for m in messages],
            status=SampleStatus.PENDING,
            truncated=result.truncated,
            tokens=result.completion_tokens,
            latency_s=result.latency_s,
            seed=seed,
            temperature=temperature,
            cached=result.cached,
        )

    def _local_candidate(self, prompt: SyntheticPrompt, index: int) -> CandidateResponse:
        temperature = candidate_temperature(
            self.generation.temperature, index, self.generation.candidates_per_prompt
        )
        seed = self.generation.seed + index
        messages = build_messages(prompt, self.config.character.system_prompt)
        cache_key = content_hash(
            {
                "kind": "local.candidate",
                "model": self.config.model.model_id,
                "revision": self.config.model.revision,
                "messages": [m.model_dump(mode="json") for m in messages],
                "temperature": temperature,
                "top_p": self.generation.top_p,
                "max_new_tokens": self.generation.max_new_tokens,
                "seed": seed,
            }
        )
        if self.cache is not None:
            cached = self.cache.get(NS_LOCAL_GEN, cache_key)
            if isinstance(cached, dict) and "text" in cached:
                return CandidateResponse(
                    candidate_id=candidate_id(prompt.prompt_id, index),
                    prompt_id=prompt.prompt_id,
                    source="local",
                    model_id=self.config.model.model_id,
                    text=str(cached.get("text", "")),
                    messages=[m.model_dump(mode="json") for m in messages],
                    status=SampleStatus.PENDING,
                    truncated=bool(cached.get("truncated", False)),
                    tokens=cached.get("tokens"),
                    latency_s=cached.get("latency_s"),
                    seed=seed,
                    temperature=temperature,
                    cached=True,
                )

        loaded = self._ensure_local()
        from autorlhf.training.models import generate_text

        started = time.monotonic()
        text, meta = generate_text(
            loaded.model,
            self._local_tokenizer,
            messages,
            max_new_tokens=self.generation.max_new_tokens,
            temperature=temperature,
            top_p=self.generation.top_p,
            seed=seed,
            device=None,
        )
        meta = meta or {}
        payload = {
            "text": text,
            "tokens": meta.get("tokens") or meta.get("completion_tokens"),
            "truncated": bool(meta.get("truncated", False)),
            "latency_s": meta.get("latency_s") or (time.monotonic() - started),
        }
        if self.cache is not None:
            self.cache.put(NS_LOCAL_GEN, cache_key, payload, model=self.config.model.model_id)
        return CandidateResponse(
            candidate_id=candidate_id(prompt.prompt_id, index),
            prompt_id=prompt.prompt_id,
            source="local",
            model_id=self.config.model.model_id,
            text=text,
            messages=[m.model_dump(mode="json") for m in messages],
            status=SampleStatus.PENDING,
            truncated=payload["truncated"],
            tokens=payload["tokens"],
            latency_s=payload["latency_s"],
            seed=seed,
            temperature=temperature,
        )

    def _generate_one(self, prompt: SyntheticPrompt, index: int, source: str) -> CandidateResponse:
        try:
            if source == "local":
                return self._local_candidate(prompt, index)
            return self._teacher_candidate(prompt, index)
        except Exception as exc:  # noqa: BLE001 - failures are data, recorded not raised
            logger.warning(
                "candidate %s generation failed: %s", candidate_id(prompt.prompt_id, index), exc
            )
            if self.bus is not None:
                self.bus.emit(
                    "warning",
                    f"generation failed for {prompt.prompt_id} candidate {index}: {exc}",
                    run_id=self.run_id,
                    phase="generate",
                    prompt_id=prompt.prompt_id,
                    candidate_index=index,
                )
            return CandidateResponse(
                candidate_id=candidate_id(prompt.prompt_id, index),
                prompt_id=prompt.prompt_id,
                source=source,
                model_id=(
                    self.generation.teacher_model
                    if source == "teacher"
                    else self.config.model.model_id
                ),
                text="",
                status=SampleStatus.FAILED,
                error=str(exc) or type(exc).__name__,
                seed=self.generation.seed + index,
                temperature=candidate_temperature(
                    self.generation.temperature, index, self.generation.candidates_per_prompt
                ),
            )

    def generate(
        self,
        prompts: list[SyntheticPrompt],
        *,
        existing: list[CandidateResponse] | None = None,
    ) -> list[CandidateResponse]:
        """Generate every candidate for ``prompts``, reusing successful ``existing`` ones."""

        per_prompt = int(self.generation.candidates_per_prompt)
        reusable: dict[str, CandidateResponse] = {
            c.candidate_id: c
            for c in (existing or [])
            if c.status is not SampleStatus.FAILED and c.text.strip()
        }
        results: dict[str, CandidateResponse] = {}
        pending: list[tuple[SyntheticPrompt, int, str]] = []
        for prompt in prompts:
            for index in range(per_prompt):
                cid = candidate_id(prompt.prompt_id, index)
                if cid in reusable:
                    results[cid] = reusable[cid]
                else:
                    pending.append((prompt, index, self._source_for(index)))

        total = len(prompts) * per_prompt
        done = len(results)

        def _report() -> None:
            if self.bus is None:
                return
            self.bus.emit(
                "progress",
                f"candidates: {done}/{total}",
                run_id=self.run_id,
                phase="generate",
                fraction=min(1.0, done / max(1, total)),
            )

        _report()
        # Local generation is serialised on the single loaded model.
        teacher_jobs = [(p, i) for p, i, src in pending if src == "teacher"]
        local_jobs = [(p, i) for p, i, src in pending if src == "local"]

        def _run_teacher() -> None:
            nonlocal done
            if not teacher_jobs:
                return
            if not wait_if_paused(self.pause_event, self.cancel_event):
                return

            def _on_done(_count: int) -> None:
                nonlocal done
                done += 1
                _report()

            outcomes = run_bounded(
                (lambda p=p, i=i: self._generate_one(p, i, "teacher") for p, i in teacher_jobs),
                max_workers=self.generation.max_concurrency,
                cancel_event=self.cancel_event,
                on_done=_on_done,
            )
            for candidate in outcomes:
                if candidate is not None:
                    results[candidate.candidate_id] = candidate

        _run_teacher()

        for prompt, index in local_jobs:
            if self.cancel_event is not None and self.cancel_event.is_set():
                break
            if not wait_if_paused(self.pause_event, self.cancel_event):
                break
            candidate = self._generate_one(prompt, index, "local")
            results[candidate.candidate_id] = candidate
            done += 1
            _report()

        ordered: list[CandidateResponse] = []
        for prompt in prompts:
            for index in range(per_prompt):
                candidate = results.get(candidate_id(prompt.prompt_id, index))
                if candidate is not None:
                    ordered.append(candidate)
        return ordered


__all__ = [
    "TEMPERATURE_STEP",
    "CandidateGenerator",
    "build_messages",
    "candidate_id",
    "candidate_temperature",
]
