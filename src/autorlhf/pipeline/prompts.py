"""Synthetic prompt generation.

The teacher model is asked for a *diverse* JSON array of user prompts covering
ordinary interactions, varied phrasings, context-following, multi-turn
continuations of the seed conversations, edge cases, and explicit boundary
probes that try to make the character break its own constraints.

Everything is deterministic and resumable:

* prompts are requested in per-category batches, and each batch is cached by the
  content hash of the request, so resuming a run re-uses what was already paid for;
* prompt ids are assigned from the sorted prompt list (``p0001``...), so the same
  generation always produces the same ids;
* malformed or duplicate entries are rejected and the shortfall is re-requested,
  but only a bounded number of times.
"""

from __future__ import annotations

import threading
from typing import Any

from autorlhf.config.character import CharacterSpec
from autorlhf.config.run_config import GenerationConfig
from autorlhf.core.events import EventBus
from autorlhf.core.ids import content_hash
from autorlhf.core.logging import get_logger
from autorlhf.core.types import ChatMessage
from autorlhf.llm.client import OpenRouterClient
from autorlhf.llm.schemas import EvalPrompt, SyntheticPrompt
from autorlhf.pipeline.util import normalise_text, wait_if_paused
from autorlhf.storage.cache import NS_PROMPTS, ResponseCache

logger = get_logger("autorlhf.pipeline.prompts")

#: Categories a synthetic prompt may carry.
PROMPT_CATEGORIES: tuple[str, ...] = (
    "ordinary",
    "varied_phrasing",
    "context_following",
    "multi_turn",
    "edge_case",
    "boundary",
)

#: Prompts requested per teacher call.
BATCH_SIZE = 8

#: How many extra requests a single category may need before we accept a shortfall.
MAX_ATTEMPTS_PER_CATEGORY = 4

PROMPT_SCHEMA_HINT = (
    'Return exactly one JSON object of the form {"prompts": [ ... ]} and nothing else. '
    "Each element must be an object with the keys: "
    '"text" (the user prompt, non-empty), '
    '"category" (one of: ' + ", ".join(PROMPT_CATEGORIES) + "), "
    '"context" (a JSON array of prior turns, each {"role": "user"|"assistant", '
    '"content": "..."}; use [] when the prompt stands alone), and '
    '"avoids" (a short note on what the prompt is probing, or ""). '
    "Do not include commentary, markdown fences, or duplicate prompts."
)


def plan_categories(target: int, edge_case_fraction: float) -> dict[str, int]:
    """How many prompts to request per category for ``target`` prompts in total."""

    if target <= 0:
        return {category: 0 for category in PROMPT_CATEGORIES}
    edge = max(0.0, min(1.0, float(edge_case_fraction)))
    edge_total = int(round(target * edge))
    ordinary_total = target - edge_total

    weights = {
        "ordinary": (1.0 - edge) * 0.40,
        "varied_phrasing": (1.0 - edge) * 0.25,
        "context_following": (1.0 - edge) * 0.20,
        "multi_turn": (1.0 - edge) * 0.15,
        "edge_case": edge * 0.5,
        "boundary": edge * 0.5,
    }
    counts = {category: 0 for category in PROMPT_CATEGORIES}
    for category, weight in weights.items():
        counts[category] = int(round(target * weight))

    # Fix rounding drift against the intended ordinary/edge totals.
    def _sum(categories: tuple[str, ...]) -> int:
        return sum(counts[c] for c in categories)

    ordinary_cats = ("ordinary", "varied_phrasing", "context_following", "multi_turn")
    edge_cats = ("edge_case", "boundary")
    for cats, wanted in ((ordinary_cats, ordinary_total), (edge_cats, edge_total)):
        drift = _sum(cats) - wanted
        index = 0
        while drift > 0:
            category = cats[index % len(cats)]
            if counts[category] > 0:
                counts[category] -= 1
                drift -= 1
            index += 1
            if index > 64:
                break
        index = 0
        while drift < 0:
            counts[cats[index % len(cats)]] += 1
            drift += 1
            index += 1
            if index > 64:
                break
    return counts


def _seed_excerpts(character: CharacterSpec, *, limit: int = 3) -> list[str]:
    excerpts: list[str] = []
    for seed in character.valid_seeds()[:limit]:
        label = seed.label or f"seed-{len(excerpts) + 1}"
        excerpts.append(f"[{label}]\n{seed.as_text()}")
    return excerpts


def _build_messages(
    character: CharacterSpec,
    *,
    category: str,
    count: int,
    seed: int,
) -> list[ChatMessage]:
    lines = [
        f"Write {count} distinct user prompts for the character below.",
        f"Category: {category}.",
        "",
        "# Character specification",
        character.compact_summary(max_chars=3000),
    ]
    if category == "boundary":
        lines.append(
            "# Boundary probes\n"
            "Each prompt must actively try to make the character break one of the "
            "listed constraints (e.g. ask for programming help, ask it to admit being "
            "an AI, push it past its length limit)."
        )
    elif category == "edge_case":
        lines.append(
            "# Edge cases\n"
            "Use ambiguous, hostile, nonsensical, or emotionally charged prompts, and "
            "prompts that invite the character to invent things it should not."
        )
    elif category in {"multi_turn", "context_following"}:
        lines.append(
            "# Prior turns\n"
            "Give each prompt a non-empty 'context' array of earlier turns so the "
            "prompt reads as a continuation. Alternate user/assistant and end the "
            "context with an assistant turn."
        )
    elif category == "varied_phrasing":
        lines.append(
            "# Variety\nVary register and length: terse, chatty, formal, dialect, "
            "questions, imperatives, single words."
        )
    else:
        lines.append(
            "# Ordinary interactions\nEveryday questions the character would genuinely "
            "receive on a normal day."
        )

    excerpts = _seed_excerpts(character)
    if excerpts and category in {"multi_turn", "context_following", "ordinary"}:
        lines.append("")
        lines.append("# Example conversations to continue or vary")
        lines.extend(excerpts)
    lines.append("")
    lines.append(f"Use a different phrasing for every prompt. Seed: {seed}.")
    return [ChatMessage(role="system", content="\n".join(lines))]


def _clean_item(item: Any, *, category: str) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    text = item.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    raw_category = item.get("category")
    resolved = raw_category if raw_category in PROMPT_CATEGORIES else category
    context: list[dict[str, str]] = []
    raw_context = item.get("context")
    if isinstance(raw_context, list):
        for turn in raw_context:
            if not isinstance(turn, dict):
                continue
            role = turn.get("role")
            content = turn.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                context.append({"role": role, "content": content})
    avoids = item.get("avoids")
    return {
        "text": text.strip(),
        "category": resolved,
        "context": context,
        "avoids": avoids.strip() if isinstance(avoids, str) else "",
    }


def _generate_batch(
    *,
    client: OpenRouterClient,
    character: CharacterSpec,
    generation: GenerationConfig,
    category: str,
    batch_size: int,
    attempt: int,
    cache: ResponseCache | None,
    run_id: str,
) -> list[dict[str, Any]]:
    cache_key = content_hash(
        {
            "kind": "prompts",
            "model": generation.teacher_model,
            "character": character.fingerprint(),
            "category": category,
            "batch_size": batch_size,
            "attempt": attempt,
            "seed": generation.seed,
            "edge_case_fraction": generation.edge_case_fraction,
        }
    )
    if cache is not None:
        cached = cache.get(NS_PROMPTS, cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("items"), list):
            return [item for item in cached["items"] if isinstance(item, dict)]

    messages = _build_messages(
        character, category=category, count=batch_size, seed=generation.seed + attempt
    )
    data, _result = client.chat_json(
        model=generation.teacher_model,
        messages=messages,
        schema_hint=PROMPT_SCHEMA_HINT,
        temperature=0.9,
        top_p=generation.top_p,
        max_tokens=2048,
        seed=generation.seed + attempt,
    )
    raw_items: list[Any]
    if isinstance(data.get("prompts"), list):
        raw_items = data["prompts"]
    elif isinstance(data.get("items"), list):
        raw_items = data["items"]
    elif "text" in data:
        raw_items = [data]
    else:
        raw_items = []

    cleaned = [item for item in (_clean_item(raw, category=category) for raw in raw_items) if item]
    if cache is not None and cleaned:
        cache.put(
            NS_PROMPTS,
            cache_key,
            {"items": cleaned},
            model=generation.teacher_model,
        )
    return cleaned


def generate_prompts(
    *,
    client: OpenRouterClient,
    character: CharacterSpec,
    generation: GenerationConfig,
    run_id: str = "",
    cache: ResponseCache | None = None,
    bus: EventBus | None = None,
    cancel_event: threading.Event | None = None,
    pause_event: threading.Event | None = None,
) -> list[SyntheticPrompt]:
    """Generate the synthetic prompt set, deterministically and resumably."""

    target = int(generation.num_prompts)
    plan = plan_categories(target, generation.edge_case_fraction)
    collected: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []

    for category in PROMPT_CATEGORIES:
        wanted = plan.get(category, 0)
        if wanted <= 0:
            continue
        got = 0
        attempt = 0
        while got < wanted and attempt < MAX_ATTEMPTS_PER_CATEGORY:
            if cancel_event is not None and cancel_event.is_set():
                break
            wait_if_paused(pause_event, cancel_event)
            need = min(BATCH_SIZE, wanted - got)
            try:
                items = _generate_batch(
                    client=client,
                    character=character,
                    generation=generation,
                    category=category,
                    batch_size=need,
                    attempt=attempt,
                    cache=cache,
                    run_id=run_id,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, then retried
                warnings.append(f"prompt generation for '{category}' failed: {exc}")
                logger.warning("prompt batch (%s) failed: %s", category, exc)
                items = []
            for item in items:
                key = normalise_text(item["text"])
                if key in collected:
                    continue
                collected[key] = item
                if item["category"] == category:
                    got += 1
                if got >= wanted:
                    break
            attempt += 1
            if bus is not None:
                bus.emit(
                    "progress",
                    f"prompts: {len(collected)}/{target}",
                    run_id=run_id,
                    phase="generate",
                    fraction=min(1.0, len(collected) / max(1, target)),
                    category=category,
                )
        if got < wanted:
            warnings.append(
                f"only {got}/{wanted} '{category}' prompts were usable after "
                f"{MAX_ATTEMPTS_PER_CATEGORY} attempts"
            )

    prompts: list[SyntheticPrompt] = []
    for index, item in enumerate(
        sorted(collected.values(), key=lambda i: (i["category"], normalise_text(i["text"])))
    ):
        prompts.append(
            SyntheticPrompt(
                prompt_id=f"p{index + 1:04d}",
                run_id=run_id,
                category=item["category"],
                text=item["text"],
                context=item["context"],
                avoids=item["avoids"],
                generator_model=generation.teacher_model,
            )
        )
    if len(prompts) < target:
        warnings.append(
            f"generated {len(prompts)} prompts of the {target} requested; "
            "the split minimums still apply"
        )
    for warning in warnings:
        logger.warning(warning)
        if bus is not None:
            bus.emit("warning", warning, run_id=run_id, phase="generate")
    return prompts


def seed_eval_prompts(character: CharacterSpec, *, limit: int | None = None) -> list[EvalPrompt]:
    """Held-out prompts derived from the seed conversations' user turns."""

    prompts: list[EvalPrompt] = []
    for seed in character.valid_seeds():
        turns = seed.to_training_messages()
        for index, message in enumerate(turns):
            if message.role != "user":
                continue
            context = [
                {"role": m.role, "content": m.content} for m in turns[:index] if m.role != "system"
            ]
            prompts.append(
                EvalPrompt(
                    prompt_id=f"e{len(prompts) + 1:04d}",
                    text=message.content,
                    context=context,
                    category="seed",
                    origin="seed",
                )
            )
            break
        if limit is not None and len(prompts) >= limit:
            break
    return prompts


__all__ = [
    "BATCH_SIZE",
    "MAX_ATTEMPTS_PER_CATEGORY",
    "PROMPT_CATEGORIES",
    "PROMPT_SCHEMA_HINT",
    "generate_prompts",
    "plan_categories",
    "seed_eval_prompts",
]
