"""Content-addressed cache for paid OpenRouter calls.

Resuming or revisiting a run must not repeat paid generation or judgement
calls.  Keys are content hashes of the request payload, so a change to the
character, rubric, prompt, model or decoding settings produces a different key
and a genuinely new call.

Records the tokens/cost reported by the API so the dashboard can show what a
run has actually cost.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from autorlhf.core.ids import content_hash
from autorlhf.storage.db import RunStore

T = TypeVar("T")

#: Namespaces kept separate so stats and invalidation are per-purpose.
NS_CHAT = "chat.completion"
NS_JEV_SCORE = "jev.score"
NS_JEV_CHOICE = "jev.choice"
NS_JEV_NOUL = "jev.noul"
NS_LOCAL_GEN = "local.generation"
NS_PROMPTS = "synthetic.prompts"


class ResponseCache:
    """Small typed wrapper over the ``cache`` table."""

    def __init__(self, store: RunStore, *, enabled: bool = True) -> None:
        self.store = store
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    # -- keys -------------------------------------------------------------- #

    @staticmethod
    def make_key(payload: Any) -> str:
        """Hash any JSON-serializable request description."""

        return content_hash(payload)

    # -- api --------------------------------------------------------------- #

    def get(self, namespace: str, key: str) -> Any | None:
        if not self.enabled:
            return None
        value = self.store.cache_get(namespace, key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        model: str = "",
        input_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        if not self.enabled:
            return
        self.store.cache_put(
            namespace,
            key,
            value,
            model=model,
            input_tokens=input_tokens,
            cost_usd=cost_usd,
        )

    def lookup_or_compute(
        self,
        namespace: str,
        key: str,
        compute: Callable[[], T],
        *,
        model: str = "",
        on_store: Callable[[T, str, str], dict[str, Any]] | None = None,
    ) -> tuple[T, bool]:
        """Return ``(value, from_cache)``.

        ``on_store`` may return ``{"input_tokens": int, "cost_usd": float,
        "model": str}`` describing the fresh response so usage is persisted with
        the cached value.
        """

        cached = self.get(namespace, key)
        if cached is not None:
            return cached, True  # type: ignore[return-value]
        value = compute()
        meta = on_store(value) if on_store else {}
        self.put(
            namespace,
            key,
            value,
            model=str(meta.get("model", model)),
            input_tokens=int(meta.get("input_tokens", 0) or 0),
            cost_usd=float(meta.get("cost_usd", 0.0) or 0.0),
        )
        return value, False

    def stats(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "namespaces": self.store.cache_stats(),
        }


__all__ = [
    "NS_CHAT",
    "NS_JEV_CHOICE",
    "NS_JEV_NOUL",
    "NS_JEV_SCORE",
    "NS_LOCAL_GEN",
    "NS_PROMPTS",
    "ResponseCache",
]
