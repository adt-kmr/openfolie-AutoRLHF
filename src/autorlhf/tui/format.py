"""Presentation helpers shared by the TUI widgets and screens.

Nothing here talks to the network, the database, or Textual: these are pure
functions so they can be unit-tested and reused from every screen.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def pct(value: float | None, *, digits: int = 0) -> str:
    """``0.42`` -> ``"42%"``; ``None`` stays explicit rather than becoming 0."""

    if value is None:
        return "n/a"
    return f"{value * 100:.{digits}f}%"


def num(value: float | None, *, digits: int = 3) -> str:
    """Compact float rendering that never hides tiny or huge magnitudes."""

    if value is None:
        return "n/a"
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e5):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def integer(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, rest = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {int(rest):02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes):02d}m"


def human_size(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def clip(text: str, limit: int) -> str:
    """Truncate on a word boundary where possible, always marking the cut."""

    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0] or text[:limit]
    return cut + " ..."


def probability_bar(
    probabilities: Mapping[str, float] | None,
    *,
    width: int = 20,
    labels: Mapping[str, str] | None = None,
) -> str:
    """Render a judge's normalised distribution as ASCII bars.

    The probabilities are the judge's own numbers; this only draws them.  When
    they are missing the caller sees that explicitly instead of a zero.
    """

    if not probabilities:
        return "(judge returned no probabilities)"
    items = [(str(key), float(value)) for key, value in probabilities.items() if value is not None]
    total = sum(max(0.0, value) for _, value in items)
    if total <= 0:
        return "(judge returned all-zero probabilities)"
    out: list[str] = []
    for key, value in items:
        share = max(0.0, value) / total
        filled = int(round(share * width))
        label = (labels or {}).get(key, key)
        out.append(f"{label:<12} {'#' * filled}{'.' * (width - filled)} {share:6.1%}")
    return "\n".join(out)


def score_scale(values: Sequence[float], *, step: int = 2) -> list[tuple[float, float]]:
    """Bucket ``values`` into ``(lower, fraction)`` pairs for a crude histogram."""

    if not values:
        return []
    low, high = min(values), max(values)
    if high <= low:
        return [(low, 1.0)]
    buckets = [0] * step
    for value in values:
        index = min(step - 1, int((value - low) / (high - low) * step))
        buckets[index] += 1
    total = float(len(values))
    return [(low + (high - low) * i / step, count / total) for i, count in enumerate(buckets)]


def overall_from_criteria(
    scores: Mapping[str, float] | None, weights: Mapping[str, float] | None
) -> float | None:
    """Weighted mean over the criteria that actually returned a score."""

    if not scores:
        return None
    total = 0.0
    weight = 0.0
    for criterion_id, score in scores.items():
        if score is None:
            continue
        item_weight = float((weights or {}).get(criterion_id, 1.0))
        total += float(score) * item_weight
        weight += item_weight
    if weight <= 0:
        return None
    return total / weight


__all__ = [
    "clip",
    "human_duration",
    "human_size",
    "integer",
    "num",
    "overall_from_criteria",
    "pct",
    "probability_bar",
    "score_scale",
]
