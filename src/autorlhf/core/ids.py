"""Identifier, slug, and content-hash helpers.

All cache keys and dataset fingerprints flow through :func:`content_hash`, so a
change in canonicalisation invalidates every cache at once — deliberate.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from typing import Any

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, max_len: int = 32) -> str:
    """Return a filesystem-safe slug (may be empty for punctuation-only input)."""

    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len].strip("-") or "run"


def new_run_id(now: float | None = None) -> str:
    """``YYYYMMDD-HHMMSS-xxxxxxxx`` — sortable and human-legible.

    The 4-byte (8 hex character) random suffix keeps the id safe as a SQLite
    primary key: two ids generated in the same second collide with probability
    ~1/4.3e9 per pair, and the timestamp prefix keeps ordering legible.
    """

    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    return f"{ts}-{secrets.token_hex(4)}"


def run_folder_name(run_id: str, slug: str) -> str:
    """``<run-id>-<slug>`` as used under ``runs/``."""

    return f"{run_id}-{slugify(slug)}"


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_default)


def content_hash(value: Any) -> str:
    """SHA-256 (first 32 hex chars) of the canonical JSON of ``value``.

    ``value`` may be any JSON-serializable object, or an already-encoded string
    of bytes (hashed verbatim so large files can be streamed in).
    """

    if isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = canonical_json(value).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def short_hash(value: Any, n: int = 8) -> str:
    return content_hash(value)[:n]


def _default(obj: Any) -> Any:
    """Fallback serializer: pydantic models, enums, sets, paths."""

    for attr in ("model_dump", "to_dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn(mode="json") if attr == "model_dump" else fn()
            except TypeError:
                return fn()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if hasattr(obj, "__fspath__"):
        return str(obj)
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum-like
        return obj.value
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")


__all__ = [
    "canonical_json",
    "content_hash",
    "new_run_id",
    "run_folder_name",
    "short_hash",
    "slugify",
]
