"""Structured logging.

Two sinks:

* a stderr handler that is **always redaction-aware** (secrets never reach it),
* an optional JSONL handler writing to ``<run>/logs/autorlhf.jsonl``.

Log lines are structured: ``logger.info("msg", extra={"extra_fields": {...}})``
is flattened into the JSONL record, and ``EventBus`` events can be mirrored by
:func:`attach_event_logging`.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

# Keys whose values are replaced anywhere they appear in a log record.
SECRET_KEYS = (
    "api_key",
    "apikey",
    "authorization",
    "hf_token",
    "token",
    "wandb_api_key",
    "weave_api_key",
    "openrouter_api_key",
    "password",
    "secret",
    "bearer",
)

#: Generic words that mark a *field name* as secret when they appear as a whole
#: word: ``my_key``, ``key_value`` and ``access_token`` are redacted, while
#: ``keywords`` and ``monkey`` are not.
GENERIC_SECRET_WORDS = frozenset(
    {
        "key",
        "keys",
        "token",
        "tokens",
        "secret",
        "secrets",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "authorization",
        "bearer",
    }
)

_WORD_SPLIT = re.compile(r"[^a-z0-9]+")

#: Matches ``name=value`` (also ``name="value"``) in free text so the *name* can
#: be judged by :func:`is_secret_name`.
_ASSIGNMENT_RE = re.compile(r"(?P<name>[A-Za-z][A-Za-z0-9_.\-]*)\s*=\s*(?P<value>[^\s,;)}\]]*)")


def is_secret_name(name: str) -> bool:
    """Whether a field name looks like it holds a credential."""

    lowered = name.lower()
    if any(fragment in lowered for fragment in SECRET_KEYS):
        return True
    return any(word in GENERIC_SECRET_WORDS for word in _WORD_SPLIT.split(lowered))


_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"),
)

REDACTED = "***REDACTED***"


class _RedactionFilter(logging.Filter):
    """Best-effort scrubber applied to every formatted record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.msg = redact_text(record.getMessage())
            record.args = ()
        except Exception:  # pragma: no cover - never break logging
            pass
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            record.extra_fields = redact_mapping(extra)
        return True


def redact_text(text: str) -> str:
    """Replace anything that looks like a credential."""

    out = text
    for pattern in _SECRET_PATTERNS:
        out = pattern.sub(REDACTED, out)
    # Redact ``name=value`` pairs whose *name* looks like a credential, using the
    # same word-boundary rule as the mapping path so ``key_value=`` is redacted
    # while ``monkey=`` and ``keywords=`` are not.
    for match in reversed(list(_ASSIGNMENT_RE.finditer(out))):
        if is_secret_name(match.group("name")):
            out = out[: match.start("value")] + REDACTED + out[match.end() :]
    return out


def redact_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact by key name and by value shape."""

    out: dict[str, Any] = {}
    for key, value in data.items():
        if is_secret_name(key):
            out[key] = REDACTED
        elif isinstance(value, dict):
            out[key] = redact_mapping(value)
        elif isinstance(value, list):
            out[key] = [
                redact_mapping(v)
                if isinstance(v, dict)
                else redact_text(v)
                if isinstance(v, str)
                else v
                for v in value
            ]
        elif isinstance(value, str):
            out[key] = redact_text(value)
        else:
            out[key] = value
    return out


class JsonlFormatter(logging.Formatter):
    """One JSON object per line, with redaction applied."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 4),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
        }
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(redact_mapping(extra))
        if record.exc_info:
            payload["exception"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable single-line stderr output; used only in headless mode."""

    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<7} {record.name}: {redact_text(record.getMessage())}"
        )
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            base += " " + json.dumps(redact_mapping(extra), default=str)
        if record.exc_info:
            base += "\n" + redact_text(self.formatException(record.exc_info))
        return base


_configured: set[str] = set()
_lock = threading.Lock()


def configure_logging(
    *,
    verbose: bool = False,
    log_file: Path | None = None,
    logger_name: str = "autorlhf",
    force: bool = False,
) -> logging.Logger:
    """Idempotently configure the ``autorlhf`` logger tree.

    Calling twice with the same ``log_file`` is a no-op so re-entering the TUI
    does not duplicate handlers.
    """

    key = f"{log_file}:{verbose}"
    logger = logging.getLogger(logger_name)
    with _lock:
        if key in _configured and not force:
            return logger
        _configured.add(key)
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.propagate = False

        redaction = _RedactionFilter()
        if not any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in logger.handlers
        ):
            stream = logging.StreamHandler(sys.stderr)
            stream.setFormatter(ConsoleFormatter())
            stream.addFilter(redaction)
            stream.setLevel(logging.DEBUG if verbose else logging.WARNING)
            logger.addHandler(stream)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            existing = {
                Path(h.baseFilename).resolve()
                for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            }
            if log_file.resolve() not in existing:
                fh = logging.FileHandler(log_file, encoding="utf-8")
                fh.setFormatter(JsonlFormatter())
                fh.addFilter(redaction)
                fh.setLevel(logging.DEBUG)
                logger.addHandler(fh)
    return logger


def get_logger(name: str = "autorlhf") -> logging.Logger:
    return logging.getLogger(name)


def attach_event_logging(bus: Any, logger_name: str = "autorlhf.events") -> Any:
    """Mirror every bus event into the logger (verbose events go to DEBUG)."""

    logger = get_logger(logger_name)

    def _listener(event: Any) -> None:
        level = (
            logging.ERROR
            if event.level == "error"
            else (
                logging.WARNING
                if event.level == "warning"
                else (logging.DEBUG if event.verbose_only else logging.INFO)
            )
        )
        logger.log(
            level,
            event.message or event.kind,
            extra={"extra_fields": {"kind": event.kind, "phase": event.phase, **event.data}},
        )

    return bus.subscribe(_listener)


__all__ = [
    "REDACTED",
    "SECRET_KEYS",
    "attach_event_logging",
    "configure_logging",
    "get_logger",
    "redact_mapping",
    "redact_text",
]
