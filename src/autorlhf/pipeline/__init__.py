"""The autorlhf data pipeline: prompts, candidates, splits, preferences, review.

Nothing in this package imports ``textual`` or ``torch`` at module import time;
the training stack is pulled in lazily inside the functions that genuinely need
a model, so the TUI can import the pipeline while it is still starting up.
"""

from __future__ import annotations

__all__: list[str] = []
