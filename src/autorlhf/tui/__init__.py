"""Textual user interface for autorlhf.

Deliberately does not import :mod:`autorlhf.tui.app` here: the CLI imports the
app lazily so ``autorlhf --doctor`` and ``autorlhf --headless`` never pull in
Textual.
"""

from __future__ import annotations

__all__: list[str] = []
