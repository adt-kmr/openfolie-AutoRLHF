"""autorlhf — local-first Jev-guided fine-tuning workbench.

The public surface is intentionally small: :func:`autorlhf.cli.main` is the only
entry point the CLI uses.  Everything else is imported directly by the TUI and
the pipeline.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
