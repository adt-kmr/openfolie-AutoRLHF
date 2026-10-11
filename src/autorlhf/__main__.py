"""``python -m autorlhf`` — same behaviour as the ``autorlhf`` console script."""

from __future__ import annotations

import sys

from autorlhf.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
