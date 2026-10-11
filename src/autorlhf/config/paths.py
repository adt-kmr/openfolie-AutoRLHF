"""Portable path resolution.

Layout (XDG on Linux, ``Library/Application Support`` on macOS, ``%APPDATA%``
on Windows, all via :mod:`platformdirs`)::

    <config>/autorlhf/settings.json          non-secret app settings
    <data>/autorlhf/config.env               secrets (0600)
    <data>/autorlhf/autorlhf.db              SQLite: runs, phases, reviews, cache
    <data>/autorlhf/runs/<run-id>-<slug>/    per-run artifacts

``AUTORLHF_HOME`` overrides the data dir (tests use this); ``AUTORLHF_CONFIG_HOME``
overrides the config dir.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

APP_NAME = "autorlhf"


def _data_root() -> Path:
    override = os.environ.get("AUTORLHF_HOME")
    if override:
        return Path(override).expanduser()
    return Path(user_data_dir(APP_NAME))


def _config_root() -> Path:
    override = os.environ.get("AUTORLHF_CONFIG_HOME")
    if override:
        return Path(override).expanduser()
    return Path(user_config_dir(APP_NAME))


@dataclass(frozen=True)
class AppPaths:
    """Resolved application paths.  Nothing here is created on construction."""

    config_dir: Path
    data_dir: Path

    # -- factory ----------------------------------------------------------- #

    @classmethod
    def default(cls) -> AppPaths:
        return cls(config_dir=_config_root(), data_dir=_data_root())

    @classmethod
    def for_tests(cls, root: Path) -> AppPaths:
        """Isolated layout rooted at ``root``."""

        return cls(config_dir=root / "config", data_dir=root / "data")

    # -- derived files ----------------------------------------------------- #

    @property
    def settings_path(self) -> Path:
        return self.config_dir / "settings.json"

    @property
    def env_path(self) -> Path:
        """Canonical secret file (0600 on POSIX)."""

        return self.data_dir / "config.env"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "autorlhf.db"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def project_env_path(self) -> Path:
        """Optional developer ``.env`` in the current working directory.

        Read-only convenience: autorlhf never writes here, so a checked-out
        repository's ``.env`` is never silently modified.
        """

        return Path.cwd() / ".env"

    # -- mutations --------------------------------------------------------- #

    def ensure(self) -> AppPaths:
        for path in (self.config_dir, self.data_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def run_dir(self, folder_name: str) -> Path:
        return self.runs_dir / folder_name

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"config={self.config_dir} data={self.data_dir}"


__all__ = ["APP_NAME", "AppPaths"]
