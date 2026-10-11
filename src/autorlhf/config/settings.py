"""Credentials and non-secret application settings.

Credential file format is dotenv so it interoperates with ``export $(cat ...)``
and the ecosystem::

    OPENROUTER_API_KEY=
    HF_TOKEN=
    WANDB_API_KEY=
    WANDB_ENTITY=
    WANDB_PROJECT=
    WEAVE_ENABLED=false
    WEAVE_PROJECT=
    WANDB_ENABLED=false

Rules enforced here:

* updates preserve unrelated keys and comments,
* the file is written atomically with ``0600`` on POSIX,
* values are never logged or rendered: :meth:`SecretStore.redacted` and
  ``AppSettings.public_view`` are the only ways the UI reads them,
* an existing project-local ``.env`` is *read* (lowest precedence) but never
  written.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from autorlhf.config.paths import AppPaths
from autorlhf.core.errors import CredentialError
from autorlhf.core.logging import redact_text

#: Every secret name autorlhf understands.  Keep in sync with ``.env.example``.
KNOWN_ENV_KEYS = (
    "OPENROUTER_API_KEY",
    "HF_TOKEN",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WEAVE_ENABLED",
    "WEAVE_PROJECT",
    "WANDB_ENABLED",
)

#: Names whose values must never be displayed.
SECRET_ENV_KEYS = frozenset({"OPENROUTER_API_KEY", "HF_TOKEN", "WANDB_API_KEY", "WEAVE_API_KEY"})


def _is_secret(name: str) -> bool:
    return name in SECRET_ENV_KEYS or "KEY" in name.upper() or "TOKEN" in name.upper()


def parse_env_text(text: str) -> dict[str, str]:
    """Parse dotenv text without importing dotenv (small, predictable)."""

    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def render_env_text(values: dict[str, str], *, existing_text: str = "") -> str:
    """Rewrite ``existing_text`` with ``values`` applied, preserving everything
    else (comments, unknown keys, ordering)."""

    lines = existing_text.splitlines()
    remaining = dict(values)
    out: list[str] = []
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(raw)
            continue
        key = stripped.split("=", 1)[0].strip().removeprefix("export ").strip()
        if key in remaining:
            out.append(f"{key}={remaining.pop(key)}")
        else:
            out.append(raw)
    if remaining:
        if out and out[-1].strip():
            out.append("")
        out.extend(f"{k}={v}" for k, v in remaining.items())
    return "\n".join(out) + "\n"


class SecretStore:
    """Read/write access to the credential file."""

    def __init__(self, path: Path, *, fallback_paths: tuple[Path, ...] = ()) -> None:
        self.path = path
        self.fallback_paths = fallback_paths
        self._cache: dict[str, str] | None = None

    # -- reading ----------------------------------------------------------- #

    def load(self, *, refresh: bool = False) -> dict[str, str]:
        if self._cache is not None and not refresh:
            return dict(self._cache)
        values: dict[str, str] = {}
        # Lowest precedence first: project .env, then the canonical file.
        for path in (*self.fallback_paths, self.path):
            try:
                if path.is_file():
                    values.update(parse_env_text(path.read_text(encoding="utf-8")))
            except OSError:
                continue
        # Real environment wins — that is how CI and `docker run -e` work.
        for key in KNOWN_ENV_KEYS:
            if os.environ.get(key):
                values[key] = os.environ[key]
        self._cache = values
        return dict(values)

    def get(self, key: str, default: str | None = None) -> str | None:
        value = self.load().get(key)
        return value if value not in (None, "") else default

    def require(self, key: str, purpose: str) -> str:
        value = self.get(key)
        if not value:
            raise CredentialError(
                f"{key} is not set. {purpose} needs it.",
                hint=f"Run `autorlhf --setup` or set {key} in {self.path} / your environment.",
            )
        return value

    def enabled(self, key: str) -> bool:
        return str(self.get(key, "false")).strip().lower() in {"1", "true", "yes", "on"}

    # -- writing ----------------------------------------------------------- #

    def update(self, values: dict[str, str]) -> None:
        """Merge ``values`` into the file, preserving unrelated content."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if self.path.is_file():
            existing = self.path.read_text(encoding="utf-8")
        new_text = render_env_text({k: v for k, v in values.items()}, existing_text=existing)
        _atomic_write_private(self.path, new_text)
        self._cache = None

    def clear(self, key: str) -> None:
        self.update({key: ""})

    def redacted(self) -> dict[str, str]:
        """Key -> masked value, safe to render in the TUI."""

        out: dict[str, str] = {}
        for key, value in self.load().items():
            out[key] = mask_secret(value) if _is_secret(key) else value
        return out

    def describe(self) -> list[dict[str, Any]]:
        """Rows for the settings screen."""

        values = self.load()
        rows = []
        for key in KNOWN_ENV_KEYS:
            value = values.get(key, "")
            rows.append(
                {
                    "key": key,
                    "present": bool(value),
                    "secret": _is_secret(key),
                    "display": mask_secret(value) if _is_secret(key) else (value or "(unset)"),
                }
            )
        return rows


def mask_secret(value: str | None) -> str:
    """Show at most 4 leading and 2 trailing characters."""

    if not value:
        return "(unset)"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}…{value[-2:]} ({len(value)} chars)"


def _atomic_write_private(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically and with ``0600`` permissions."""

    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".env.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name == "posix":
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        os.replace(tmp, path)
        if os.name == "posix":
            try:
                os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:  # pragma: no cover - e.g. exotic filesystems
                pass
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class AppSettings(BaseModel):
    """Non-secret, app-wide preferences (``settings.json``)."""

    schema_version: int = 1
    first_run_complete: bool = False
    #: Default feedback mode when neither ``-i`` nor a saved default applies.
    default_feedback_mode: str = "automated"
    verbose: bool = False
    cache_enabled: bool = True
    #: Concurrent OpenRouter requests during generation/judging.
    max_concurrency: int = 4
    request_timeout_s: float = 120.0
    max_retries: int = 4
    default_hf_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    default_teacher_model: str = "openai/gpt-4o-mini"
    default_judge_model: str = "typesafe/jev-1.13"
    judge_model_alternates: list[str] = Field(
        default_factory=lambda: ["typesafe/jev-1.13", "~typesafe/jev-latest"]
    )
    wandb_enabled: bool = False
    weave_enabled: bool = False
    wandb_project: str = ""
    weave_project: str = ""
    #: Consent recorded when Weave tracing was last enabled.
    weave_consent: bool = False
    #: Preset chosen last time; a hint only, never silently reused for a new run.
    last_preset: str | None = None

    # -- io ---------------------------------------------------------------- #

    @classmethod
    def load(cls, path: Path) -> AppSettings:
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        if not isinstance(raw, dict):
            return cls()
        known = set(cls.model_fields)
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".settings.", suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.model_dump(mode="json"), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def public_view(self) -> dict[str, Any]:
        """Settings with any accidental secret-looking string redacted."""

        return {k: redact_text(str(v)) for k, v in self.model_dump(mode="json").items()}


class Environment:
    """Bundle of paths + secrets + settings, passed to every subsystem."""

    def __init__(self, paths: AppPaths | None = None) -> None:
        self.paths = (paths or AppPaths.default()).ensure()
        self.secrets = SecretStore(
            self.paths.env_path,
            fallback_paths=(self.paths.project_env_path,),
        )
        self.settings = AppSettings.load(self.paths.settings_path)

    # -- convenience ------------------------------------------------------- #

    def refresh(self) -> None:
        self.secrets.load(refresh=True)
        self.settings = AppSettings.load(self.paths.settings_path)

    def save_settings(self) -> None:
        self.settings.save(self.paths.settings_path)

    def openrouter_key(self) -> str | None:
        return self.secrets.get("OPENROUTER_API_KEY")

    def require_openrouter(self) -> str:
        return self.secrets.require(
            "OPENROUTER_API_KEY", "Talking to OpenRouter (generation and Jev)"
        )

    def hf_token(self) -> str | None:
        return self.secrets.get("HF_TOKEN")

    def env_with_secrets(self) -> dict[str, str]:
        """Environment for child processes (never logged)."""

        env = dict(os.environ)
        for key, value in self.secrets.load().items():
            if value:
                env[key] = value
        return env


__all__ = [
    "KNOWN_ENV_KEYS",
    "SECRET_ENV_KEYS",
    "AppSettings",
    "Environment",
    "SecretStore",
    "mask_secret",
    "parse_env_text",
    "render_env_text",
]
