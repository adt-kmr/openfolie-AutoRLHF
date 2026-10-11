"""Environment diagnostics for ``autorlhf --doctor``.

Every check is best-effort and self-contained: a broken install is *reported*,
never raised.  ``run_doctor`` is guaranteed not to raise, so the CLI can always
print a report and derive an exit code from :attr:`DoctorReport.worst`.

No credential value is ever rendered.  Secrets are summarised through
``Environment.secrets.describe()`` (masked) or as a simple present/absent flag.
"""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import stat
import sys
import tempfile
from collections.abc import Callable

from pydantic import BaseModel, Field

from autorlhf.config.settings import Environment

#: Packages that must import for the core app to run.
_CORE_MODULES = ("textual", "pydantic", "httpx")
#: Packages that must import for training to run.
_TRAINING_MODULES = ("torch", "transformers", "trl", "peft", "datasets", "accelerate")
#: Optional extras: reported present-but-optional, never a failure.
_OPTIONAL_MODULES = ("bitsandbytes", "wandb", "weave")


class DoctorCheck(BaseModel):
    name: str
    #: "ok" | "warn" | "fail"
    status: str
    detail: str
    remediation: str = ""


class DoctorReport(BaseModel):
    checks: list[DoctorCheck] = Field(default_factory=list)

    @property
    def worst(self) -> str:
        if any(check.status == "fail" for check in self.checks):
            return "fail"
        if any(check.status == "warn" for check in self.checks):
            return "warn"
        return "ok"

    def render(self) -> str:
        ok = sum(1 for c in self.checks if c.status == "ok")
        warn = sum(1 for c in self.checks if c.status == "warn")
        fail = sum(1 for c in self.checks if c.status == "fail")
        header = (
            f"autorlhf doctor - {self.worst.upper()} "
            f"({ok} ok, {warn} warning(s), {fail} failure(s))"
        )
        lines = [header, "-" * len(header)]
        tags = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
        for check in self.checks:
            lines.append(f"{tags.get(check.status, '[????]')} {check.name}: {check.detail}")
            if check.remediation:
                lines.append(f"        -> {check.remediation}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _module_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except BaseException as exc:  # noqa: BLE001 - report, never raise
        return f"<import failed: {exc.__class__.__name__}: {exc}>"
    return str(getattr(module, "__version__", "unknown"))


def _check_python() -> DoctorCheck:
    version = sys.version_info
    text = platform.python_version()
    if (3, 11) <= (version.major, version.minor) < (3, 13):
        return DoctorCheck(name="python version", status="ok", detail=text)
    return DoctorCheck(
        name="python version",
        status="fail",
        detail=f"{text} (autorlhf requires >=3.11,<3.13)",
        remediation="Install Python 3.11 or 3.12 and recreate the venv: `uv sync --all-extras`.",
    )


def _check_uv_venv() -> DoctorCheck:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    uv = shutil.which("uv")
    detail = f"virtualenv={'yes' if in_venv else 'no'}, uv={'yes' if uv else 'no'}"
    if in_venv and uv:
        return DoctorCheck(name="uv / virtualenv", status="ok", detail=detail)
    problems = []
    if not in_venv:
        problems.append("not running inside a virtualenv")
    if not uv:
        problems.append("the `uv` binary is not on PATH")
    return DoctorCheck(
        name="uv / virtualenv",
        status="warn",
        detail=detail + " (" + "; ".join(problems) + ")",
        remediation="Run `uv sync --all-extras` and invoke commands via `uv run ...`.",
    )


def _check_modules(name: str, modules: tuple[str, ...], *, fatal: bool) -> DoctorCheck:
    parts: list[str] = []
    missing: list[str] = []
    for module in modules:
        version = _module_version(module)
        if version is not None and version.startswith("<import failed"):
            missing.append(f"{module} ({version})")
        else:
            parts.append(f"{module} {version}")
    if missing:
        remediation = (
            "Reinstall the environment: `uv sync --all-extras` "
            "(a broken compile of a compiled dependency usually needs a rebuild)."
        )
        return DoctorCheck(
            name=name,
            status="fail" if fatal else "warn",
            detail="; ".join(missing),
            remediation=remediation,
        )
    return DoctorCheck(name=name, status="ok", detail="; ".join(parts))


def _check_optional_modules() -> DoctorCheck:
    parts: list[str] = []
    for module in _OPTIONAL_MODULES:
        present = importlib.util.find_spec(module) is not None
        parts.append(f"{module} {'available' if present else 'not installed'}")
    return DoctorCheck(
        name="optional extras",
        status="ok",
        detail="; ".join(parts) + " (all optional; both trackers default to off)",
    )


def _check_hardware() -> DoctorCheck:
    try:
        from autorlhf.training.hardware import detect_hardware
    except ImportError as exc:
        return DoctorCheck(
            name="hardware",
            status="warn",
            detail=f"training.hardware could not be imported ({exc})",
            remediation="Training is optional for dataset preparation; install the training deps.",
        )
    try:
        hw = detect_hardware()
    except BaseException as exc:  # noqa: BLE001 - broken torch must be reported
        return DoctorCheck(
            name="hardware",
            status="warn",
            detail=f"hardware detection failed: {exc.__class__.__name__}: {exc}",
            remediation=(
                "A broken torch install is the usual cause; reinstall with `uv sync --all-extras`."
            ),
        )
    detail = (
        f"device={hw.device_type} ({hw.device_name or 'unknown'}), "
        f"devices={hw.device_count}, cpu={hw.cpu_count}, "
        f"ram={hw.total_ram_mib} MiB"
    )
    if hw.total_vram_mib:
        detail += f", vram={hw.total_vram_mib} MiB"
    detail += (
        f", bf16={hw.bf16_supported}, fp16={hw.fp16_supported}, "
        f"bitsandbytes={hw.bitsandbytes_available}"
    )
    return DoctorCheck(name="hardware", status="ok", detail=detail)


def _check_dirs_writable(env: Environment) -> DoctorCheck:
    targets = {
        "config": env.paths.config_dir,
        "data": env.paths.data_dir,
        "runs": env.paths.runs_dir,
    }
    failures: list[str] = []
    ok: list[str] = []
    for label, directory in targets.items():
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=str(directory), prefix=".doctor.", delete=True):
                pass
            ok.append(label)
        except OSError as exc:
            failures.append(f"{label} ({directory}): {exc}")
    if failures:
        return DoctorCheck(
            name="writable directories",
            status="fail",
            detail="; ".join(failures),
            remediation="Fix the ownership/permissions of the config and data directories, "
            "or set AUTORLHF_HOME to a writable path.",
        )
    return DoctorCheck(name="writable directories", status="ok", detail="config, data, runs")


def _check_secret_permissions(env: Environment) -> DoctorCheck:
    path = env.paths.env_path
    if not path.is_file():
        return DoctorCheck(
            name="secret file permissions",
            status="ok",
            detail=f"{path} not created yet (run `autorlhf --setup`)",
        )
    mode = stat.S_IMODE(path.stat().st_mode)
    if os.name != "posix":
        return DoctorCheck(
            name="secret file permissions",
            status="ok",
            detail=f"{path} mode {mode:o} (permission check is POSIX-only)",
        )
    if mode & 0o077:
        return DoctorCheck(
            name="secret file permissions",
            status="warn",
            detail=f"{path} is mode {mode:o}; credentials should be 0600",
            remediation="Restrict access: `chmod 600` the file (autorlhf rewrites it 0600 on update).",
        )
    return DoctorCheck(name="secret file permissions", status="ok", detail=f"mode {mode:o}")


def _check_credentials(env: Environment) -> DoctorCheck:
    rows = env.secrets.describe()
    by_key = {row["key"]: row for row in rows}

    def fmt(key: str) -> str:
        row = by_key.get(key, {"present": False, "display": "(unset)"})
        return f"{key}={row['display']}" if row["present"] else f"{key}=(unset)"

    detail = "; ".join(
        fmt(key)
        for key in (
            "OPENROUTER_API_KEY",
            "HF_TOKEN",
            "WANDB_API_KEY",
            "WANDB_ENABLED",
            "WEAVE_ENABLED",
        )
    )
    if not by_key.get("OPENROUTER_API_KEY", {}).get("present"):
        return DoctorCheck(
            name="credentials",
            status="warn",
            detail=detail,
            remediation="Set the OpenRouter key: `autorlhf --setup` (or export OPENROUTER_API_KEY).",
        )
    return DoctorCheck(name="credentials", status="ok", detail=detail)


def _check_openrouter_live(env: Environment) -> DoctorCheck:
    try:
        from autorlhf.llm.client import OpenRouterClient
    except ImportError as exc:
        return DoctorCheck(
            name="openrouter (live)",
            status="warn",
            detail=f"llm.client could not be imported ({exc})",
            remediation="Install/sync the package: `uv sync --all-extras`.",
        )
    key = env.openrouter_key()
    if not key:
        return DoctorCheck(
            name="openrouter (live)",
            status="warn",
            detail="no OpenRouter key to verify",
            remediation="Run `autorlhf --setup` to record OPENROUTER_API_KEY.",
        )
    client = None
    try:
        client = OpenRouterClient(key)
        valid, message = client.verify_credentials()
    except BaseException as exc:  # noqa: BLE001 - report, never raise
        return DoctorCheck(
            name="openrouter (live)",
            status="warn",
            detail=f"credential check failed: {exc.__class__.__name__}: {exc}",
            remediation="Check network access and that the key is current; retry `autorlhf --doctor --live`.",
        )
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - defensive
                pass
    if valid:
        return DoctorCheck(
            name="openrouter (live)", status="ok", detail=message or "credential accepted"
        )
    return DoctorCheck(
        name="openrouter (live)",
        status="fail",
        detail=message or "credential rejected",
        remediation="Refresh the key at openrouter.ai/keys, then re-run `autorlhf --setup`.",
    )


def _check_hf_live(env: Environment) -> DoctorCheck:
    token = env.hf_token()
    if not token:
        return DoctorCheck(
            name="hugging face (live)",
            status="ok",
            detail="no HF_TOKEN set (only needed for gated/private models)",
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        return DoctorCheck(
            name="hugging face (live)",
            status="warn",
            detail=f"huggingface_hub unavailable ({exc})",
            remediation="Install the training deps: `uv sync --all-extras`.",
        )
    try:
        who = HfApi().whoami(token=token)
    except BaseException as exc:  # noqa: BLE001 - report, never raise
        return DoctorCheck(
            name="hugging face (live)",
            status="warn",
            detail=f"token rejected or unreachable: {exc.__class__.__name__}: {exc}",
            remediation="Create a read token at hf.co/settings/tokens and re-run `autorlhf --setup`.",
        )
    name = who.get("name") if isinstance(who, dict) else None
    return DoctorCheck(
        name="hugging face (live)",
        status="ok",
        detail=f"token valid for {name}" if name else "token valid",
    )


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #


def _safe(name: str, fn: Callable[[], DoctorCheck]) -> DoctorCheck:
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001 - a check must never crash the doctor
        return DoctorCheck(
            name=name,
            status="warn",
            detail=f"check crashed: {exc.__class__.__name__}: {exc}",
            remediation="This is a bug; please report it. The rest of the report is still valid.",
        )


def run_doctor(*, env: Environment, live: bool = False) -> DoctorReport:
    """Collect all checks into a report.  Never raises."""

    checks: list[DoctorCheck] = [
        _safe("python version", _check_python),
        _safe("uv / virtualenv", _check_uv_venv),
        _safe("core imports", lambda: _check_modules("core imports", _CORE_MODULES, fatal=True)),
        _safe(
            "training stack",
            lambda: _check_modules("training stack", _TRAINING_MODULES, fatal=True),
        ),
        _safe("optional extras", _check_optional_modules),
        _safe("hardware", _check_hardware),
        _safe("writable directories", lambda: _check_dirs_writable(env)),
        _safe("secret file permissions", lambda: _check_secret_permissions(env)),
        _safe("credentials", lambda: _check_credentials(env)),
    ]
    if live:
        checks.append(_safe("openrouter (live)", lambda: _check_openrouter_live(env)))
        checks.append(_safe("hugging face (live)", lambda: _check_hf_live(env)))
    return DoctorReport(checks=checks)


__all__ = ["DoctorCheck", "DoctorReport", "run_doctor"]
