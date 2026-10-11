"""Interactive credential/setup wizard for ``autorlhf --setup``.

Secrets are entered with :func:`getpass.getpass`, so they are never echoed, and
written through :class:`~autorlhf.config.settings.SecretStore`, which preserves
every unrelated line in the file and writes ``0600`` on POSIX.

The OpenRouter key is validated with a cheap live request when supplied; a
failure prints an actionable (doctor-style) message but still lets the user
continue without a working key.
"""

from __future__ import annotations

import getpass
from collections.abc import Callable

from autorlhf.config.settings import AppSettings, Environment
from autorlhf.integrations.tracking import WEAVE_EGRESS_NOTICE

#: Printed before Weave is enabled.  Frames the egress in plain terms.
WEAVE_SETUP_NOTICE = (
    "Weave tracing transmits data off this machine: the prompts sent to the model, "
    "the model's responses, and the Jev judgements recorded against them are uploaded "
    "to an external Weights & Biases / Weave project. Only enable it if you accept that."
)


def _read_line(label: str) -> str:
    try:
        return input(label)
    except EOFError:
        return ""


def prompt_secret(label: str, *, allow_empty: bool = True) -> str:
    """Read a secret without echoing it.  Re-prompts unless empty is allowed."""

    while True:
        try:
            value = getpass.getpass(label)
        except EOFError:
            value = ""
        value = value.strip()
        if value or allow_empty:
            return value
        print("A value is required (press Ctrl-C to abort).")


def prompt_text(label: str, *, default: str = "") -> str:
    """Read a plain, non-secret line, falling back to ``default`` on empty input."""

    value = _read_line(label).strip()
    return value or default


def prompt_yes_no(label: str, *, default: bool = False) -> bool:
    """Read a yes/no answer; empty input returns ``default``."""

    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        answer = _read_line(label + suffix).strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _validate_openrouter(key: str) -> tuple[bool, str]:
    """Cheap live credential check; degrades to a skip if the client is absent."""

    try:
        from autorlhf.llm.client import OpenRouterClient
    except ImportError:
        return True, "skipped validation (llm.client not importable yet)"
    client = None
    try:
        client = OpenRouterClient(key)
        valid, message = client.verify_credentials()
        return bool(valid), message or ("accepted" if valid else "rejected")
    except Exception as exc:  # noqa: BLE001 - surface, never crash setup
        return False, f"{exc.__class__.__name__}: {exc}"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - defensive
                pass


def test_wandb_connection(api_key: str, project: str, entity: str = "") -> tuple[bool, str]:
    """Verify W&B credentials and confirm the configured project is reachable."""

    try:
        import wandb
    except ImportError:
        return False, "The 'wandb' package is not installed. Install with: uv sync --extra tracking"
    try:
        api = wandb.Api(api_key=api_key)
        viewer = api.viewer
        if viewer is None:
            return False, "W&B did not return an authenticated user. Check the API key."
        owner = entity or getattr(viewer, "username", "")
        if not owner or api.project(project, owner) is None:
            return (
                False,
                f"Project '{owner or 'current user'}/{project}' is not accessible to this account.",
            )
    except Exception:
        return (
            False,
            "W&B connection failed. Check the API key, network, entity, and project access.",
        )
    return True, f"Connected to W&B as {getattr(viewer, 'username', 'authenticated user')}"


def configure_wandb(env: Environment) -> None:
    """Prompt for W&B settings, test connectivity, and persist them."""

    enabled = prompt_yes_no(
        "Log training metrics to Weights & Biases?", default=env.settings.wandb_enabled
    )
    if not enabled:
        env.secrets.update({"WANDB_ENABLED": "false"})
        env.settings.wandb_enabled = False
        env.save_settings()
        print("W&B disabled.")
        return

    api_key = prompt_secret(
        "W&B API key (hidden; Enter to keep the existing key): ", allow_empty=True
    ) or env.secrets.get("WANDB_API_KEY", "")
    if not api_key:
        env.settings.wandb_enabled = False
        env.secrets.update({"WANDB_ENABLED": "false"})
        env.save_settings()
        print("W&B was not enabled: an API key is required. Re-run setup after obtaining a key.")
        return
    project = prompt_text("W&B project name: ", default=env.settings.wandb_project or "autorlhf")
    entity = prompt_text(
        "W&B entity/team (optional): ", default=env.secrets.get("WANDB_ENTITY", "")
    )
    if prompt_yes_no("Test W&B connection now?", default=True):
        valid, message = test_wandb_connection(api_key, project, entity)
        print(message)
        if not valid:
            print("Credentials and project settings were saved so you can correct them and retry.")
    env.secrets.update(
        {
            "WANDB_API_KEY": api_key,
            "WANDB_ENABLED": "true",
            "WANDB_PROJECT": project,
            "WANDB_ENTITY": entity,
        }
    )
    env.settings.wandb_enabled = True
    env.settings.wandb_project = project
    env.save_settings()
    print(f"W&B enabled for project '{project}'.")


def configure_weave(env: Environment, *, project: str = "") -> bool:
    """Prompt for and record Weave consent.  Returns the recorded consent value."""

    print(WEAVE_SETUP_NOTICE)
    if not prompt_yes_no("Enable Weave tracing and accept the data egress above?", default=False):
        env.secrets.update({"WEAVE_ENABLED": "false"})
        env.settings.weave_enabled = False
        env.settings.weave_consent = False
        env.save_settings()
        print(WEAVE_EGRESS_NOTICE)
        return False
    if not project:
        project = prompt_text(
            "Weave project name: ", default=env.settings.weave_project or "autorlhf"
        )
    env.secrets.update({"WEAVE_ENABLED": "true", "WEAVE_PROJECT": project})
    env.settings.weave_enabled = True
    env.settings.weave_project = project
    env.settings.weave_consent = True
    env.save_settings()
    print(f"Weave tracing enabled for project '{project}'.")
    return True


def interactive_setup(
    *,
    env: Environment,
    ask_wandb: bool = True,
    force: bool = False,
    validate: Callable[[str], tuple[bool, str]] | None = None,
) -> AppSettings:
    """Run the headless credential wizard and return the updated settings."""

    if env.settings.first_run_complete and not force:
        print("autorlhf is already set up; re-running will update existing credentials.")

    print("autorlhf setup - credentials are written to " + str(env.paths.env_path))
    print("(hidden input; Enter skips a prompt and keeps the current value)")

    key = prompt_secret("OpenRouter API key (hidden): ", allow_empty=True)
    if key:
        checker = validate or _validate_openrouter
        valid, message = checker(key)
        if valid:
            env.secrets.update({"OPENROUTER_API_KEY": key})
            print(f"OpenRouter key stored ({message}).")
        else:
            print(f"Warning: OpenRouter key could not be validated: {message}")
            print("  -> Check network access and the key at openrouter.ai/keys, then re-run setup.")
            if prompt_yes_no("Store this key anyway?", default=False):
                env.secrets.update({"OPENROUTER_API_KEY": key})
                print("Stored unvalidated key.")
            else:
                print("Key not stored; you can still continue.")

    hf_token = prompt_secret("Hugging Face token (hidden, optional): ", allow_empty=True)
    if hf_token:
        env.secrets.update({"HF_TOKEN": hf_token})
        print("Hugging Face token stored.")

    if ask_wandb:
        configure_wandb(env)
        configure_weave(env)

    env.settings.first_run_complete = True
    env.save_settings()
    env.refresh()
    return env.settings


__all__ = [
    "WEAVE_SETUP_NOTICE",
    "configure_wandb",
    "configure_weave",
    "interactive_setup",
    "prompt_secret",
    "prompt_text",
    "prompt_yes_no",
]
