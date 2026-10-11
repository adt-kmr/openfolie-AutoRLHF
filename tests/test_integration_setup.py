"""Integration: the credential setup wizard and secret handling."""

from __future__ import annotations

import os
import stat

from autorlhf import setup_env
from autorlhf.config.settings import Environment, render_env_text


def test_render_env_text_preserves_comments_and_unknown_keys():
    existing = "# leading comment\nFOO=bar\n\n# inline note\nBAZ=qux\nexport QUX=1\n"
    out = render_env_text({"FOO": "new"}, existing_text=existing)
    assert "# leading comment" in out
    assert "# inline note" in out
    assert "FOO=new" in out
    assert "BAZ=qux" in out
    assert "QUX=1" in out
    # A brand-new key is appended.
    out2 = render_env_text({"ADDED": "yes"}, existing_text=out)
    assert "ADDED=yes" in out2
    assert "# leading comment" in out2


def test_secret_update_preserves_unrelated_lines_and_sets_mode(app_paths):
    env = Environment(app_paths)
    path = env.secrets.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# my comment\nOPENROUTER_API_KEY=old-key\nUNKNOWN_KEY=keepme\n", encoding="utf-8"
    )
    if os.name == "posix":
        os.chmod(path, 0o644)

    env.secrets.update({"WANDB_API_KEY": "wandb-secret"})

    text = path.read_text(encoding="utf-8")
    assert "# my comment" in text
    assert "OPENROUTER_API_KEY=old-key" in text
    assert "UNKNOWN_KEY=keepme" in text
    assert "WANDB_API_KEY=wandb-secret" in text
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_configure_wandb_persists_and_preserves(app_paths, monkeypatch):
    env = Environment(app_paths)
    path = env.secrets.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("HF_TOKEN=keep-me\n", encoding="utf-8")

    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: True)
    monkeypatch.setattr(setup_env, "prompt_secret", lambda *a, **k: "wb-key-123")
    monkeypatch.setattr(setup_env, "prompt_text", lambda label, default="", **k: "demo-project")
    checked = []
    monkeypatch.setattr(
        setup_env,
        "test_wandb_connection",
        lambda key, project, entity: checked.append((key, project, entity)) or (True, "connected"),
    )

    setup_env.configure_wandb(env)

    text = path.read_text(encoding="utf-8")
    assert "HF_TOKEN=keep-me" in text
    assert "WANDB_API_KEY=wb-key-123" in text
    assert "WANDB_ENABLED=true" in text
    assert env.settings.wandb_enabled is True
    assert env.settings.wandb_project == "demo-project"
    assert checked == [("wb-key-123", "demo-project", "demo-project")]
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_configure_wandb_reuses_saved_key_for_connection_test(app_paths, monkeypatch):
    env = Environment(app_paths)
    env.secrets.update({"WANDB_API_KEY": "saved-key"})
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: True)
    monkeypatch.setattr(setup_env, "prompt_secret", lambda *a, **k: "")
    monkeypatch.setattr(setup_env, "prompt_text", lambda label, default="", **k: "demo-project")
    checked = []
    monkeypatch.setattr(
        setup_env,
        "test_wandb_connection",
        lambda key, project, entity: checked.append((key, project, entity)) or (True, "connected"),
    )
    setup_env.configure_wandb(env)
    assert checked == [("saved-key", "demo-project", "demo-project")]
    assert env.secrets.get("WANDB_API_KEY") == "saved-key"
    assert env.settings.wandb_project == "demo-project"


def test_test_wandb_connection_reports_missing_optional_dependency(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def import_without_wandb(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_wandb)
    valid, message = setup_env.test_wandb_connection("key", "project")
    assert valid is False
    assert "uv sync --extra tracking" in message


def test_configure_wandb_does_not_enable_without_any_api_key(app_paths, monkeypatch):
    env = Environment(app_paths)
    env.settings.wandb_enabled = True
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: True)
    monkeypatch.setattr(setup_env, "prompt_secret", lambda *a, **k: "")
    monkeypatch.setattr(setup_env, "prompt_text", lambda label, default="", **k: default)
    setup_env.configure_wandb(env)
    assert env.settings.wandb_enabled is False
    assert env.secrets.get("WANDB_ENABLED") == "false"
    assert env.secrets.get("WANDB_API_KEY") is None


def test_test_wandb_connection_checks_project_access(monkeypatch):
    import sys
    from types import SimpleNamespace

    class Api:
        def __init__(self, *, api_key):
            assert api_key == "key"
            self.viewer = SimpleNamespace(username="tester")

        def project(self, project, entity):
            assert (project, entity) == ("experiment", "team")
            return object()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Api=Api))
    valid, message = setup_env.test_wandb_connection("key", "experiment", "team")
    assert valid is True
    assert "tester" in message


def test_configure_weave_requires_explicit_consent(app_paths, monkeypatch):
    env = Environment(app_paths)

    # Declining: no consent, disabled, returns False.
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: False)
    assert setup_env.configure_weave(env) is False
    assert env.settings.weave_consent is False
    assert env.settings.weave_enabled is False
    assert env.secrets.get("WEAVE_ENABLED") == "false"

    # Accepting: consent recorded, enabled, returns True.
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: True)
    monkeypatch.setattr(setup_env, "prompt_text", lambda label, default="", **k: "traces")
    assert setup_env.configure_weave(env) is True
    assert env.settings.weave_consent is True
    assert env.settings.weave_enabled is True
    assert env.secrets.get("WEAVE_PROJECT") == "traces"
    assert env.secrets.get("WEAVE_ENABLED") == "true"


def test_interactive_setup_stores_credentials(app_paths, monkeypatch):
    env = Environment(app_paths)

    def _secret(label: str, *, allow_empty: bool = True) -> str:
        if "OpenRouter" in label:
            return "or-key-abcdef"
        if "Hugging" in label:
            return "hf-token-xyz"
        return ""

    monkeypatch.setattr(setup_env, "prompt_secret", _secret)
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(setup_env, "prompt_text", lambda label, default="", **k: default)

    calls: list[str] = []

    def _validate(key: str) -> tuple[bool, str]:
        calls.append(key)
        return True, "accepted"

    settings = setup_env.interactive_setup(env=env, ask_wandb=True, force=True, validate=_validate)

    assert calls == ["or-key-abcdef"]
    assert env.secrets.get("OPENROUTER_API_KEY") == "or-key-abcdef"
    assert env.secrets.get("HF_TOKEN") == "hf-token-xyz"
    assert settings.first_run_complete is True
    assert env.settings.first_run_complete is True


def test_interactive_setup_handles_rejected_key(app_paths, monkeypatch):
    env = Environment(app_paths)

    def _secret(label: str, *, allow_empty: bool = True) -> str:
        return "bad-key-000000" if "OpenRouter" in label else ""

    monkeypatch.setattr(setup_env, "prompt_secret", _secret)
    # Decline "store anyway", decline W&B/Weave.
    monkeypatch.setattr(setup_env, "prompt_yes_no", lambda *a, **k: False)

    setup_env.interactive_setup(
        env=env, ask_wandb=False, force=True, validate=lambda key: (False, "rejected")
    )
    # The rejected key must not be stored when the user declines.
    assert env.secrets.get("OPENROUTER_API_KEY") is None
    assert env.settings.first_run_complete is True
