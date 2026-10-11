"""CLI dispatch: help, version, list-runs, doctor, export and headless."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_tui_support import make_run, run_python, stub_hooks

from autorlhf.cli import main
from autorlhf.config.run_config import RunConfig
from autorlhf.doctor import DoctorCheck, DoctorReport
from autorlhf.tui.worker import PipelineHooks


def isolate(monkeypatch, tmp_path: Path) -> None:
    """Point every supported spelling of the home override at a temp dir."""

    for name in ("AUTORLHF_HOME", "AUTORHLF_HOME"):
        monkeypatch.setenv(name, str(tmp_path / "data"))
    for name in ("AUTORLHF_CONFIG_HOME", "AUTORHLF_CONFIG_HOME"):
        monkeypatch.setenv(name, str(tmp_path / "config"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def test_help_exits_zero(capsys):
    assert main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "autorlhf" in captured.out
    assert "--headless" in captured.out


def test_version_exits_zero(capsys):
    assert main(["--version"]) == 0
    assert "autorlhf" in capsys.readouterr().out


def test_unknown_flag_is_a_usage_error(capsys):
    assert main(["--not-a-flag"]) == 2


def test_list_runs_without_runs_exits_zero(monkeypatch, tmp_path, capsys):
    isolate(monkeypatch, tmp_path)
    assert main(["--list-runs"]) == 0
    assert "no runs yet" in capsys.readouterr().out


def test_list_runs_prints_a_table(monkeypatch, tmp_path, capsys, store, app_paths, run_config):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    run_id, _paths = make_run(store, app_paths, run_config)
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    assert main(["--list-runs"]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "STATUS" in out and "created" in out


def _env(paths):
    from autorlhf.config.settings import Environment

    return Environment(paths)


def test_doctor_reports_and_maps_the_exit_code(monkeypatch, tmp_path, capsys):
    isolate(monkeypatch, tmp_path)
    healthy = DoctorReport(checks=[DoctorCheck(name="python version", status="ok", detail="3.12")])
    monkeypatch.setattr("autorlhf.doctor.run_doctor", lambda *, env, live=False: healthy)
    assert main(["--doctor"]) == 0
    assert "python version" in capsys.readouterr().out

    broken = DoctorReport(
        checks=[
            DoctorCheck(
                name="training stack",
                status="fail",
                detail="torch missing",
                remediation="uv sync --all-extras",
            )
        ]
    )
    monkeypatch.setattr("autorlhf.doctor.run_doctor", lambda *, env, live=False: broken)
    assert main(["--doctor"]) == 1
    assert "uv sync" in capsys.readouterr().out


def test_doctor_survives_a_broken_probe(monkeypatch, tmp_path, capsys):
    isolate(monkeypatch, tmp_path)

    def explode(*, env, live=False):
        raise RuntimeError("probe blew up")

    monkeypatch.setattr("autorlhf.doctor.run_doctor", explode)
    assert main(["--doctor"]) == 2
    assert "probe blew up" in capsys.readouterr().err


def test_setup_delegates_to_the_wizard(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    recorded: list[object] = []
    monkeypatch.setattr(
        "autorlhf.setup_env.interactive_setup",
        lambda *, env, **kwargs: recorded.append(env) or env.settings,
    )
    assert main(["--setup"]) == 0
    assert len(recorded) == 1


def test_export_unknown_run_fails(monkeypatch, tmp_path, store, app_paths, capsys):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    assert main(["--export", "nope", str(tmp_path / "dest")]) == 1
    assert "unknown run" in capsys.readouterr().err


def test_export_writes_a_bundle(monkeypatch, tmp_path, store, app_paths, run_config, capsys):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    run_id, run_paths = make_run(store, app_paths, run_config)
    (run_paths.root / "notes.txt").write_text("hello", encoding="utf-8")
    destination = tmp_path / "bundle"
    assert main(["--export", run_id, str(destination)]) == 0
    assert "exported" in capsys.readouterr().out
    assert (destination / "config.json").is_file()
    # credentials are never exported
    assert not (destination / "config.env").exists()


def test_headless_without_a_target_is_a_usage_error(monkeypatch, tmp_path, capsys):
    isolate(monkeypatch, tmp_path)
    assert main(["--headless"]) == 2
    assert "--config" in capsys.readouterr().err


def test_headless_rejects_an_invalid_config(monkeypatch, tmp_path, capsys, run_config):
    isolate(monkeypatch, tmp_path)
    bad = run_config.model_copy(deep=True)
    bad.generation.num_prompts = 2
    config_path = tmp_path / "bad.json"
    config_path.write_text(bad.to_json(), encoding="utf-8")
    assert main(["--headless", "--config", str(config_path)]) == 2
    assert "invalid" in capsys.readouterr().err


def test_headless_runs_the_worker_and_creates_a_run(
    monkeypatch, tmp_path, capsys, app_paths, store, run_config
):
    isolate(monkeypatch, tmp_path)
    config = run_config.model_copy(deep=True)
    config.generation.num_prompts = 25  # self-consistent splits
    config_path = tmp_path / "run.json"
    config_path.write_text(config.to_json(), encoding="utf-8")

    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    monkeypatch.setattr("autorlhf.tui.worker.default_hooks", lambda: stub_hooks())

    assert main(["--headless", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "run complete" in out
    assert "adapter" in out

    records = store.list_runs()
    assert len(records) == 1
    paths = app_paths.run_dir(records[0].folder)
    assert (paths / "config.json").is_file()
    saved = RunConfig.load(paths / "config.json")
    assert saved.generation.num_prompts == 25


def test_headless_uses_stub_hooks_only_for_the_documented_sequence(
    monkeypatch, tmp_path, app_paths, store, run_config
):
    isolate(monkeypatch, tmp_path)
    config = run_config.model_copy(deep=True)
    config.generation.num_prompts = 25
    config_path = tmp_path / "run.json"
    config_path.write_text(config.to_json(), encoding="utf-8")

    record: dict = {}
    hooks = stub_hooks(record)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    monkeypatch.setattr("autorlhf.tui.worker.default_hooks", lambda: hooks)

    assert main(["--headless", "--config", str(config_path)]) == 0
    assert isinstance(hooks, PipelineHooks)
    assert record["job"].kwargs["cancel_event"] is not None
    assert record["adapter_path"] == Path("/tmp/stub-adapter")


def test_headless_with_an_unknown_run(monkeypatch, tmp_path, app_paths, store, capsys):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    assert main(["--headless", "--run-id", "missing"]) == 2
    assert "unknown run" in capsys.readouterr().err


def test_cli_help_does_not_import_textual():
    """--help must work on a box with no TTY, so nothing imports Textual."""

    result = run_python(
        "import sys\n"
        "import autorlhf.cli\n"
        "assert 'textual' not in sys.modules, sorted(n for n in sys.modules if 'textual' in n)\n"
        "assert 'autorlhf.tui.app' not in sys.modules\n"
        "assert autorlhf.cli.main(['--help']) == 0\n"
        "assert 'textual' not in sys.modules\n"
    )
    assert result.returncode == 0, result.stderr


def test_module_entry_point_is_importable():
    from autorlhf import __main__ as entry

    assert callable(entry.main)


def test_doctor_live_flag_is_passed_through(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    seen: dict = {}

    def fake_doctor(*, env, live=False):
        seen["live"] = live
        return DoctorReport(checks=[DoctorCheck(name="x", status="ok", detail="ok")])

    monkeypatch.setattr("autorlhf.doctor.run_doctor", fake_doctor)
    assert main(["--doctor", "--live"]) == 0
    assert seen["live"] is True


@pytest.mark.parametrize("flags", [["--list-runs"], ["--doctor"], ["--help"], ["--version"]])
def test_no_command_touches_the_network(flags, monkeypatch, tmp_path):
    """Every inspection command must work with the environment wiped."""

    isolate(monkeypatch, tmp_path)
    if flags == ["--doctor"]:
        monkeypatch.setattr(
            "autorlhf.doctor.run_doctor",
            lambda *, env, live=False: DoctorReport(
                checks=[DoctorCheck(name="python version", status="ok", detail="3.12")]
            ),
        )
    assert main(flags) in (0, 1)


def test_list_runs_never_prints_credentials(
    monkeypatch, tmp_path, capsys, store, app_paths, run_config
):
    isolate(monkeypatch, tmp_path)
    monkeypatch.setattr("autorlhf.cli.Environment", lambda paths=None: _env(app_paths))
    monkeypatch.setattr("autorlhf.cli.RunStore", lambda db_path: store)
    app_env = _env(app_paths)
    app_env.secrets.update({"OPENROUTER_API_KEY": "sk-or-v1-secret-value-0123456789"})
    make_run(store, app_paths, run_config)
    assert main(["--list-runs"]) == 0
    out = capsys.readouterr().out
    assert "sk-or-v1-secret-value" not in out
    assert "OPENROUTER_API_KEY" not in out
