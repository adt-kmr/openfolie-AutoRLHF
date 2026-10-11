"""Integration: the environment doctor."""

from __future__ import annotations

from autorlhf import doctor
from autorlhf.config.settings import Environment
from autorlhf.doctor import DoctorCheck, run_doctor

EXPECTED_CHECKS = {
    "python version",
    "uv / virtualenv",
    "core imports",
    "training stack",
    "optional extras",
    "hardware",
    "writable directories",
    "secret file permissions",
    "credentials",
}


def test_report_has_expected_checks(app_paths):
    env = Environment(app_paths)
    report = run_doctor(env=env)
    names = {check.name for check in report.checks}
    assert EXPECTED_CHECKS <= names
    assert report.worst in {"ok", "warn", "fail"}
    text = report.render()
    assert "autorlhf doctor" in text
    for check in report.checks:
        assert check.status in {"ok", "warn", "fail"}
        if check.status in {"warn", "fail"}:
            assert check.remediation, f"missing remediation for {check.name}"


def test_never_renders_credential_value(app_paths):
    env = Environment(app_paths)
    secret = "sk-or-v1-FAKEKEY1234567890abcdef"
    env.secrets.update({"OPENROUTER_API_KEY": secret})
    env.refresh()

    report = run_doctor(env=env)
    text = report.render()
    assert secret not in text
    assert "FAKEKEY1234567890" not in text
    # The key is present and masked, not missing.
    creds = next(c for c in report.checks if c.name == "credentials")
    assert "OPENROUTER_API_KEY=(unset)" not in creds.detail


def test_never_raises_even_if_a_check_crashes(app_paths, monkeypatch):
    env = Environment(app_paths)

    def _boom() -> DoctorCheck:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(doctor, "_check_python", _boom)
    report = run_doctor(env=env)
    assert report.worst in {"ok", "warn", "fail"}
    crashed = next(c for c in report.checks if c.name == "python version")
    assert crashed.status == "warn"
    assert crashed.remediation


def test_live_checks_never_raise(app_paths):
    env = Environment(app_paths)
    report = run_doctor(env=env, live=True)
    names = {check.name for check in report.checks}
    assert "openrouter (live)" in names
    assert "hugging face (live)" in names
    assert report.worst in {"ok", "warn", "fail"}
