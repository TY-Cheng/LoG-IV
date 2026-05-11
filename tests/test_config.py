from pathlib import Path
from typing import Any

from log_iv.config import (
    DEFAULT_DATA_DIR,
    DEFAULT_REPORTS_DIR,
    PROJECT_ROOT,
    ProjectSettings,
    env_key_status,
    load_research_config,
)


def test_project_settings_from_env_defaults(monkeypatch: Any) -> None:
    monkeypatch.delenv("PROJECT_NAME", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("REPORTS_DIR", raising=False)

    settings = ProjectSettings.from_env()

    assert settings.project_name == "log-iv"
    assert settings.data_dir == DEFAULT_DATA_DIR
    assert settings.reports_dir == DEFAULT_REPORTS_DIR


def test_project_settings_resolves_relative_env_paths_from_repo_root(monkeypatch: Any) -> None:
    monkeypatch.setenv("DATA_DIR", "local-data")
    monkeypatch.setenv("REPORTS_DIR", "local-reports")

    settings = ProjectSettings.from_env()

    assert settings.data_dir == PROJECT_ROOT / "local-data"
    assert settings.reports_dir == PROJECT_ROOT / "local-reports"


def test_project_settings_keeps_absolute_env_paths(monkeypatch: Any, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    reports_dir = tmp_path / "reports"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(reports_dir))

    settings = ProjectSettings.from_env()

    assert settings.data_dir == data_dir
    assert settings.reports_dir == reports_dir


def test_load_research_config() -> None:
    config = load_research_config()

    assert config["project_slug"] == "log-iv"
    assert config["data_dir"] == "data"
    assert config["reports_dir"] == "reports"
    assert "SPY" in config["universe"]["us_mvp"]["underlyings"]


def test_env_key_status_accepts_key_file(monkeypatch: Any, tmp_path: Path) -> None:
    key_file = tmp_path / "massive.key"
    key_file.write_text("secret\n")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(key_file))

    assert env_key_status(["MASSIVE_API_KEY"]) == {"MASSIVE_API_KEY": True}
    assert env_key_status(["MASSIVE_API_KEY_FILE"]) == {"MASSIVE_API_KEY_FILE": True}


def test_env_key_status_rejects_missing_key_file(monkeypatch: Any, tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.key"
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("MASSIVE_API_KEY_FILE", str(missing_file))

    assert env_key_status(["MASSIVE_API_KEY"]) == {"MASSIVE_API_KEY": False}


def test_env_key_status_ignores_direct_key_values(monkeypatch: Any) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "direct-secret-is-not-supported")
    monkeypatch.delenv("MASSIVE_API_KEY_FILE", raising=False)

    assert env_key_status(["MASSIVE_API_KEY"]) == {"MASSIVE_API_KEY": False}
