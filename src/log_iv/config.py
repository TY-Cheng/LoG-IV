from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/default.toml"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_BRONZE_DATA_DIR = DEFAULT_DATA_DIR / "bronze"
DEFAULT_SILVER_DATA_DIR = DEFAULT_DATA_DIR / "silver"
DEFAULT_GOLD_DATA_DIR = DEFAULT_DATA_DIR / "gold"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"


class ProjectSettings(BaseModel):
    """Runtime paths loaded from environment variables."""

    project_name: str = Field(default="log-iv")
    data_dir: Path = Field(default=DEFAULT_DATA_DIR)
    reports_dir: Path = Field(default=DEFAULT_REPORTS_DIR)
    log_level: str = Field(default="INFO")

    @classmethod
    def from_env(cls) -> ProjectSettings:
        return cls(
            project_name=os.environ.get("PROJECT_NAME", "log-iv"),
            data_dir=path_from_env("DATA_DIR", DEFAULT_DATA_DIR),
            reports_dir=path_from_env("REPORTS_DIR", DEFAULT_REPORTS_DIR),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def path_from_env(name: str, default: Path) -> Path:
    """Resolve env paths relative to the repo root unless they are absolute."""

    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        path = default
    else:
        path = Path(os.path.expanduser(os.path.expandvars(raw_value)))
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_research_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the static research scaffold configuration."""

    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data


def env_key_status(keys: list[str]) -> dict[str, bool]:
    """Return whether key-file environment variables point to existing files.

    The project intentionally ignores direct secret values. Pass either logical
    names such as ``MASSIVE_API_KEY`` or exact file variables such as
    ``MASSIVE_API_KEY_FILE``; only the corresponding file path is considered
    configured.
    """

    status: dict[str, bool] = {}
    for key in keys:
        key_file_var = key if key.endswith("_FILE") else f"{key}_FILE"
        key_file_value = os.environ.get(key_file_var)
        key_file_configured = False
        if key_file_value:
            key_file_path = Path(os.path.expanduser(os.path.expandvars(key_file_value)))
            key_file_configured = key_file_path.is_file()
        status[key] = key_file_configured
    return status
