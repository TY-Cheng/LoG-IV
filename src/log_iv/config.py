from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ProjectSettings(BaseModel):
    """Runtime paths loaded from environment variables."""

    project_name: str = Field(default="log-iv")
    data_dir: Path = Field(default=Path("data"))
    reports_dir: Path = Field(default=Path("reports"))
    log_level: str = Field(default="INFO")

    @classmethod
    def from_env(cls) -> ProjectSettings:
        return cls(
            project_name=os.environ.get("PROJECT_NAME", "log-iv"),
            data_dir=Path(os.environ.get("DATA_DIR", "data")),
            reports_dir=Path(os.environ.get("REPORTS_DIR", "reports")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )


def load_research_config(path: Path = Path("config/default.toml")) -> dict[str, Any]:
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
