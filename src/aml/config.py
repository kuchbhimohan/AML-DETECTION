"""Load project configuration files (configs/*.yaml)."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def load_yaml(name: str) -> dict[str, Any]:
    """Read configs/<name>.yaml and return it as a dict."""
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema() -> dict[str, Any]:
    return load_yaml("schema")


def load_train_config() -> dict[str, Any]:
    return load_yaml("train")
