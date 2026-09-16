from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_default_config() -> tuple[dict[str, Any], dict[str, Any]]:
    root = repo_root()
    return (
        load_yaml(root / "config" / "interest_profile.yml"),
        load_yaml(root / "config" / "sources.yml"),
    )

