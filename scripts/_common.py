"""Shared command-line and project-path helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

PROJECT_DIR = Path(__file__).resolve().parents[1]
SIZING_CONFIG = PROJECT_DIR / "netlists" / "strongarm_2ghz" / "sizing.yaml"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def load_config(path: str | Path = SIZING_CONFIG) -> dict:
    """Load a YAML mapping from the requested configuration file."""
    import yaml

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a YAML mapping: {path}")
    return config


def write_json(path: str | Path, values: Mapping) -> None:
    """Write a mapping as readable UTF-8 JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(values), indent=2, sort_keys=True) + "\n", encoding="utf-8")
