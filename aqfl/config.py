"""Configuration loading and deterministic runtime setup."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def resolve_data_root(config: dict[str, Any]) -> Path:
    """Resolve the raw dataset root without embedding a personal path."""
    env_name = str(config["data"].get("root_env", "BEIJING_AQ_DATA_DIR"))
    raw = os.getenv(env_name)
    if not raw:
        raise RuntimeError(
            f"Environment variable {env_name} is required and must point to the "
            "Beijing multi-site air-quality dataset directory."
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {path}")
    return path


def project_root(config: dict[str, Any] | None = None) -> Path:
    """Return the installed project root."""
    if config and config.get("_config_path"):
        return Path(config["_config_path"]).resolve().parent.parent
    return Path(__file__).resolve().parent.parent


def resolve_project_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve a project-relative configuration path."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root(config) / path).resolve()


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for CPU experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def canonical_json(data: Any) -> str:
    """Serialize data in a stable form for hashes and audit records."""
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
