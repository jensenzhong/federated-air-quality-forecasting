"""Memory-mapped station datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from aqfl.config import load_config, resolve_project_path


class StationWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cache_root: Path, station: str, split: str):
        self.cache_dir = cache_root / station / split
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f"Prepared cache is missing: {self.cache_dir}")
        self.x = np.load(self.cache_dir / "x.npy", mmap_mode="r")
        self.y = np.load(self.cache_dir / "y.npy", mmap_mode="r")
        self.y_raw = np.load(self.cache_dir / "y_raw.npy", mmap_mode="r")
        self.target_timestamp_ns = np.load(self.cache_dir / "target_timestamp_ns.npy", mmap_mode="r")
        self.persistence = np.load(self.cache_dir / "persistence.npy", mmap_mode="r")
        self.seasonal_naive = np.load(self.cache_dir / "seasonal_naive.npy", mmap_mode="r")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.array(self.x[index], copy=True)), torch.from_numpy(np.array(self.y[index], copy=True))


def load_cache_metadata(config: dict[str, Any]) -> dict[str, Any]:
    root = resolve_project_path(config, config["data"]["cache_dir"])
    path = root / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError("Prepared cache metadata is missing; run aqfl.data.prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def list_stations(config: dict[str, Any]) -> list[str]:
    return list(load_cache_metadata(config)["stations"])


def load_station_dataset(
    config_or_path: dict[str, Any] | str | Path,
    station: str,
    split: str,
) -> StationWindowDataset:
    config = load_config(config_or_path) if not isinstance(config_or_path, dict) else config_or_path
    root = resolve_project_path(config, config["data"]["cache_dir"])
    return StationWindowDataset(root, station, split)
