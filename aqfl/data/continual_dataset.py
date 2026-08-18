"""Client-local views over the frozen continual benchmark schedule.

The adapter only indexes the station's local prepared cache.  It never sends
timestamps, row indices, or task matrices to the server.  A caller may use the
returned datasets for local training/evaluation and place only a fixed-size
summary behind the existing SecAgg+ boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from aqfl.config import resolve_project_path
from aqfl.data.continual_schedule import (
    BENCHMARK_BASE_TEST_END,
    BENCHMARK_BASE_TEST_START,
    benchmark_evaluation_window,
    benchmark_phase_window,
    task_key,
)
from aqfl.data.dataset import StationWindowDataset

ContinualSplit = Literal["train", "test"]


def _datetime_ns(value: datetime) -> int:
    return int(np.datetime64(value, "ns").astype(np.int64))


class IndexedStationWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """A deterministic, timestamp-ordered view over one or more cache splits."""

    def __init__(
        self,
        sources: Sequence[StationWindowDataset],
        source_indices: Sequence[np.ndarray],
        *,
        task_id: int,
        split: ContinualSplit,
    ) -> None:
        if len(sources) != len(source_indices) or not sources:
            raise ValueError("A continual dataset requires at least one cache source")
        self._sources = tuple(sources)
        self._indices = tuple(
            np.asarray(indices, dtype=np.int64).reshape(-1)
            for indices in source_indices
        )
        if any(np.any(indices < 0) for indices in self._indices):
            raise ValueError("Continual cache indices must be non-negative")
        if any(
            np.any(indices >= len(source))
            for source, indices in zip(self._sources, self._indices, strict=True)
        ):
            raise ValueError("Continual cache index exceeds source length")
        self.task_id = int(task_id)
        self.split = split
        self._lengths = tuple(int(indices.size) for indices in self._indices)
        self._offsets = np.cumsum((0, *self._lengths), dtype=np.int64)
        if int(self._offsets[-1]) == 0:
            raise ValueError("Continual task split contains no windows")

    def __len__(self) -> int:
        return int(self._offsets[-1])

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_position = int(np.searchsorted(self._offsets, index, side="right") - 1)
        local_index = index - int(self._offsets[source_position])
        return source_position, int(self._indices[source_position][local_index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source_position, local_index = self._locate(index)
        return self._sources[source_position][local_index]

    @property
    def target_timestamp_ns(self) -> np.ndarray:
        chunks = [
            np.asarray(source.target_timestamp_ns[indices], dtype=np.int64)
            for source, indices in zip(self._sources, self._indices, strict=True)
            if indices.size
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)

    @property
    def y_raw(self) -> np.ndarray:
        chunks = [
            np.asarray(source.y_raw[indices])
            for source, indices in zip(self._sources, self._indices, strict=True)
            if indices.size
        ]
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)


def _candidate_cache_splits(config: dict[str, Any]) -> tuple[str, ...]:
    configured = config.get("data", {}).get("splits")
    if configured is None:
        return ("train", "val", "test")
    values = tuple(str(value) for value in configured)
    if not values or any(value not in {"train", "val", "test"} for value in values):
        raise ValueError("Continual cache splits must be drawn from train/val/test")
    return values


def _ordered_entries(
    sources: Sequence[StationWindowDataset],
    *,
    start: datetime,
    end: datetime,
) -> list[tuple[int, int, int]]:
    start_ns, end_ns = _datetime_ns(start), _datetime_ns(end)
    entries: list[tuple[int, int, int]] = []
    for source_position, source in enumerate(sources):
        timestamps = np.asarray(source.target_timestamp_ns, dtype=np.int64)
        selected = np.flatnonzero((timestamps >= start_ns) & (timestamps <= end_ns))
        entries.extend(
            (int(timestamp), source_position, int(index))
            for timestamp, index in zip(timestamps[selected], selected, strict=True)
        )
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    timestamps = [item[0] for item in entries]
    if len(timestamps) != len(set(timestamps)):
        raise RuntimeError("Continual cache contains duplicate target timestamps")
    return entries


def load_station_continual_dataset(
    config: dict[str, Any],
    station: str,
    task_id: int,
    split: ContinualSplit,
    *,
    train_ratio: float = 0.8,
) -> IndexedStationWindowDataset:
    """Build a local base/task train or test view from prepared cache windows.

    Task 0 follows the supplied benchmark's separate base and base-test day.
    Tasks 1..11 use the exact public window and split chronologically at
    ``train_ratio`` after lagged windows are indexed, matching the notebook's
    ``int(len(Xt) * SPLIT_RATIO)`` convention without exposing row metadata.
    """
    if split not in {"train", "test"}:
        raise ValueError("Continual split must be train or test")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one")
    if task_id < 0 or task_id > 11:
        raise ValueError("Unknown continual task ID")
    cache_root = resolve_project_path(config, config["data"]["cache_dir"])
    cache_sources: list[StationWindowDataset] = []
    for cache_split in _candidate_cache_splits(config):
        source = StationWindowDataset(cache_root, station, cache_split)
        cache_sources.append(source)
    if task_id == 0:
        start, end = (
            (BENCHMARK_BASE_TEST_START, BENCHMARK_BASE_TEST_END)
            if split == "test"
            else benchmark_phase_window(0)
        )
        entries = _ordered_entries(cache_sources, start=start, end=end)
    else:
        start, end = benchmark_phase_window(task_id)
        entries = _ordered_entries(cache_sources, start=start, end=end)
        split_at = int(len(entries) * train_ratio)
        split_at = min(max(split_at, 1), len(entries) - 1)
        entries = entries[:split_at] if split == "train" else entries[split_at:]
    if not entries:
        window = benchmark_evaluation_window(task_id)
        raise RuntimeError(
            f"No local cache windows for {task_key(task_id, split)} "
            f"in [{window[0].isoformat()}, {window[1].isoformat()}]"
        )
    grouped: list[list[int]] = [[] for _ in cache_sources]
    for _, source_position, index in entries:
        grouped[source_position].append(index)
    selected_sources = [
        source for source, indices in zip(cache_sources, grouped, strict=True) if indices
    ]
    selected_indices = [
        np.asarray(indices, dtype=np.int64) for indices in grouped if indices
    ]
    # Grouping by source would destroy the global chronological order.  Keep a
    # single source order for the normal cache layout; mixed cache splits are
    # rejected until a timestamp-aware concat view is needed by the runner.
    if len(selected_sources) > 1:
        source_by_entry = [source_position for _, source_position, _ in entries]
        if source_by_entry != sorted(source_by_entry):
            raise RuntimeError(
                "Continual cache split boundaries interleave; rebuild a unified cache"
            )
    return IndexedStationWindowDataset(
        selected_sources,
        selected_indices,
        task_id=task_id,
        split=split,
    )
