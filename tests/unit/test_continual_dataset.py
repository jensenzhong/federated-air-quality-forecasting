from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pytest

from aqfl.data.continual_dataset import load_station_continual_dataset
from aqfl.data.continual_schedule import (
    BENCHMARK_BASE_TEST_START,
    benchmark_phase_window,
)
from aqfl.federated.client_app import _continual_task_id


def _write_cache(root, timestamps: list[datetime]) -> None:
    station = root / "StationA"
    split = station / "train"
    split.mkdir(parents=True)
    count = len(timestamps)
    np.save(split / "x.npy", np.zeros((count, 24, 31), dtype=np.float32))
    np.save(split / "y.npy", np.zeros((count, 1), dtype=np.float32))
    np.save(split / "y_raw.npy", np.arange(count, dtype=np.float32).reshape(-1, 1))
    np.save(
        split / "target_timestamp_ns.npy",
        np.asarray(
            [np.datetime64(value, "ns").astype(np.int64) for value in timestamps],
            dtype=np.int64,
        ),
    )
    np.save(split / "persistence.npy", np.zeros(count, dtype=np.float32))
    np.save(split / "seasonal_naive.npy", np.zeros(count, dtype=np.float32))


def _config(cache_root) -> dict:
    return {"data": {"cache_dir": str(cache_root), "splits": ["train"]}}


def test_local_adapter_preserves_frozen_window_and_chronological_task_split(tmp_path) -> None:
    task_start, _ = benchmark_phase_window(1)
    timestamps = [task_start + timedelta(hours=offset) for offset in range(10)]
    _write_cache(tmp_path / "cache", timestamps)
    config = _config(tmp_path / "cache")
    train = load_station_continual_dataset(config, "StationA", 1, "train")
    test = load_station_continual_dataset(config, "StationA", 1, "test")
    assert len(train) == 8
    assert len(test) == 2
    assert train.target_timestamp_ns[-1] < test.target_timestamp_ns[0]
    assert train.task_id == test.task_id == 1


def test_base_test_is_separate_from_base_train(tmp_path) -> None:
    base = datetime.fromisoformat("2013-05-01T00:00:00")
    timestamps = [base, base + timedelta(hours=1), BENCHMARK_BASE_TEST_START]
    _write_cache(tmp_path / "cache", timestamps)
    config = _config(tmp_path / "cache")
    base_train = load_station_continual_dataset(config, "StationA", 0, "train")
    base_test = load_station_continual_dataset(config, "StationA", 0, "test")
    assert len(base_train) == 2
    assert len(base_test) == 1
    assert base_test.target_timestamp_ns[0] == np.datetime64(
        BENCHMARK_BASE_TEST_START, "ns"
    ).astype(np.int64)


def test_adapter_fails_closed_when_local_window_is_missing(tmp_path) -> None:
    _write_cache(tmp_path / "cache", [datetime.fromisoformat("2013-05-01T00:00:00")])
    with pytest.raises(RuntimeError, match="No local cache windows"):
        load_station_continual_dataset(
            _config(tmp_path / "cache"),
            "StationA",
            1,
            "train",
        )


def test_clientapp_continual_task_request_is_explicit_and_bounded() -> None:
    config = {"continual": {"enabled": False, "task_count": 11}}
    assert _continual_task_id(config, {"continual-enabled": False}) is None
    assert _continual_task_id(
        config,
        {"continual-enabled": True, "continual-task-id": 3, "continual-task-count": 11},
    ) == 3
    with pytest.raises(RuntimeError, match="invalid task ID"):
        _continual_task_id(
            config,
            {"continual-enabled": True, "continual-task-id": 12, "continual-task-count": 11},
        )
