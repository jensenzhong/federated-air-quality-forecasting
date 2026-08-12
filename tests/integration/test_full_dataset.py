from __future__ import annotations

import os

import numpy as np
import pytest

from aqfl.config import load_config, resolve_data_root
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.manifest import validate_and_load_raw


@pytest.mark.full_data
def test_complete_raw_dataset_and_prepared_cache() -> None:
    if not os.getenv("BEIJING_AQ_DATA_DIR"):
        pytest.skip("BEIJING_AQ_DATA_DIR is not configured")
    config = load_config("configs/base.yaml")
    frames, records = validate_and_load_raw(config, resolve_data_root(config))
    assert len(frames) == 12
    assert len(records) == 12
    assert sum(len(frame) for frame in frames.values()) == 420_768
    metadata = load_cache_metadata(config)
    assert metadata["feature_dim"] == 31
    assert len(list_stations(config)) == 12
    target_sets = []
    for station in list_stations(config):
        for split in ("train", "val", "test"):
            dataset = load_station_dataset(config, station, split)
            assert dataset.x.shape[1:] == (24, 31)
            target_sets.append((station, split, set(np.asarray(dataset.target_timestamp_ns).tolist())))
    for station in list_stations(config):
        station_sets = [values for st, _, values in target_sets if st == station]
        assert station_sets[0].isdisjoint(station_sets[1])
        assert station_sets[1].isdisjoint(station_sets[2])
