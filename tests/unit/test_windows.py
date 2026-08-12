from __future__ import annotations

import numpy as np
import pandas as pd

from aqfl.data.windows import build_window_split


def test_window_shape_alignment_and_no_leakage() -> None:
    timestamps = pd.Series(pd.date_range("2013-03-01", periods=72, freq="h"))
    features = np.arange(72 * 31, dtype=np.float32).reshape(72, 31)
    target = np.arange(72, dtype=np.float32)
    split = build_window_split(
        features,
        target,
        np.ones(72, dtype=bool),
        timestamps,
        target,
        "2013-03-02 00:00",
        "2013-03-02 23:00",
    )
    assert split.x.shape == (24, 24, 31)
    assert np.array_equal(split.x[0, -1], features[23])
    assert split.y_raw[0, 0] == 24
    assert split.persistence[0] == 23
    assert split.seasonal_naive[0] == 0
    assert np.all(split.target_timestamp_ns > timestamps.iloc[:24].astype("int64").max())


def test_missing_target_excluded_and_split_targets_disjoint() -> None:
    timestamps = pd.Series(pd.date_range("2013-03-01", periods=100, freq="h"))
    features = np.zeros((100, 31), dtype=np.float32)
    target = np.arange(100, dtype=np.float32)
    observed = np.ones(100, dtype=bool)
    observed[30] = False
    first = build_window_split(features, target, observed, timestamps, target, str(timestamps[24]), str(timestamps[49]))
    second = build_window_split(features, target, observed, timestamps, target, str(timestamps[50]), str(timestamps[80]))
    assert np.datetime64(timestamps[30], "ns").astype(np.int64) not in first.target_timestamp_ns
    assert set(first.target_timestamp_ns).isdisjoint(second.target_timestamp_ns)
