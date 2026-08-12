"""Leakage-safe sliding-window construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WindowSplit:
    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    target_timestamp_ns: np.ndarray
    persistence: np.ndarray
    seasonal_naive: np.ndarray


def build_window_split(
    features_scaled: np.ndarray,
    target_raw_imputed: np.ndarray,
    target_observed: np.ndarray,
    timestamps: pd.Series,
    target_scaled: np.ndarray,
    split_start: str,
    split_end: str,
    window: int = 24,
    horizon: int = 1,
    stride: int = 1,
) -> WindowSplit:
    if len(features_scaled) != len(target_raw_imputed) or len(timestamps) != len(features_scaled):
        raise ValueError("Feature, target, and timestamp lengths must match")
    timestamp_values = timestamps.to_numpy(dtype="datetime64[ns]")
    start = np.datetime64(split_start, "ns")
    end = np.datetime64(split_end, "ns")
    target_indices = np.arange(window - 1 + horizon, len(features_scaled), stride)
    target_times = timestamp_values[target_indices]
    selected = target_indices[(target_times >= start) & (target_times <= end) & target_observed[target_indices]]
    if len(selected) == 0:
        shape = (0, window, features_scaled.shape[1])
        return WindowSplit(
            np.empty(shape, dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    offsets = np.arange(window - 1, -1, -1)
    end_indices = selected - horizon
    input_indices = end_indices[:, None] - offsets[None, :]
    x = features_scaled[input_indices].astype(np.float32)
    y = target_scaled[selected, None].astype(np.float32)
    y_raw = target_raw_imputed[selected, None].astype(np.float32)
    persistence = target_raw_imputed[end_indices].astype(np.float32)
    seasonal_index = selected - 24
    if (seasonal_index < 0).any():
        raise ValueError("Seasonal-naive index precedes the dataset")
    seasonal = target_raw_imputed[seasonal_index].astype(np.float32)

    input_max = timestamp_values[end_indices]
    if not np.all(input_max < timestamp_values[selected]):
        raise AssertionError("Window contains the target or future information")
    return WindowSplit(
        x=x,
        y=y,
        y_raw=y_raw,
        target_timestamp_ns=timestamp_values[selected].astype(np.int64),
        persistence=persistence,
        seasonal_naive=seasonal,
    )
