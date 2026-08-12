from __future__ import annotations

import numpy as np

from aqfl.data.preprocessing import (
    aggregate_feature_stats,
    aggregate_scalar_stats,
    fit_global_scaler,
    fit_station_imputer,
    invalidate_physical_values,
    scale_features,
    transform_station,
)
from aqfl.data.schema import CONTINUOUS_COLUMNS, FEATURE_COLUMNS


def test_negative_physical_values_are_invalidated(raw_frame) -> None:
    raw_frame.loc[0, "PM2.5"] = -1
    raw_frame.loc[1, "TEMP"] = -10
    clean = invalidate_physical_values(raw_frame)
    assert np.isnan(clean.loc[0, "PM2.5"])
    assert clean.loc[1, "TEMP"] == -10


def test_causal_fill_does_not_read_future(raw_frame) -> None:
    train = raw_frame.iloc[:24].copy()
    state = fit_station_imputer("A", train)
    changed = raw_frame.copy()
    changed.loc[24:30, "PM2.5"] = np.nan
    left, observed_left = transform_station(changed, state, max_forward_fill_hours=6)
    changed.loc[31:, "PM2.5"] = 99999
    right, observed_right = transform_station(changed, state, max_forward_fill_hours=6)
    assert np.array_equal(left.loc[:30, "PM2.5"], right.loc[:30, "PM2.5"])
    assert not observed_left[24]
    assert np.array_equal(observed_left[:31], observed_right[:31])


def test_feature_encoding_and_missing_indicators(raw_frame) -> None:
    train = raw_frame.iloc[:24]
    state = fit_station_imputer("A", train)
    raw_frame.loc[3, "wd"] = np.nan
    raw_frame.loc[4, "PM10"] = np.nan
    features, _ = transform_station(raw_frame, state)
    assert list(features.columns) == FEATURE_COLUMNS
    assert features.shape == (72, 31)
    assert features.loc[3, "wd_missing"] == 1
    assert features.loc[4, "PM10_missing"] == 1
    assert np.isclose(features.loc[0, "wind_dir_sin"], 0)
    assert np.isclose(features.loc[0, "wind_dir_cos"], 1)


def test_federated_stats_match_pooled_continuous_data(raw_frame) -> None:
    state = fit_station_imputer("A", raw_frame.iloc[:48])
    features, _ = transform_station(raw_frame, state)
    masks = {"a": np.arange(72) < 36, "b": np.arange(72) >= 36}
    frames = {"a": raw_frame.copy(), "b": raw_frame.copy()}
    station_features = {"a": features.copy(), "b": features.copy()}
    scaler = fit_global_scaler(station_features, frames, masks)
    pooled = np.concatenate(
        [features.loc[masks[name], CONTINUOUS_COLUMNS].to_numpy() for name in ("a", "b")]
    )
    assert np.allclose(np.asarray(scaler.feature_mean)[:11], pooled.mean(axis=0))
    assert np.allclose(np.asarray(scaler.feature_mean)[11:], 0)
    assert np.allclose(np.asarray(scaler.feature_std)[11:], 1)
    scaled = scale_features(features, scaler)
    assert scaled.dtype == np.float32
    assert scaled.shape[1] == 31


def test_stat_aggregation() -> None:
    mean, std = aggregate_feature_stats(
        [
            {"sum": np.array([2.0]), "sum_sq": np.array([4.0]), "count": 1},
            {"sum": np.array([4.0]), "sum_sq": np.array([16.0]), "count": 1},
        ]
    )
    assert np.allclose(mean, [3])
    assert np.allclose(std, [1])
    scalar_mean, scalar_std = aggregate_scalar_stats(
        [{"sum": 2.0, "sum_sq": 4.0, "count": 1}, {"sum": 4.0, "sum_sq": 16.0, "count": 1}]
    )
    assert scalar_mean == 3
    assert scalar_std == 1
