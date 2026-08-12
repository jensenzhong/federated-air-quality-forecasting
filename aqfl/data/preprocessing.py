"""Causal imputation, feature encoding, and federated standardization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aqfl.data.schema import (
    CONTINUOUS_COLUMNS,
    FEATURE_COLUMNS,
    MISSING_SOURCE_COLUMNS,
    NONNEGATIVE_COLUMNS,
    TARGET_COLUMN,
    WIND_DEGREES,
)


@dataclass
class StationImputerState:
    station: str
    medians_by_month_hour: dict[str, dict[str, float]]
    overall_medians: dict[str, float]
    wind_mode: str


@dataclass
class GlobalScalerState:
    feature_columns: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: float
    target_std: float
    pollution_p90: float


def invalidate_physical_values(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    for column in NONNEGATIVE_COLUMNS:
        clean.loc[clean[column] < 0, column] = np.nan
    clean.loc[~clean["wd"].isin(WIND_DEGREES), "wd"] = np.nan
    return clean


def _group_key(month: int, hour: int) -> str:
    return f"{int(month):02d}-{int(hour):02d}"


def fit_station_imputer(station: str, train_frame: pd.DataFrame) -> StationImputerState:
    train = invalidate_physical_values(train_frame)
    grouped = train.groupby(["month", "hour"], observed=True)
    medians: dict[str, dict[str, float]] = {column: {} for column in CONTINUOUS_COLUMNS}
    for column in CONTINUOUS_COLUMNS:
        series = grouped[column].median()
        medians[column] = {
            _group_key(month, hour): float(value)
            for (month, hour), value in series.items()
            if pd.notna(value)
        }
    overall = {column: float(train[column].median()) for column in CONTINUOUS_COLUMNS}
    if any(not np.isfinite(value) for value in overall.values()):
        raise ValueError(f"Training data cannot fit all medians for station {station}")
    wind_modes = train["wd"].dropna().mode()
    if wind_modes.empty:
        raise ValueError(f"Training data has no valid wind direction for station {station}")
    return StationImputerState(station, medians, overall, str(wind_modes.iloc[0]))


def transform_station(
    frame: pd.DataFrame,
    state: StationImputerState,
    max_forward_fill_hours: int = 6,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply strictly causal short-gap filling, then train-fitted seasonal statistics."""
    clean = invalidate_physical_values(frame).sort_values("timestamp").reset_index(drop=True)
    missing_flags = clean[MISSING_SOURCE_COLUMNS].isna().astype(np.float32)
    target_observed = clean[TARGET_COLUMN].notna().to_numpy()

    for column in CONTINUOUS_COLUMNS:
        clean[column] = clean[column].ffill(limit=max_forward_fill_hours)
        missing = clean[column].isna()
        if missing.any():
            keys = [
                _group_key(month, hour)
                for month, hour in zip(clean.loc[missing, "month"], clean.loc[missing, "hour"], strict=False)
            ]
            replacements = [
                state.medians_by_month_hour[column].get(key, state.overall_medians[column])
                for key in keys
            ]
            clean.loc[missing, column] = replacements

    clean["wd"] = clean["wd"].ffill(limit=max_forward_fill_hours).fillna(state.wind_mode)
    radians = np.deg2rad(clean["wd"].map(WIND_DEGREES).astype(float).to_numpy())

    features = pd.DataFrame(index=clean.index)
    for column in CONTINUOUS_COLUMNS:
        features[column] = clean[column].astype(np.float32)
    features["wind_dir_sin"] = np.sin(radians).astype(np.float32)
    features["wind_dir_cos"] = np.cos(radians).astype(np.float32)
    features["hour_sin"] = np.sin(2 * np.pi * clean["hour"] / 24).astype(np.float32)
    features["hour_cos"] = np.cos(2 * np.pi * clean["hour"] / 24).astype(np.float32)
    weekday = clean["timestamp"].dt.dayofweek
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7).astype(np.float32)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7).astype(np.float32)
    features["month_sin"] = np.sin(2 * np.pi * (clean["month"] - 1) / 12).astype(np.float32)
    features["month_cos"] = np.cos(2 * np.pi * (clean["month"] - 1) / 12).astype(np.float32)
    for column in MISSING_SOURCE_COLUMNS:
        features[f"{column}_missing"] = missing_flags[column]

    features = features[FEATURE_COLUMNS]
    if features.isna().any().any():
        raise ValueError(f"Imputation left missing features for station {state.station}")
    return features, target_observed


def local_feature_stats(features: pd.DataFrame, train_mask: np.ndarray) -> dict[str, Any]:
    """Return the only statistics uploaded for continuous feature scaling."""
    values = features.loc[train_mask, CONTINUOUS_COLUMNS].to_numpy(dtype=np.float64)
    return {"sum": values.sum(axis=0), "sum_sq": np.square(values).sum(axis=0), "count": len(values)}


def local_target_stats(frame: pd.DataFrame, train_mask: np.ndarray) -> dict[str, float | int]:
    values = frame.loc[train_mask, TARGET_COLUMN].dropna().to_numpy(dtype=np.float64)
    return {
        "sum": float(values.sum()),
        "sum_sq": float(np.square(values).sum()),
        "count": int(len(values)),
    }


def aggregate_feature_stats(local_stats: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    count = sum(int(stats["count"]) for stats in local_stats)
    total = sum(np.asarray(stats["sum"], dtype=np.float64) for stats in local_stats)
    total_sq = sum(np.asarray(stats["sum_sq"], dtype=np.float64) for stats in local_stats)
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return mean, np.sqrt(variance)


def aggregate_scalar_stats(local_stats: list[dict[str, float | int]]) -> tuple[float, float]:
    count = sum(int(stats["count"]) for stats in local_stats)
    if count <= 0:
        raise ValueError("Cannot aggregate empty scalar statistics")
    total = sum(float(stats["sum"]) for stats in local_stats)
    total_sq = sum(float(stats["sum_sq"]) for stats in local_stats)
    mean = total / count
    variance = max(total_sq / count - mean**2, 1e-12)
    return mean, float(np.sqrt(variance))


def fit_global_scaler(
    station_features: dict[str, pd.DataFrame],
    station_frames: dict[str, pd.DataFrame],
    train_masks: dict[str, np.ndarray],
) -> GlobalScalerState:
    stats = [local_feature_stats(station_features[s], train_masks[s]) for s in station_features]
    continuous_mean, continuous_std = aggregate_feature_stats(stats)
    mean = np.zeros(len(FEATURE_COLUMNS), dtype=np.float64)
    std = np.ones(len(FEATURE_COLUMNS), dtype=np.float64)
    for index, column in enumerate(CONTINUOUS_COLUMNS):
        feature_index = FEATURE_COLUMNS.index(column)
        mean[feature_index] = continuous_mean[index]
        std[feature_index] = continuous_std[index]
    target_stats = [local_target_stats(station_frames[s], train_masks[s]) for s in station_frames]
    target_mean, target_std = aggregate_scalar_stats(target_stats)
    target_parts = [
        station_frames[s].loc[train_masks[s], TARGET_COLUMN].dropna().to_numpy(dtype=np.float64)
        for s in station_frames
    ]
    target = np.concatenate(target_parts)
    return GlobalScalerState(
        feature_columns=FEATURE_COLUMNS,
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
        target_mean=float(target_mean),
        target_std=float(max(target_std, 1e-6)),
        pollution_p90=float(np.quantile(target, 0.90)),
    )


def scale_features(features: pd.DataFrame, scaler: GlobalScalerState) -> np.ndarray:
    values = features[scaler.feature_columns].to_numpy(dtype=np.float32)
    return ((values - np.asarray(scaler.feature_mean, dtype=np.float32)) / np.asarray(scaler.feature_std, dtype=np.float32)).astype(np.float32)


def scale_target(values: np.ndarray, scaler: GlobalScalerState) -> np.ndarray:
    return ((values - scaler.target_mean) / scaler.target_std).astype(np.float32)


def inverse_target(values: np.ndarray, scaler: GlobalScalerState) -> np.ndarray:
    return values * scaler.target_std + scaler.target_mean
