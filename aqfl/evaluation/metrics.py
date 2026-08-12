"""Regression, fairness, and high-pollution metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred_unclipped: np.ndarray,
    pollution_threshold: float | None = None,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    raw_pred = np.asarray(y_pred_unclipped, dtype=np.float64).reshape(-1)
    if len(true) == 0 or len(true) != len(raw_pred):
        raise ValueError("Metric inputs must be non-empty and have equal length")
    pred = np.maximum(raw_pred, 0.0)
    error = pred - true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    denominator = np.abs(true) + np.abs(pred)
    smape = float(np.mean(2.0 * np.abs(error) / np.maximum(denominator, 1e-8)))
    total = float(np.sum(np.square(true - true.mean())))
    r2 = float(1.0 - np.sum(np.square(error)) / total) if total > 0 else 0.0
    result = {
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "r2": r2,
        "negative_prediction_rate": float(np.mean(raw_pred < 0)),
        "num_examples": float(len(true)),
    }
    if pollution_threshold is not None:
        mask = true >= pollution_threshold
        result["high_pollution_mae"] = float(np.mean(np.abs(error[mask]))) if mask.any() else float("nan")
        result["high_pollution_examples"] = float(mask.sum())
    return result


def aggregate_station_metrics(station_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not station_metrics:
        raise ValueError("At least one station metric record is required")
    maes = np.asarray([metrics["mae"] for metrics in station_metrics.values()], dtype=np.float64)
    counts = np.asarray([metrics["num_examples"] for metrics in station_metrics.values()], dtype=np.float64)
    macro_mae = float(maes.mean())
    result: dict[str, Any] = {
        "macro_mae": macro_mae,
        "micro_mae": float(np.average(maes, weights=counts)),
        "worst_station_mae": float(maes.max()),
        "station_mae_std": float(maes.std(ddof=0)),
        "station_mae_cv": float(maes.std(ddof=0) / macro_mae) if macro_mae > 0 else 0.0,
        "station_count": len(station_metrics),
        "per_station": station_metrics,
    }
    for metric in ("rmse", "smape", "r2", "negative_prediction_rate", "high_pollution_mae"):
        values = np.asarray([record.get(metric, np.nan) for record in station_metrics.values()], dtype=np.float64)
        if np.isfinite(values).any():
            result[f"macro_{metric}"] = float(np.nanmean(values))
    return result
