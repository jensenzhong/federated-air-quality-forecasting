"""Pre-registered paired, station-stratified block bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_station_block_bootstrap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    n_resamples: int = 10_000,
    block_hours: int = 24,
    seed: int = 42,
) -> dict[str, Any]:
    keys = ["seed", "station", "target_timestamp_ns"]
    merged = left[keys + ["y_true", "y_pred"]].merge(
        right[keys + ["y_true", "y_pred"]],
        on=keys,
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )
    if len(merged) != len(left) or len(merged) != len(right):
        raise ValueError("Paired bootstrap requires identical seed/station/timestamp predictions")
    merged["block"] = merged.groupby(["seed", "station"]).cumcount() // block_hours
    merged["difference"] = (
        np.abs(merged["y_pred_left"] - merged["y_true_left"])
        - np.abs(merged["y_pred_right"] - merged["y_true_right"])
    )
    block_differences = merged.groupby(["seed", "station", "block"], observed=True)["difference"].mean()
    groups = [values.to_numpy() for _, values in block_differences.groupby(level=[0, 1])]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        group_means = [float(rng.choice(values, size=len(values), replace=True).mean()) for values in groups]
        draws[index] = np.mean(group_means)
    estimate = float(block_differences.groupby(level=[0, 1]).mean().mean())
    return {
        "left_minus_right_macro_mae": estimate,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "n_resamples": n_resamples,
        "block_hours": block_hours,
        "paired_examples": len(merged),
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda name: p_values[name])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, name in enumerate(ordered):
        value = min(1.0, (count - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def paired_seed_wilcoxon(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if len(left) != len(right):
        raise ValueError("Paired seed arrays must have equal length")
    statistic, p_value = wilcoxon(left, right, zero_method="wilcox", alternative="two-sided")
    return {"statistic": float(statistic), "p_value": float(p_value)}
