"""Auditable dynamic client-weight calculations."""

from __future__ import annotations

import numpy as np


def normalize(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Weights must be finite, non-negative, and have a positive sum")
    return weights / weights.sum()


def project_bounded_simplex(values: np.ndarray, lower: float = 0.04, upper: float = 0.16) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0 or lower * n > 1 + 1e-12 or upper * n < 1 - 1e-12:
        raise ValueError("Weight bounds are infeasible for the client count")
    lo = float(np.min(values - upper))
    hi = float(np.max(values - lower))
    for _ in range(100):
        midpoint = (lo + hi) / 2
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > 1:
            lo = midpoint
        else:
            hi = midpoint
    projected = np.clip(values - (lo + hi) / 2, lower, upper)
    projected /= projected.sum()
    if np.any(projected < lower - 1e-8) or np.any(projected > upper + 1e-8):
        raise AssertionError("Bounded simplex projection violated requested bounds")
    return projected


def client_weights(
    sample_counts: np.ndarray,
    validation_mae: np.ndarray,
    strategy: str,
    lower: float = 0.04,
    upper: float = 0.16,
) -> np.ndarray:
    size = normalize(sample_counts)
    performance = normalize(1.0 / np.maximum(validation_mae, 1e-8))
    if strategy == "size_only":
        return size
    if strategy == "perf_only":
        return performance
    hybrid = 0.5 * size + 0.5 * performance
    if strategy == "hybrid":
        return normalize(hybrid)
    if strategy == "fairness_clip":
        return project_bounded_simplex(hybrid, lower, upper)
    raise ValueError(f"Unsupported aggregation strategy: {strategy}")
