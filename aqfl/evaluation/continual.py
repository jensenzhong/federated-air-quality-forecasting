"""Aggregate-only metrics for federated continual forecasting benchmarks.

Clients build their task-performance matrix locally.  Only a fixed-size sum and
sample count may cross the SecAgg+ boundary; no task row, client identifier, or
timestamp is represented here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ContinualMetricSummary:
    average_forgetting: float
    average_plasticity: float
    average_performance: float
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("Continual task matrix must be square")
    if values.shape[0] < 2:
        raise ValueError("Continual task matrix requires at least two tasks")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Continual task matrix must contain finite non-negative metrics")
    return values


def continual_metrics(matrix: np.ndarray) -> ContinualMetricSummary:
    """Compute the benchmark's AF/AP/AvgPerf convention (lower is better)."""
    values = _validate_matrix(matrix)
    diagonal = np.diag(values)
    forgetting = float(np.mean(values[-1, :-1] - diagonal[:-1]))
    plasticity = float(np.mean(diagonal))
    performance = float(np.mean(values[-1, :]))
    return ContinualMetricSummary(forgetting, plasticity, performance, int(values.shape[0]))


def secure_aggregate_continual_metrics(
    summed_matrix: np.ndarray,
    cohort_size: int,
    *,
    minimum_cohort_size: int,
) -> ContinualMetricSummary:
    """Decode a SecAgg+ sum only after enforcing the minimum group gate."""
    if cohort_size < minimum_cohort_size or minimum_cohort_size < 2:
        raise RuntimeError("Continual metrics require a minimum secure cohort")
    if not isinstance(cohort_size, int):
        raise TypeError("cohort_size must be an integer")
    values = np.asarray(summed_matrix, dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise RuntimeError("Invalid secure continual metric aggregate")
    return continual_metrics(values / float(cohort_size))


def encode_task_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return a fixed-size local summary suitable for SecAgg+ summation."""
    values = _validate_matrix(matrix)
    return values.astype(np.float32, copy=True).reshape(-1)


def decode_task_matrix_sum(vector: np.ndarray, task_count: int) -> np.ndarray:
    if task_count < 2:
        raise ValueError("task_count must be at least two")
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != task_count * task_count:
        raise ValueError("Task matrix aggregate has an unexpected fixed length")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Task matrix aggregate must be finite and non-negative")
    return values.reshape(task_count, task_count)
