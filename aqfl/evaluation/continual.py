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


class LocalContinualTaskLedger:
    """Keep a task-performance matrix inside one client process only.

    The benchmark reports tasks 1..N; the base phase (T0) initializes the
    model and is evaluated separately.  This ledger deliberately has no client
    identity or serialization method.  Callers may pass its fixed-size encoded
    matrix to the existing SecAgg+ array path only after the local row is
    complete and the server-side minimum-cohort gate is satisfied.
    """

    def __init__(self, task_count: int = 11) -> None:
        if task_count < 2:
            raise ValueError("Continual task ledger requires at least two tasks")
        self.task_count = int(task_count)
        self._matrix = np.full((self.task_count, self.task_count), np.nan, dtype=np.float64)

    def record(self, task_id: int, evaluated_task_id: int, metric: float) -> None:
        if not 1 <= task_id <= self.task_count:
            raise ValueError("Unknown continual task ID")
        if not 1 <= evaluated_task_id <= self.task_count:
            raise ValueError("Unknown evaluated task ID")
        value = float(metric)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Continual metric must be finite and non-negative")
        row, column = task_id - 1, evaluated_task_id - 1
        previous = self._matrix[row, column]
        if np.isfinite(previous) and not np.isclose(previous, value):
            raise RuntimeError("Continual task metric overwrite rejected")
        self._matrix[row, column] = value

    def matrix(self, *, through_task: int | None = None) -> np.ndarray:
        last_task = self.task_count if through_task is None else int(through_task)
        if not 1 <= last_task <= self.task_count:
            raise ValueError("Unknown continual task ID")
        values = self._matrix[:last_task, :last_task]
        if not np.isfinite(values).all():
            raise RuntimeError("Local continual task matrix is incomplete")
        return values.copy()

    def encode_for_secagg(self) -> np.ndarray:
        """Return only the fixed-size numeric payload suitable for SecAgg+."""
        return encode_task_matrix(self.matrix())


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
