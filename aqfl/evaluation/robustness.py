"""Deterministic stress transforms; never applied during model selection."""

from __future__ import annotations

import numpy as np

POLLUTANT_FEATURE_INDICES = np.arange(6)


def missing_blocks(x: np.ndarray, fraction: float = 0.10, block_hours: int = 6, seed: int = 42) -> np.ndarray:
    output = np.array(x, copy=True)
    rng = np.random.default_rng(seed)
    total_slots = output.shape[0] * output.shape[1]
    blocks = max(1, int(total_slots * fraction / block_hours))
    for _ in range(blocks):
        sample = int(rng.integers(0, output.shape[0]))
        start = int(rng.integers(0, max(1, output.shape[1] - block_hours + 1)))
        output[sample, start : start + block_hours, :11] = 0.0
        output[sample, start : start + block_hours, 19:30] = 1.0
    return output


def sensor_drift(x: np.ndarray, factor: float = 1.10) -> np.ndarray:
    output = np.array(x, copy=True)
    output[..., POLLUTANT_FEATURE_INDICES] *= factor
    return output


def dropped_client_indices(round_number: int, num_clients: int = 12, drop_count: int = 3, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed + round_number)
    return sorted(rng.choice(num_clients, size=drop_count, replace=False).tolist())
