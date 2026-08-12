from __future__ import annotations

import numpy as np
import pytest

from aqfl.federated.aggregation import client_weights, normalize, project_bounded_simplex


def test_fixed_aggregation_weights() -> None:
    counts = np.array([1, 3], dtype=float)
    maes = np.array([1, 2], dtype=float)
    assert np.allclose(client_weights(counts, maes, "size_only"), [0.25, 0.75])
    assert np.allclose(client_weights(counts, maes, "perf_only"), [2 / 3, 1 / 3])
    assert np.isclose(client_weights(counts, maes, "hybrid").sum(), 1)


def test_bounded_simplex_for_twelve_clients() -> None:
    values = np.array([0.9] + [0.1 / 11] * 11)
    projected = project_bounded_simplex(values)
    assert np.isclose(projected.sum(), 1)
    assert projected.min() >= 0.04 - 1e-8
    assert projected.max() <= 0.16 + 1e-8


def test_bounded_simplex_changes_with_client_count() -> None:
    projected = project_bounded_simplex(np.ones(10) / 10, lower=0.05, upper=0.2)
    assert np.allclose(projected, np.ones(10) / 10)
    with pytest.raises(ValueError, match="infeasible"):
        project_bounded_simplex(np.ones(3), lower=0.04, upper=0.16)


def test_invalid_weights_and_strategy() -> None:
    with pytest.raises(ValueError):
        normalize(np.array([-1, 2]))
    with pytest.raises(ValueError):
        client_weights(np.ones(2), np.ones(2), "unknown")
