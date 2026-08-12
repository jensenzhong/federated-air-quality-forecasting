from __future__ import annotations

import numpy as np
import pytest

from aqfl.evaluation.metrics import aggregate_station_metrics, regression_metrics


def test_regression_metrics_clip_only_predictions() -> None:
    metrics = regression_metrics(np.array([0, 2]), np.array([-1, 4]), pollution_threshold=1)
    assert metrics["mae"] == 1
    assert metrics["rmse"] == pytest.approx(np.sqrt(2))
    assert metrics["negative_prediction_rate"] == 0.5
    assert metrics["high_pollution_mae"] == 2


def test_smape_and_r2_edges() -> None:
    metrics = regression_metrics(np.zeros(2), np.zeros(2))
    assert metrics["smape"] == 0
    assert metrics["r2"] == 0
    with pytest.raises(ValueError):
        regression_metrics(np.array([]), np.array([]))


def test_station_macro_micro_and_fairness() -> None:
    result = aggregate_station_metrics(
        {
            "a": {"mae": 1.0, "rmse": 1.0, "smape": 0.1, "r2": 0.5, "num_examples": 1},
            "b": {"mae": 3.0, "rmse": 3.0, "smape": 0.3, "r2": 0.1, "num_examples": 3},
        }
    )
    assert result["macro_mae"] == 2
    assert result["micro_mae"] == 2.5
    assert result["worst_station_mae"] == 3
    assert result["station_mae_cv"] == 0.5
