from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aqfl.evaluation.bootstrap import (
    holm_adjust,
    paired_seed_wilcoxon,
    paired_station_block_bootstrap,
)
from aqfl.evaluation.robustness import dropped_client_indices, missing_blocks, sensor_drift


def prediction_frame(offset: float) -> pd.DataFrame:
    rows = []
    for seed in [42, 123]:
        for station in ["a", "b"]:
            for timestamp in range(48):
                rows.append(
                    {
                        "seed": seed,
                        "station": station,
                        "target_timestamp_ns": timestamp,
                        "y_true": 10.0,
                        "y_pred": 10.0 + offset,
                    }
                )
    return pd.DataFrame(rows)


def test_paired_block_bootstrap_and_pair_validation() -> None:
    result = paired_station_block_bootstrap(prediction_frame(1), prediction_frame(2), n_resamples=100)
    assert result["left_minus_right_macro_mae"] == -1
    broken = prediction_frame(2).iloc[:-1]
    with pytest.raises(ValueError, match="identical"):
        paired_station_block_bootstrap(prediction_frame(1), broken, n_resamples=10)


def test_holm_and_wilcoxon() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert 0 <= adjusted["a"] <= adjusted["c"] <= 1
    result = paired_seed_wilcoxon(np.array([1, 2, 3]), np.array([2, 3, 4]))
    assert 0 <= result["p_value"] <= 1


def test_robustness_transforms_are_deterministic() -> None:
    x = np.ones((10, 24, 31), dtype=np.float32)
    missing_a = missing_blocks(x, seed=7)
    missing_b = missing_blocks(x, seed=7)
    assert np.array_equal(missing_a, missing_b)
    assert (missing_a[..., :11] == 0).any()
    drift = sensor_drift(x)
    assert np.allclose(drift[..., :6], 1.1)
    assert np.allclose(drift[..., 6:], 1.0)
    dropped = dropped_client_indices(1)
    assert len(dropped) == 3
    assert dropped == dropped_client_indices(1)
