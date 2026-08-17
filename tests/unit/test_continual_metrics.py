from __future__ import annotations

import numpy as np
import pytest

from aqfl.evaluation.continual import (
    continual_metrics,
    decode_task_matrix_sum,
    encode_task_matrix,
    secure_aggregate_continual_metrics,
)


def test_continual_metrics_match_benchmark_convention() -> None:
    matrix = np.asarray(
        [
            [1.0, 4.0, 5.0],
            [2.0, 1.5, 3.0],
            [3.0, 2.5, 1.0],
        ]
    )
    summary = continual_metrics(matrix)
    assert summary.average_forgetting == pytest.approx((2.0 + 1.0) / 2.0)
    assert summary.average_plasticity == pytest.approx((1.0 + 1.5 + 1.0) / 3.0)
    assert summary.average_performance == pytest.approx((3.0 + 2.5 + 1.0) / 3.0)


def test_secure_continual_summary_only_accepts_minimum_cohort() -> None:
    matrix = np.eye(2, dtype=np.float32)
    vector = encode_task_matrix(matrix)
    decoded = decode_task_matrix_sum(vector * 3, 2)
    summary = secure_aggregate_continual_metrics(
        decoded,
        3,
        minimum_cohort_size=2,
    )
    assert summary.task_count == 2
    with pytest.raises(RuntimeError, match="minimum secure cohort"):
        secure_aggregate_continual_metrics(decoded, 1, minimum_cohort_size=2)


def test_continual_metric_codec_rejects_non_square_or_private_metadata() -> None:
    with pytest.raises(ValueError, match="square"):
        encode_task_matrix(np.ones((2, 3)))
    assert "client" not in str(continual_metrics(np.eye(2)).to_dict()).lower()
