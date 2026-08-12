from __future__ import annotations

from flwr.app import MetricRecord, RecordDict

from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics


def test_training_metric_aggregation() -> None:
    records = [
        RecordDict(
            {
                "metrics": MetricRecord(
                    {"num-examples": 1, "train_loss": 2.0, "val_mae": 4.0, "update_norm": 1.0}
                )
            }
        ),
        RecordDict(
            {
                "metrics": MetricRecord(
                    {"num-examples": 3, "train_loss": 1.0, "val_mae": 2.0, "update_norm": 3.0}
                )
            }
        ),
    ]
    result = aggregate_training_metrics(records, "num-examples")
    assert result["train_loss"] == 1.25
    assert result["val_macro_mae"] == 3
    assert result["num-clients"] == 2
    assert result["total-upload-bytes"] == 0


def test_evaluation_metric_aggregation() -> None:
    records = [
        RecordDict(
            {
                "metrics": MetricRecord(
                    {"num-examples": count, "mae": mae, "rmse": mae + 1, "smape": mae / 10}
                )
            }
        )
        for count, mae in [(1, 1.0), (3, 3.0)]
    ]
    result = aggregate_evaluation_metrics(records, "num-examples")
    assert result["macro_mae"] == 2
    assert result["micro_mae"] == 2.5
    assert result["worst_station_mae"] == 3
