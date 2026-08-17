"""Flower MetricRecord aggregation for station-level forecasting."""

from __future__ import annotations

import numpy as np
from flwr.app import MetricRecord, RecordDict


def _metric_records(records: list[RecordDict]) -> list[MetricRecord]:
    return [next(iter(record.metric_records.values())) for record in records]


def aggregate_training_metrics(records: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    metrics = _metric_records(records)
    counts = np.asarray([float(item[weighted_by_key]) for item in metrics])
    losses = np.asarray([float(item["train_loss"]) for item in metrics])
    maes = np.asarray([float(item["val_mae"]) for item in metrics])
    norms = np.asarray([float(item["update_norm"]) for item in metrics])
    tail_count = max(1, int(np.ceil(len(maes) * 0.25)))
    return MetricRecord({
        "train_loss": float(np.average(losses, weights=counts)),
        "val_macro_mae": float(maes.mean()),
        "val_worst_mae": float(maes.max()),
        "val_mae_cv": float(maes.std(ddof=0) / max(maes.mean(), 1e-8)),
        "val_station_cvar25_mae": float(np.sort(maes)[-tail_count:].mean()),
        "update_norm_cv": float(norms.std(ddof=0) / max(norms.mean(), 1e-8)),
        "num-examples": int(counts.sum()),
        "num-clients": len(metrics),
        "total-upload-bytes": int(
            sum(
                array.numpy().nbytes
                for record in records
                for array_record in record.array_records.values()
                for array in array_record.values()
            )
        ),
    })


def aggregate_evaluation_metrics(records: list[RecordDict], weighted_by_key: str) -> MetricRecord:
    metrics = _metric_records(records)
    counts = np.asarray([float(item[weighted_by_key]) for item in metrics])
    maes = np.asarray([float(item["mae"]) for item in metrics])
    rmses = np.asarray([float(item["rmse"]) for item in metrics])
    smapes = np.asarray([float(item["smape"]) for item in metrics])
    macro = float(maes.mean())
    tail_count = max(1, int(np.ceil(len(maes) * 0.25)))
    result = {
        "macro_mae": macro,
        "micro_mae": float(np.average(maes, weights=counts)),
        "worst_station_mae": float(maes.max()),
        "station_mae_std": float(maes.std(ddof=0)),
        "station_mae_cv": float(maes.std(ddof=0) / max(macro, 1e-8)),
        "station_cvar25_mae": float(np.sort(maes)[-tail_count:].mean()),
        "macro_rmse": float(rmses.mean()),
        "macro_smape": float(smapes.mean()),
        "num-examples": int(counts.sum()),
        "num-clients": len(metrics),
    }
    high_pollution = np.asarray(
        [float(item.get("high-pollution-mae", np.nan)) for item in metrics]
    )
    if np.isfinite(high_pollution).any():
        result["macro_high_pollution_mae"] = float(np.nanmean(high_pollution))
    return MetricRecord(result)
