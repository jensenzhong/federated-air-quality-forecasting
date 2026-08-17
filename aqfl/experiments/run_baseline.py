"""Run deterministic and neural non-federated baselines."""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset

from aqfl.config import load_config, set_seed
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.preprocessing import GlobalScalerState
from aqfl.evaluation.metrics import aggregate_station_metrics, regression_metrics
from aqfl.models import build_model
from aqfl.models.training import evaluate_model, fit_with_early_stopping
from aqfl.reporting.artifacts import ProcessResourceSampler, RunArtifacts

METHODS = {"persistence", "seasonal_naive", "local_gru", "centralized_gru", "mlp"}


def _scaler(config: dict[str, Any]) -> GlobalScalerState:
    return GlobalScalerState(**load_cache_metadata(config)["scaler"])


def _evaluate_naive(
    config: dict[str, Any], method: str, artifacts: RunArtifacts, split: str
) -> dict[str, Any]:
    station_metrics = {}
    threshold = _scaler(config).pollution_p90
    for station in list_stations(config):
        dataset = load_station_dataset(config, station, split)
        prediction = np.asarray(getattr(dataset, method))
        y_true = np.asarray(dataset.y_raw).reshape(-1)
        station_metrics[station] = regression_metrics(y_true, prediction, threshold)
        artifacts.save_predictions(station, split, dataset.target_timestamp_ns, y_true, prediction)
    return aggregate_station_metrics(station_metrics)


def _run_centralized(
    config: dict[str, Any],
    method: str,
    artifacts: RunArtifacts,
    evaluation_split: str,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, float]]]:
    stations = list_stations(config)
    train_sets = [load_station_dataset(config, station, "train") for station in stations]
    val_sets = {station: load_station_dataset(config, station, "val") for station in stations}
    evaluation_sets = {
        station: load_station_dataset(config, station, evaluation_split) for station in stations
    }
    model = build_model(config, "mlp" if method == "mlp" else "gru")
    model, history = fit_with_early_stopping(
        model,
        ConcatDataset(train_sets),
        val_sets,
        _scaler(config),
        int(config["training"]["centralized_epochs"]),
        int(config["training"]["patience"]),
        float(config["training"]["learning_rate"]),
        int(config["training"]["batch_size"]),
        float(config["training"]["weight_decay"]),
    )
    station_metrics = {}
    scaler = _scaler(config)
    for station, dataset in evaluation_sets.items():
        metrics, prediction = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
        station_metrics[station] = metrics
        artifacts.save_predictions(
            station, evaluation_split, dataset.target_timestamp_ns, dataset.y_raw.reshape(-1), prediction
        )
    return model, aggregate_station_metrics(station_metrics), history


def _run_local(
    config: dict[str, Any], artifacts: RunArtifacts, evaluation_split: str
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, Any]]]:
    scaler = _scaler(config)
    station_metrics = {}
    histories: list[dict[str, Any]] = []
    checkpoint_models = torch.nn.ModuleDict()
    for station in list_stations(config):
        model = build_model(config)
        model, station_history = fit_with_early_stopping(
            model,
            load_station_dataset(config, station, "train"),
            {station: load_station_dataset(config, station, "val")},
            scaler,
            int(config["training"]["centralized_epochs"]),
            int(config["training"]["patience"]),
            float(config["training"]["learning_rate"]),
            int(config["training"]["batch_size"]),
            float(config["training"]["weight_decay"]),
        )
        dataset = load_station_dataset(config, station, evaluation_split)
        metrics, prediction = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
        station_metrics[station] = metrics
        histories.extend({"station": station, **record} for record in station_history)
        artifacts.save_predictions(
            station, evaluation_split, dataset.target_timestamp_ns, dataset.y_raw.reshape(-1), prediction
        )
        checkpoint_models[station] = model
    return checkpoint_models, aggregate_station_metrics(station_metrics), histories


def run(
    method: str,
    config_path: str,
    seed: int,
    evaluation_split: str = "val",
    protocol_frozen: bool = False,
    max_epochs: int | None = None,
    hidden_size: int | None = None,
    learning_rate: float | None = None,
) -> str:
    if method not in METHODS:
        raise ValueError(f"Unsupported baseline: {method}")
    if max_epochs is not None and max_epochs < 1:
        raise ValueError("--max-epochs must be at least 1")
    if hidden_size is not None and hidden_size < 1:
        raise ValueError("--hidden-size must be at least 1")
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    has_validation_override = any(
        value is not None for value in (max_epochs, hidden_size, learning_rate)
    )
    if has_validation_override and (evaluation_split == "test" or protocol_frozen):
        raise RuntimeError("Training overrides are validation-only screening controls")
    config = load_config(config_path)
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {evaluation_split}")
    if evaluation_split == "test" and not protocol_frozen:
        raise RuntimeError("Test evaluation requires --protocol-frozen")
    if max_epochs is not None:
        config["training"]["centralized_epochs"] = max_epochs
    if hidden_size is not None:
        config["model"]["hidden_size"] = hidden_size
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate
    config["project"]["seed"] = seed
    config["runtime"] = {
        "method": method,
        "evaluation_split": evaluation_split,
        "protocol_frozen": protocol_frozen,
        "smoke_epochs_override": max_epochs,
        "hidden_size_override": hidden_size,
        "learning_rate_override": learning_rate,
    }
    set_seed(seed)
    artifacts = RunArtifacts(config, method, seed)
    try:
        with ProcessResourceSampler(artifacts.path / "system_metrics.jsonl") as telemetry:
            if method in {"persistence", "seasonal_naive"}:
                summary = _evaluate_naive(config, method, artifacts, evaluation_split)
                model = build_model(config)
                history: list[dict[str, Any]] = []
            elif method == "local_gru":
                model, summary, history = _run_local(config, artifacts, evaluation_split)
            else:
                model, summary, history = _run_centralized(
                    config, method, artifacts, evaluation_split
                )
        summary["peak_rss_gb"] = round(telemetry.peak_rss_gb, 6)
        summary["minimum_available_memory_observed_gb"] = round(
            telemetry.minimum_available_memory_gb, 6
        )
        if method in {"local_gru", "centralized_gru", "mlp"} and history:
            summary["configured_epochs"] = int(config["training"]["centralized_epochs"])
            summary["effective_epochs"] = int(max(record["epoch"] for record in history))
        summary["evaluation_split"] = evaluation_split
        summary["protocol_frozen"] = protocol_frozen
        artifacts.finalize(model, summary, round_metrics=history)
    except Exception as exc:
        artifacts.invalidate(str(exc))
        raise
    return str(artifacts.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--protocol-frozen", action="store_true")
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Validation-only smoke override for Local/Centralized training epochs.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help="Validation-only GRU screening override.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Validation-only learning-rate screening override.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "run_dir": run(
                    args.method,
                    args.config,
                    args.seed,
                    args.split,
                    args.protocol_frozen,
                    args.max_epochs,
                    args.hidden_size,
                    args.learning_rate,
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
