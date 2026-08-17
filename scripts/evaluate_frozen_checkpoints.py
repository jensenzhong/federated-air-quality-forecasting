"""Evaluate frozen validation-selected checkpoints on the test split once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from aqfl.config import project_root, set_seed
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.preprocessing import GlobalScalerState
from aqfl.evaluation.metrics import aggregate_station_metrics
from aqfl.models import build_model
from aqfl.models.training import evaluate_model
from aqfl.reporting.artifacts import ProcessResourceSampler, RunArtifacts


def _load_source(source_run_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = project_root()
    source_path = root / "artifacts" / "runs" / source_run_id
    summary_path = source_path / "summary.json"
    config_path = source_path / "resolved_config.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing source run artifacts: {source_run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(summary, dict):
        raise ValueError(f"Invalid source metadata: {source_run_id}")
    if summary.get("status") != "completed":
        raise RuntimeError(f"Source run is not completed: {source_run_id}")
    if summary.get("evaluation_split") != "val" or summary.get("protocol_frozen"):
        raise RuntimeError("Frozen test evaluation requires a completed validation-only source run")
    config["_config_path"] = str((root / "configs" / "base.yaml").resolve())
    return config, summary, source_path


def evaluate_source(source_run_id: str) -> str:
    config, source_summary, source_path = _load_source(source_run_id)
    seed = int(source_summary["seed"])
    method = str(source_summary["method"])
    config["project"]["seed"] = seed
    config["runtime"] = {
        "method": method,
        "source_run_id": source_run_id,
        "evaluation_split": "test",
        "protocol_frozen": True,
        "execution_mode": "frozen_checkpoint_evaluation",
    }
    set_seed(seed)
    artifacts = RunArtifacts(config, method, seed)
    model = build_model(config)
    checkpoint = torch.load(source_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint)
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    station_metrics: dict[str, dict[str, Any]] = {}
    client_rows: list[dict[str, Any]] = []
    try:
        with ProcessResourceSampler(artifacts.path / "system_metrics.jsonl") as telemetry:
            for station in list_stations(config):
                dataset = load_station_dataset(config, station, "test")
                metrics, prediction = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
                station_metrics[station] = metrics
                client_rows.append({"station": station, "split": "test", **metrics})
                artifacts.save_predictions(
                    station,
                    "test",
                    dataset.target_timestamp_ns,
                    dataset.y_raw.reshape(-1),
                    prediction,
                )
        test_summary = aggregate_station_metrics(station_metrics)
        summary = {
            "protocol": "frozen_checkpoint_single_test_evaluation",
            "execution_mode": "frozen_checkpoint_evaluation",
            "source_run_id": source_run_id,
            "source_best_round": source_summary.get("best_round"),
            "source_best_validation_macro_mae": source_summary.get("best_validation_macro_mae"),
            "num_clients": len(station_metrics),
            "checkpoint_selection": "source_validation_macro_mae",
            "evaluation_split": "test",
            "protocol_frozen": True,
            "test_metrics": test_summary,
            "peak_rss_gb": round(telemetry.peak_rss_gb, 6),
            "minimum_available_memory_observed_gb": round(
                telemetry.minimum_available_memory_gb, 6
            ),
        }
        artifacts.finalize(
            model,
            summary,
            round_metrics=[{"source_best_round": source_summary.get("best_round")}],
            client_metrics=client_rows,
        )
    except Exception as exc:
        artifacts.invalidate(str(exc))
        raise
    return str(artifacts.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", action="append", required=True)
    args = parser.parse_args()
    for source_run_id in args.source_run_id:
        print(json.dumps({"source_run_id": source_run_id, "run_dir": evaluate_source(source_run_id)}))


if __name__ == "__main__":
    main()
