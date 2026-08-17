"""Immutable, auditable experiment artifact management."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import torch
import yaml

from aqfl.config import canonical_json, project_root, resolve_project_path

REQUIRED_ARTIFACTS = [
    "resolved_config.yaml",
    "dataset_manifest.json",
    "environment.json",
    "round_metrics.parquet",
    "client_metrics.parquet",
    "predictions",
    "decisions.jsonl",
    "system_metrics.jsonl",
    "checkpoint.pt",
    "summary.json",
]


def git_short_sha(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def make_run_id(method: str, seed: int, root: Path) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{method}-{seed}-{timestamp}-{git_short_sha(root)}"


class RunArtifacts:
    def __init__(self, config: dict[str, Any], method: str, seed: int) -> None:
        self.config = config
        self.root = project_root(config)
        output_root = resolve_project_path(config, config["project"]["output_dir"])
        self.run_id = make_run_id(method, seed, self.root)
        self.path = output_root / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "predictions").mkdir()
        self.method = method
        self.seed = seed
        self.started_at = datetime.now(UTC)
        self.write_initial_files()

    def write_initial_files(self) -> None:
        clean_config = {key: value for key, value in self.config.items() if key != "_config_path"}
        (self.path / "resolved_config.yaml").write_text(yaml.safe_dump(clean_config, sort_keys=False), encoding="utf-8")
        manifest_path = resolve_project_path(self.config, "data/manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        (self.path / "dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        dataset_manifest_sha256 = hashlib.sha256(
            canonical_json(manifest).encode("utf-8")
        ).hexdigest()
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": psutil.cpu_count(logical=True),
            "physical_cpus": psutil.cpu_count(logical=False),
            "total_memory_gb": psutil.virtual_memory().total / 1024**3,
            "torch": torch.__version__,
            "dependencies": {
                "numpy": __import__("numpy").__version__,
                "pandas": pd.__version__,
                "flower": __import__("flwr").__version__,
            },
            "config_sha256": hashlib.sha256(canonical_json(clean_config).encode()).hexdigest(),
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "git_sha": git_short_sha(self.root),
        }
        (self.path / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        (self.path / "decisions.jsonl").touch()
        (self.path / "system_metrics.jsonl").touch()
        with (self.path / "system_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "run_start",
                        "logical_cpus": environment["logical_cpus"],
                        "available_memory_gb": psutil.virtual_memory().available / 1024**3,
                    }
                )
                + "\n"
            )
        pd.DataFrame().to_parquet(self.path / "round_metrics.parquet")
        pd.DataFrame().to_parquet(self.path / "client_metrics.parquet")

    def save_predictions(self, station: str, split: str, timestamps: Any, y_true: Any, y_pred: Any) -> None:
        frame = pd.DataFrame({
            "station": station,
            "split": split,
            "target_timestamp_ns": timestamps,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        frame.to_parquet(self.path / "predictions" / f"{station}_{split}.parquet", index=False)

    def finalize(
        self,
        model: torch.nn.Module,
        summary: dict[str, Any],
        round_metrics: list[dict[str, Any]] | None = None,
        client_metrics: list[dict[str, Any]] | None = None,
        status: str = "completed",
    ) -> None:
        torch.save(model.state_dict(), self.path / "checkpoint.pt")
        pd.DataFrame(round_metrics or []).to_parquet(self.path / "round_metrics.parquet", index=False)
        pd.DataFrame(client_metrics or []).to_parquet(self.path / "client_metrics.parquet", index=False)
        summary = dict(summary)
        summary.update({
            "run_id": self.run_id,
            "method": self.method,
            "seed": self.seed,
            "status": status,
            "started_at_utc": self.started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        })
        (self.path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        with (self.path / "system_metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "run_end",
                        "available_memory_gb": psutil.virtual_memory().available / 1024**3,
                    }
                )
                + "\n"
            )

    def invalidate(self, reason: str) -> None:
        summary = {"run_id": self.run_id, "method": self.method, "seed": self.seed, "status": "invalid", "reason": reason}
        (self.path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


class ProcessResourceSampler:
    """Sample process RSS and host availability during non-federated runs."""

    def __init__(self, path: Path, interval_seconds: float = 0.5) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.path = path
        self.interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._sample_until_stopped,
            name="process-resource-sampler",
            daemon=True,
        )
        self._started_at = time.monotonic()
        self.peak_rss_gb = 0.0
        self.minimum_available_memory_gb = float("inf")

    def __enter__(self) -> ProcessResourceSampler:
        self._write_sample("process_sampler_start")
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._write_sample("process_sampler_end")

    def _sample_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write_sample("process_sample")

    def _write_sample(self, event: str) -> None:
        try:
            rss_gb = float(self._process.memory_info().rss / 1024**3)
            available_memory_gb = float(psutil.virtual_memory().available / 1024**3)
        except (psutil.Error, OSError):
            return
        self.peak_rss_gb = max(self.peak_rss_gb, rss_gb)
        self.minimum_available_memory_gb = min(
            self.minimum_available_memory_gb, available_memory_gb
        )
        payload = {
            "event": event,
            "elapsed_seconds": time.monotonic() - self._started_at,
            "rss_gb": rss_gb,
            "available_memory_gb": available_memory_gb,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def validate_artifact_directory(path: Path) -> list[str]:
    errors = []
    for name in REQUIRED_ARTIFACTS:
        if not (path / name).exists():
            errors.append(f"missing:{name}")
    summary_path = path / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") not in {"completed", "validated", "invalid"}:
                errors.append("invalid_status")
            if summary.get("status") == "invalid":
                errors.append("run_marked_invalid")
            if summary.get("evaluation_split") != "test":
                errors.append("not_test_evaluation")
            if not summary.get("protocol_frozen", False):
                errors.append("protocol_not_frozen")
        except json.JSONDecodeError:
            errors.append("invalid_summary_json")
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if str(summary.get("method", "")).startswith("pafa_"):
                if summary.get("secure_aggregation") != "flower_secaggplus":
                    errors.append("pafa_missing_verified_secaggplus")
                if summary.get("coordinator_visibility") != "cohort_summary_only":
                    errors.append("pafa_coordinator_visibility_violation")
                if summary.get("client_metrics_persisted_on_server") is not False:
                    errors.append("pafa_client_metrics_persisted")
                event_path = path / "agentic_events.jsonl"
                if not event_path.is_file():
                    errors.append("missing:agentic_events.jsonl")
                else:
                    event_types: set[str] = set()
                    for line in event_path.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        event = json.loads(line)
                        event_types.add(str(event.get("event", "")))
                        serialized = json.dumps(event).lower()
                        if any(
                            token in serialized
                            for token in ('"client_id"', '"node_id"', '"station"', '"partition_id"')
                        ):
                            errors.append("pafa_event_contains_client_identity")
                    if "cohort_summary" not in event_types:
                        errors.append("incomplete_agentic_event_trace")
                client_metrics_path = path / "client_metrics.parquet"
                if client_metrics_path.is_file():
                    try:
                        if not pd.read_parquet(client_metrics_path).empty:
                            errors.append("pafa_client_metrics_not_empty")
                    except Exception:
                        errors.append("pafa_client_metrics_unreadable")
        except json.JSONDecodeError:
            errors.append("invalid_agentic_event_json")
    manifest_path = path / "dataset_manifest.json"
    environment_path = path / "environment.json"
    if manifest_path.is_file() and environment_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
            actual = hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()
            if environment.get("dataset_manifest_sha256") != actual:
                errors.append("dataset_manifest_hash_mismatch")
        except json.JSONDecodeError:
            errors.append("invalid_manifest_or_environment_json")
    return errors
