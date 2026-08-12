from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from aqfl.reporting.artifacts import REQUIRED_ARTIFACTS, RunArtifacts, validate_artifact_directory
from aqfl.reporting.build_report import build_report
from aqfl.reporting.registry import ExperimentRegistry


def test_artifact_validator(tmp_path) -> None:
    errors = validate_artifact_directory(tmp_path)
    assert "missing:summary.json" in errors
    for name in REQUIRED_ARTIFACTS:
        path = tmp_path / name
        path.mkdir() if name == "predictions" else path.touch()
    (tmp_path / "summary.json").write_text(
        json.dumps({"status": "completed", "evaluation_split": "test", "protocol_frozen": True}),
        encoding="utf-8",
    )
    assert "invalid_manifest_or_environment_json" in validate_artifact_directory(tmp_path)


def test_registry_state_machine(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.csv")
    registry.transition("run", "planned", method="fedavg", seed=42)
    registry.transition("run", "running")
    registry.transition("run", "completed")
    registry.transition("run", "validated")
    assert registry.frame.loc[0, "status"] == "validated"
    with pytest.raises(ValueError):
        registry.transition("run", "running")


def test_report_reads_validated_only(tmp_path) -> None:
    valid_dir = tmp_path / "valid"
    invalid_dir = tmp_path / "invalid"
    valid_dir.mkdir()
    invalid_dir.mkdir()
    (valid_dir / "summary.json").write_text(json.dumps({"method": "fedavg", "seed": 42, "macro_mae": 1}), encoding="utf-8")
    (invalid_dir / "summary.json").write_text(json.dumps({"method": "bad", "macro_mae": 999}), encoding="utf-8")
    registry = tmp_path / "registry.csv"
    pd.DataFrame(
        [
            {"status": "validated", "run_dir": valid_dir, "method": "fedavg", "seed": 42},
            {"status": "invalid", "run_dir": invalid_dir, "method": "bad", "seed": 1},
        ]
    ).to_csv(registry, index=False)
    output = tmp_path / "report.md"
    build_report(registry, output)
    text = output.read_text(encoding="utf-8")
    assert "fedavg" in text
    assert "999" not in text


def test_run_artifacts_are_complete_and_immutable(tmp_path) -> None:
    config_path = tmp_path / "configs" / "base.yaml"
    config_path.parent.mkdir()
    config_path.touch()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "manifest.json").write_text(json.dumps({"status": "prepared"}), encoding="utf-8")
    config = {
        "_config_path": str(config_path),
        "project": {"output_dir": "artifacts/runs"},
        "data": {"cache_dir": "data/cache"},
    }
    artifacts = RunArtifacts(config, "test", 42)
    artifacts.save_predictions("a", "test", [1], [2.0], [3.0])
    model = torch.nn.Linear(1, 1)
    artifacts.finalize(
        model,
        {"macro_mae": 1.0, "evaluation_split": "test", "protocol_frozen": True},
        round_metrics=[{"round": 1}],
    )
    assert validate_artifact_directory(artifacts.path) == []
    summary = json.loads((artifacts.path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    with pytest.raises(FileExistsError):
        artifacts.path.mkdir(exist_ok=False)
