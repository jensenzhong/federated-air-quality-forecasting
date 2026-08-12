"""Validate one run directory and optionally promote it in the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aqfl.reporting.artifacts import validate_artifact_directory
from aqfl.reporting.registry import ExperimentRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--registry", default="artifacts/experiment_registry.csv")
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    errors = validate_artifact_directory(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").is_file() else {}
    if errors:
        print(json.dumps({"status": "invalid", "errors": errors}, indent=2))
        raise SystemExit(1)
    registry = ExperimentRegistry(Path(args.registry))
    run_id = str(summary.get("run_id", run_dir.name))
    existing = registry.frame[registry.frame["run_id"] == run_id]
    if existing.empty:
        registry.transition(run_id, "planned", method=summary.get("method", ""), seed=summary.get("seed", ""), run_dir=str(run_dir))
        registry.transition(run_id, "running")
        registry.transition(run_id, "completed")
    if registry.frame.loc[registry.frame["run_id"] == run_id, "status"].iloc[0] == "completed":
        registry.transition(run_id, "validated")
        summary["status"] = "validated"
        (run_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps({"status": "validated", "run_id": run_id}, indent=2))


if __name__ == "__main__":
    main()
