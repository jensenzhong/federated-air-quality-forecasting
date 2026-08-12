"""Build a Markdown report from validated experiment runs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_report(registry_path: Path, output: Path) -> None:
    registry = pd.read_csv(registry_path)
    validated = registry[registry["status"] == "validated"]
    rows = []
    for _, record in validated.iterrows():
        summary_path = Path(record["run_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append({
            "Method": summary.get("method", record["method"]),
            "Seed": summary.get("seed", record["seed"]),
            "Macro MAE": summary.get("macro_mae", "TBD"),
            "Worst Station MAE": summary.get("worst_station_mae", "TBD"),
            "Station MAE CV": summary.get("station_mae_cv", "TBD"),
            "Status": summary.get("status", "TBD"),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).to_markdown(index=False) if rows else "No validated experiment results are available."
    output.write_text(
        "# AQ-MAS-FL experiment report\n\n" + table + "\n\nAll missing results remain `TBD`; no value is fabricated.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="artifacts/experiment_registry.csv")
    parser.add_argument("--output", default="artifacts/reports/main_results.md")
    args = parser.parse_args()
    build_report(Path(args.registry), Path(args.output))


if __name__ == "__main__":
    main()
