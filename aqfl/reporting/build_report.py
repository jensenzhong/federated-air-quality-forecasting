"""Build a Markdown report from validated experiment runs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aqfl.reporting.method_catalog import method_info

_GROUP_ORDER = {
    "非联邦参考": 0,
    "传统联邦参考": 1,
    "待协议审计的传统参考": 2,
    "传统联邦主表": 3,
    "同动作空间控制器对比": 4,
    "控制器消融": 5,
    "机制上界对比": 6,
    "持续学习独立表": 7,
    "历史 v1 参考": 8,
    "开发候选（不进主表）": 9,
}


def build_report(registry_path: Path, output: Path) -> None:
    registry = pd.read_csv(registry_path)
    validated = registry[registry["status"] == "validated"]
    rows = []
    for _, record in validated.iterrows():
        summary_path = Path(record["run_dir"]) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        info = method_info(str(summary.get("method", record["method"])))
        rows.append({
            "Comparison group": info.comparison_group,
            "Layer": info.layer,
            "Method": info.display_name,
            "Internal ID": info.method_id,
            "Seed": summary.get("seed", record["seed"]),
            "Macro MAE": summary.get("macro_mae", "TBD"),
            "Worst Station MAE": summary.get("worst_station_mae", "TBD"),
            "Station MAE CV": summary.get("station_mae_cv", "TBD"),
            # The registry is the promotion authority. A stale summary must not
            # downgrade or otherwise misrepresent a validated run in a report.
            "Status": record["status"],
            "_group_order": _GROUP_ORDER.get(info.comparison_group, 99),
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = pd.DataFrame(rows).sort_values(
            ["_group_order", "Layer", "Method", "Seed"],
            kind="stable",
        ).drop(columns=["_group_order"])
        table = frame.to_markdown(index=False)
    else:
        table = "No validated experiment results are available."
    output.write_text(
        "# AQ-MAS-FL experiment report\n\n"
        "The table is grouped by comparison role. Methods in different groups "
        "are not direct controller comparisons. Internal IDs are shown only for "
        "artifact traceability.\n\n"
        + table
        + "\n\nAll missing results remain `TBD`; no value is fabricated.\n",
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
