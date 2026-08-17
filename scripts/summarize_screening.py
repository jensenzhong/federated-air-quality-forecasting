"""Summarize the pre-registered validation-only screening queue.

The screening report is intentionally separate from the formal test report.  It
matches queue items to immutable run artifacts, applies the registered
lexicographic selection rule, and refuses to consume test metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from aqfl.config import project_root

FEDERATED_PARAMETER_KEYS = {
    "fedprox": "proximal_mu",
    "qfedavg": "q",
    "fedadam": "server_lr",
}


def _utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _close(left: Any, right: Any) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-12
    except (TypeError, ValueError):
        return left == right


def _load_run(path: Path) -> dict[str, Any] | None:
    summary_path = path / "summary.json"
    config_path = path / "resolved_config.yaml"
    if not summary_path.is_file() or not config_path.is_file():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError):
        return None
    if summary.get("status") != "completed":
        return None
    return {"path": path, "summary": summary, "config": config}


def _matches(item: dict[str, Any], run: dict[str, Any]) -> bool:
    summary = run["summary"]
    config = run["config"]
    runtime = config.get("runtime", {})
    if summary.get("method") != item["method"] or int(summary.get("seed", -1)) != int(item["seed"]):
        return False
    if item["method"] == "centralized_gru":
        return (
            int(config.get("model", {}).get("hidden_size", -1)) == int(item["hidden_size"])
            and _close(config.get("training", {}).get("learning_rate"), item["learning_rate"])
            and int(runtime.get("smoke_epochs_override", -1)) == int(item["centralized_epochs"])
        )
    parameter_key = FEDERATED_PARAMETER_KEYS[item["method"]]
    return (
        int(runtime.get("num_rounds", -1)) == int(item["rounds"])
        and _close(runtime.get(parameter_key), item[parameter_key])
        and runtime.get("evaluation_split") == "val"
    )


def _metric_row(run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    method = summary["method"]
    if method == "centralized_gru":
        source = summary
        primary = float(summary["macro_mae"])
        best_round = None
    else:
        best_round = int(summary["best_round"])
        frame = pd.read_parquet(run["path"] / "round_metrics.parquet")
        rows = frame.loc[frame["round"] == best_round]
        if rows.empty:
            raise ValueError(f"best round {best_round} missing in {run['path']}")
        source = rows.iloc[0].to_dict()
        primary = float(summary["best_validation_macro_mae"])
    started = _utc(summary["started_at_utc"])
    completed = _utc(summary["completed_at_utc"])
    rss_values: list[float] = []
    telemetry = run["path"] / "system_metrics.jsonl"
    if telemetry.is_file():
        for line in telemetry.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "rss_gb" in event:
                rss_values.append(float(event["rss_gb"]))
    return {
        "run_id": summary["run_id"],
        "run_dir": str(run["path"].as_posix()),
        "method": method,
        "seed": int(summary["seed"]),
        "evaluation_split": summary.get("evaluation_split"),
        "protocol_frozen": bool(summary.get("protocol_frozen", False)),
        "status": summary.get("status"),
        "best_round": best_round,
        "macro_mae": primary,
        "worst_station_mae": float(source.get("worst_station_mae", summary.get("worst_station_mae", float("nan")))),
        "station_mae_cv": float(source.get("station_mae_cv", summary.get("station_mae_cv", float("nan")))),
        "elapsed_seconds": (completed - started).total_seconds(),
        "peak_rss_gb": max(rss_values) if rss_values else None,
        "test_metrics": summary.get("test_metrics", "TBD"),
    }


def collect(plan_path: Path, queue_path: Path, runs_dir: Path) -> dict[str, Any]:
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    runs = [run for path in runs_dir.iterdir() if path.is_dir() for run in [_load_run(path)] if run is not None]
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    used: set[str] = set()
    for item in queue:
        matches = [run for run in runs if str(run["path"]) not in used and _matches(item, run)]
        if not matches:
            missing.append(item)
            continue
        run = max(matches, key=lambda candidate: candidate["summary"].get("started_at_utc", ""))
        used.add(str(run["path"]))
        row = _metric_row(run)
        row.update({key: value for key, value in item.items() if key not in {"status"}})
        if row["evaluation_split"] != "val" or row["test_metrics"] != "TBD":
            raise ValueError(f"screening run accessed test data: {row['run_id']}")
        rows.append(row)
    if missing:
        raise RuntimeError(f"Missing screening artifacts for {len(missing)} queue item(s): {missing}")

    selection = plan["screening"]["selection"]
    sort_keys = [selection["primary_metric"], *selection["tie_breakers"]]
    winners: dict[str, dict[str, Any]] = {}
    for method in sorted({row["method"] for row in rows}):
        candidates = [row for row in rows if row["method"] == method]
        candidates.sort(key=lambda row: tuple(row[key] for key in sort_keys))
        winners[method] = candidates[0]
    return {
        "report_type": "screening_validation_only",
        "formal_results_status": "screening_only_nonformal",
        "selection": selection,
        "queue_path": str(queue_path.as_posix()),
        "candidate_count": len(rows),
        "completed_count": len(rows),
        "test_access": "prohibited_and_checked",
        "candidates": sorted(rows, key=lambda row: (row["method"], row["macro_mae"])),
        "winners": winners,
        "resource_summary": {
            "total_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
            "max_peak_rss_gb": max(row["peak_rss_gb"] for row in rows if row["peak_rss_gb"] is not None),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# W3 Screening Results",
        "",
        "> Validation-only, single-seed (42), 3-round federated / 1-epoch centralized screening. These are non-formal selection results; test metrics were not read.",
        "",
        "## Selection rule",
        "",
        f"Primary: `{report['selection']['primary_metric']}`; tie-breakers: "
        + ", ".join(f"`{key}`" for key in report["selection"]["tie_breakers"])
        + ".",
        "",
        "## Candidates",
        "",
        "| Method | Parameter | Macro MAE | Worst station MAE | Station MAE CV | Best round | Elapsed (s) | Peak RSS (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    parameter_names = {"centralized_gru": "hidden/lr", "fedprox": "proximal_mu", "qfedavg": "q", "fedadam": "server_lr"}
    for row in report["candidates"]:
        if row["method"] == "centralized_gru":
            parameter = f"{row['hidden_size']}/{row['learning_rate']}"
        else:
            parameter = str(row[parameter_names[row["method"]]])
        lines.append(
            f"| {row['method']} | {parameter} | {row['macro_mae']:.4f} | {row['worst_station_mae']:.4f} | "
            f"{row['station_mae_cv']:.4f} | {row['best_round'] or '—'} | {row['elapsed_seconds']:.1f} | "
            f"{row['peak_rss_gb']:.3f} |" if row["peak_rss_gb"] is not None else
            f"| {row['method']} | {parameter} | {row['macro_mae']:.4f} | {row['worst_station_mae']:.4f} | "
            f"{row['station_mae_cv']:.4f} | {row['best_round'] or '—'} | {row['elapsed_seconds']:.1f} | — |"
        )
    lines.extend(["", "## Selected validation candidates", "", "| Method | Run | Parameter | Macro MAE |", "|---|---|---:|---:|"])
    for method, row in report["winners"].items():
        parameter = (
            f"{row['hidden_size']}/{row['learning_rate']}" if method == "centralized_gru"
            else str(row[parameter_names[method]])
        )
        lines.append(f"| {method} | `{row['run_id']}` | {parameter} | {row['macro_mae']:.4f} |")
    lines.extend([
        "",
        f"Total measured artifact time: {report['resource_summary']['total_elapsed_seconds'] / 60:.1f} min; maximum recorded RSS: {report['resource_summary']['max_peak_rss_gb']:.3f} GB.",
        "",
        "The selected candidates require a 30-round confirmation run before any formal test evaluation or protocol freeze.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = project_root()
    parser.add_argument("--plan", type=Path, default=root / "configs/experiments/formal.yaml")
    parser.add_argument("--queue", type=Path, default=root / "artifacts/queue_screening.json")
    parser.add_argument("--runs-dir", type=Path, default=root / "artifacts/runs")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/reports")
    args = parser.parse_args()
    report = collect(args.plan, args.queue, args.runs_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "screening_results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "screening_results.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"candidates": report["candidate_count"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
