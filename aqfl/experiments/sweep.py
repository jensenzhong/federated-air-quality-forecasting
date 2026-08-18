"""Create and optionally execute the pre-registered staged experiment queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from aqfl.config import project_root
from aqfl.reporting.registry import ExperimentRegistry

FEDERATED_METHODS = {
    "fedavg", "fedprox", "qfedavg", "fedadam", "rule_mas", "mas_llm",
    "mas_llm_dynamic_only", "mas_llm_no_fairness", "fedprox_budget_matched",
    "pafa_rule", "pafa_bandit", "pafa_bandit_fedadam", "pafa_probe_oracle", "pafa_llm", "pafa_llm_no_probe",
    "pafa_fedavg", "pafa_fedprox", "pafa_fedadam", "pafa_fedprox_budget_matched",
}
DETERMINISTIC_METHODS = {"persistence", "seasonal_naive"}


def load_plan(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_queue(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in plan["stage_order"]:
        raise ValueError(f"Unknown experiment stage: {stage}")
    if stage == "screening":
        screening = plan["screening"]
        screening_rounds = int(screening["rounds"])
        centralized_epochs = int(screening["centralized_epochs"])
        queue: list[dict[str, Any]] = []
        for hidden_size in screening["model"]["hidden_size"]:
            for learning_rate in screening["model"]["learning_rate"]:
                queue.append({
                    "method": "centralized_gru",
                    "seed": 42,
                    "stage": stage,
                    "status": "planned",
                    "hidden_size": hidden_size,
                    "learning_rate": learning_rate,
                    "centralized_epochs": centralized_epochs,
                })
        queue.extend(
            {
                "method": "fedprox",
                "seed": 42,
                "stage": stage,
                "status": "planned",
                "proximal_mu": value,
                "rounds": screening_rounds,
            }
            for value in screening["fedprox_mu"]
        )
        queue.extend(
            {
                "method": "qfedavg",
                "seed": 42,
                "stage": stage,
                "status": "planned",
                "q": value,
                "rounds": screening_rounds,
            }
            for value in screening["qfedavg_q"]
        )
        queue.extend(
            {
                "method": "fedadam",
                "seed": 42,
                "stage": stage,
                "status": "planned",
                "server_lr": value,
                "rounds": screening_rounds,
            }
            for value in screening["fedadam_server_lr"]
        )
        return queue
    seeds = [42] if stage in {"smoke", "screening", "single_seed_full"} else list(plan["seeds"])
    queue = []
    for method in plan["methods"]:
        method_seeds = [42] if method in DETERMINISTIC_METHODS else seeds
        for seed in method_seeds:
            queue.append({"method": method, "seed": seed, "stage": stage, "status": "planned"})
    return queue


def command_for(item: dict[str, Any]) -> list[str]:
    method = item["method"]
    seed = str(item["seed"])
    if method in FEDERATED_METHODS:
        rounds = str(item.get("rounds", 3 if item["stage"] == "smoke" else 30))
        evaluation_split = "val" if item["stage"] in {"smoke", "screening", "single_seed_full"} else "test"
        command = [
            sys.executable,
            "scripts/run_flower_sequential.py",
            "--method",
            method,
            "--seed",
            seed,
            "--rounds",
            rounds,
            "--evaluation-split",
            evaluation_split,
        ]
        if evaluation_split == "test":
            command.append("--protocol-frozen")
        for item_key, argument in (
            ("proximal_mu", "--proximal-mu"),
            ("q", "--q"),
            ("qffl_lr", "--qffl-lr"),
            ("server_lr", "--server-lr"),
        ):
            if item_key in item:
                command.extend([argument, str(item[item_key])])
        if "budget_trace" in item:
            command.extend(["--budget-trace", str(item["budget_trace"])])
        return command
    evaluation_split = "val" if item["stage"] in {"smoke", "screening", "single_seed_full"} else "test"
    command = [
        sys.executable,
        "-m",
        "aqfl.experiments.run_baseline",
        "--method",
        method,
        "--seed",
        seed,
        "--split",
        evaluation_split,
    ]
    if evaluation_split == "test":
        command.append("--protocol-frozen")
    if "hidden_size" in item:
        command.extend(["--hidden-size", str(item["hidden_size"])])
    if "learning_rate" in item:
        command.extend(["--learning-rate", str(item["learning_rate"])])
    if "centralized_epochs" in item:
        command.extend(["--max-epochs", str(item["centralized_epochs"])])
    return command


def ensure_planned(
    registry: ExperimentRegistry,
    queue_id: str,
    item: dict[str, Any],
) -> None:
    existing = registry.frame[registry.frame["run_id"] == queue_id]
    if existing.empty:
        registry.transition(
            queue_id,
            "planned",
            method=item["method"],
            seed=item["seed"],
        )
        return
    current = str(existing.iloc[0]["status"])
    if current != "planned":
        raise RuntimeError(
            f"Queue item {queue_id} is already {current}; refusing to overwrite its state"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/experiments/formal.yaml")
    parser.add_argument("--stage", default="smoke")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = project_root()
    plan = load_plan(Path(args.plan))
    queue = build_queue(plan, args.stage)
    queue_path = root / "artifacts" / f"queue_{args.stage}.json"
    queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
    registry = ExperimentRegistry(root / "artifacts" / "experiment_registry.csv")
    started = time.monotonic()
    for index, item in enumerate(queue):
        queue_id = f"queue-{args.stage}-{index:03d}-{item['method']}-{item['seed']}"
        ensure_planned(registry, queue_id, item)
        if args.execute:
            registry.transition(queue_id, "running")
            runs_dir = root / "artifacts" / "runs"
            before_runs = {path for path in runs_dir.iterdir() if path.is_dir()} if runs_dir.is_dir() else set()
            result = subprocess.run(command_for(item), cwd=root, check=False)
            after_runs = {path for path in runs_dir.iterdir() if path.is_dir()} if runs_dir.is_dir() else set()
            new_runs = sorted(after_runs - before_runs, key=lambda path: path.stat().st_mtime)
            run_dir = str(new_runs[-1].relative_to(root)) if new_runs else ""
            status = "completed" if result.returncode == 0 else "invalid"
            registry.transition(
                queue_id,
                status,
                run_dir=run_dir,
                reason="" if result.returncode == 0 else f"exit={result.returncode}",
            )
            queue[index]["status"] = status
            if run_dir:
                queue[index]["run_dir"] = run_dir
            queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")
            elapsed_days = (time.monotonic() - started) / 86400
            if elapsed_days > float(plan["wall_clock_stop_days"]):
                raise RuntimeError("Pre-registered seven-day wall-clock stop condition reached")
    print(json.dumps({"queue": str(queue_path), "jobs": len(queue), "executed": args.execute}, indent=2))


if __name__ == "__main__":
    main()
