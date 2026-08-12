"""Create and optionally execute the pre-registered staged experiment queue."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

from aqfl.config import project_root
from aqfl.reporting.registry import ExperimentRegistry

FEDERATED_METHODS = {
    "fedavg", "fedprox", "qfedavg", "fedadam", "rule_mas", "mas_llm",
    "mas_llm_dynamic_only", "mas_llm_no_fairness", "fedprox_budget_matched",
}
DETERMINISTIC_METHODS = {"persistence", "seasonal_naive"}


def load_plan(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_queue(plan: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in plan["stage_order"]:
        raise ValueError(f"Unknown experiment stage: {stage}")
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
        rounds = "3" if item["stage"] == "smoke" else "30"
        return ["flwr", "run", ".", "local-12", "--run-config", f"method={method} seed={seed} num-server-rounds={rounds}"]
    return [
        "python",
        "-m",
        "aqfl.experiments.run_baseline",
        "--method",
        method,
        "--seed",
        seed,
        "--split",
        "val",
    ]


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
        registry.transition(queue_id, "planned", method=item["method"], seed=item["seed"])
        if args.execute:
            registry.transition(queue_id, "running")
            result = subprocess.run(command_for(item), cwd=root, check=False)
            registry.transition(queue_id, "completed" if result.returncode == 0 else "invalid", reason="" if result.returncode == 0 else f"exit={result.returncode}")
            elapsed_days = (time.monotonic() - started) / 86400
            if elapsed_days > float(plan["wall_clock_stop_days"]):
                raise RuntimeError("Pre-registered seven-day wall-clock stop condition reached")
    print(json.dumps({"queue": str(queue_path), "jobs": len(queue), "executed": args.execute}, indent=2))


if __name__ == "__main__":
    main()
