"""Run the real Flower ServerApp/ClientApp stack in one process, sequentially."""

from __future__ import annotations

import argparse
from typing import Any

from flwr.app import Context, RecordDict

from aqfl.config import load_config
from aqfl.data.dataset import list_stations
from aqfl.federated.client_app import app as client_app
from aqfl.federated.sequential import SequentialGrid
from aqfl.federated.server_app import main as server_main

METHODS = (
    "fedavg",
    "fedprox",
    "qfedavg",
    "fedadam",
    "rule_mas",
    "mas_llm",
    "mas_llm_dynamic_only",
    "mas_llm_no_fairness",
    "fedprox_budget_matched",
    "pafa_rule",
    "pafa_bandit",
    "pafa_probe_oracle",
    "pafa_llm",
    "pafa_llm_no_probe",
)


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": args.method,
        "seed": args.seed,
        "num-server-rounds": args.rounds,
        "local-epochs": args.local_epochs,
        "lr": args.lr,
        "batch-size": args.batch_size,
        "proximal-mu": args.proximal_mu,
        "q": args.q,
        "qffl-lr": args.qffl_lr,
        "server-lr": args.server_lr,
        "config-path": args.config_path,
        "strict-llm": args.strict_llm,
        "enforce-resource-check": args.enforce_resource_check,
        "budget-trace": args.budget_trace,
        "evaluation-split": args.evaluation_split,
        "protocol-frozen": args.protocol_frozen,
        "client-state-isolated": False,
        "execution-mode": "low_memory_sequential",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="fedprox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--qffl-lr", type=float, default=1.0)
    parser.add_argument("--server-lr", type=float, default=0.1)
    parser.add_argument("--config-path", default="configs/base.yaml")
    parser.add_argument("--budget-trace", default="")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--protocol-frozen", action="store_true")
    parser.add_argument("--strict-llm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enforce-resource-check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    stations = list_stations(config)
    expected_clients = int(config["federated"]["num_clients"])
    if len(stations) != expected_clients:
        raise RuntimeError(f"Expected {expected_clients} stations, found {len(stations)}")
    run_config = build_run_config(args)
    node_configs = {
        index: {
            "partition-id": index,
            "num-partitions": expected_clients,
            "station": station,
        }
        for index, station in enumerate(stations)
    }
    grid = SequentialGrid(
        client_app,
        run_id=1,
        node_configs=node_configs,
        run_config=run_config,
    )
    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )
    server_main(grid, context)


if __name__ == "__main__":
    main()
