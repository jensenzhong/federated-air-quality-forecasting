"""Run the real Flower ServerApp/ClientApp stack in one process, sequentially."""

from __future__ import annotations

import argparse
import json
from typing import Any

from flwr.app import Context, RecordDict

from aqfl.config import load_config
from aqfl.data.dataset import list_stations
from aqfl.federated.client_app import app as client_app
from aqfl.federated.resources import enforce_sequential_resource_gate
from aqfl.federated.secure_aggregation import validate_secagg_numeric_policy
from aqfl.federated.sequential import SequentialGrid
from aqfl.federated.server_app import main as server_main
from aqfl.privacy import enforce_client_level_llm_policy, enforce_pafa_run_mode

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


def validate_preflight(
    config: dict[str, Any],
    run_config: dict[str, Any],
    stations: list[str],
) -> dict[str, Any]:
    """Validate a sequential run without exposing station identities or training."""
    expected_clients = int(config["federated"]["num_clients"])
    if len(stations) != expected_clients:
        raise RuntimeError(f"Expected {expected_clients} stations, found {len(stations)}")
    if len(set(stations)) != len(stations):
        raise RuntimeError("Sequential preflight requires unique private station bindings")

    method = str(run_config["method"])
    is_pafa = method.startswith("pafa_")
    quantization_step: float | None = None
    if is_pafa:
        enforce_pafa_run_mode(
            method,
            evaluation_split=str(run_config["evaluation-split"]),
            protocol_frozen=bool(run_config["protocol-frozen"]),
            secure_aggregation_active=True,
            client_state_isolated=False,
        )
        minimum_cohort = int(config["privacy"]["coordinator_min_cohort_size"])
        if expected_clients < minimum_cohort:
            raise RuntimeError("PAFA cohort is smaller than coordinator_min_cohort_size")
        quantization_step = validate_secagg_numeric_policy(
            config["privacy"]["secaggplus"], expected_clients
        )
        if method in {"pafa_llm", "pafa_llm_no_probe"}:
            enforce_client_level_llm_policy(config)

    if bool(run_config["enforce-resource-check"]):
        enforce_sequential_resource_gate(config)

    return {
        "status": "passed_nonformal_preflight" if is_pafa else "passed_preflight",
        "method": method,
        "station_count": expected_clients,
        "evaluation_split": str(run_config["evaluation-split"]),
        "execution_mode": "low_memory_sequential",
        "secure_aggregation": "flower_secaggplus" if is_pafa else "method_specific",
        "client_state_isolated": False,
        "formal_eligible": False,
        "resource_gate_enforced": bool(run_config["enforce-resource-check"]),
        "quantization_step": quantization_step,
        "training_started": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config_path)
    stations = list_stations(config)
    run_config = build_run_config(args)
    report = validate_preflight(config, run_config, stations)
    expected_clients = int(config["federated"]["num_clients"])
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
    if args.preflight_only:
        print(json.dumps(report, sort_keys=True))
        return report

    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )
    server_main(grid, context)
    return report


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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the run and exit before ServerApp or client training starts.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
