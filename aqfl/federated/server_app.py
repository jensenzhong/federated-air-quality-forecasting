"""Flower ServerApp for fixed and multi-agent federated strategies."""

from __future__ import annotations

import json
from typing import Any

from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp

from aqfl.agents import (
    BudgetReplayPlanner,
    LLMPlanningAgent,
    RulePlanningAgent,
)
from aqfl.config import load_config, resolve_project_path, set_seed
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.preprocessing import GlobalScalerState
from aqfl.evaluation.metrics import aggregate_station_metrics
from aqfl.federated.dynamic_strategy import DynamicAggregationStrategy
from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.resources import (
    enforce_resource_gate,
    enforce_sequential_resource_gate,
    resource_snapshot,
)
from aqfl.federated.secure_aggregation import run_secure_pafa
from aqfl.federated.strict import StrictFedAdam, StrictFedAvg, StrictFedProx, StrictQFedAvg
from aqfl.models import build_model
from aqfl.models.training import evaluate_model
from aqfl.privacy import enforce_pafa_run_mode
from aqfl.reporting.artifacts import RunArtifacts

app = ServerApp()

DEFAULT_QFFL_LR = 1.0


def resolve_qffl_lr(run_config: dict[str, Any], config: dict[str, Any]) -> float:
    """Return the frozen q-FFL Lipschitz eta. This is not the local AdamW lr."""
    raw = run_config.get("qffl-lr") if hasattr(run_config, "get") else None
    if raw is not None:
        return float(raw)
    federated = config.get("federated", {})
    if "qfedavg_qffl_lr" in federated:
        return float(federated["qfedavg_qffl_lr"])
    return DEFAULT_QFFL_LR


def _base_kwargs(expected_clients: int) -> dict[str, Any]:
    return {
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": expected_clients,
        "min_evaluate_nodes": expected_clients,
        "min_available_nodes": expected_clients,
        "train_metrics_aggr_fn": aggregate_training_metrics,
        "evaluate_metrics_aggr_fn": aggregate_evaluation_metrics,
    }


def _write_grid_telemetry(grid: Grid, path: Any) -> None:
    rows = getattr(grid, "telemetry", [])
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


@app.main()
def main(grid: Grid, context: Context) -> None:
    config = load_config(str(context.run_config.get("config-path", "configs/base.yaml")))
    method = str(context.run_config["method"])
    seed = int(context.run_config["seed"])
    config["project"]["seed"] = seed
    config["runtime"] = {
        "method": method,
        "num_rounds": int(context.run_config["num-server-rounds"]),
        "local_epochs": int(context.run_config["local-epochs"]),
        "learning_rate": float(context.run_config["lr"]),
        "proximal_mu": float(context.run_config["proximal-mu"]),
        "q": float(context.run_config["q"]),
        "qffl_lr": resolve_qffl_lr(context.run_config, config),
        "server_lr": float(context.run_config["server-lr"]),
        "budget_trace": str(context.run_config.get("budget-trace", "")),
        "evaluation_split": str(context.run_config.get("evaluation-split", "val")),
        "protocol_frozen": bool(context.run_config.get("protocol-frozen", False)),
        "execution_mode": str(context.run_config.get("execution-mode", "full_concurrency")),
        "probe_enabled": method != "pafa_llm_no_probe",
    }
    if bool(context.run_config.get("continual-enabled", False)):
        config.setdefault("continual", {}).update(
            {
                "enabled": True,
                "task_count": int(context.run_config.get("continual-task-count", 11)),
                "base_rounds": int(context.run_config.get("continual-base-rounds", 1)),
                "rounds_per_task": int(
                    context.run_config.get("continual-rounds-per-task", 1)
                ),
            }
        )
    set_seed(seed)
    expected_clients = len(list_stations(config))
    if expected_clients != int(config["federated"]["num_clients"]):
        raise RuntimeError("Prepared station count does not match federated.num_clients")
    execution_mode = str(context.run_config.get("execution-mode", "full_concurrency"))
    if bool(context.run_config.get("enforce-resource-check", True)):
        if execution_mode == "low_memory_sequential":
            enforce_sequential_resource_gate(config)
        else:
            enforce_resource_gate(config)

    artifacts = RunArtifacts(config, method, seed)
    with (artifacts.path / "system_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({"event": "server_start", "execution_mode": execution_mode, **resource_snapshot()})
            + "\n"
        )
    model = build_model(config)
    arrays = ArrayRecord(model.state_dict())
    kwargs = _base_kwargs(expected_clients)
    base_lr = float(context.run_config["lr"])
    evaluation_split = str(context.run_config.get("evaluation-split", "val"))
    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation-split: {evaluation_split}")
    if evaluation_split == "test" and not bool(context.run_config.get("protocol-frozen", False)):
        raise RuntimeError("Test evaluation requires protocol-frozen=true")
    enforce_pafa_run_mode(
        method,
        evaluation_split=evaluation_split,
        protocol_frozen=bool(context.run_config.get("protocol-frozen", False)),
        secure_aggregation_active=method.startswith("pafa_"),
        client_state_isolated=bool(context.run_config.get("client-state-isolated", False)),
    )

    if method.startswith("pafa_"):
        event_log = artifacts.path / "agentic_events.jsonl"
        event_log.touch()
        try:
            secure_result = run_secure_pafa(
                grid=grid,
                context=context,
                initial_arrays=arrays,
                config=config,
                method=method,
                num_rounds=int(context.run_config["num-server-rounds"]),
                base_lr=base_lr,
                batch_size=int(context.run_config["batch-size"]),
                strict_llm=bool(context.run_config["strict-llm"]),
                probe_enabled=method != "pafa_llm_no_probe",
                event_log=event_log,
            )
            model.load_state_dict(secure_result.best_arrays.to_torch_state_dict())
            summary = {
                "protocol": "flower_secaggplus_client_local_agents",
                "execution_mode": execution_mode,
                "num_clients": expected_clients,
                "num_rounds": int(context.run_config["num-server-rounds"]),
                "checkpoint_selection": "secure_cohort_validation_macro_mae",
                "evaluation_split": evaluation_split,
                "protocol_frozen": bool(context.run_config.get("protocol-frozen", False)),
                "best_round": secure_result.best_round,
                "best_validation_macro_mae": secure_result.best_macro_mae,
                "agentic_v2": True,
                "agent_location": "client_context_state_only",
                "secure_aggregation": "flower_secaggplus",
                "coordinator_visibility": "cohort_summary_only",
                "client_metrics_persisted_on_server": False,
                "differential_privacy": False,
                "test_metrics": "TBD",
                "final_validation": secure_result.round_metrics[-1]
                if secure_result.round_metrics
                else {},
                "continual_metrics": secure_result.continual_metrics or "TBD",
            }
            _write_grid_telemetry(grid, artifacts.path / "system_metrics.jsonl")
            artifacts.finalize(
                model,
                summary,
                round_metrics=secure_result.round_metrics,
                client_metrics=[],
                status="completed",
            )
        except Exception:
            _write_grid_telemetry(grid, artifacts.path / "system_metrics.jsonl")
            artifacts.invalidate("PAFA_PRIVACY_GATE_FAILED")
            raise RuntimeError(
                "PAFA secure cohort failed closed; inspect institution-local client logs"
            ) from None
        return

    if method == "fedavg":
        strategy = StrictFedAvg(expected_clients=expected_clients, **kwargs)
    elif method == "fedprox":
        strategy = StrictFedProx(
            expected_clients=expected_clients,
            proximal_mu=float(context.run_config["proximal-mu"]),
            **kwargs,
        )
    elif method == "qfedavg":
        strategy = StrictQFedAvg(
            expected_clients=expected_clients,
            client_learning_rate=float(config["runtime"]["qffl_lr"]),
            q=float(context.run_config["q"]),
            **kwargs,
        )
    elif method == "fedadam":
        strategy = StrictFedAdam(
            expected_clients=expected_clients,
            eta=float(context.run_config["server-lr"]),
            eta_l=base_lr,
            **kwargs,
        )
    elif method in {
        "rule_mas",
        "mas_llm",
        "mas_llm_dynamic_only",
        "mas_llm_no_fairness",
        "fedprox_budget_matched",
    }:
        bounds = config["federated"]["weight_bounds"]
        if method == "rule_mas":
            planner: Any = RulePlanningAgent()
        elif method in {"mas_llm", "mas_llm_dynamic_only", "mas_llm_no_fairness"}:
            cache_dir = resolve_project_path(config, config["llm"]["cache_dir"])
            planner = LLMPlanningAgent(
                config,
                cache_dir,
                bool(context.run_config["strict-llm"]),
                fixed_budget=method == "mas_llm_dynamic_only",
                include_fairness=method != "mas_llm_no_fairness",
            )
        else:
            trace_value = str(context.run_config.get("budget-trace", ""))
            if not trace_value:
                raise ValueError("fedprox_budget_matched requires --run-config budget-trace='<decisions.jsonl>'")
            planner = BudgetReplayPlanner(resolve_project_path(config, trace_value))
        strategy = DynamicAggregationStrategy(
            planner=planner,
            expected_clients=expected_clients,
            base_lr=base_lr,
            decision_log=artifacts.path / "decisions.jsonl",
            lower_weight=float(bounds[0]),
            upper_weight=float(bounds[1]),
            proximal_mu=float(context.run_config["proximal-mu"]),
            **kwargs,
        )
    else:
        artifacts.invalidate(f"Unsupported federated method: {method}")
        raise ValueError(f"Unsupported federated method: {method}")

    try:
        result = strategy.start(
            grid=grid,
            initial_arrays=arrays,
            train_config=ConfigRecord({
                "lr": base_lr,
                "local-epochs": int(context.run_config["local-epochs"]),
            }),
            evaluate_config=ConfigRecord({"split": evaluation_split}),
            num_rounds=int(context.run_config["num-server-rounds"]),
        )
        selected_arrays = strategy.best_arrays if strategy.best_arrays is not None else result.arrays
        final_state = selected_arrays.to_torch_state_dict()
        model.load_state_dict(final_state)
        round_rows = []
        for round_number, metric_record in result.evaluate_metrics_clientapp.items():
            round_rows.append({"round": round_number, **dict(metric_record)})
        client_rows = []
        evaluation_summary: dict[str, Any] = {"test_metrics": "TBD"}
        if evaluation_split == "test":
            scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
            station_metrics = {}
            for station in list_stations(config):
                dataset = load_station_dataset(config, station, "test")
                metrics, prediction = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
                station_metrics[station] = metrics
                client_rows.append({"station": station, "split": "test", **metrics})
                artifacts.save_predictions(
                    station,
                    "test",
                    dataset.target_timestamp_ns,
                    dataset.y_raw.reshape(-1),
                    prediction,
                )
            evaluation_summary = aggregate_station_metrics(station_metrics)
        summary = {
            "protocol": "flower_serverapp_clientapp",
            "execution_mode": execution_mode,
            "client_schedule": "deterministic_sequential" if execution_mode == "low_memory_sequential" else "flower_transport",
            "num_clients": expected_clients,
            "num_rounds": int(context.run_config["num-server-rounds"]),
            "checkpoint_selection": "validation_macro_mae",
            "evaluation_split": evaluation_split,
            "protocol_frozen": bool(context.run_config.get("protocol-frozen", False)),
            "best_round": strategy.best_round,
            "best_validation_macro_mae": strategy.best_macro_mae,
            "agentic_v2": method.startswith("pafa_"),
            "probe_enabled": method.startswith("pafa_") and method != "pafa_llm_no_probe",
            "final_validation": round_rows[-1] if round_rows else {},
            **evaluation_summary,
        }
        _write_grid_telemetry(grid, artifacts.path / "system_metrics.jsonl")
        artifacts.finalize(model, summary, round_metrics=round_rows, client_metrics=client_rows)
    except Exception as exc:
        _write_grid_telemetry(grid, artifacts.path / "system_metrics.jsonl")
        artifacts.invalidate(str(exc))
        raise
