"""Synthetic equivalence checks for the low-memory Flower transport."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.serverapp import Grid

from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.sequential import SequentialGrid
from aqfl.federated.strict import StrictFedAdam, StrictFedAvg, StrictFedProx, StrictQFedAvg


class _ReferenceGrid(Grid):
    """Deterministic in-memory reference transport for equivalence checks."""

    def __init__(self, client_app: ClientApp, node_configs: dict[int, dict[str, Any]], run_config: dict[str, Any], trace: list[int]) -> None:
        self._client_app = client_app
        self._node_configs = node_configs
        self._run_config = run_config
        self._states = {node_id: RecordDict() for node_id in node_configs}

    def set_run(self, run_id: int) -> None:
        del run_id

    @property
    def run(self) -> Any:
        raise RuntimeError("ReferenceGrid does not expose a SuperLink Run")

    def create_message(self, content: RecordDict, message_type: str, dst_node_id: int, group_id: str, ttl: float | None = None) -> Message:
        return Message(content=content, message_type=message_type, dst_node_id=dst_node_id, group_id=group_id, ttl=ttl)

    def get_node_ids(self) -> Iterable[int]:
        return tuple(sorted(self._node_configs))

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        del messages
        return ()

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        del message_ids
        return ()

    def send_and_receive(self, messages: Iterable[Message], *, timeout: float | None = None) -> Iterable[Message]:
        del timeout
        replies = []
        for message in messages:
            node_id = int(message.metadata.dst_node_id)
            context = Context(
                run_id=1,
                node_id=node_id,
                node_config=self._node_configs[node_id],
                state=self._states[node_id],
                run_config=self._run_config,
            )
            replies.append(self._client_app(message, context))
        return replies


def _synthetic_client(trace: list[int]) -> ClientApp:
    app = ClientApp()

    @app.train()
    def train(message: Message, context: Context) -> Message:
        node_id = int(context.node_config["partition-id"])
        trace.append(node_id)
        state = message.content["arrays"].to_torch_state_dict()
        proximal_mu = float(message.content["config"].get("proximal-mu", 0.0))
        delta = float(node_id + 1) + proximal_mu
        updated = OrderedDict((key, value + delta) for key, value in state.items())
        metrics = MetricRecord(
            {
                "train_loss": float(node_id + 1),
                "val_mae": float(node_id + 2),
                "update_norm": abs(delta),
                "num-examples": node_id + 1,
                "partition-id": node_id,
            }
        )
        return Message(content=RecordDict({"arrays": ArrayRecord(updated), "metrics": metrics}), reply_to=message)

    @app.evaluate()
    def evaluate(message: Message, context: Context) -> Message:
        node_id = int(context.node_config["partition-id"])
        trace.append(node_id)
        metrics = MetricRecord(
            {
                "mae": float(node_id + 1),
                "rmse": float(node_id + 2),
                "smape": float(node_id + 1) / 10,
                "r2": 0.0,
                "num-examples": node_id + 1,
                "partition-id": node_id,
            }
        )
        return Message(content=RecordDict({"metrics": metrics}), reply_to=message)

    return app


def _strategy(method: str, expected_clients: int) -> Any:
    kwargs: dict[str, Any] = {
        "expected_clients": expected_clients,
        "fraction_train": 1.0,
        "fraction_evaluate": 1.0,
        "min_train_nodes": expected_clients,
        "min_evaluate_nodes": expected_clients,
        "min_available_nodes": expected_clients,
        "train_metrics_aggr_fn": aggregate_training_metrics,
        "evaluate_metrics_aggr_fn": aggregate_evaluation_metrics,
    }
    if method == "fedavg":
        return StrictFedAvg(**kwargs)
    if method == "fedprox":
        return StrictFedProx(proximal_mu=0.1, **kwargs)
    if method == "qfedavg":
        return StrictQFedAvg(client_learning_rate=0.001, q=1.0, **kwargs)
    if method == "fedadam":
        return StrictFedAdam(eta=0.1, eta_l=0.001, **kwargs)
    raise ValueError(f"Unsupported equivalence method: {method}")


def _run(method: str, expected_clients: int, transport: str) -> tuple[Any, list[int]]:
    trace: list[int] = []
    app = _synthetic_client(trace)
    node_configs = {
        node_id: {"partition-id": node_id, "num-partitions": expected_clients}
        for node_id in range(expected_clients)
    }
    run_config = {"local-epochs": 1, "batch-size": 1, "lr": 0.001, "proximal-mu": 0.1}
    if transport == "sequential":
        grid: Grid = SequentialGrid(
            app,
            run_id=1,
            node_configs=node_configs,
            run_config=run_config,
        )
    else:
        grid = _ReferenceGrid(app, node_configs, run_config, trace)
    result = _strategy(method, expected_clients).start(
        grid=grid,
        initial_arrays=ArrayRecord(OrderedDict({"weight": torch.tensor([0.0])})),
        num_rounds=1,
        train_config=ConfigRecord({"lr": 0.001, "local-epochs": 1}),
        evaluate_config=ConfigRecord({"split": "val"}),
    )
    return result, trace


def run_equivalence_suite(report_path: Path | None = None) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for expected_clients in (2, 12):
        for method in ("fedavg", "fedprox"):
            reference, reference_trace = _run(method, expected_clients, "reference")
            sequential, sequential_trace = _run(method, expected_clients, "sequential")
            reference_arrays = reference.arrays.to_numpy_ndarrays()
            sequential_arrays = sequential.arrays.to_numpy_ndarrays()
            max_array_diff = max(
                float(np.max(np.abs(left - right)))
                for left, right in zip(reference_arrays, sequential_arrays, strict=True)
            )
            reference_metrics = dict(reference.evaluate_metrics_clientapp[1])
            sequential_metrics = dict(sequential.evaluate_metrics_clientapp[1])
            metric_diffs = {
                key: abs(float(reference_metrics[key]) - float(sequential_metrics[key]))
                for key in reference_metrics
                if isinstance(reference_metrics[key], int | float)
            }
            if max_array_diff > 1e-6 or max(metric_diffs.values(), default=0.0) > 1e-6:
                raise AssertionError(
                    f"{method}/{expected_clients} equivalence failed: "
                    f"array_diff={max_array_diff}, metric_diffs={metric_diffs}"
                )
            expected_trace = list(range(expected_clients)) * 2
            if sequential_trace != expected_trace or reference_trace != expected_trace:
                raise AssertionError(
                    f"{method}/{expected_clients} partition order mismatch: "
                    f"reference={reference_trace}, sequential={sequential_trace}"
                )
            cases.append(
                {
                    "method": method,
                    "num_clients": expected_clients,
                    "max_array_abs_diff": max_array_diff,
                    "max_metric_abs_diff": max(metric_diffs.values(), default=0.0),
                    "partition_order": "complete_unique_deterministic",
                    "artifact_fields": ["arrays", "metrics", "partition-id", "num-examples"],
                }
            )
    report = {
        "status": "passed",
        "tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "cases": cases,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
