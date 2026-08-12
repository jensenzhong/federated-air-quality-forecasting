"""Dynamic aggregation controlled by a rule or LLM planning agent."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp.strategy import FedAvg

from aqfl.agents.decision import Decision
from aqfl.federated.aggregation import client_weights
from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics


class DynamicAggregationStrategy(FedAvg):
    def __init__(
        self,
        *,
        planner: Any,
        expected_clients: int,
        base_lr: float,
        decision_log: Path,
        lower_weight: float = 0.04,
        upper_weight: float = 0.16,
        proximal_mu: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.planner = planner
        self.expected_clients = expected_clients
        self.base_lr = base_lr
        self.decision_log = decision_log
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        self.proximal_mu = proximal_mu
        self.history: list[dict[str, float]] = []
        self.last_metrics: dict[str, float] = {}
        self.latest_train_metrics: dict[str, float] = {}
        self.current_decision: Decision | None = None
        self.latest_arrays: ArrayRecord | None = None
        self.best_arrays: ArrayRecord | None = None
        self.best_macro_mae = float("inf")
        self.best_round: int | None = None

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Any) -> Iterable[Message]:
        current = self.last_metrics or {"macro_mae": 0.0, "worst_station_mae": 0.0, "station_mae_cv": 0.0, "update_norm_cv": 0.0}
        decision = self.planner.choose(server_round, self.history, current)
        self.current_decision = decision
        config["lr"] = self.base_lr * decision.lr_scale
        config["local-epochs"] = decision.local_epochs
        config["aggregation-strategy"] = decision.strategy
        config["proximal-mu"] = self.proximal_mu
        config["server-round"] = server_round
        with self.decision_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"round": server_round, **decision.to_dict()}, ensure_ascii=False) + "\n")
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        valid, errors = self._check_and_log_replies(replies_list, is_train=True)
        if errors or len(valid) != self.expected_clients:
            raise RuntimeError(f"Dynamic formal round invalid: {len(valid)}/{self.expected_clients} clients succeeded")
        if self.current_decision is None:
            raise RuntimeError("Missing current planning decision")
        contents = [message.content for message in valid]
        metric_records = [next(iter(content.metric_records.values())) for content in contents]
        counts = np.asarray([float(record["num-examples"]) for record in metric_records])
        maes = np.asarray([float(record["val_mae"]) for record in metric_records])
        weights = client_weights(counts, maes, self.current_decision.strategy, self.lower_weight, self.upper_weight)
        aggregated: OrderedDict[str, np.ndarray] = OrderedDict()
        for content, weight in zip(contents, weights, strict=False):
            array_record = next(iter(content.array_records.values()))
            for key, value in array_record.items():
                weighted = value.numpy() * weight
                aggregated[key] = weighted if key not in aggregated else aggregated[key] + weighted
        arrays = ArrayRecord(OrderedDict((key, Array(np.asarray(value))) for key, value in aggregated.items()))
        self.latest_arrays = arrays
        metrics = aggregate_training_metrics(contents, "num-examples")
        self.latest_train_metrics = {
            key: float(value) for key, value in metrics.items() if isinstance(value, int | float)
        }
        metrics["min_aggregation_weight"] = float(weights.min())
        metrics["max_aggregation_weight"] = float(weights.max())
        return arrays, metrics

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        replies_list = list(replies)
        valid, errors = self._check_and_log_replies(replies_list, is_train=False)
        if errors or len(valid) != self.expected_clients:
            raise RuntimeError(f"Dynamic evaluation invalid: {len(valid)}/{self.expected_clients} clients succeeded")
        metrics = aggregate_evaluation_metrics([message.content for message in valid], "num-examples")
        self.last_metrics = {
            **self.latest_train_metrics,
            **{
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, int | float)
            },
        }
        self.history.append({"round": float(server_round), **self.last_metrics})
        macro = float(metrics["macro_mae"])
        if self.latest_arrays is not None and macro < self.best_macro_mae:
            self.best_macro_mae = macro
            self.best_arrays = self.latest_arrays
            self.best_round = server_round
        return metrics
