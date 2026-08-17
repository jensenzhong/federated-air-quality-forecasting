"""PAFA strategy: propose candidates, verify locally, aggregate safely, assign credit."""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp.strategy import FedAvg

from aqfl.agents.action_library import build_action_library
from aqfl.agents.memory import EpisodicMemory
from aqfl.agents.v2_contracts import (
    ActionProposal,
    ClientStateCapsule,
    CreditRecord,
    ExecutionDecision,
    ProbeOutcome,
)
from aqfl.agents.v2_proposers import ActionProposer
from aqfl.federated.aggregation import normalize, project_bounded_simplex
from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.strict import DeterministicSchedulingMixin


def _initial_capsule(client_id: str) -> ClientStateCapsule:
    return ClientStateCapsule(
        client_id=client_id,
        round_number=0,
        val_mae=0.0,
        val_rmse=0.0,
        high_pollution_mae=0.0,
        train_loss=0.0,
        update_norm=0.0,
        update_cosine=1.0,
        mae_ema=0.0,
        mae_slope=0.0,
        mae_oscillation=0.0,
        drift_score=0.0,
        previous_action_id="none",
        previous_realized_gain=0.0,
        train_seconds=0.0,
        local_epochs=0,
    )


def _execution_from_json(raw: str) -> ExecutionDecision:
    data = json.loads(raw)
    selected = data["selected_action"]
    from aqfl.agents.v2_contracts import ClientAction

    return ExecutionDecision(
        client_id=str(data["client_id"]),
        selected_action=ClientAction.from_dict(selected),
        accepted=bool(data["accepted"]),
        reason=str(data["reason"]),
        conservative_gain=float(data["conservative_gain"]),
        probe_outcomes=tuple(ProbeOutcome.from_dict(item) for item in data["probe_outcomes"]),
    )


def _update_cosines(
    global_arrays: ArrayRecord,
    client_arrays: list[ArrayRecord],
) -> list[float]:
    updates = []
    for record in client_arrays:
        pieces = []
        for key, global_value in global_arrays.items():
            pieces.append((record[key].numpy() - global_value.numpy()).reshape(-1))
        updates.append(np.concatenate(pieces).astype(np.float64, copy=False))
    mean_update = np.mean(np.stack(updates), axis=0)
    mean_norm = float(np.linalg.norm(mean_update))
    cosines = []
    for update in updates:
        denominator = float(np.linalg.norm(update)) * mean_norm
        cosines.append(float(np.dot(update, mean_update) / denominator) if denominator > 0 else 1.0)
    return cosines


def _agentic_weights(
    counts: np.ndarray,
    maes: np.ndarray,
    cosines: np.ndarray,
    gates: Sequence[str],
    lower: float,
    upper: float,
) -> np.ndarray:
    base = normalize(counts)
    median_mae = float(np.median(maes))
    multipliers = np.ones(len(counts), dtype=np.float64)
    for index, gate in enumerate(gates):
        if gate == "downweight_conflict":
            multipliers[index] = max(0.25, (float(cosines[index]) + 1.0) / 2.0)
        elif gate == "protect_tail":
            multipliers[index] = 1.0 + max(float(maes[index]) / max(median_mae, 1e-8) - 1.0, 0.0)
        elif gate != "normal":
            raise ValueError(f"Unsupported aggregation gate: {gate}")
    return project_bounded_simplex(normalize(base * multipliers), lower, upper)


class AgenticAggregationStrategy(DeterministicSchedulingMixin, FedAvg):
    def __init__(
        self,
        *,
        proposer: ActionProposer,
        expected_clients: int,
        base_lr: float,
        event_log: Path,
        probe_enabled: bool = True,
        memory_records_per_client: int = 20,
        credit_extra_epoch_penalty: float = 0.0,
        lower_weight: float = 0.04,
        upper_weight: float = 0.16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.proposer = proposer
        self.expected_clients = expected_clients
        self.base_lr = base_lr
        self.event_log = event_log
        self.probe_enabled = probe_enabled
        self.lower_weight = lower_weight
        self.upper_weight = upper_weight
        if credit_extra_epoch_penalty < 0:
            raise ValueError("credit_extra_epoch_penalty must be non-negative")
        self.memory = EpisodicMemory(memory_records_per_client)
        self.credit_extra_epoch_penalty = float(credit_extra_epoch_penalty)
        self.latest_capsules: dict[str, ClientStateCapsule] = {}
        self.current_proposals: dict[str, ActionProposal] = {}
        self.latest_arrays: ArrayRecord | None = None
        self.best_arrays: ArrayRecord | None = None
        self.best_macro_mae = float("inf")
        self.best_round: int | None = None
        self.latest_train_metrics: dict[str, float] = {}
        self._round_global_arrays: ArrayRecord | None = None
        raise RuntimeError(
            "Rejected privacy architecture: server-side per-client PAFA proposals and "
            "individual reply inspection are permanently disabled"
        )

    def _log(self, event: dict[str, Any]) -> None:
        with self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Any,
    ) -> Iterable[Message]:
        node_ids = sorted(int(node_id) for node_id in grid.get_node_ids())
        if len(node_ids) != self.expected_clients:
            raise RuntimeError(
                f"PAFA round expected {self.expected_clients} nodes, found {len(node_ids)}"
            )
        capsules = [
            self.latest_capsules.get(str(node_id), _initial_capsule(str(node_id)))
            for node_id in node_ids
        ]
        proposals = self.proposer.propose(server_round, capsules, self.memory)
        if set(proposals) != {str(node_id) for node_id in node_ids}:
            raise RuntimeError("PAFA proposer did not return exactly one proposal per client")
        self.current_proposals = proposals
        self._round_global_arrays = arrays.copy()
        messages = []
        for node_id in node_ids:
            client_id = str(node_id)
            client_config = ConfigRecord(dict(config))
            client_config["server-round"] = server_round
            client_config["base-lr"] = self.base_lr
            client_config["probe-enabled"] = self.probe_enabled
            client_config["agentic-proposal"] = json.dumps(
                proposals[client_id].to_dict(), ensure_ascii=False, separators=(",", ":")
            )
            record = RecordDict(
                {self.arrayrecord_key: arrays, self.configrecord_key: client_config}
            )
            messages.extend(self._construct_messages(record, [node_id], MessageType.TRAIN))
            self._log(
                {
                    "event": "proposal",
                    "round": server_round,
                    **proposals[client_id].to_dict(),
                }
            )
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        replies_list = list(replies)
        valid, errors = self._check_and_log_replies(replies_list, is_train=True)
        if errors or len(valid) != self.expected_clients:
            raise RuntimeError(
                f"PAFA formal round invalid: {len(valid)}/{self.expected_clients} clients succeeded"
            )
        if self._round_global_arrays is None:
            raise RuntimeError("PAFA is missing the pre-round global arrays")
        ordered = sorted(
            valid,
            key=lambda message: int(
                next(iter(message.content.metric_records.values()))["partition-id"]
            ),
        )
        contents = [message.content for message in ordered]
        metric_records = [next(iter(content.metric_records.values())) for content in contents]
        array_records = [next(iter(content.array_records.values())) for content in contents]
        client_ids = [str(record["agentic-client-id"]) for record in metric_records]
        if len(set(client_ids)) != self.expected_clients or set(client_ids) != set(
            self.current_proposals
        ):
            raise RuntimeError("PAFA client identifiers are incomplete or duplicated")
        cosines = np.asarray(
            _update_cosines(self._round_global_arrays, array_records), dtype=np.float64
        )
        executions = []
        gates = []
        for client_id, record in zip(client_ids, metric_records, strict=True):
            raw = str(record["agentic-execution"])
            if not raw:
                raise RuntimeError("PAFA client omitted its execution transcript")
            execution = _execution_from_json(raw)
            if execution.client_id != client_id:
                raise RuntimeError("PAFA execution/client identifier mismatch")
            proposed_ids = {
                action.action_id for action in self.current_proposals[client_id].candidates
            }
            if (
                execution.accepted
                and execution.selected_action.action_id not in proposed_ids
            ):
                raise RuntimeError("PAFA client executed an action that was not proposed")
            if (
                not execution.accepted
                and execution.selected_action.action_id
                != build_action_library()["safe_default"].action_id
            ):
                raise RuntimeError("Rejected PAFA proposals must use the safe fallback")
            executions.append(execution)
            gates.append(execution.selected_action.aggregation_gate)
            self.memory.add_execution(server_round, execution)
        counts = np.asarray([float(record["num-examples"]) for record in metric_records])
        maes = np.asarray([float(record["val_mae"]) for record in metric_records])
        weights = _agentic_weights(
            counts,
            maes,
            cosines,
            gates,
            self.lower_weight,
            self.upper_weight,
        )
        aggregated: OrderedDict[str, np.ndarray] = OrderedDict()
        for array_record, weight in zip(array_records, weights, strict=True):
            for key, value in array_record.items():
                weighted = value.numpy() * weight
                aggregated[key] = weighted if key not in aggregated else aggregated[key] + weighted
        arrays = ArrayRecord(
            OrderedDict((key, Array(np.asarray(value))) for key, value in aggregated.items())
        )
        self.latest_arrays = arrays
        for index, (client_id, record, execution) in enumerate(
            zip(client_ids, metric_records, executions, strict=True)
        ):
            previous = self.memory.capsules(client_id)
            if previous:
                credit = CreditRecord(
                    client_id=client_id,
                    round_number=server_round,
                    action_id=execution.selected_action.action_id,
                    predicted_gain=execution.conservative_gain,
                    realized_gain=(
                        previous[-1].val_mae
                        - float(record["val_mae"])
                        - self.credit_extra_epoch_penalty
                        * (execution.selected_action.local_epochs - 1)
                    ),
                    accepted=execution.accepted,
                )
                self.memory.add_credit(credit)
                self.proposer.observe(credit)
                self._log({"event": "credit", **credit.to_dict()})
            capsule = self.memory.build_capsule(
                client_id=client_id,
                round_number=server_round,
                val_mae=float(record["val_mae"]),
                val_rmse=float(record["val_rmse"]),
                high_pollution_mae=float(record["high_pollution_mae"]),
                train_loss=float(record["train_loss"]),
                update_norm=float(record["update_norm"]),
                update_cosine=float(cosines[index]),
                train_seconds=float(record["train_seconds"]),
                local_epochs=int(record["local-epochs-used"]),
            )
            self.memory.add_capsule(capsule)
            self.latest_capsules[client_id] = capsule
            self._log(
                {
                    "event": "execution",
                    "round": server_round,
                    "capsule": capsule.to_dict(),
                    "aggregation_weight": float(weights[index]),
                    **execution.to_dict(),
                }
            )
        metrics = aggregate_training_metrics(contents, "num-examples")
        metrics["min_aggregation_weight"] = float(weights.min())
        metrics["max_aggregation_weight"] = float(weights.max())
        metrics["probe_batches"] = int(
            sum(int(record["probe-batches"]) for record in metric_records)
        )
        metrics["accepted_interventions"] = int(sum(item.accepted for item in executions))
        self.latest_train_metrics = {
            key: float(value) for key, value in metrics.items() if isinstance(value, int | float)
        }
        return arrays, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> MetricRecord | None:
        replies_list = list(replies)
        valid, errors = self._check_and_log_replies(replies_list, is_train=False)
        if errors or len(valid) != self.expected_clients:
            raise RuntimeError(
                f"PAFA evaluation invalid: {len(valid)}/{self.expected_clients} clients succeeded"
            )
        metrics = aggregate_evaluation_metrics(
            [message.content for message in valid], "num-examples"
        )
        macro = float(metrics["macro_mae"])
        if self.latest_arrays is not None and macro < self.best_macro_mae:
            self.best_macro_mae = macro
            self.best_arrays = self.latest_arrays
            self.best_round = server_round
        return metrics
