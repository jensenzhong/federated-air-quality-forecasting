"""Privacy-preserving PAFA transport and aggregate-only coordination."""

from __future__ import annotations

import copy
import json
import math
import timeit
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.client.mod import secaggplus_mod
from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.constant import MessageType
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    Key as SecAggKey,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    Stage as SecAggStage,
)
from flwr.server import ServerConfig, SimpleClientManager
from flwr.server.compat.grid_client_proxy import GridClientProxy
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.strategy import FedAvg
from flwr.server.workflow import SecAggPlusWorkflow
from flwr.server.workflow.constant import (
    MAIN_CONFIGS_RECORD,
    MAIN_PARAMS_RECORD,
)
from flwr.server.workflow.constant import (
    Key as WorkflowKey,
)
from flwr.serverapp import Grid

from aqfl.agents.v2_contracts import (
    ActionProposal,
    CohortDirective,
    DirectivePhase,
    DirectivePriority,
    ExecutionDecision,
)
from aqfl.data.continual_schedule import continual_task_id_for_round
from aqfl.evaluation.continual import (
    CONTINUAL_METRIC_SCALE,
    decode_task_matrix_sum,
    secure_aggregate_continual_metrics,
)

COHORT_SUMMARY_ARRAY = "__pafa_cohort_summary__"
COHORT_CONTINUAL_TASK_MATRIX_ARRAY = "__pafa_continual_task_matrix__"
SECAGG_COLLECT_GUARD = "__pafa_secagg_collect_guard__"
SECAGG_SESSION_GUARD = "__pafa_secagg_session_guard__"
ACTION_IDS = ("safe_default", "cautious", "adapt_fast", "tail_focus")
DIAGNOSES = ("stable", "underfit", "overfit", "drift", "conflict", "tail_risk")
SOURCES = ("rule", "bandit", "llm", "cache", "fallback")
GATES = ("normal", "downweight_conflict", "protect_tail")


class StrictCohortGrid(Grid):
    """Enforce one identity-bound reply per expected client at every SecAgg+ stage."""

    def __init__(self, delegate: Grid, expected_node_ids: set[int]) -> None:
        self.delegate = delegate
        self.expected_node_ids = frozenset(expected_node_ids)

    def set_run(self, run_id: int) -> None:
        self.delegate.set_run(run_id)

    @property
    def run(self) -> Any:
        return self.delegate.run

    def create_message(
        self,
        content: RecordDict,
        message_type: str,
        dst_node_id: int,
        group_id: str,
        ttl: float | None = None,
    ) -> Message:
        return self.delegate.create_message(
            content,
            message_type,
            dst_node_id,
            group_id,
            ttl,
        )

    def get_node_ids(self) -> Iterable[int]:
        return self.delegate.get_node_ids()

    def push_messages(self, messages: Iterable[Message]) -> Iterable[str]:
        return self.delegate.push_messages(messages)

    def pull_messages(self, message_ids: Iterable[str]) -> Iterable[Message]:
        return self.delegate.pull_messages(message_ids)

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        instructions = list(messages)
        destinations = [int(message.metadata.dst_node_id) for message in instructions]
        if (
            len(destinations) != len(self.expected_node_ids)
            or len(set(destinations)) != len(destinations)
            or set(destinations) != self.expected_node_ids
        ):
            raise RuntimeError(
                "PAFA SecAgg+ stage must address the complete unique cohort"
            )
        replies = list(self.delegate.send_and_receive(instructions, timeout=timeout))
        sources = [int(reply.metadata.src_node_id) for reply in replies]
        if (
            len(replies) != len(instructions)
            or len(set(sources)) != len(sources)
            or set(sources) != self.expected_node_ids
        ):
            raise RuntimeError(
                "PAFA SecAgg+ stage requires one reply from every cohort member"
            )
        return replies


class CohortSummaryCodec:
    """Encode private client metrics into one fixed, SecAgg+-summed vector."""

    continuous_size = 16
    size = continuous_size + len(ACTION_IDS) + len(DIAGNOSES) + len(SOURCES) + len(GATES)

    @classmethod
    def encode(
        cls,
        *,
        train_loss: float,
        val_mae: float,
        val_rmse: float,
        high_pollution_mae: float,
        update_norm: float,
        train_seconds: float,
        local_epochs: int,
        probe_batches: int,
        max_probe_batches: int,
        contribution_scale: float,
        clipping_violation: bool,
        proposal: ActionProposal,
        execution: ExecutionDecision,
        directive: CohortDirective | None = None,
    ) -> np.ndarray:
        values = np.zeros(cls.size, dtype=np.float32)
        probe_gains = np.asarray(
            [outcome.estimated_gain for outcome in execution.probe_outcomes],
            dtype=np.float64,
        )
        probe_mean = float(np.mean(probe_gains)) if probe_gains.size else 0.0
        probe_std = float(np.std(probe_gains)) if probe_gains.size else 0.0
        active_directive = directive or CohortDirective(
            phase="initial",
            priority="accuracy",
            lr_scale_cap=1.5,
            allow_adapt_fast=True,
            allow_tail_focus=True,
            directive_round=0,
        )
        directive_compliant = (
            execution.selected_action.lr_scale <= active_directive.lr_scale_cap
            and (active_directive.allow_adapt_fast or execution.selected_action.action_id != "adapt_fast")
            and (active_directive.allow_tail_focus or execution.selected_action.action_id != "tail_focus")
        )
        priority_aligned = {
            "accuracy": execution.selected_action.aggregation_gate == "normal",
            "tail_recovery": execution.selected_action.aggregation_gate == "protect_tail",
            "conflict_recovery": execution.selected_action.aggregation_gate == "downweight_conflict",
            "efficiency": execution.selected_action.action_id == "safe_default",
        }[active_directive.priority]
        values[: cls.continuous_size] = np.asarray(
            [
                np.clip(train_loss / 5.0, 0.0, 1.0),
                np.clip(val_mae / 100.0, 0.0, 1.0),
                np.clip(val_rmse / 150.0, 0.0, 1.0),
                np.clip(high_pollution_mae / 150.0, 0.0, 1.0),
                np.clip(np.log1p(update_norm) / 10.0, 0.0, 1.0),
                np.clip(train_seconds / 600.0, 0.0, 1.0),
                np.clip(local_epochs / 2.0, 0.0, 1.0),
                np.clip(probe_batches / max(max_probe_batches, 1), 0.0, 1.0),
                float(execution.accepted),
                np.clip(contribution_scale / 1.25, 0.0, 1.0),
                float(clipping_violation),
                np.clip((probe_mean + 5.0) / 10.0, 0.0, 1.0),
                np.clip(probe_std / 5.0, 0.0, 1.0),
                float(directive_compliant),
                float(proposal.diagnosis == "tail_risk"),
                float(priority_aligned),
            ],
            dtype=np.float32,
        )
        offset = cls.continuous_size
        values[offset + ACTION_IDS.index(execution.selected_action.action_id)] = 1.0
        offset += len(ACTION_IDS)
        values[offset + DIAGNOSES.index(proposal.diagnosis)] = 1.0
        offset += len(DIAGNOSES)
        values[offset + SOURCES.index(proposal.source)] = 1.0
        offset += len(SOURCES)
        values[offset + GATES.index(execution.selected_action.aggregation_gate)] = 1.0
        return values

    @classmethod
    def decode(cls, vector: np.ndarray) -> dict[str, float]:
        array = np.asarray(vector, dtype=np.float64).reshape(-1)
        if array.size != cls.size or not np.isfinite(array).all():
            raise RuntimeError("Invalid SecAgg+ cohort summary vector")
        tolerance = 1e-3
        if np.any(array < -tolerance) or np.any(array > 1.0 + tolerance):
            raise RuntimeError("SecAgg+ cohort summary escaped its normalized bounds")
        group_sizes = (len(ACTION_IDS), len(DIAGNOSES), len(SOURCES), len(GATES))
        offset = cls.continuous_size
        for group_size in group_sizes:
            if not math.isclose(
                float(array[offset : offset + group_size].sum()),
                1.0,
                abs_tol=tolerance,
            ):
                raise RuntimeError("SecAgg+ cohort categorical rates do not sum to one")
            offset += group_size
        result = {
            "cohort_train_loss": float(array[0] * 5.0),
            "cohort_val_macro_mae": float(array[1] * 100.0),
            "cohort_val_macro_rmse": float(array[2] * 150.0),
            "cohort_high_pollution_mae": float(array[3] * 150.0),
            "cohort_update_norm": float(np.expm1(array[4] * 10.0)),
            "cohort_train_seconds": float(array[5] * 600.0),
            "cohort_local_epochs": float(array[6] * 2.0),
            "cohort_probe_fraction": float(array[7]),
            "cohort_action_acceptance_rate": float(array[8]),
            "cohort_contribution_scale": float(array[9] * 1.25),
            "cohort_clipping_violation_rate": float(array[10]),
            "cohort_probe_gain_mean": float(array[11] * 10.0 - 5.0),
            "cohort_probe_gain_std": float(array[12] * 5.0),
            "cohort_directive_compliance_rate": float(array[13]),
            "cohort_tail_risk_rate": float(array[14]),
            "cohort_priority_alignment_rate": float(array[15]),
        }
        offset = cls.continuous_size
        for name in ACTION_IDS:
            result[f"action_rate_{name}"] = float(array[offset])
            offset += 1
        for name in DIAGNOSES:
            result[f"diagnosis_rate_{name}"] = float(array[offset])
            offset += 1
        for name in SOURCES:
            result[f"source_rate_{name}"] = float(array[offset])
            offset += 1
        for name in GATES:
            result[f"gate_rate_{name}"] = float(array[offset])
            offset += 1
        return result


def sanitize_secagg_collect_reply(content: RecordDict) -> None:
    """Remove every client-level application field after the official mod masks arrays."""
    if "fitres.parameters" not in content.array_records:
        raise RuntimeError("SecAgg+ collect reply omitted the masked FitRes placeholder")
    for key in list(content.array_records):
        if key != "fitres.parameters":
            del content.array_records[key]
    content.metric_records.clear()
    content.metric_records["fitres.num_examples"] = MetricRecord({"num_examples": 1})
    for key in list(content.config_records):
        if key not in {RECORD_KEY_CONFIGS, "fitres.metrics", "fitres.status"}:
            del content.config_records[key]
    content.config_records["fitres.metrics"] = ConfigRecord({})
    content.config_records["fitres.status"] = ConfigRecord(
        {"code": Code.OK.value, "message": ""}
    )


def _claim_pafa_secagg_stage(msg: Message, context: Context, stage: str) -> None:
    """Bind every PAFA SecAgg+ stage to one client, run, and monotonic round."""
    method = str(context.run_config.get("method", ""))
    if not method.startswith("pafa_"):
        return
    if msg.metadata.dst_node_id != context.node_id:
        raise RuntimeError("PAFA SecAgg+ destination does not match the local client")
    if msg.metadata.run_id != context.run_id:
        raise RuntimeError("PAFA SecAgg+ message belongs to a different Flower run")
    try:
        round_number = int(msg.metadata.group_id)
    except ValueError as exc:
        raise RuntimeError("PAFA SecAgg+ group_id must be the numeric server round") from exc
    if round_number < 1:
        raise RuntimeError("PAFA SecAgg+ server round must be positive")

    prior = context.state.config_records.get(SECAGG_SESSION_GUARD)
    last_completed = int(prior.get("last_completed_round", 0)) if prior else 0
    active_round = int(prior.get("active_round", 0)) if prior else 0
    last_stage = str(prior.get("last_stage", "")) if prior else ""
    if stage == SecAggStage.SETUP:
        if active_round or round_number != last_completed + 1:
            raise RuntimeError("PAFA SecAgg+ setup replay or round skip rejected")
    else:
        expected_stage = {
            SecAggStage.SETUP: SecAggStage.SHARE_KEYS,
            SecAggStage.SHARE_KEYS: SecAggStage.COLLECT_MASKED_VECTORS,
            SecAggStage.COLLECT_MASKED_VECTORS: SecAggStage.UNMASK,
        }.get(last_stage)
        if active_round != round_number or stage != expected_stage:
            raise RuntimeError("PAFA SecAgg+ stage replay or reordering rejected")
    if stage == SecAggStage.COLLECT_MASKED_VECTORS:
        fit_config = msg.content.config_records.get("fitins.config")
        if fit_config is None:
            raise RuntimeError("PAFA SecAgg+ collect stage omitted fit configuration")
        if str(fit_config.get("method", "")) != method:
            raise RuntimeError("PAFA SecAgg+ method binding mismatch")
        if int(fit_config.get("server-round", 0)) != round_number:
            raise RuntimeError("PAFA SecAgg+ collect round binding mismatch")
    context.state.config_records[SECAGG_SESSION_GUARD] = ConfigRecord(
        {
            "last_completed_round": last_completed,
            "active_round": round_number,
            "last_stage": stage,
        }
    )


def _complete_pafa_secagg_stage(context: Context, stage: str) -> None:
    if stage != SecAggStage.UNMASK:
        return
    guard = context.state.config_records.get(SECAGG_SESSION_GUARD)
    if guard is None:
        return
    round_number = int(guard["active_round"])
    context.state.config_records[SECAGG_SESSION_GUARD] = ConfigRecord(
        {
            "last_completed_round": round_number,
            "active_round": 0,
            "last_stage": SecAggStage.UNMASK,
        }
    )


def validate_secagg_numeric_policy(privacy: dict[str, Any], cohort_size: int) -> float:
    """Validate PAFA's equal-weight quantization capacity and return its step size."""
    if cohort_size < 2:
        raise RuntimeError("SecAgg+ PAFA cohort must contain at least two clients")
    clipping_range = float(privacy["clipping_range"])
    quantization_value = privacy["quantization_range"]
    modulus_value = privacy["modulus_range"]
    if type(quantization_value) is not int or type(modulus_value) is not int:
        raise RuntimeError("SecAgg+ quantization and modulus ranges must be integers")
    quantization_range = quantization_value
    modulus_range = modulus_value
    max_weight = float(privacy["max_weight"])
    if not math.isfinite(clipping_range) or clipping_range <= 0:
        raise RuntimeError("SecAgg+ clipping_range must be finite and positive")
    if quantization_range <= 0 or quantization_range > np.iinfo(np.int32).max:
        raise RuntimeError("SecAgg+ quantization_range exceeds the int32-safe range")
    if modulus_range <= 0 or modulus_range & (modulus_range - 1):
        raise RuntimeError("SecAgg+ modulus_range must be a positive power of two")
    if modulus_range <= cohort_size * quantization_range:
        raise RuntimeError("SecAgg+ modulus capacity is insufficient for this cohort")
    if max_weight != 1.0:
        raise RuntimeError("PAFA requires max_weight=1 for station-equal aggregation")
    return 2.0 * clipping_range / quantization_range


def private_secaggplus_mod(
    msg: Message,
    context: Context,
    call_next: Callable[[Message, Context], Message],
) -> Message:
    """Apply Flower SecAgg+ only to workflow messages and sanitize collect replies."""
    context.state.config_records.pop(SECAGG_COLLECT_GUARD, None)
    if (
        msg.metadata.message_type != MessageType.TRAIN
        or RECORD_KEY_CONFIGS not in msg.content.config_records
    ):
        fit_config = msg.content.config_records.get("fitins.config")
        method = str(fit_config.get("method", "")) if fit_config is not None else ""
        if method.startswith("pafa_"):
            raise RuntimeError(
                "PAFA client request omitted the Flower SecAgg+ protocol record"
            )
        return call_next(msg, context)
    stage = str(msg.content.config_records[RECORD_KEY_CONFIGS][SecAggKey.STAGE])
    _claim_pafa_secagg_stage(msg, context, stage)
    if stage == SecAggStage.COLLECT_MASKED_VECTORS:
        context.state.config_records[SECAGG_COLLECT_GUARD] = ConfigRecord(
            {"active": True}
        )
    try:
        reply = secaggplus_mod(msg, context, call_next)
    finally:
        context.state.config_records.pop(SECAGG_COLLECT_GUARD, None)
    if stage == SecAggStage.COLLECT_MASKED_VECTORS:
        sanitize_secagg_collect_reply(reply.content)
    _complete_pafa_secagg_stage(context, stage)
    return reply


def is_verified_secagg_collect(context: Context) -> bool:
    """Return whether execution is inside Flower's authenticated collect stage."""
    record = context.state.config_records.get(SECAGG_COLLECT_GUARD)
    return record is not None and record.get("active") is True


class AggregateCoordinatorAgent:
    """Generate bounded cohort-wide signals from SecAgg+-only summaries."""

    def __init__(self, minimum_cohort_size: int) -> None:
        if minimum_cohort_size < 2:
            raise ValueError("Aggregate coordinator cohort must contain at least two clients")
        self.minimum_cohort_size = minimum_cohort_size
        self.history: list[dict[str, Any]] = []
        self.signal: dict[str, Any] = {
            "cohort-phase": "initial",
            "cohort-lr-scale-cap": 1.0,
            "cohort-round": 0,
        }
        self.directive = CohortDirective(
            phase="initial",
            priority="accuracy",
            lr_scale_cap=1.0,
            allow_adapt_fast=True,
            allow_tail_focus=True,
            directive_round=0,
        )

    def observe(
        self,
        round_number: int,
        cohort_size: int,
        summary: dict[str, float],
    ) -> dict[str, Any]:
        if cohort_size < self.minimum_cohort_size:
            raise RuntimeError(
                f"Aggregate coordinator requires cohort >= {self.minimum_cohort_size}, "
                f"received {cohort_size}"
            )
        current = float(summary["cohort_val_macro_mae"])
        previous = (
            float(self.history[-1]["summary"]["cohort_val_macro_mae"])
            if self.history
            else current
        )
        relative_change = (current - previous) / max(previous, 1e-8)
        drift_rate = float(summary.get("diagnosis_rate_drift", 0.0))
        conflict_rate = float(summary.get("diagnosis_rate_conflict", 0.0))
        probe_gain = float(summary.get("cohort_probe_gain_mean", 0.0))
        compliance = float(summary.get("cohort_directive_compliance_rate", 1.0))
        if (
            relative_change > 0.01
            or drift_rate + conflict_rate > 0.35
            or probe_gain < -0.25
            or compliance < 0.8
        ):
            phase, cap = "volatile", 0.5
            priority = "conflict_recovery" if conflict_rate >= drift_rate else "tail_recovery"
            allow_fast, allow_tail = False, True
        elif abs(relative_change) <= 0.002 and round_number > 1:
            phase, cap = "stagnating", 1.0
            priority = "efficiency"
            allow_fast, allow_tail = False, True
        else:
            phase, cap = "improving", 1.5
            priority = "accuracy"
            allow_fast, allow_tail = True, False
        self.directive = CohortDirective(
            phase=cast(DirectivePhase, phase),
            priority=cast(DirectivePriority, priority),
            lr_scale_cap=cap,
            allow_adapt_fast=allow_fast,
            allow_tail_focus=allow_tail,
            directive_round=round_number,
        )
        self.signal = {
            "cohort-phase": phase,
            "cohort-lr-scale-cap": cap,
            "cohort-round": round_number,
        }
        event = {
            "event": "cohort_summary",
            "round": round_number,
            "cohort_size": cohort_size,
            "summary": summary,
            "signal": dict(self.signal),
            "directive": self.directive.to_dict(),
        }
        self.history.append(event)
        return dict(self.signal)


class SecureAggregateOnlyFedAvg(FedAvg):
    """Legacy strategy adapter receiving only the already-unmasked cohort vector."""

    def __init__(
        self,
        *,
        expected_clients: int,
        expected_node_ids: set[int],
        model_array_count: int,
        coordinator: AggregateCoordinatorAgent,
        on_fit_config_fn: Callable[[int], dict[str, Any]],
        continual_task_count: int | None = None,
        continual_base_rounds: int = 1,
        continual_rounds_per_task: int = 1,
        server_optimizer: str = "fedavg",
        server_learning_rate: float = 1.0,
        server_beta1: float = 0.9,
        server_beta2: float = 0.99,
        server_tau: float = 1e-3,
    ) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=expected_clients,
            min_evaluate_clients=0,
            min_available_clients=expected_clients,
            accept_failures=False,
            on_fit_config_fn=on_fit_config_fn,
        )
        self.expected_clients = expected_clients
        self.expected_node_ids = frozenset(expected_node_ids)
        self.model_array_count = model_array_count
        self.coordinator = coordinator
        self.continual_task_count = continual_task_count
        self.continual_base_rounds = continual_base_rounds
        self.continual_rounds_per_task = continual_rounds_per_task
        if server_optimizer not in {"fedavg", "fedadam"}:
            raise ValueError("Unsupported secure server optimizer")
        if server_learning_rate <= 0 or not 0 < server_beta1 < 1 or not 0 < server_beta2 < 1 or server_tau <= 0:
            raise ValueError("Invalid secure server optimizer hyperparameters")
        self.server_optimizer = server_optimizer
        self.server_learning_rate = float(server_learning_rate)
        self.server_beta1 = float(server_beta1)
        self.server_beta2 = float(server_beta2)
        self.server_tau = float(server_tau)
        self._server_round = 0
        self._server_model: list[np.ndarray] | None = None
        self._server_first_moment: list[np.ndarray] | None = None
        self._server_second_moment: list[np.ndarray] | None = None
        self.latest_summary: dict[str, float] | None = None
        self.latest_continual_metrics: dict[str, float] | None = None
        self.last_server_round = 0

    def aggregate_fit(self, server_round: int, results: list[Any], failures: list[Any]) -> Any:
        if server_round != self.last_server_round + 1:
            raise RuntimeError("SecAgg+ PAFA aggregate round replay or skip rejected")
        if failures or len(results) != self.expected_clients:
            raise RuntimeError(
                f"SecAgg+ PAFA round requires {self.expected_clients} clients and no failures"
            )
        result_node_ids = [int(proxy.node_id) for proxy, _ in results]
        if (
            len(set(result_node_ids)) != len(result_node_ids)
            or set(result_node_ids) != self.expected_node_ids
        ):
            raise RuntimeError("SecAgg+ PAFA client identities must be unique and complete")
        if any(result.num_examples != 1 or result.metrics for _, result in results):
            raise RuntimeError("Client-level metadata survived the SecAgg+ privacy sanitizer")
        arrays = parameters_to_ndarrays(results[0][1].parameters)
        expected_array_count = self.model_array_count + 1 + (
            1 if self.continual_task_count is not None else 0
        )
        if len(arrays) != expected_array_count:
            raise RuntimeError("SecAgg+ result has an unexpected fixed array layout")
        summary_index = self.model_array_count
        self.latest_summary = CohortSummaryCodec.decode(arrays[summary_index])
        self.latest_continual_metrics = None
        if self.continual_task_count is not None:
            task_id = continual_task_id_for_round(
                server_round,
                base_rounds=self.continual_base_rounds,
                rounds_per_task=self.continual_rounds_per_task,
                task_count=self.continual_task_count,
            )
            if task_id == self.continual_task_count:
                summed = decode_task_matrix_sum(
                    np.asarray(arrays[-1], dtype=np.float64)
                    * float(self.expected_clients),
                    self.continual_task_count,
                ) * CONTINUAL_METRIC_SCALE
                self.latest_continual_metrics = secure_aggregate_continual_metrics(
                    summed,
                    self.expected_clients,
                    minimum_cohort_size=self.coordinator.minimum_cohort_size,
                ).to_dict()
            elif not np.allclose(arrays[-1], 0.0, atol=1e-6):
                raise RuntimeError(
                    "Continual task matrix must remain zero before the final task"
                )
        if (
            self.latest_summary["cohort_clipping_violation_rate"]
            > 0.5 / self.expected_clients
        ):
            raise RuntimeError("SecAgg+ parameter clipping detected in at least one client")
        model_arrays = [np.asarray(array, dtype=np.float64) for array in arrays[: self.model_array_count]]
        if self._server_model is None:
            self._server_model = [array.copy() for array in model_arrays]
            self._server_first_moment = [np.zeros_like(array) for array in model_arrays]
            self._server_second_moment = [np.zeros_like(array) for array in model_arrays]
        if self.server_optimizer == "fedadam":
            if self._server_first_moment is None or self._server_second_moment is None:
                raise RuntimeError("Secure FedAdam moments were not initialized")
            self._server_round += 1
            corrected: list[np.ndarray] = []
            for index, averaged in enumerate(model_arrays):
                reference = self._server_model[index]
                delta = averaged - reference
                first = (
                    self.server_beta1 * self._server_first_moment[index]
                    + (1.0 - self.server_beta1) * delta
                )
                second = (
                    self.server_beta2 * self._server_second_moment[index]
                    + (1.0 - self.server_beta2) * np.square(delta)
                )
                self._server_first_moment[index] = first
                self._server_second_moment[index] = second
                first_hat = first / (1.0 - self.server_beta1**self._server_round)
                second_hat = second / (1.0 - self.server_beta2**self._server_round)
                updated = reference + self.server_learning_rate * first_hat / (
                    np.sqrt(second_hat) + self.server_tau
                )
                corrected.append(updated.astype(np.float32))
            self._server_model = [array.astype(np.float64) for array in corrected]
            arrays[: self.model_array_count] = corrected
        else:
            self._server_model = model_arrays
        self.coordinator.observe(server_round, len(results), self.latest_summary)
        self.last_server_round = server_round
        metrics = {key: float(value) for key, value in self.latest_summary.items()}
        metrics["cohort_size"] = len(results)
        metrics = dict(metrics)
        if self.latest_continual_metrics is not None:
            metrics.update(
                {
                    f"continual_{key}": float(value)
                    for key, value in self.latest_continual_metrics.items()
                    if key != "task_count"
                }
            )
        return ndarrays_to_parameters(arrays[: self.model_array_count]), metrics

    def set_initial_model(self, arrays: ArrayRecord) -> None:
        """Set the public global model used by the optional server optimizer."""
        model = [np.asarray(array.numpy(), dtype=np.float64) for array in arrays.values()]
        if len(model) != self.model_array_count:
            raise ValueError("Initial model array count does not match the strategy")
        self._server_model = [array.copy() for array in model]
        self._server_first_moment = [np.zeros_like(array) for array in model]
        self._server_second_moment = [np.zeros_like(array) for array in model]


@dataclass
class SecurePafaResult:
    arrays: ArrayRecord
    best_arrays: ArrayRecord
    best_round: int
    best_macro_mae: float
    round_metrics: list[dict[str, Any]]
    coordinator_events: list[dict[str, Any]]
    continual_metrics: dict[str, float] | None


def run_secure_pafa(
    *,
    grid: Grid,
    context: Context,
    initial_arrays: ArrayRecord,
    config: dict[str, Any],
    method: str,
    num_rounds: int,
    base_lr: float,
    batch_size: int,
    server_learning_rate: float = 1.0,
    strict_llm: bool,
    probe_enabled: bool,
    event_log: Path,
) -> SecurePafaResult:
    node_ids = sorted(int(item) for item in grid.get_node_ids())
    expected_clients = int(config["federated"]["num_clients"])
    minimum_cohort = int(config["privacy"]["coordinator_min_cohort_size"])
    if len(node_ids) != expected_clients or expected_clients < minimum_cohort:
        raise RuntimeError(
            f"SecAgg+ PAFA requires {expected_clients} clients with cohort >= {minimum_cohort}"
        )
    coordinator = AggregateCoordinatorAgent(minimum_cohort)
    privacy = config["privacy"]["secaggplus"]
    validate_secagg_numeric_policy(privacy, expected_clients)
    strict_grid = StrictCohortGrid(grid, set(node_ids))
    continual = config.get("continual", {})
    continual_enabled = bool(continual.get("enabled", False))
    continual_task_count = int(continual.get("task_count", 11))
    continual_base_rounds = int(continual.get("base_rounds", 1))
    continual_rounds_per_task = int(continual.get("rounds_per_task", 1))
    if continual_enabled:
        expected_rounds = continual_base_rounds + (
            continual_task_count * continual_rounds_per_task
        )
        if num_rounds != expected_rounds:
            raise RuntimeError(
                "Continual PAFA requires exactly base_rounds + task_count * "
                "rounds_per_task communication rounds"
            )

    def fit_config(round_number: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "method": method,
            "base-lr": base_lr,
            "lr": base_lr,
            "local-epochs": 1,
            "batch-size": batch_size,
            "server-round": round_number,
            "strict-llm": strict_llm,
            "probe-enabled": probe_enabled,
            "cohort-directive": coordinator.directive.to_json(),
            **coordinator.signal,
        }
        if continual_enabled:
            task_id = continual_task_id_for_round(
                round_number,
                base_rounds=continual_base_rounds,
                rounds_per_task=continual_rounds_per_task,
                task_count=continual_task_count,
            )
            next_task_id = (
                continual_task_id_for_round(
                    round_number + 1,
                    base_rounds=continual_base_rounds,
                    rounds_per_task=continual_rounds_per_task,
                    task_count=continual_task_count,
                )
                if round_number < num_rounds
                else task_id
            )
            values.update(
                {
                    "continual-enabled": True,
                    "continual-task-id": task_id,
                    "continual-task-count": continual_task_count,
                    "continual-task-final": task_id > 0 and next_task_id != task_id,
                }
            )
        return values

    strategy = SecureAggregateOnlyFedAvg(
        expected_clients=expected_clients,
        expected_node_ids=set(node_ids),
        model_array_count=len(initial_arrays),
        coordinator=coordinator,
        on_fit_config_fn=fit_config,
        continual_task_count=continual_task_count if continual_enabled else None,
        continual_base_rounds=continual_base_rounds,
        continual_rounds_per_task=continual_rounds_per_task,
        server_optimizer="fedadam" if method == "pafa_fedadam" else "fedavg",
        server_learning_rate=server_learning_rate,
    )
    strategy.set_initial_model(initial_arrays)
    manager = SimpleClientManager()
    for node_id in node_ids:
        if not manager.register(GridClientProxy(node_id, strict_grid, context.run_id)):
            raise RuntimeError("Duplicate Flower node registration rejected")
    legacy = LegacyContext(
        context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_manager=manager,
    )
    legacy.state.array_records[MAIN_PARAMS_RECORD] = copy.deepcopy(initial_arrays)
    model_keys = list(initial_arrays.keys())
    started = timeit.default_timer()
    workflow = SecAggPlusWorkflow(
        num_shares=float(privacy["num_shares"]),
        reconstruction_threshold=minimum_cohort / expected_clients,
        max_weight=float(privacy["max_weight"]),
        clipping_range=float(privacy["clipping_range"]),
        quantization_range=int(privacy["quantization_range"]),
        modulus_range=int(privacy["modulus_range"]),
    )
    best_arrays = copy.deepcopy(initial_arrays)
    best_round = 0
    best_macro = math.inf
    rows: list[dict[str, Any]] = []
    for round_number in range(1, num_rounds + 1):
        legacy.state.config_records[MAIN_CONFIGS_RECORD] = ConfigRecord(
            {
                WorkflowKey.CURRENT_ROUND: round_number,
                WorkflowKey.START_TIME: started,
            }
        )
        workflow(strict_grid, legacy)
        if strategy.latest_summary is None or len(coordinator.history) != round_number:
            raise RuntimeError(f"SecAgg+ PAFA round {round_number} did not produce a cohort summary")
        workflow_arrays = legacy.state.array_records[MAIN_PARAMS_RECORD]
        if len(workflow_arrays) != len(model_keys):
            raise RuntimeError("SecAgg+ changed the number of model arrays")
        arrays = ArrayRecord(
            OrderedDict(
                (key, value)
                for key, value in zip(model_keys, workflow_arrays.values(), strict=True)
            )
        )
        legacy.state.array_records[MAIN_PARAMS_RECORD] = arrays
        macro = float(strategy.latest_summary["cohort_val_macro_mae"])
        row = {"round": round_number, **strategy.latest_summary, **coordinator.signal}
        rows.append(row)
        if macro < best_macro:
            best_macro = macro
            best_round = round_number
            best_arrays = copy.deepcopy(arrays)
    with event_log.open("a", encoding="utf-8") as handle:
        for event in coordinator.history:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    return SecurePafaResult(
        arrays=copy.deepcopy(legacy.state.array_records[MAIN_PARAMS_RECORD]),
        best_arrays=best_arrays,
        best_round=best_round,
        best_macro_mae=best_macro,
        round_metrics=rows,
        coordinator_events=list(coordinator.history),
        continual_metrics=strategy.latest_continual_metrics,
    )
