"""Flower ClientApp: one SuperNode is permanently bound to one monitoring station."""

from __future__ import annotations

import copy
import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Code, FitRes, Status
from flwr.common import recorddict_compat as compat

from aqfl.agents.action_library import build_action_library, resolve_action_ids
from aqfl.agents.executor import ProbeBudget, SafeActionExecutor
from aqfl.agents.local_runtime import (
    LOCAL_CLIENT_TOKEN,
    apply_cohort_lr_cap,
    load_private_agent_state,
    propose_local_actions,
    record_local_outcome,
    save_private_agent_state,
)
from aqfl.agents.v2_contracts import ActionProposal, CohortDirective, ProbeOutcome
from aqfl.agents.v2_proposers import ActionProposer, RuleActionProposer
from aqfl.config import load_config, set_seed
from aqfl.data.continual_dataset import load_station_continual_dataset
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.preprocessing import GlobalScalerState
from aqfl.evaluation.continual import LocalContinualTaskLedger
from aqfl.federated.probe_runtime import probe_candidates
from aqfl.federated.resources import limit_client_threads
from aqfl.federated.secure_aggregation import (
    COHORT_CONTINUAL_TASK_MATRIX_ARRAY,
    COHORT_SUMMARY_ARRAY,
    CohortSummaryCodec,
    is_verified_secagg_collect,
    private_secaggplus_mod,
)
from aqfl.models import build_model
from aqfl.models.training import evaluate_model, train_local_model

limit_client_threads(1)
app = ClientApp(mods=[private_secaggplus_mod])

LOCAL_CONTINUAL_LEDGER_ARRAY = "__pafa_local_continual_task_ledger__"


def _config(context: Context) -> dict[str, Any]:
    return load_config(str(context.run_config.get("config-path", "configs/base.yaml")))


def _station(context: Context, config: dict[str, Any]) -> str:
    stations = list_stations(config)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    if num_partitions != len(stations):
        raise ValueError(f"Expected {len(stations)} partitions, received {num_partitions}")
    if not 0 <= partition_id < len(stations):
        raise ValueError(f"Invalid partition-id: {partition_id}")
    station = stations[partition_id]
    configured_station = context.node_config.get("station")
    if configured_station is not None and str(configured_station) != station:
        raise ValueError(f"SuperNode station binding mismatch: {configured_station} != {station}")
    return station


def _update_norm(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in before:
        delta = after[key].detach().cpu() - before[key].detach().cpu()
        total += float(torch.sum(delta * delta))
    return float(np.sqrt(total))


def _update_sketch(
    before: dict[str, torch.Tensor],
    after: dict[str, torch.Tensor],
) -> tuple[float, ...]:
    sketch: list[float] = []
    for key in before:
        delta = after[key].detach().float().cpu() - before[key].detach().float().cpu()
        sketch.extend((float(delta.mean()), float(torch.sqrt(torch.mean(delta * delta)))))
    return tuple(sketch)


def _sketch_cosine(current: tuple[float, ...], previous: tuple[float, ...]) -> float:
    if not current or len(current) != len(previous):
        return 1.0
    left = np.asarray(current, dtype=np.float64)
    right = np.asarray(previous, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def _apply_private_contribution_scale(
    model: torch.nn.Module,
    global_state: dict[str, torch.Tensor],
    scale: float,
) -> None:
    scaled = {}
    for key, local_value in model.state_dict().items():
        global_value = global_state[key].to(local_value.device)
        scaled[key] = global_value + scale * (local_value - global_value)
    model.load_state_dict(scaled)


def _would_clip_parameters(model: torch.nn.Module, clipping_range: float) -> bool:
    maximum = 0.0
    for value in model.state_dict().values():
        tensor = value.detach().float()
        if not torch.isfinite(tensor).all():
            raise RuntimeError("PAFA client model contains non-finite parameters")
        if tensor.numel():
            maximum = max(maximum, float(tensor.abs().max()))
    return maximum > clipping_range


def _is_secure_pafa_request(msg: Message, context: Context) -> bool:
    return (
        "fitins.parameters" in msg.content.array_records
        and is_verified_secagg_collect(context)
    )


def _state_dict_from_array_record(
    model: torch.nn.Module,
    record: ArrayRecord,
) -> OrderedDict[str, torch.Tensor]:
    incoming = record.to_torch_state_dict()
    expected_keys = list(model.state_dict())
    if list(incoming) == expected_keys:
        return OrderedDict((key, incoming[key]) for key in expected_keys)
    numeric_keys = [str(index) for index in range(len(expected_keys))]
    if list(incoming) == numeric_keys:
        return OrderedDict(
            (key, value)
            for key, value in zip(expected_keys, incoming.values(), strict=True)
        )
    raise RuntimeError("Flower array record keys do not match the local model schema")


def _continual_task_id(
    config: dict[str, Any],
    train_config: Any,
) -> int | None:
    settings = config.get("continual", {})
    enabled = bool(train_config.get("continual-enabled", settings.get("enabled", False)))
    if not enabled:
        return None
    task_id = int(train_config.get("continual-task-id", -1))
    task_count = int(train_config.get("continual-task-count", settings.get("task_count", 11)))
    if task_id < 0 or task_id > task_count or task_count < 2 or task_count > 11:
        raise RuntimeError("Continual ClientApp request has an invalid task ID")
    return task_id


def _load_local_continual_ledger(
    context: Context,
    task_count: int,
) -> LocalContinualTaskLedger:
    record = context.state.array_records.get(LOCAL_CONTINUAL_LEDGER_ARRAY)
    if record is None:
        return LocalContinualTaskLedger(task_count)
    arrays = record.to_numpy_ndarrays()
    if len(arrays) != 1:
        raise RuntimeError("Local continual ledger state has an unexpected shape")
    return LocalContinualTaskLedger.from_private_matrix(arrays[0])


def _save_local_continual_ledger(
    context: Context,
    ledger: LocalContinualTaskLedger,
) -> None:
    context.state.array_records[LOCAL_CONTINUAL_LEDGER_ARRAY] = ArrayRecord(
        {"matrix": Array(ledger.private_matrix().astype(np.float32, copy=False))}
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = _config(context)
    station = _station(context, config)
    seed = int(context.run_config.get("seed", config["project"]["seed"]))
    set_seed(seed + int(context.node_config["partition-id"]))
    model = build_model(config)
    secure_pafa = _is_secure_pafa_request(msg, context)
    if secure_pafa:
        incoming_record = msg.content.array_records["fitins.parameters"]
        train_config = msg.content.config_records["fitins.config"]
    else:
        incoming_record = msg.content["arrays"]
        train_config = msg.content["config"]
    incoming = _state_dict_from_array_record(model, incoming_record)
    model.load_state_dict(incoming)
    before = copy.deepcopy(model.state_dict())
    continual_task_id = _continual_task_id(config, train_config)
    continual_settings = config.get("continual", {})
    continual_task_count = int(
        train_config.get("continual-task-count", continual_settings.get("task_count", 11))
    )
    continual_task_final = bool(train_config.get("continual-task-final", False))
    continual_ledger: LocalContinualTaskLedger | None = None
    if continual_task_id is not None and continual_task_id > 0:
        continual_ledger = _load_local_continual_ledger(context, continual_task_count)
    if continual_task_id is None:
        train_dataset = load_station_dataset(config, station, "train")
    else:
        train_ratio = float(config.get("continual", {}).get("train_ratio", 0.8))
        train_dataset = load_station_continual_dataset(
            config,
            station,
            continual_task_id,
            "train",
            train_ratio=train_ratio,
        )
    val_dataset = load_station_dataset(config, station, "val")
    started = time.perf_counter()
    base_lr = float(train_config.get("base-lr", train_config.get("lr", context.run_config["lr"])))
    local_epochs = int(train_config.get("local-epochs", context.run_config["local-epochs"]))
    learning_rate = float(train_config.get("lr", context.run_config["lr"]))
    proximal_mu = float(train_config.get("proximal-mu", context.run_config.get("proximal-mu", 0.0)))
    method = str(train_config.get("method", context.run_config.get("method", "")))
    if method.startswith("pafa_") and not secure_pafa:
        raise RuntimeError(
            "PAFA client training requires the SecAgg+ collect-masked-vectors workflow"
        )
    probe_batches_used = 0
    proposal = None
    execution = None
    private_state = None
    proposer: ActionProposer | None = None
    contribution_scale = 1.0
    if secure_pafa:
        agentic = config["agentic_v2"]
        round_number = int(train_config["server-round"])
        directive = CohortDirective.from_json(str(train_config["cohort-directive"]))
        directive.validate_for_round(round_number)
        private_state = load_private_agent_state(
            context, int(agentic["memory_records_per_client"])
        )
        static_baseline = method in {"pafa_fedavg", "pafa_fedprox", "pafa_fedadam"}
        if static_baseline:
            proposer = RuleActionProposer()
            proposal = ActionProposal(
                LOCAL_CLIENT_TOKEN,
                "stable",
                ("fixed static baseline action",),
                resolve_action_ids(("safe_default",)),
                "rule",
            )
        else:
            proposal, proposer = propose_local_actions(
                method,
                round_number,
                config,
                private_state,
                strict_llm=bool(train_config.get("strict-llm", True)),
                directive=directive,
            )
        probe_enabled = bool(train_config.get("probe-enabled", True)) and not static_baseline
        outcomes: tuple[ProbeOutcome, ...] = ()
        if probe_enabled:
            outcomes = probe_candidates(
                model,
                train_dataset,
                val_dataset,
                proposal.candidates,
                base_lr=base_lr,
                batch_size=int(context.run_config["batch-size"]),
                weight_decay=float(config["training"]["weight_decay"]),
                global_state=incoming,
                train_batches=int(agentic["probe_train_batches"]),
                val_batches=int(agentic["probe_val_batches"]),
                seed=seed + int(train_config["server-round"]) * 1000,
            )
        budget = ProbeBudget(
            max_candidates=int(agentic["max_candidates"]),
            batches_per_candidate=int(agentic["probe_train_batches"]),
        )
        executor = SafeActionExecutor(
            build_action_library()["safe_default"],
            minimum_gain=float(agentic["minimum_probe_gain"]),
            uncertainty_margin=float(agentic["probe_uncertainty_margin"]),
            extra_epoch_penalty=float(agentic["extra_epoch_penalty"]),
        )
        execution = executor.select(
            proposal,
            outcomes,
            budget,
            probe_enabled=probe_enabled,
        )
        if not static_baseline:
            execution = apply_cohort_lr_cap(execution, directive.lr_scale_cap)
        selected = execution.selected_action
        learning_rate = base_lr * selected.lr_scale
        local_epochs = selected.local_epochs
        proximal_mu = 0.0 if method in {"pafa_fedavg", "pafa_fedadam"} else selected.proximal_mu
        probe_batches_used = budget.consumed_batches
        contribution_scale = {
            "normal": 1.0,
            "downweight_conflict": 0.5,
            "protect_tail": 1.25,
        }[selected.aggregation_gate]
    train_loss = train_local_model(
        model,
        train_dataset,
        local_epochs,
        learning_rate,
        int(context.run_config["batch-size"]),
        float(config["training"]["weight_decay"]),
        proximal_mu=proximal_mu,
        global_state=incoming,
    )
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    val_metrics, _ = evaluate_model(model, val_dataset, scaler, scaler.pollution_p90)
    high_pollution_mae = float(val_metrics.get("high_pollution_mae", val_metrics["mae"]))
    if not np.isfinite(high_pollution_mae):
        high_pollution_mae = float(val_metrics["mae"])
    elapsed = time.perf_counter() - started
    update_norm = _update_norm(before, model.state_dict())
    continual_task_matrix_vector: np.ndarray | None = None
    if (
        continual_task_id is not None
        and continual_task_id > 0
        and continual_task_final
    ):
        if continual_ledger is None:
            raise RuntimeError("Continual ledger was not initialized for a task checkpoint")
        for evaluated_task_id in range(1, continual_task_id + 1):
            task_test_dataset = load_station_continual_dataset(
                config,
                station,
                evaluated_task_id,
                "test",
                train_ratio=float(continual_settings.get("train_ratio", 0.8)),
            )
            task_metrics, _ = evaluate_model(
                model,
                task_test_dataset,
                scaler,
                scaler.pollution_p90,
            )
            continual_ledger.record(
                continual_task_id,
                evaluated_task_id,
                float(task_metrics["mae"]),
            )
        _save_local_continual_ledger(context, continual_ledger)
        if continual_task_id == continual_task_count:
            continual_task_matrix_vector = continual_ledger.encode_for_secagg()
    if secure_pafa:
        if private_state is None or proposer is None or proposal is None or execution is None:
            raise RuntimeError("Client-private PAFA state was not initialized")
        current_sketch = _update_sketch(before, model.state_dict())
        update_cosine = _sketch_cosine(current_sketch, private_state.previous_update_sketch)
        record_local_outcome(
            private_state,
            proposer,
            execution,
            round_number=int(train_config["server-round"]),
            val_mae=float(val_metrics["mae"]),
        )
        capsule = private_state.memory.build_capsule(
            client_id=LOCAL_CLIENT_TOKEN,
            round_number=int(train_config["server-round"]),
            val_mae=float(val_metrics["mae"]),
            val_rmse=float(val_metrics["rmse"]),
            high_pollution_mae=high_pollution_mae,
            train_loss=float(train_loss),
            update_norm=update_norm,
            update_cosine=update_cosine,
            train_seconds=elapsed,
            local_epochs=local_epochs,
        )
        private_state.memory.add_capsule(capsule)
        private_state.previous_update_sketch = current_sketch
        save_private_agent_state(context, private_state)
        _apply_private_contribution_scale(model, incoming, contribution_scale)
        clipping_violation = _would_clip_parameters(
            model,
            float(config["privacy"]["secaggplus"]["clipping_range"]),
        )
        max_probe_batches = int(agentic["max_candidates"]) * int(
            agentic["probe_train_batches"]
        )
        summary_vector = CohortSummaryCodec.encode(
            train_loss=float(train_loss),
            val_mae=float(val_metrics["mae"]),
            val_rmse=float(val_metrics["rmse"]),
            high_pollution_mae=high_pollution_mae,
            update_norm=update_norm,
            train_seconds=elapsed,
            local_epochs=local_epochs,
            probe_batches=probe_batches_used,
            max_probe_batches=max_probe_batches,
            contribution_scale=contribution_scale,
            clipping_violation=clipping_violation,
            proposal=proposal,
            execution=execution,
            directive=directive,
        )
        arrays = ArrayRecord.from_torch_state_dict(model.state_dict())
        arrays[COHORT_SUMMARY_ARRAY] = Array(summary_vector)
        if continual_task_id is not None:
            if continual_task_matrix_vector is None:
                continual_task_matrix_vector = np.zeros(
                    continual_task_count * continual_task_count,
                    dtype=np.float32,
                )
            arrays[COHORT_CONTINUAL_TASK_MATRIX_ARRAY] = Array(
                continual_task_matrix_vector
            )
        fitres = FitRes(
            status=Status(Code.OK, ""),
            parameters=compat.arrayrecord_to_parameters(arrays, keep_input=True),
            num_examples=1,
            metrics={},
        )
        return Message(content=compat.fitres_to_recorddict(fitres, True), reply_to=msg)
    metrics = MetricRecord({
        "train_loss": train_loss,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "high_pollution_mae": high_pollution_mae,
        "update_norm": update_norm,
        "train_seconds": elapsed,
        "local-epochs-used": local_epochs,
        "proximal-mu-used": proximal_mu,
        "num-examples": len(train_dataset),
        "partition-id": int(context.node_config["partition-id"]),
    })
    content = RecordDict(
        {
            "arrays": ArrayRecord.from_torch_state_dict(model.state_dict()),
            "metrics": metrics,
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config = _config(context)
    station = _station(context, config)
    model = build_model(config)
    model.load_state_dict(_state_dict_from_array_record(model, msg.content["arrays"]))
    split = str(msg.content["config"].get("split", "val"))
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    dataset = load_station_dataset(config, station, split)
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    metrics, _ = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
    high_pollution_mae = float(metrics.get("high_pollution_mae", metrics["mae"]))
    if not np.isfinite(high_pollution_mae):
        high_pollution_mae = float(metrics["mae"])
    record = MetricRecord({
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "smape": metrics["smape"],
        "r2": metrics["r2"],
        "negative-prediction-rate": metrics["negative_prediction_rate"],
        "high-pollution-mae": high_pollution_mae,
        "num-examples": len(dataset),
        "partition-id": int(context.node_config["partition-id"]),
    })
    return Message(content=RecordDict({"metrics": record}), reply_to=msg)
