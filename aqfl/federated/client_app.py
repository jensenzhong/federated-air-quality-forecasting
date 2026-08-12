"""Flower ClientApp: one SuperNode is permanently bound to one monitoring station."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from aqfl.config import load_config, set_seed
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset
from aqfl.data.preprocessing import GlobalScalerState
from aqfl.federated.resources import limit_client_threads
from aqfl.models import build_model
from aqfl.models.training import evaluate_model, train_local_model

limit_client_threads(1)
app = ClientApp()


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


@app.train()
def train(msg: Message, context: Context) -> Message:
    config = _config(context)
    station = _station(context, config)
    seed = int(context.run_config.get("seed", config["project"]["seed"]))
    set_seed(seed + int(context.node_config["partition-id"]))
    model = build_model(config)
    incoming = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(incoming)
    before = copy.deepcopy(model.state_dict())
    train_dataset = load_station_dataset(config, station, "train")
    val_dataset = load_station_dataset(config, station, "val")
    proximal_mu = float(msg.content["config"].get("proximal-mu", context.run_config.get("proximal-mu", 0.0)))
    train_loss = train_local_model(
        model,
        train_dataset,
        int(msg.content["config"].get("local-epochs", context.run_config["local-epochs"])),
        float(msg.content["config"].get("lr", context.run_config["lr"])),
        int(context.run_config["batch-size"]),
        float(config["training"]["weight_decay"]),
        proximal_mu=proximal_mu,
        global_state=incoming,
    )
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    val_metrics, _ = evaluate_model(model, val_dataset, scaler, scaler.pollution_p90)
    metrics = MetricRecord({
        "train_loss": train_loss,
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "update_norm": _update_norm(before, model.state_dict()),
        "num-examples": len(train_dataset),
        "partition-id": int(context.node_config["partition-id"]),
    })
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    config = _config(context)
    station = _station(context, config)
    model = build_model(config)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    split = str(msg.content["config"].get("split", "val"))
    if split not in {"val", "test"}:
        raise ValueError(f"Unsupported evaluation split: {split}")
    dataset = load_station_dataset(config, station, split)
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    metrics, _ = evaluate_model(model, dataset, scaler, scaler.pollution_p90)
    record = MetricRecord({
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "smape": metrics["smape"],
        "r2": metrics["r2"],
        "negative-prediction-rate": metrics["negative_prediction_rate"],
        "num-examples": len(dataset),
        "partition-id": int(context.node_config["partition-id"]),
    })
    return Message(content=RecordDict({"metrics": record}), reply_to=msg)
