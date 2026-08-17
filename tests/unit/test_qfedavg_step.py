from __future__ import annotations

from collections import OrderedDict

import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.server_app import DEFAULT_QFFL_LR, resolve_qffl_lr
from aqfl.federated.strict import StrictQFedAvg


def _request() -> Message:
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(OrderedDict({"w": torch.tensor([0.0])})),
                "config": ConfigRecord(),
            }
        ),
        message_type="train",
        dst_node_id=0,
        group_id="1",
    )


def _reply(request: Message, value: float, loss: float, partition_id: int) -> Message:
    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(OrderedDict({"w": torch.tensor([value])})),
                "metrics": MetricRecord(
                    {
                        "train_loss": loss,
                        "val_mae": 1.0,
                        "update_norm": abs(value),
                        "num-examples": 10,
                        "partition-id": partition_id,
                    }
                ),
            }
        ),
        reply_to=request,
    )


def _aggregate(qffl_lr: float, q: float, values: list[float], losses: list[float]) -> float:
    strategy = StrictQFedAvg(
        expected_clients=len(values),
        client_learning_rate=qffl_lr,
        q=q,
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=len(values),
        min_evaluate_nodes=len(values),
        min_available_nodes=len(values),
        train_metrics_aggr_fn=aggregate_training_metrics,
        evaluate_metrics_aggr_fn=aggregate_evaluation_metrics,
    )
    strategy.current_arrays = ArrayRecord(OrderedDict({"w": torch.tensor([0.0])}))
    request = _request()
    replies = [
        _reply(request, value, loss, partition_id)
        for partition_id, (value, loss) in enumerate(zip(values, losses, strict=True))
    ]
    arrays, _ = strategy.aggregate_train(1, replies)
    assert arrays is not None
    return float(arrays.to_numpy_ndarrays()[0].reshape(-1)[0])


def test_q_zero_with_unit_lipschitz_recovers_uniform_mean() -> None:
    updated = _aggregate(qffl_lr=1.0, q=0.0, values=[2.0, 4.0], losses=[1.0, 1.0])
    assert updated == 3.0


def test_legacy_adamw_lr_as_lipschitz_crushes_the_step() -> None:
    values = [2.0, 4.0]
    losses = [0.4, 0.5]
    healthy = _aggregate(qffl_lr=1.0, q=0.1, values=values, losses=losses)
    crushed = _aggregate(qffl_lr=0.001, q=0.1, values=values, losses=losses)
    assert healthy > 0.5
    assert crushed < 0.05
    assert healthy / max(crushed, 1e-12) > 20


def test_resolve_qffl_lr_is_independent_of_adamw_lr() -> None:
    assert resolve_qffl_lr({"qffl-lr": 1.0, "lr": 0.001}, {}) == 1.0
    assert resolve_qffl_lr({"lr": 0.001}, {"federated": {"qfedavg_qffl_lr": 1.0}}) == 1.0
    assert resolve_qffl_lr({}, {}) == DEFAULT_QFFL_LR
