from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.strict import StrictFedAvg


class InMemoryTwoClientGrid:
    """Exercise one complete Flower strategy round without network or Ray."""

    def __init__(self) -> None:
        self.node_ids = [10, 11]

    def get_node_ids(self):
        return self.node_ids

    def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
        return Message(
            content=content,
            message_type=message_type,
            dst_node_id=dst_node_id,
            group_id=group_id,
            ttl=ttl,
        )

    def send_and_receive(self, messages, *, timeout=None):
        del timeout
        replies = []
        for message in messages:
            partition_id = self.node_ids.index(message.metadata.dst_node_id)
            if message.metadata.message_type == "train":
                state = message.content["arrays"].to_torch_state_dict()
                updated = OrderedDict((name, tensor + float(partition_id + 1)) for name, tensor in state.items())
                content = RecordDict(
                    {
                        "arrays": ArrayRecord(updated),
                        "metrics": MetricRecord(
                            {
                                "train_loss": float(2 - partition_id),
                                "val_mae": float(3 - partition_id),
                                "update_norm": float(partition_id + 1),
                                "num-examples": partition_id + 1,
                                "partition-id": partition_id,
                            }
                        ),
                    }
                )
            else:
                content = RecordDict(
                    {
                        "metrics": MetricRecord(
                            {
                                "mae": float(partition_id + 1),
                                "rmse": float(partition_id + 2),
                                "smape": float(partition_id + 1) / 10,
                                "r2": 0.0,
                                "num-examples": partition_id + 1,
                                "partition-id": partition_id,
                            }
                        )
                    }
                )
            replies.append(Message(content=content, reply_to=message))
        return replies


@pytest.mark.integration
def test_two_client_one_round_flower_protocol() -> None:
    strategy = StrictFedAvg(
        expected_clients=2,
        fraction_train=1,
        fraction_evaluate=1,
        min_train_nodes=2,
        min_evaluate_nodes=2,
        min_available_nodes=2,
        train_metrics_aggr_fn=aggregate_training_metrics,
        evaluate_metrics_aggr_fn=aggregate_evaluation_metrics,
    )
    initial = ArrayRecord(OrderedDict({"weight": torch.tensor([0.0])}))
    result = strategy.start(
        grid=InMemoryTwoClientGrid(),
        initial_arrays=initial,
        num_rounds=1,
        train_config=ConfigRecord({"lr": 0.001}),
        evaluate_config=ConfigRecord({"split": "val"}),
    )
    assert np.allclose(result.arrays["weight"].numpy(), [5 / 3])
    assert result.evaluate_metrics_clientapp[1]["macro_mae"] == 1.5
    assert strategy.best_round == 1
    assert strategy.best_arrays is not None


@pytest.mark.integration
def test_formal_round_rejects_missing_client() -> None:
    strategy = StrictFedAvg(expected_clients=2, min_train_nodes=2, min_available_nodes=2)
    message = Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(OrderedDict({"weight": torch.tensor([1.0])})),
                "metrics": MetricRecord({"num-examples": 1, "partition-id": 0}),
            }
        ),
        dst_node_id=1,
        message_type="train",
    )
    reply = Message(content=message.content, reply_to=message)
    with pytest.raises(RuntimeError, match="expected 2"):
        strategy.aggregate_train(1, [reply])
