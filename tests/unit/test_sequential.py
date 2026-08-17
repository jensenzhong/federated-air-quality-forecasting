from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, Metadata, RecordDict
from flwr.clientapp import ClientApp

from aqfl.federated.equivalence import _run
from aqfl.federated.metrics import aggregate_evaluation_metrics, aggregate_training_metrics
from aqfl.federated.sequential import SequentialGrid
from aqfl.federated.strict import StrictFedAvg


def test_sequential_grid_dispatches_in_sorted_order() -> None:
    visited: list[int] = []
    app = ClientApp()

    @app.train()
    def train(message: Message, context: Context) -> Message:
        visited.append(context.node_id)
        return Message(content=message.content, reply_to=message)

    run_config = {"local-epochs": 1}
    grid = SequentialGrid(
        app,
        run_id=1,
        node_configs={2: {"partition-id": 2}, 0: {"partition-id": 0}, 1: {"partition-id": 1}},
        run_config=run_config,
    )
    content = RecordDict({"arrays": ArrayRecord(OrderedDict({"w": torch.tensor([1.0])})), "config": ConfigRecord()})
    messages = [
        grid.create_message(content, "train", node_id, "1")
        for node_id in (2, 0, 1)
    ]
    replies = list(grid.send_and_receive(messages))
    assert visited == [0, 1, 2]
    assert [reply.metadata.reply_to_message_id for reply in replies]


def test_strict_strategy_configures_fixed_order() -> None:
    class GridStub:
        def get_node_ids(self):
            return [2, 0, 1]

    strategy = StrictFedAvg(
        expected_clients=3,
        fraction_train=1.0,
        min_train_nodes=3,
        min_available_nodes=3,
        train_metrics_aggr_fn=aggregate_training_metrics,
        evaluate_metrics_aggr_fn=aggregate_evaluation_metrics,
    )
    initial = ArrayRecord(OrderedDict({"w": torch.tensor([0.0])}))
    messages = strategy.configure_train(1, initial, ConfigRecord(), GridStub())
    assert [message.metadata.dst_node_id for message in messages] == [0, 1, 2]


def test_qfedavg_and_fedadam_keep_strategy_state_in_sequential_mode() -> None:
    for method in ("qfedavg", "fedadam"):
        result, trace = _run(method, 2, "sequential")
        assert result.arrays is not None
        assert trace == [0, 1, 0, 1]


def test_private_grid_rejects_incomplete_or_duplicate_dispatch_without_ids() -> None:
    app = ClientApp()
    grid = SequentialGrid(
        app,
        run_id=1,
        node_configs={
            node_id: {
                "partition-id": node_id,
                "num-partitions": 2,
                "station": f"private-{node_id}",
            }
            for node_id in range(2)
        },
        run_config={"method": "pafa_rule"},
    )
    content = RecordDict({"config": ConfigRecord()})
    messages = [grid.create_message(content, "train", node_id, "1") for node_id in range(2)]
    for invalid in ([messages[0]], [messages[0], messages[1], messages[1]]):
        with pytest.raises(RuntimeError, match="every client exactly once") as exc_info:
            list(grid.send_and_receive(invalid))
        assert "missing=" not in str(exc_info.value)
        assert "extra=" not in str(exc_info.value)


def test_private_grid_requires_unique_complete_station_bindings() -> None:
    app = ClientApp()
    with pytest.raises(RuntimeError, match="unique, complete private station bindings"):
        SequentialGrid(
            app,
            run_id=1,
            node_configs={
                10: {"partition-id": 0, "num-partitions": 2, "station": "same"},
                11: {"partition-id": 0, "num-partitions": 2, "station": "same"},
            },
            run_config={"method": "pafa_rule"},
        )


def test_private_grid_cannot_reuse_client_state_across_runs() -> None:
    grid = SequentialGrid(
        ClientApp(),
        run_id=1,
        node_configs={
            0: {"partition-id": 0, "num-partitions": 1, "station": "private"}
        },
        run_config={"method": "pafa_rule"},
    )
    grid.set_run(1)
    with pytest.raises(RuntimeError, match="cannot be reused across runs"):
        grid.set_run(2)


@pytest.mark.parametrize(
    "forged_field",
    [
        "run_id",
        "src_node_id",
        "dst_node_id",
        "group_id",
        "message_type",
        "reply_to_message_id",
    ],
)
def test_private_grid_rejects_forged_reply_metadata_without_ids(
    forged_field: str,
) -> None:
    private_node_id = 91_071
    private_station = "station-secret-3d826"
    private_client = "client-secret-7a914"
    app = ClientApp()

    @app.train()
    def train(message: Message, context: Context) -> Message:
        del context
        valid_reply = Message(content=message.content, reply_to=message)
        metadata = valid_reply.metadata
        forged_metadata = Metadata(
            run_id=metadata.run_id + 1
            if forged_field == "run_id"
            else metadata.run_id,
            message_id=metadata.message_id,
            src_node_id=metadata.src_node_id + 1
            if forged_field == "src_node_id"
            else metadata.src_node_id,
            dst_node_id=metadata.dst_node_id + 1
            if forged_field == "dst_node_id"
            else metadata.dst_node_id,
            reply_to_message_id="forged-reply-id"
            if forged_field == "reply_to_message_id"
            else metadata.reply_to_message_id,
            group_id=f"{metadata.group_id}-forged"
            if forged_field == "group_id"
            else metadata.group_id,
            created_at=metadata.created_at,
            ttl=metadata.ttl,
            message_type="evaluate"
            if forged_field == "message_type"
            else metadata.message_type,
        )
        return Message(content=valid_reply.content, metadata=forged_metadata)

    grid = SequentialGrid(
        app,
        run_id=41,
        node_configs={
            private_node_id: {
                "partition-id": 0,
                "num-partitions": 1,
                "station": private_station,
                "client-id": private_client,
            }
        },
        run_config={"method": "pafa_rule"},
    )
    instruction = grid.create_message(
        RecordDict({"config": ConfigRecord()}),
        "train",
        private_node_id,
        "5",
    )

    with pytest.raises(
        RuntimeError,
        match="Sequential PAFA ClientApp failed during secure cohort dispatch",
    ) as exc_info:
        list(grid.send_and_receive([instruction]))

    public_error = str(exc_info.value)
    for private_identifier in (
        str(private_node_id),
        private_station,
        private_client,
        "partition-id=0",
        "partition 0",
    ):
        assert private_identifier not in public_error
