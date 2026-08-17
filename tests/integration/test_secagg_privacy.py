from __future__ import annotations

import json

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.common import recorddict_compat as compat

from aqfl.federated.secure_aggregation import (
    ACTION_IDS,
    DIAGNOSES,
    GATES,
    SOURCES,
    CohortSummaryCodec,
    private_secaggplus_mod,
    run_secure_pafa,
)
from aqfl.federated.sequential import SequentialGrid


def _cohort_vector(normalized_mae: float, *, clipping_violation: bool = False) -> np.ndarray:
    vector = np.zeros(CohortSummaryCodec.size, dtype=np.float32)
    vector[1] = normalized_mae
    vector[10] = float(clipping_violation)
    offset = CohortSummaryCodec.continuous_size
    for group in (ACTION_IDS, DIAGNOSES, SOURCES, GATES):
        vector[offset] = 1.0
        offset += len(group)
    return vector


def test_real_flower_secaggplus_exposes_only_cohort_aggregate(tmp_path) -> None:
    client_app = ClientApp(mods=[private_secaggplus_mod])

    @client_app.train()
    def train(msg: Message, context: Context) -> Message:
        fitins = compat.recorddict_to_fitins(msg.content, keep_input=True)
        del fitins
        private_value = float(context.node_id + 1)
        cohort_vector = _cohort_vector(private_value / 10.0)
        fitres = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(
                [
                    np.asarray([private_value], dtype=np.float32),
                    cohort_vector,
                ]
            ),
            num_examples=1,
            metrics={"private_client_metric": private_value},
        )
        return Message(content=compat.fitres_to_recorddict(fitres, True), reply_to=msg)

    run_config = {
        "method": "pafa_rule",
        "seed": 42,
        "lr": 0.001,
        "batch-size": 2,
    }
    grid = SequentialGrid(
        client_app,
        run_id=1,
        node_configs={
            node_id: {
                "partition-id": node_id,
                "num-partitions": 3,
                "station": f"synthetic-{node_id}",
            }
            for node_id in range(3)
        },
        run_config=run_config,
    )
    context = Context(
        run_id=1,
        node_id=0,
        node_config={},
        state=RecordDict(),
        run_config=run_config,
    )
    event_log = tmp_path / "cohort-events.jsonl"
    result = run_secure_pafa(
        grid=grid,
        context=context,
        initial_arrays=ArrayRecord(
            {"weight": Array(np.asarray([0.0], dtype=np.float32))}
        ),
        config={
            "federated": {"num_clients": 3},
            "privacy": {
                "coordinator_min_cohort_size": 2,
                "secaggplus": {
                    "num_shares": 1.0,
                    "max_weight": 1.0,
                    "clipping_range": 8.0,
                    "quantization_range": 2**22,
                    "modulus_range": 2**32,
                },
            },
        },
        method="pafa_rule",
        num_rounds=1,
        base_lr=0.001,
        batch_size=2,
        strict_llm=True,
        probe_enabled=True,
        event_log=event_log,
    )

    quantization_step = 2 * 8.0 / (2**22)
    assert result.arrays["weight"].numpy().item() == pytest.approx(
        2.0, abs=1.05 * quantization_step
    )
    assert result.round_metrics[0]["cohort_val_macro_mae"] == pytest.approx(
        20.0, abs=1.05 * 100.0 * quantization_step
    )
    serialized = event_log.read_text(encoding="utf-8").lower()
    assert "client_id" not in serialized
    assert "node_id" not in serialized
    assert "station" not in serialized
    event = json.loads(serialized)
    assert event["cohort_size"] == 3
    assert all("node_id" not in row for row in grid.telemetry)
