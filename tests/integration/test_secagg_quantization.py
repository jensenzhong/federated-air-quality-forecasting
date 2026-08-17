from __future__ import annotations

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

CLIPPING_RANGE = 8.0
QUANTIZATION_RANGE = 2**22
MODULUS_RANGE = 2**32


def _summary(normalized_mae: float) -> np.ndarray:
    vector = np.zeros(CohortSummaryCodec.size, dtype=np.float32)
    vector[1] = normalized_mae
    offset = CohortSummaryCodec.continuous_size
    for group in (ACTION_IDS, DIAGNOSES, SOURCES, GATES):
        vector[offset] = 1.0
        offset += len(group)
    return vector


def test_twelve_client_secagg_quantization_stays_within_hard_error_bound(
    tmp_path,
) -> None:
    client_app = ClientApp(mods=[private_secaggplus_mod])
    client_arrays = {
        node_id: np.asarray(
            [
                -7.99991 + node_id * 0.00001,
                -1.234567 + node_id * 0.01,
                0.1234567 - node_id * 0.002,
                4.567891 + node_id * 0.003,
                7.99989 - node_id * 0.00001,
            ],
            dtype=np.float32,
        )
        for node_id in range(12)
    }

    @client_app.train()
    def train(msg: Message, context: Context) -> Message:
        compat.recorddict_to_fitins(msg.content, keep_input=True)
        node_id = context.node_id
        fitres = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(
                [client_arrays[node_id], _summary((node_id + 1) / 20.0)]
            ),
            num_examples=1,
            metrics={},
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
                "num-partitions": 12,
                "station": f"synthetic-{node_id}",
            }
            for node_id in range(12)
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
    result = run_secure_pafa(
        grid=grid,
        context=context,
        initial_arrays=ArrayRecord({
            "weight": Array(np.zeros(5, dtype=np.float32))
        }),
        config={
            "federated": {"num_clients": 12},
            "privacy": {
                "coordinator_min_cohort_size": 10,
                "secaggplus": {
                    "num_shares": 1.0,
                    "max_weight": 1.0,
                    "clipping_range": CLIPPING_RANGE,
                    "quantization_range": QUANTIZATION_RANGE,
                    "modulus_range": MODULUS_RANGE,
                },
            },
        },
        method="pafa_rule",
        num_rounds=1,
        base_lr=0.001,
        batch_size=2,
        strict_llm=True,
        probe_enabled=True,
        event_log=tmp_path / "quantization-events.jsonl",
    )

    expected = np.mean(np.stack(list(client_arrays.values())), axis=0)
    actual = result.arrays["weight"].numpy()
    quantization_step = 2 * CLIPPING_RANGE / QUANTIZATION_RANGE
    assert float(np.max(np.abs(actual - expected))) <= 1.05 * quantization_step
    expected_macro_mae = np.mean([(node_id + 1) / 20.0 for node_id in range(12)]) * 100
    assert result.round_metrics[0]["cohort_val_macro_mae"] == pytest.approx(
        expected_macro_mae,
        abs=1.05 * 100 * quantization_step,
    )
    assert result.round_metrics[0]["cohort_clipping_violation_rate"] == pytest.approx(
        0.0,
        abs=1.05 * quantization_step,
    )
