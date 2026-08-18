from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.common import recorddict_compat as compat

from aqfl.federated.client_app import _config_bool
from aqfl.federated.secure_aggregation import (
    ACTION_IDS,
    COHORT_CONTINUAL_TASK_MATRIX_ARRAY,
    COHORT_SUMMARY_ARRAY,
    DIAGNOSES,
    GATES,
    CohortSummaryCodec,
    private_secaggplus_mod,
    run_secure_pafa,
)
from aqfl.federated.sequential import SequentialGrid


def _summary() -> np.ndarray:
    vector = np.zeros(CohortSummaryCodec.size, dtype=np.float32)
    offset = CohortSummaryCodec.continuous_size
    for group in (ACTION_IDS, DIAGNOSES, ("rule", "bandit", "llm", "cache", "fallback"), GATES):
        vector[offset] = 1.0
        offset += len(group)
    return vector


def test_secagg_continual_matrix_is_fixed_length_and_decoded_only_at_final_task(
    tmp_path,
) -> None:
    client_app = ClientApp(mods=[private_secaggplus_mod])

    @client_app.train()
    def train(msg: Message, context: Context) -> Message:
        fit_config = msg.content.config_records["fitins.config"]
        task_id = int(fit_config["continual-task-id"])
        task_final = _config_bool(fit_config["continual-task-final"])
        matrix = (
            np.zeros(4, dtype=np.float32)
            if not (task_final and task_id == 2)
            else np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float32)
        )
        fitres = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(
                [
                    np.asarray([1.0], dtype=np.float32),
                    _summary(),
                    matrix,
                ]
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
    result = run_secure_pafa(
        grid=grid,
        context=context,
        initial_arrays=ArrayRecord({"weight": Array(np.asarray([0.0], dtype=np.float32))}),
        config={
            "federated": {"num_clients": 3},
            "continual": {
                "enabled": True,
                "task_count": 2,
                "base_rounds": 1,
                "rounds_per_task": 1,
            },
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
        num_rounds=3,
        base_lr=0.001,
        batch_size=2,
        strict_llm=True,
        probe_enabled=True,
        event_log=tmp_path / "continual-events.jsonl",
    )
    assert COHORT_SUMMARY_ARRAY not in result.arrays
    assert COHORT_CONTINUAL_TASK_MATRIX_ARRAY not in result.arrays
    assert result.continual_metrics is not None
    assert result.continual_metrics["average_performance"] == pytest.approx(3.5, abs=1e-3)
    assert result.continual_metrics["average_forgetting"] == pytest.approx(2.0, abs=1e-3)
