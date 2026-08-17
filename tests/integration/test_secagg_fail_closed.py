from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from flwr.app import Context, Message, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.common import recorddict_compat as compat
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS
from flwr.common.secure_aggregation.secaggplus_constants import Key as SecAggKey
from flwr.common.secure_aggregation.secaggplus_constants import Stage as SecAggStage

from aqfl.federated import server_app
from aqfl.federated.secure_aggregation import (
    ACTION_IDS,
    DIAGNOSES,
    GATES,
    SOURCES,
    CohortSummaryCodec,
    private_secaggplus_mod,
)
from aqfl.federated.sequential import SequentialGrid


def _cohort_vector(normalized_mae: float) -> np.ndarray:
    vector = np.zeros(CohortSummaryCodec.size, dtype=np.float32)
    vector[1] = normalized_mae
    offset = CohortSummaryCodec.continuous_size
    for group in (ACTION_IDS, DIAGNOSES, SOURCES, GATES):
        vector[offset] = 1.0
        offset += len(group)
    return vector


class _DroppingCollectReplyGrid(SequentialGrid):
    """Drop one reply only after every client completed the collect stage."""

    dropped_collect_reply = False

    def send_and_receive(
        self,
        messages: Iterable[Message],
        *,
        timeout: float | None = None,
    ) -> Iterable[Message]:
        instructions = list(messages)
        replies = list(super().send_and_receive(instructions, timeout=timeout))
        if not instructions or self.dropped_collect_reply:
            return replies
        configs = instructions[0].content.config_records.get(RECORD_KEY_CONFIGS)
        if configs is not None and configs.get(SecAggKey.STAGE) == SecAggStage.COLLECT_MASKED_VECTORS:
            self.dropped_collect_reply = True
            return replies[:-1]
        return replies


class _TrackingArtifacts:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.mkdir(parents=True)
        self.finalize_called = False
        self.invalidation_reason: str | None = None

    def finalize(self, *_args: Any, **_kwargs: Any) -> None:
        self.finalize_called = True
        (self.path / "checkpoint.pt").touch()
        (self.path / "summary.json").write_text(
            json.dumps({"status": "completed"}),
            encoding="utf-8",
        )

    def invalidate(self, reason: str) -> None:
        self.invalidation_reason = reason
        (self.path / "summary.json").write_text(
            json.dumps({"status": "invalid", "reason": reason}),
            encoding="utf-8",
        )


def test_missing_collect_reply_produces_no_coordinator_event_or_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client_app = ClientApp(mods=[private_secaggplus_mod])

    @client_app.train()
    def train(msg: Message, context: Context) -> Message:
        compat.recorddict_to_fitins(msg.content, keep_input=True)
        value = float(context.node_id + 1)
        fitres = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(
                [
                    np.asarray([[value]], dtype=np.float32),
                    _cohort_vector(value / 10.0),
                ]
            ),
            num_examples=1,
            metrics={},
        )
        return Message(content=compat.fitres_to_recorddict(fitres, True), reply_to=msg)

    run_config = {
        "method": "pafa_rule",
        "seed": 42,
        "num-server-rounds": 1,
        "local-epochs": 1,
        "lr": 0.001,
        "batch-size": 2,
        "proximal-mu": 0.01,
        "q": 1.0,
        "qffl-lr": 1.0,
        "server-lr": 0.1,
        "strict-llm": True,
        "enforce-resource-check": False,
        "budget-trace": "",
        "evaluation-split": "val",
        "protocol-frozen": False,
        "execution-mode": "low_memory_sequential",
        "client-state-isolated": True,
        "config-path": "unused-in-test.yaml",
    }
    grid = _DroppingCollectReplyGrid(
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
    config = {
        "project": {"seed": 42},
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
    }
    artifacts = _TrackingArtifacts(tmp_path / "failed-run")
    monkeypatch.setattr(server_app, "load_config", lambda _path: copy.deepcopy(config))
    monkeypatch.setattr(
        server_app,
        "list_stations",
        lambda _config: ["synthetic-0", "synthetic-1", "synthetic-2"],
    )
    monkeypatch.setattr(
        server_app,
        "build_model",
        lambda _config: torch.nn.Linear(1, 1, bias=False),
    )
    monkeypatch.setattr(server_app, "RunArtifacts", lambda *_args: artifacts)

    with pytest.raises(RuntimeError, match="secure cohort failed closed"):
        server_app.main(grid, context)

    assert grid.dropped_collect_reply is True
    assert artifacts.finalize_called is False
    assert artifacts.invalidation_reason == "PAFA_PRIVACY_GATE_FAILED"
    assert not (artifacts.path / "checkpoint.pt").exists()
    assert not (artifacts.path / "round_metrics.parquet").exists()
    failure_summary = json.loads(
        (artifacts.path / "summary.json").read_text(encoding="utf-8")
    )
    assert failure_summary == {
        "status": "invalid",
        "reason": "PAFA_PRIVACY_GATE_FAILED",
    }
    event_log = artifacts.path / "agentic_events.jsonl"
    assert event_log.is_file()
    assert event_log.read_text(encoding="utf-8") == ""
