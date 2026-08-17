from __future__ import annotations

from pathlib import Path

import pytest

from aqfl.experiments.sweep import build_queue, command_for, ensure_planned, load_plan
from aqfl.reporting.registry import ExperimentRegistry


def test_screening_queue_expands_preregistered_grid() -> None:
    plan = load_plan(Path("configs/experiments/formal.yaml"))

    queue = build_queue(plan, "screening")

    assert len(queue) == 13
    assert {item.get("rounds") for item in queue if item["method"] != "centralized_gru"} == {3}
    assert {item.get("centralized_epochs") for item in queue if item["method"] == "centralized_gru"} == {1}
    assert sum(item["method"] == "centralized_gru" for item in queue) == 4
    assert sum(item["method"] == "fedprox" for item in queue) == 3
    assert sum(item["method"] == "qfedavg" for item in queue) == 3
    assert sum(item["method"] == "fedadam" for item in queue) == 3


def test_screening_commands_include_selected_parameters() -> None:
    baseline = command_for({
        "method": "centralized_gru",
        "seed": 42,
        "stage": "screening",
        "hidden_size": 32,
        "learning_rate": 0.0005,
        "centralized_epochs": 1,
    })
    fedprox = command_for({
        "method": "fedprox",
        "seed": 42,
        "stage": "screening",
        "proximal_mu": 0.1,
        "rounds": 3,
    })

    assert "--hidden-size" in baseline and "32" in baseline
    assert "--learning-rate" in baseline and "0.0005" in baseline
    assert "scripts/run_flower_sequential.py" in fedprox
    assert "--proximal-mu" in fedprox
    assert "0.1" in fedprox
    assert "--rounds" in fedprox
    assert "3" in fedprox
    assert "--max-epochs" in baseline
    qfedavg = command_for({
        "method": "qfedavg",
        "seed": 42,
        "stage": "screening",
        "q": 0.1,
        "qffl_lr": 1.0,
        "rounds": 3,
    })
    assert "--q" in qfedavg and "0.1" in qfedavg
    assert "--qffl-lr" in qfedavg and "1.0" in qfedavg


def test_planning_is_idempotent_but_does_not_overwrite_progress(tmp_path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.csv")
    item = {"method": "fedprox", "seed": 42}

    ensure_planned(registry, "queue-screening-001", item)
    ensure_planned(registry, "queue-screening-001", item)
    registry.transition("queue-screening-001", "running")

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        ensure_planned(registry, "queue-screening-001", item)


def test_pafa_methods_use_the_real_sequential_flower_entrypoint() -> None:
    for method in (
        "pafa_rule",
        "pafa_bandit",
        "pafa_probe_oracle",
        "pafa_llm",
        "pafa_llm_no_probe",
    ):
        command = command_for({"method": method, "seed": 42, "stage": "single_seed_full"})
        assert "scripts/run_flower_sequential.py" in command
        assert command[command.index("--method") + 1] == method
