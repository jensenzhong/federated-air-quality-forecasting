from __future__ import annotations

import argparse
import json

import pytest

from scripts import run_flower_sequential as runner


def _config() -> dict:
    return {
        "federated": {"num_clients": 2},
        "privacy": {
            "mode": "strict_federated",
            "client_llm_allowed_hosts": [],
            "coordinator_min_cohort_size": 2,
            "secaggplus": {
                "clipping_range": 8.0,
                "quantization_range": 2**22,
                "modulus_range": 2**32,
                "max_weight": 1.0,
            },
        },
        "llm": {
            "base_url": "https://api.deepseek.com",
            "client_base_url": "http://127.0.0.1:11434/v1",
        },
    }


def _args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "method": "pafa_rule",
        "seed": 42,
        "rounds": 1,
        "local_epochs": 1,
        "lr": 0.001,
        "batch_size": 128,
        "proximal_mu": 0.01,
        "q": 1.0,
        "qffl_lr": 1.0,
        "server_lr": 0.1,
        "config_path": "unused.yaml",
        "budget_trace": "",
        "evaluation_split": "val",
        "protocol_frozen": False,
        "strict_llm": True,
        "enforce_resource_check": False,
        "preflight_only": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_preflight_only_exits_before_server_and_deidentifies_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_stations = ["private-station-a", "private-station-b"]
    server_called = False

    def fail_if_server_starts(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal server_called
        server_called = True
        raise AssertionError("ServerApp must not start during preflight")

    monkeypatch.setattr(runner, "load_config", lambda _: _config())
    monkeypatch.setattr(runner, "list_stations", lambda _: private_stations)
    monkeypatch.setattr(runner, "server_main", fail_if_server_starts)

    report = runner.run(_args())

    serialized = capsys.readouterr().out.strip()
    assert json.loads(serialized) == report
    assert report["status"] == "passed_nonformal_preflight"
    assert report["station_count"] == 2
    assert report["training_started"] is False
    assert report["formal_eligible"] is False
    assert server_called is False
    assert all(station not in serialized for station in private_stations)


def test_preflight_rejects_pafa_test_split() -> None:
    run_config = runner.build_run_config(_args(evaluation_split="test"))
    with pytest.raises(RuntimeError, match="test evaluation remains blocked"):
        runner.validate_preflight(_config(), run_config, ["a", "b"])


def test_preflight_rejects_public_client_level_llm() -> None:
    config = _config()
    config["llm"]["client_base_url"] = "https://api.deepseek.com/v1"
    run_config = runner.build_run_config(_args(method="pafa_llm"))
    with pytest.raises(RuntimeError, match="external LLM endpoint"):
        runner.validate_preflight(config, run_config, ["a", "b"])
