from __future__ import annotations

import os

import pytest

from aqfl.federated import resources


def test_thread_limits(monkeypatch) -> None:
    for name in resources.THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    resources.limit_client_threads(1)
    assert all(os.environ[name] == "1" for name in resources.THREAD_ENV_VARS)


def test_resource_gate_fails_without_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        resources,
        "resource_snapshot",
        lambda: {"logical_cpus": 16, "physical_cpus": 8, "available_memory_gb": 2.0, "total_memory_gb": 16.0},
    )
    with pytest.raises(RuntimeError, match="no automatic downgrade"):
        resources.enforce_resource_gate(
            {"resources": {"min_logical_cpus": 13, "min_available_memory_gb": 10}}
        )
