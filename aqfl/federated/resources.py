"""Host resource preflight for the pre-registered 12-process protocol."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import psutil

THREAD_ENV_VARS = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]


def limit_client_threads(count: int = 1) -> None:
    for name in THREAD_ENV_VARS:
        os.environ[name] = str(count)


def resource_snapshot() -> dict[str, float | int]:
    memory = psutil.virtual_memory()
    return {
        "logical_cpus": int(psutil.cpu_count(logical=True) or 0),
        "physical_cpus": int(psutil.cpu_count(logical=False) or 0),
        "available_memory_gb": float(memory.available / 1024**3),
        "total_memory_gb": float(memory.total / 1024**3),
    }


def enforce_resource_gate(config: dict[str, Any]) -> dict[str, float | int]:
    snapshot = resource_snapshot()
    requirements = config["resources"]
    if snapshot["logical_cpus"] < int(requirements["min_logical_cpus"]):
        raise RuntimeError(
            f"Full-concurrency protocol requires {requirements['min_logical_cpus']} logical CPUs; "
            f"found {snapshot['logical_cpus']}"
        )
    if snapshot["available_memory_gb"] < float(requirements["min_available_memory_gb"]):
        raise RuntimeError(
            f"Full-concurrency protocol requires {requirements['min_available_memory_gb']} GB free RAM; "
            f"found {snapshot['available_memory_gb']:.2f} GB. Free memory and retry; no automatic downgrade is allowed."
        )
    return snapshot


def enforce_sequential_resource_gate(config: dict[str, Any]) -> dict[str, float | int]:
    """Preflight the single-process route without applying full-concurrency limits."""
    snapshot = resource_snapshot()
    requirements = config["resources"]
    min_cpus = int(requirements.get("sequential_min_logical_cpus", 1))
    min_memory = float(requirements.get("sequential_min_available_memory_gb", 0.5))
    if snapshot["logical_cpus"] < min_cpus:
        raise RuntimeError(
            f"Sequential protocol requires {min_cpus} logical CPU; "
            f"found {snapshot['logical_cpus']}"
        )
    if snapshot["available_memory_gb"] < min_memory:
        raise RuntimeError(
            f"Sequential protocol requires {min_memory:g} GB free RAM; "
            f"found {snapshot['available_memory_gb']:.2f} GB"
        )
    return snapshot


def append_system_snapshot(path: Path, event: str) -> None:
    payload = {"event": event, **resource_snapshot()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
