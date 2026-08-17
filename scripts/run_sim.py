#!/usr/bin/env python
"""Low-memory 12-client federated run (in-process, batched sequential scheduling).

All 12 stations still participate every round (station partitioning unchanged);
clients run one-by-one inside a single process instead of 12 concurrent
processes, so peak memory fits comfortably on a 16GB host.

Implements the exact aggregation semantics of these pre-registered methods:
- fedavg: sample-weighted average
- fedprox: FedAvg aggregation + client-side proximal term (mu)

QFedAvg, FedAdam, Rule-MAS, and LLM-MAS deliberately remain unsupported here;
their Flower strategy semantics must be exercised by the formal runtime.

Results are DEBUG/screening evidence only, not formal M4/M6 evidence
(the pre-registered formal protocol requires 12 concurrent processes on
adequate hardware). The run directory is deliberately marked
protocol_frozen=false and evaluation_split=val so that validate_run rejects
it for formal reporting.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

# Conservative thread limits BEFORE importing torch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import torch  # noqa: E402

from aqfl.config import load_config, set_seed  # noqa: E402
from aqfl.data.dataset import list_stations, load_cache_metadata, load_station_dataset  # noqa: E402
from aqfl.data.preprocessing import GlobalScalerState  # noqa: E402
from aqfl.models import build_model  # noqa: E402
from aqfl.models.training import evaluate_model, train_local_model  # noqa: E402
from aqfl.reporting.artifacts import RunArtifacts  # noqa: E402

METHODS_FIXED = {"fedavg", "fedprox"}


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            if key.strip() and value.strip():
                os.environ[key.strip()] = value.strip()


def _state_dict_f64(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().to(torch.float64) for k, v in model.state_dict().items()}


def _fedavg_aggregate(
    client_states: list[dict[str, torch.Tensor]],
    client_sizes: list[int],
) -> dict[str, torch.Tensor]:
    total_n = sum(client_sizes)
    new_state: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        acc = torch.zeros_like(client_states[0][key], dtype=torch.float64)
        for state, n in zip(client_states, client_sizes, strict=True):
            acc += state[key] * n
        new_state[key] = acc / total_n
    return new_state


def main() -> None:
    method = sys.argv[1] if len(sys.argv) > 1 else "fedprox"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    if method not in METHODS_FIXED:
        supported = ", ".join(sorted(METHODS_FIXED))
        raise SystemExit(
            f"unsupported sequential simulation method: {method}; supported: {supported}"
        )

    _load_env(PROJECT / ".env")
    os.environ.setdefault(
        "BEIJING_AQ_DATA_DIR",
        r"C:\Users\23079\Downloads\beijing+multi+site+air+quality+data",
    )

    config = load_config(PROJECT / "configs" / "base.yaml")
    config["project"]["seed"] = seed
    # Hard firewall: this runner only ever evaluates on val; test requires the
    # formal 12-process protocol and a frozen protocol.
    config["runtime"] = {
        "method": method,
        "evaluation_split": "val",
        "protocol_frozen": False,
        "execution_mode": "sequential_simulation",
    }
    set_seed(seed)

    fed = config["federated"]
    training = config["training"]
    local_epochs = int(os.environ.get("AQFL_LOCAL_EPOCHS", str(training["local_epochs"])))
    lr = float(training["learning_rate"])
    batch_size = int(training["batch_size"])
    weight_decay = float(training["weight_decay"])
    proximal_mu = float(fed["fedprox_mu"]) if method == "fedprox" else 0.0
    stations = list_stations(config)
    print(f"[sim] method={method} seed={seed} rounds={rounds} local_epochs={local_epochs} "
          f"mu={proximal_mu} "
          f"stations={len(stations)} (sequential clients)")
    print("[sim] NOTE: sequential-client results are debug/screening evidence, NOT formal evidence.")

    device = torch.device("cpu")
    global_model = build_model(config).to(device)
    scaler = GlobalScalerState(**load_cache_metadata(config)["scaler"])
    artifacts = RunArtifacts(config, method, seed)
    round_records: list[dict[str, Any]] = []
    client_records: list[dict[str, Any]] = []
    best_macro = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_round = 0

    for rnd in range(1, rounds + 1):
        t0 = time.monotonic()
        client_states: list[dict[str, torch.Tensor]] = []
        client_sizes: list[int] = []
        for station in stations:
            model = build_model(config).to(device)
            model.load_state_dict(global_model.state_dict())

            train_set = load_station_dataset(config, station, "train")
            train_loss = train_local_model(
                model,
                train_set,
                local_epochs,
                lr,
                batch_size,
                weight_decay,
                device=device,
                proximal_mu=proximal_mu,
                global_state=global_model.state_dict() if proximal_mu > 0 else None,
            )
            client_states.append(_state_dict_f64(model))
            client_sizes.append(len(train_set))
            client_records.append({
                "round": rnd,
                "station": station,
                "train_loss": float(train_loss),
                "num_examples": len(train_set),
            })
            del model, train_set

        new_state = _fedavg_aggregate(client_states, client_sizes)

        global_model.load_state_dict({k: v.to(torch.float32) for k, v in new_state.items()})

        # Validation (macro MAE across stations) — screening only, never test
        val_maes: list[float] = []
        for station in stations:
            val_set = load_station_dataset(config, station, "val")
            metrics, _ = evaluate_model(global_model, val_set, scaler, scaler.pollution_p90)
            val_maes.append(float(metrics["mae"]))
            del val_set
        macro_mae = sum(val_maes) / len(val_maes)

        if macro_mae < best_macro:
            best_macro = macro_mae
            best_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}
            best_round = rnd

        elapsed = time.monotonic() - t0
        round_records.append({
            "round": rnd,
            "macro_val_mae": macro_mae,
            "elapsed_sec": elapsed,
        })
        print(f"[sim] round {rnd}/{rounds}  macro_val_mae={macro_mae:.4f}  "
              f"({elapsed/60:.1f} min, clients sequential)", flush=True)

    assert best_state is not None
    global_model.load_state_dict(best_state)

    summary: dict[str, Any] = {
        "evaluation_split": "val",
        "protocol_frozen": False,
        "execution_mode": "sequential_simulation",
        "best_round": best_round,
        "best_macro_val_mae": best_macro,
        "num_rounds": rounds,
        "num_clients": len(stations),
        "local_epochs": local_epochs,
        "proximal_mu": proximal_mu,
    }
    artifacts.finalize(
        global_model,
        summary,
        round_metrics=round_records,
        client_metrics=client_records,
    )
    # Explicitly mark the run as NOT for formal reporting
    summary_path = artifacts.path / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    data["formal_eligible"] = False
    summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    print(f"[sim] done. artifacts: {artifacts.path}")
    print(f"[sim] best_round={best_round} best_macro_val_mae={best_macro:.4f}")


if __name__ == "__main__":
    main()
