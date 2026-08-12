"""Budget replay planner for a compute-matched FedProx control."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aqfl.agents.decision import Decision


class BudgetReplayPlanner:
    def __init__(self, trace_path: Path) -> None:
        if not trace_path.is_file():
            raise FileNotFoundError(f"Budget-matched FedProx requires a validated decision trace: {trace_path}")
        records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.by_round = {int(record["round"]): record for record in records}
        if not self.by_round:
            raise ValueError("Budget decision trace is empty")

    def choose(self, round_number: int, history: list[dict[str, Any]], current: dict[str, float]) -> Decision:
        del history, current
        if round_number not in self.by_round:
            raise ValueError(f"Budget trace is missing round {round_number}")
        source = self.by_round[round_number]
        return Decision(
            "size_only",
            float(source["lr_scale"]),
            int(source["local_epochs"]),
            "FedProx compute-budget replay of the paired MAS run",
            str(source.get("prompt_hash", "")),
            "cache",
        )
