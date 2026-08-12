"""Experiment registry with a constrained state machine."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_TRANSITIONS = {
    None: {"planned"},
    "planned": {"running", "invalid"},
    "running": {"completed", "invalid"},
    "completed": {"validated", "invalid"},
    "validated": set(),
    "invalid": set(),
}


class ExperimentRegistry:
    COLUMNS = ["run_id", "method", "seed", "status", "run_dir", "updated_at_utc", "reason"]

    def __init__(self, path: Path):
        self.path = path
        if path.is_file():
            self.frame = pd.read_csv(path, dtype={"run_id": str, "method": str, "status": str, "run_dir": str, "reason": str})
        else:
            self.frame = pd.DataFrame(columns=self.COLUMNS)

    def transition(self, run_id: str, status: str, **fields: Any) -> None:
        matching = self.frame.index[self.frame["run_id"] == run_id].tolist()
        current = str(self.frame.loc[matching[0], "status"]) if matching else None
        if status not in ALLOWED_TRANSITIONS[current]:
            raise ValueError(f"Invalid registry transition: {current} -> {status}")
        row = {
            "run_id": run_id,
            "method": fields.get("method", ""),
            "seed": fields.get("seed", ""),
            "status": status,
            "run_dir": fields.get("run_dir", ""),
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "reason": fields.get("reason", ""),
        }
        if matching:
            for key, value in row.items():
                if value != "" or key in {"status", "updated_at_utc"}:
                    self.frame.loc[matching[0], key] = value
        else:
            self.frame = pd.concat([self.frame, pd.DataFrame([row])], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(self.path, index=False)
