"""Validated MAS decision contract."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import jsonschema

StrategyName = Literal["size_only", "perf_only", "hybrid", "fairness_clip"]
DecisionSource = Literal["llm", "rule", "cache", "fallback"]
ALLOWED_LR_SCALES = {0.5, 1.0, 1.5}
ALLOWED_LOCAL_EPOCHS = {1, 2}
ALLOWED_STRATEGIES = {"size_only", "perf_only", "hybrid", "fairness_clip"}
ALLOWED_SOURCES = {"llm", "rule", "cache", "fallback"}


@dataclass(frozen=True)
class Decision:
    strategy: StrategyName
    lr_scale: float
    local_epochs: int
    reason: str
    prompt_hash: str
    source: DecisionSource

    def __post_init__(self) -> None:
        if self.strategy not in ALLOWED_STRATEGIES:
            raise ValueError(f"Invalid strategy: {self.strategy}")
        if float(self.lr_scale) not in ALLOWED_LR_SCALES:
            raise ValueError(f"Invalid lr_scale: {self.lr_scale}")
        if int(self.local_epochs) not in ALLOWED_LOCAL_EPOCHS:
            raise ValueError(f"Invalid local_epochs: {self.local_epochs}")
        if self.source not in ALLOWED_SOURCES:
            raise ValueError(f"Invalid source: {self.source}")
        if not self.reason.strip():
            raise ValueError("Decision reason cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source: DecisionSource | None = None, prompt_hash: str | None = None) -> Decision:
        schema_path = Path(__file__).resolve().parents[2] / "configs" / "decision.schema.json"
        if schema_path.is_file():
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            try:
                jsonschema.validate(data, schema)
            except jsonschema.ValidationError as exc:
                raise ValueError(f"Decision schema validation failed: {exc.message}") from exc
        required = {"strategy", "lr_scale", "local_epochs", "reason"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Decision is missing keys: {sorted(missing)}")
        return cls(
            strategy=data["strategy"],
            lr_scale=float(data["lr_scale"]),
            local_epochs=int(data["local_epochs"]),
            reason=str(data["reason"]),
            prompt_hash=prompt_hash if prompt_hash is not None else str(data.get("prompt_hash", "")),
            source=source if source is not None else data.get("source", "rule"),
        )
