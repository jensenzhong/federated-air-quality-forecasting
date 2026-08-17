"""Fail-closed contracts for the PAFA agentic control loop."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

DiagnosisTag = Literal["stable", "underfit", "overfit", "drift", "conflict", "tail_risk"]
AggregationGate = Literal["normal", "downweight_conflict", "protect_tail"]
ProposalSource = Literal["rule", "bandit", "llm", "cache", "fallback"]
DirectivePhase = Literal["initial", "improving", "stagnating", "volatile"]
DirectivePriority = Literal["accuracy", "tail_recovery", "conflict_recovery", "efficiency"]

ALLOWED_DIAGNOSES = {"stable", "underfit", "overfit", "drift", "conflict", "tail_risk"}
ALLOWED_GATES = {"normal", "downweight_conflict", "protect_tail"}
ALLOWED_PROPOSAL_SOURCES = {"rule", "bandit", "llm", "cache", "fallback"}
ALLOWED_DIRECTIVE_PHASES = {"initial", "improving", "stagnating", "volatile"}
ALLOWED_DIRECTIVE_PRIORITIES = {
    "accuracy",
    "tail_recovery",
    "conflict_recovery",
    "efficiency",
}
ALLOWED_LR_SCALES = {0.5, 1.0, 1.5}
ALLOWED_LOCAL_EPOCHS = {1, 2}
ALLOWED_PROXIMAL_MU = {0.001, 0.01}


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class CohortDirective:
    """Fixed, identity-free public instruction produced from one SecAgg+ aggregate."""

    phase: DirectivePhase
    priority: DirectivePriority
    lr_scale_cap: float
    allow_adapt_fast: bool
    allow_tail_focus: bool
    directive_round: int

    def __post_init__(self) -> None:
        if self.phase not in ALLOWED_DIRECTIVE_PHASES:
            raise ValueError(f"Invalid directive phase: {self.phase}")
        if self.priority not in ALLOWED_DIRECTIVE_PRIORITIES:
            raise ValueError(f"Invalid directive priority: {self.priority}")
        if float(self.lr_scale_cap) not in {0.5, 1.0, 1.5}:
            raise ValueError(f"Invalid directive lr_scale_cap: {self.lr_scale_cap}")
        if not isinstance(self.allow_adapt_fast, bool) or not isinstance(self.allow_tail_focus, bool):
            raise ValueError("Directive action permissions must be boolean")
        if self.directive_round < 0:
            raise ValueError("directive_round must be non-negative")

    def validate_for_round(self, round_number: int) -> None:
        if round_number < 1 or self.directive_round != round_number - 1:
            raise RuntimeError(
                "CohortDirective round binding mismatch: expected previous completed round"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CohortDirective:
        required = {
            "phase",
            "priority",
            "lr_scale_cap",
            "allow_adapt_fast",
            "allow_tail_focus",
            "directive_round",
        }
        if set(data) != required:
            raise ValueError("CohortDirective contains missing or unexpected keys")
        if not isinstance(data["allow_adapt_fast"], bool) or not isinstance(
            data["allow_tail_focus"], bool
        ):
            raise ValueError("CohortDirective action permissions must be booleans")
        return cls(
            phase=str(data["phase"]),  # type: ignore[arg-type]
            priority=str(data["priority"]),  # type: ignore[arg-type]
            lr_scale_cap=float(data["lr_scale_cap"]),
            allow_adapt_fast=data["allow_adapt_fast"],
            allow_tail_focus=data["allow_tail_focus"],
            directive_round=int(data["directive_round"]),
        )

    @classmethod
    def from_json(cls, raw: str) -> CohortDirective:
        import json

        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("CohortDirective JSON must encode an object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class ClientAction:
    action_id: str
    lr_scale: float
    local_epochs: int
    proximal_mu: float
    aggregation_gate: AggregationGate
    rationale: str

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("action_id cannot be empty")
        if float(self.lr_scale) not in ALLOWED_LR_SCALES:
            raise ValueError(f"Invalid lr_scale: {self.lr_scale}")
        if int(self.local_epochs) not in ALLOWED_LOCAL_EPOCHS:
            raise ValueError(f"Invalid local_epochs: {self.local_epochs}")
        if float(self.proximal_mu) not in ALLOWED_PROXIMAL_MU:
            raise ValueError(f"Invalid proximal_mu: {self.proximal_mu}")
        if self.aggregation_gate not in ALLOWED_GATES:
            raise ValueError(f"Invalid aggregation_gate: {self.aggregation_gate}")
        if not self.rationale.strip():
            raise ValueError("Action rationale cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClientAction:
        required = {
            "action_id",
            "lr_scale",
            "local_epochs",
            "proximal_mu",
            "aggregation_gate",
            "rationale",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"ClientAction is missing keys: {sorted(missing)}")
        extra = data.keys() - required
        if extra:
            raise ValueError(f"ClientAction contains unexpected keys: {sorted(extra)}")
        return cls(
            action_id=str(data["action_id"]),
            lr_scale=float(data["lr_scale"]),
            local_epochs=int(data["local_epochs"]),
            proximal_mu=float(data["proximal_mu"]),
            aggregation_gate=str(data["aggregation_gate"]),  # type: ignore[arg-type]
            rationale=str(data["rationale"]),
        )


@dataclass(frozen=True)
class ClientStateCapsule:
    client_id: str
    round_number: int
    val_mae: float
    val_rmse: float
    high_pollution_mae: float
    train_loss: float
    update_norm: float
    update_cosine: float
    mae_ema: float
    mae_slope: float
    mae_oscillation: float
    drift_score: float
    previous_action_id: str
    previous_realized_gain: float
    train_seconds: float
    local_epochs: int

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id cannot be empty")
        if self.round_number < 0:
            raise ValueError("round_number must be non-negative")
        for name in (
            "val_mae",
            "val_rmse",
            "high_pollution_mae",
            "train_loss",
            "update_norm",
            "update_cosine",
            "mae_ema",
            "mae_slope",
            "mae_oscillation",
            "drift_score",
            "previous_realized_gain",
            "train_seconds",
        ):
            _finite(name, getattr(self, name))
        if self.val_mae < 0 or self.val_rmse < 0 or self.update_norm < 0:
            raise ValueError("Error and update-norm values must be non-negative")
        if not -1.0 <= self.update_cosine <= 1.0:
            raise ValueError("update_cosine must be in [-1, 1]")
        if self.mae_oscillation < 0 or self.drift_score < 0 or self.train_seconds < 0:
            raise ValueError("Oscillation, drift, and cost values must be non-negative")
        if self.local_epochs not in {0, 1, 2}:
            raise ValueError("local_epochs must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClientStateCapsule:
        return cls(**data)


@dataclass(frozen=True)
class ActionProposal:
    client_id: str
    diagnosis: DiagnosisTag
    evidence: tuple[str, ...]
    candidates: tuple[ClientAction, ...]
    source: ProposalSource
    prompt_hash: str = ""

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id cannot be empty")
        if self.diagnosis not in ALLOWED_DIAGNOSES:
            raise ValueError(f"Invalid diagnosis: {self.diagnosis}")
        if not 1 <= len(self.candidates) <= 3:
            raise ValueError("Each proposal must contain 1 to 3 candidates")
        action_ids = [action.action_id for action in self.candidates]
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Candidate action IDs must be unique")
        if not self.evidence or any(not item.strip() for item in self.evidence):
            raise ValueError("Proposal evidence must contain non-empty entries")
        if self.source not in ALLOWED_PROPOSAL_SOURCES:
            raise ValueError(f"Invalid proposal source: {self.source}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "diagnosis": self.diagnosis,
            "evidence": list(self.evidence),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "source": self.source,
            "prompt_hash": self.prompt_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionProposal:
        required = {"client_id", "diagnosis", "evidence", "candidates", "source", "prompt_hash"}
        if set(data) != required:
            raise ValueError("ActionProposal contains missing or unexpected keys")
        return cls(
            client_id=str(data["client_id"]),
            diagnosis=str(data["diagnosis"]),  # type: ignore[arg-type]
            evidence=tuple(str(item) for item in data["evidence"]),
            candidates=tuple(ClientAction.from_dict(item) for item in data["candidates"]),
            source=str(data["source"]),  # type: ignore[arg-type]
            prompt_hash=str(data["prompt_hash"]),
        )


@dataclass(frozen=True)
class ProbeOutcome:
    action_id: str
    baseline_loss: float
    probed_loss: float
    estimated_gain: float
    cost_batches: int
    update_norm: float

    def __post_init__(self) -> None:
        if not self.action_id.strip():
            raise ValueError("Probe action_id cannot be empty")
        for name in ("baseline_loss", "probed_loss", "estimated_gain", "update_norm"):
            _finite(name, getattr(self, name))
        if self.baseline_loss < 0 or self.probed_loss < 0 or self.cost_batches < 0:
            raise ValueError("Probe losses and cost must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProbeOutcome:
        return cls(**data)


@dataclass(frozen=True)
class ExecutionDecision:
    client_id: str
    selected_action: ClientAction
    accepted: bool
    reason: str
    conservative_gain: float
    probe_outcomes: tuple[ProbeOutcome, ...]

    def __post_init__(self) -> None:
        if not self.client_id.strip() or not self.reason.strip():
            raise ValueError("Execution client_id and reason cannot be empty")
        _finite("conservative_gain", self.conservative_gain)
        outcome_ids = {outcome.action_id for outcome in self.probe_outcomes}
        if self.accepted and self.probe_outcomes and self.selected_action.action_id not in outcome_ids:
            raise ValueError("Accepted action must have a probe outcome")

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "selected_action": self.selected_action.to_dict(),
            "accepted": self.accepted,
            "reason": self.reason,
            "conservative_gain": self.conservative_gain,
            "probe_outcomes": [outcome.to_dict() for outcome in self.probe_outcomes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionDecision:
        return cls(
            client_id=str(data["client_id"]),
            selected_action=ClientAction.from_dict(data["selected_action"]),
            accepted=bool(data["accepted"]),
            reason=str(data["reason"]),
            conservative_gain=float(data["conservative_gain"]),
            probe_outcomes=tuple(ProbeOutcome.from_dict(item) for item in data["probe_outcomes"]),
        )


@dataclass(frozen=True)
class CreditRecord:
    client_id: str
    round_number: int
    action_id: str
    predicted_gain: float
    realized_gain: float
    accepted: bool

    def __post_init__(self) -> None:
        if not self.client_id.strip() or not self.action_id.strip():
            raise ValueError("Credit identifiers cannot be empty")
        if self.round_number < 1:
            raise ValueError("Credit round_number must be positive")
        _finite("predicted_gain", self.predicted_gain)
        _finite("realized_gain", self.realized_gain)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreditRecord:
        return cls(**data)
