"""Protocol qualification registry for strong federated baselines.

The registry separates algorithmic compatibility from implementation evidence.
It is intentionally conservative: a method cannot enter a formal comparison
merely because a non-secure Flower strategy exists for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProtocolCompatibility = Literal["compatible", "conditional", "incompatible"]
FormalStatus = Literal[
    "verified_secagg",
    "pending_secagg_adapter",
    "pending_protocol_audit",
    "incompatible_client_signal",
]


@dataclass(frozen=True)
class BaselineContract:
    method: str
    family: str
    protocol_compatibility: ProtocolCompatibility
    formal_status: FormalStatus
    requires_client_level_server_signal: bool
    requires_per_client_update_visibility: bool
    implementation_note: str
    budget_class: str

    @property
    def formal_eligible(self) -> bool:
        return (
            self.formal_status == "verified_secagg"
            and not self.requires_client_level_server_signal
            and not self.requires_per_client_update_visibility
        )


_VERIFIED_Pafa = {
    "pafa_rule": "rule proposer with aggregate-only blackboard",
    "pafa_bandit": "contextual bandit proposer with aggregate-only blackboard",
    "pafa_probe_oracle": "probe oracle control with aggregate-only blackboard",
    "pafa_llm": "local LLM proposer with aggregate-only blackboard",
    "pafa_llm_no_probe": "local LLM no-probe ablation with aggregate-only blackboard",
    "pafa_fedavg": "static FedAvg baseline through the verified aggregate-only PAFA transport",
    "pafa_fedprox": "static FedProx baseline through the verified aggregate-only PAFA transport",
    "pafa_fedadam": "static FedAdam baseline with aggregate-only server moments",
}


def _build_registry() -> dict[str, BaselineContract]:
    registry: dict[str, BaselineContract] = {
        method: BaselineContract(
            method=method,
            family="secure_baseline" if method.startswith("pafa_fed") else "agentic_control",
            protocol_compatibility="compatible",
            formal_status="verified_secagg",
            requires_client_level_server_signal=False,
            requires_per_client_update_visibility=False,
            implementation_note=note,
            budget_class="pafa_equal_client_probe_budget",
        )
        for method, note in _VERIFIED_Pafa.items()
    }
    for method, family in {
        "fedavg": "static_aggregation",
        "fedprox": "proximal_local_objective",
        "fedadam": "server_adaptive_optimizer",
        "qfedavg": "client_utility_fairness",
    }.items():
        registry[method] = BaselineContract(
            method=method,
            family=family,
            protocol_compatibility="compatible",
            formal_status="pending_secagg_adapter",
            requires_client_level_server_signal=False,
            requires_per_client_update_visibility=False,
            implementation_note="Existing strict path is non-secure; aggregate-only adapter required before formal use",
            budget_class="fixed_local_step_budget",
        )
    for method, family in {
        "scaffold": "control_variate_correction",
        "feddyn": "dynamic_regularization",
        "flash": "drift_aware_optimization",
    }.items():
        registry[method] = BaselineContract(
            method=method,
            family=family,
            protocol_compatibility="conditional",
            formal_status="pending_protocol_audit",
            requires_client_level_server_signal=False,
            requires_per_client_update_visibility=False,
            implementation_note="Client state/control statistics must remain local and be aggregated through fixed SecAgg+ arrays",
            budget_class="fixed_local_step_budget",
        )
    for method in ("aaggff", "fedaware", "fedawa", "selective_collaboration"):
        registry[method] = BaselineContract(
            method=method,
            family="client_selection_or_weighting",
            protocol_compatibility="incompatible",
            formal_status="incompatible_client_signal",
            requires_client_level_server_signal=True,
            requires_per_client_update_visibility=True,
            implementation_note="Original protocol relies on linkable client utility/update signals; no silent privacy-weakened port",
            budget_class="not_comparable_until_protocol_adapted",
        )
    return registry


BASELINE_CONTRACTS = _build_registry()


def baseline_contract(method: str) -> BaselineContract:
    key = str(method).lower()
    try:
        return BASELINE_CONTRACTS[key]
    except KeyError as exc:
        raise ValueError(f"No baseline protocol contract is registered for {method}") from exc


def require_formal_baseline(method: str) -> BaselineContract:
    contract = baseline_contract(method)
    if not contract.formal_eligible:
        raise RuntimeError(
            f"Baseline {method} is not formal-eligible: "
            f"status={contract.formal_status}, "
            f"protocol={contract.protocol_compatibility}"
        )
    return contract
