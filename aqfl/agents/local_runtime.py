"""Client-private PAFA agent runtime.

All state in this module is stored in the ClientApp ``Context.state``. It is never
serialized into a server message or a run artifact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from flwr.app import ConfigRecord, Context

from aqfl.agents.action_library import build_action_library
from aqfl.agents.memory import EpisodicMemory
from aqfl.agents.v2_contracts import (
    ActionProposal,
    CohortDirective,
    CreditRecord,
    ExecutionDecision,
)
from aqfl.agents.v2_proposers import (
    ActionProposer,
    ContextualBanditProposer,
    LLMActionProposer,
    ProbeOracleProposer,
    RuleActionProposer,
)

PRIVATE_STATE_RECORD = "pafa_private_agent_state"
LOCAL_CLIENT_TOKEN = "local"


@dataclass
class PrivateAgentState:
    memory: EpisodicMemory
    previous_update_sketch: tuple[float, ...] = ()
    last_proposal: ActionProposal | None = None
    bandit_a: list[list[float]] | None = None
    bandit_b: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory": self.memory.to_dict(),
            "previous_update_sketch": list(self.previous_update_sketch),
            "last_proposal": self.last_proposal.to_dict() if self.last_proposal else None,
            "bandit_a": self.bandit_a,
            "bandit_b": self.bandit_b,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PrivateAgentState:
        proposal = data.get("last_proposal")
        return cls(
            memory=EpisodicMemory.from_dict(data["memory"]),
            previous_update_sketch=tuple(
                float(item) for item in data.get("previous_update_sketch", [])
            ),
            last_proposal=ActionProposal.from_dict(proposal) if proposal else None,
            bandit_a=data.get("bandit_a"),
            bandit_b=data.get("bandit_b"),
        )


def load_private_agent_state(context: Context, max_records: int) -> PrivateAgentState:
    if PRIVATE_STATE_RECORD not in context.state.config_records:
        return PrivateAgentState(EpisodicMemory(max_records))
    record = context.state.config_records[PRIVATE_STATE_RECORD]
    payload = json.loads(str(record["json"]))
    state = PrivateAgentState.from_dict(payload)
    if state.memory.max_records_per_client != max_records:
        raise RuntimeError("Client-private memory capacity changed during a PAFA run")
    return state


def save_private_agent_state(context: Context, state: PrivateAgentState) -> None:
    context.state.config_records[PRIVATE_STATE_RECORD] = ConfigRecord(
        {"json": json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))}
    )


def _build_proposer(
    method: str,
    config: dict[str, Any],
    state: PrivateAgentState,
    *,
    strict_llm: bool,
    directive: CohortDirective | None = None,
) -> ActionProposer:
    if method == "pafa_rule":
        return RuleActionProposer()
    if method == "pafa_probe_oracle":
        return ProbeOracleProposer()
    if method == "pafa_bandit":
        bandit_proposer = ContextualBanditProposer(
            alpha=float(config["agentic_v2"]["bandit_alpha"]),
            ridge=float(config["agentic_v2"]["bandit_ridge"]),
        )
        if state.bandit_a is not None and state.bandit_b is not None:
            bandit_proposer.a = np.asarray(state.bandit_a, dtype=np.float64)
            bandit_proposer.b = np.asarray(state.bandit_b, dtype=np.float64)
        return bandit_proposer
    if method in {"pafa_llm", "pafa_llm_no_probe"}:
        llm_proposer = LLMActionProposer(config, None, strict=strict_llm)
        if state.last_proposal is not None:
            llm_proposer.last_proposals = {LOCAL_CLIENT_TOKEN: state.last_proposal}
        return llm_proposer
    raise ValueError(f"Unsupported client-local PAFA method: {method}")


def propose_local_actions(
    method: str,
    round_number: int,
    config: dict[str, Any],
    state: PrivateAgentState,
    *,
    strict_llm: bool,
    directive: CohortDirective | None = None,
) -> tuple[ActionProposal, ActionProposer]:
    proposer = _build_proposer(method, config, state, strict_llm=strict_llm)
    capsules = list(state.memory.capsules(LOCAL_CLIENT_TOKEN))
    if capsules:
        capsule = capsules[-1]
    else:
        capsule = state.memory.build_capsule(
            client_id=LOCAL_CLIENT_TOKEN,
            round_number=0,
            val_mae=0.0,
            val_rmse=0.0,
            high_pollution_mae=0.0,
            train_loss=0.0,
            update_norm=0.0,
            update_cosine=1.0,
            train_seconds=0.0,
            local_epochs=0,
        )
    proposal = proposer.propose(
        round_number, [capsule], state.memory, directive
    )[LOCAL_CLIENT_TOKEN]
    state.last_proposal = proposal
    return proposal, proposer


def apply_cohort_lr_cap(
    execution: ExecutionDecision,
    lr_scale_cap: float,
) -> ExecutionDecision:
    if execution.selected_action.lr_scale <= lr_scale_cap:
        return execution
    library = build_action_library()
    replacement = library["cautious"] if lr_scale_cap <= 0.5 else library["safe_default"]
    return ExecutionDecision(
        client_id=execution.client_id,
        selected_action=replacement,
        accepted=False,
        reason=(
            f"aggregate-only coordinator capped lr_scale at {lr_scale_cap:g}; "
            f"{execution.reason}"
        ),
        conservative_gain=execution.conservative_gain,
        probe_outcomes=execution.probe_outcomes,
    )


def record_local_outcome(
    state: PrivateAgentState,
    proposer: ActionProposer,
    execution: ExecutionDecision,
    *,
    round_number: int,
    val_mae: float,
) -> CreditRecord:
    previous = state.memory.capsules(LOCAL_CLIENT_TOKEN)
    realized_gain = 0.0 if not previous else previous[-1].val_mae - float(val_mae)
    credit = CreditRecord(
        client_id=LOCAL_CLIENT_TOKEN,
        round_number=round_number,
        action_id=execution.selected_action.action_id,
        predicted_gain=execution.conservative_gain,
        realized_gain=realized_gain,
        accepted=execution.accepted,
    )
    state.memory.add_credit(credit)
    state.memory.add_execution(round_number, execution)
    proposer.observe(credit)
    if isinstance(proposer, ContextualBanditProposer):
        state.bandit_a = proposer.a.tolist()
        state.bandit_b = proposer.b.tolist()
    return credit
