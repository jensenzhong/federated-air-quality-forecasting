from __future__ import annotations

import json

import numpy as np
import pytest

from aqfl.agents.action_library import build_action_library, resolve_action_ids
from aqfl.agents.executor import ProbeBudget, SafeActionExecutor
from aqfl.agents.memory import EpisodicMemory
from aqfl.agents.v2_contracts import ActionProposal, CohortDirective
from aqfl.agents.v2_proposers import (
    ContextualBanditProposer,
    LLMActionProposer,
    RuleActionProposer,
)
from aqfl.federated.secure_aggregation import (
    AggregateCoordinatorAgent,
    CohortSummaryCodec,
)


def _capsule(client_id: str = "local"):
    return EpisodicMemory().build_capsule(
        client_id=client_id,
        round_number=1,
        val_mae=10.0,
        val_rmse=12.0,
        high_pollution_mae=11.0,
        train_loss=0.5,
        update_norm=1.0,
        update_cosine=0.8,
        train_seconds=1.0,
        local_epochs=1,
    )


def _directive() -> CohortDirective:
    return CohortDirective(
        phase="volatile",
        priority="conflict_recovery",
        lr_scale_cap=0.5,
        allow_adapt_fast=False,
        allow_tail_focus=False,
        directive_round=4,
    )


def test_directive_tampering_and_replay_fail_closed() -> None:
    directive = _directive()
    with pytest.raises(ValueError, match="unexpected keys"):
        CohortDirective.from_dict({**directive.to_dict(), "client_id": "station-7"})
    with pytest.raises(ValueError, match="booleans"):
        CohortDirective.from_dict({**directive.to_dict(), "allow_tail_focus": "false"})
    with pytest.raises(RuntimeError, match="round binding"):
        directive.validate_for_round(3)
    directive.validate_for_round(5)


def test_rule_bandit_and_llm_schema_consume_the_same_public_directive() -> None:
    capsule = _capsule()
    memory = EpisodicMemory()
    directive = _directive()
    rule = RuleActionProposer().propose(5, [capsule], memory, directive)["local"]
    bandit = ContextualBanditProposer(alpha=0.0).propose(
        5, [capsule], memory, directive
    )["local"]
    prompt = LLMActionProposer.build_prompt(5, [capsule], memory, directive)
    assert {action.action_id for action in rule.candidates} <= {
        "safe_default",
        "cautious",
    }
    assert {action.action_id for action in bandit.candidates} <= {
        "safe_default",
        "cautious",
    }
    assert '"directive_round":4' in prompt.replace(" ", "")
    assert "station-7" not in prompt


def test_aggregate_summary_exposes_only_fixed_rates_and_directive_quality() -> None:
    proposal = ActionProposal(
        "local",
        "conflict",
        ("private update conflict",),
        resolve_action_ids(["cautious", "safe_default"]),
        "rule",
    )
    execution = SafeActionExecutor(build_action_library()["safe_default"]).select(
        proposal,
        (),
        ProbeBudget(2, 2),
        probe_enabled=False,
    )
    encoded = CohortSummaryCodec.encode(
        train_loss=0.5,
        val_mae=10.0,
        val_rmse=12.0,
        high_pollution_mae=11.0,
        update_norm=1.0,
        train_seconds=1.0,
        local_epochs=1,
        probe_batches=0,
        max_probe_batches=4,
        contribution_scale=1.0,
        clipping_violation=False,
        proposal=proposal,
        execution=execution,
        directive=_directive(),
    )
    decoded = CohortSummaryCodec.decode(encoded)
    serialized = json.dumps(decoded, sort_keys=True)
    assert decoded["cohort_directive_compliance_rate"] == pytest.approx(1.0)
    assert decoded["cohort_priority_alignment_rate"] == pytest.approx(1.0)
    assert "client" not in serialized.lower()
    assert "station" not in serialized.lower()
    assert np.isfinite(encoded).all()


def test_coordinator_history_contains_public_directive_without_client_identity() -> None:
    coordinator = AggregateCoordinatorAgent(2)
    coordinator.observe(
        1,
        2,
        {
            "cohort_val_macro_mae": 10.0,
            "diagnosis_rate_drift": 0.0,
            "diagnosis_rate_conflict": 0.5,
            "cohort_probe_gain_mean": 0.0,
            "cohort_directive_compliance_rate": 1.0,
        },
    )
    event = coordinator.history[-1]
    assert set(event["directive"]) == {
        "phase",
        "priority",
        "lr_scale_cap",
        "allow_adapt_fast",
        "allow_tail_focus",
        "directive_round",
    }
    serialized = json.dumps(event, sort_keys=True).lower()
    assert "client_id" not in serialized
    assert "station" not in serialized
