from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    Metadata,
    MetricRecord,
    RecordDict,
)
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from flwr.common import recorddict_compat as compat
from flwr.common.secure_aggregation.secaggplus_constants import (
    RECORD_KEY_CONFIGS,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    Key as SecAggKey,
)
from flwr.common.secure_aggregation.secaggplus_constants import (
    Stage as SecAggStage,
)
from torch.utils.data import TensorDataset

from aqfl.agents.action_library import build_action_library, resolve_action_ids
from aqfl.agents.executor import ProbeBudget, SafeActionExecutor
from aqfl.agents.memory import EpisodicMemory
from aqfl.agents.v2_contracts import (
    ActionProposal,
    ClientAction,
    ClientStateCapsule,
    CreditRecord,
    ProbeOutcome,
)
from aqfl.agents.v2_proposers import (
    ContextualBanditProposer,
    LLMActionProposer,
    RuleActionProposer,
)
from aqfl.federated.agentic_strategy import AgenticAggregationStrategy
from aqfl.federated.client_app import _would_clip_parameters
from aqfl.federated.probe_runtime import probe_candidates
from aqfl.federated.secure_aggregation import (
    SECAGG_SESSION_GUARD,
    AggregateCoordinatorAgent,
    CohortSummaryCodec,
    SecureAggregateOnlyFedAvg,
    private_secaggplus_mod,
    sanitize_secagg_collect_reply,
    validate_secagg_numeric_policy,
)


def capsule(client_id: str = "0", round_number: int = 1, **overrides: float) -> ClientStateCapsule:
    values = {
        "val_mae": 10.0,
        "val_rmse": 12.0,
        "high_pollution_mae": 11.0,
        "train_loss": 0.5,
        "update_norm": 1.0,
        "update_cosine": 0.8,
        "mae_ema": 10.0,
        "mae_slope": 0.0,
        "mae_oscillation": 0.0,
        "drift_score": 0.0,
        "previous_realized_gain": 0.0,
        "train_seconds": 1.0,
    }
    values.update(overrides)
    return ClientStateCapsule(
        client_id=client_id,
        round_number=round_number,
        previous_action_id="none",
        local_epochs=1,
        **values,
    )


def _setup_message(*, run_id: int = 1, dst_node_id: int = 1, round_number: int = 1) -> Message:
    content = RecordDict({
        RECORD_KEY_CONFIGS: ConfigRecord({
            SecAggKey.STAGE: SecAggStage.SETUP,
            SecAggKey.SAMPLE_NUMBER: 3,
            SecAggKey.SHARE_NUMBER: 3,
            SecAggKey.THRESHOLD: 2,
            SecAggKey.CLIPPING_RANGE: 8.0,
            SecAggKey.TARGET_RANGE: 2**22,
            SecAggKey.MOD_RANGE: 2**32,
            SecAggKey.MAX_WEIGHT: 1.0,
        })
    })
    return Message(
        content=content,
        metadata=Metadata(
            run_id=run_id,
            message_id="setup-instruction",
            src_node_id=0,
            dst_node_id=dst_node_id,
            reply_to_message_id="",
            group_id=str(round_number),
            created_at=0.0,
            ttl=3600.0,
            message_type="train",
        ),
    )


def _private_context() -> Context:
    return Context(
        run_id=1,
        node_id=1,
        node_config={},
        state=RecordDict(),
        run_config={"method": "pafa_rule"},
    )


def test_action_contract_fails_closed() -> None:
    library = build_action_library()
    assert set(library) == {"safe_default", "cautious", "adapt_fast", "tail_focus"}
    with pytest.raises(ValueError, match="Invalid lr_scale"):
        ClientAction("freeform", 2.0, 1, 0.01, "normal", "invalid")
    with pytest.raises(ValueError, match="Unknown action"):
        resolve_action_ids(["invented"])
    with pytest.raises(ValueError, match="unexpected"):
        ClientAction.from_dict({**library["safe_default"].to_dict(), "raw_weight": 0.9})


def test_safe_executor_selects_verified_gain_and_falls_back() -> None:
    library = build_action_library()
    proposal = ActionProposal(
        "0",
        "underfit",
        ("mae slope is negative",),
        resolve_action_ids(["adapt_fast", "safe_default"]),
        "rule",
    )
    outcomes = (
        ProbeOutcome("adapt_fast", 1.0, 0.8, 0.2, 2, 0.1),
        ProbeOutcome("safe_default", 1.0, 0.9, 0.1, 2, 0.1),
    )
    budget = ProbeBudget(2, 2)
    selected = SafeActionExecutor(library["safe_default"], minimum_gain=0.05).select(
        proposal, outcomes, budget
    )
    assert selected.accepted
    assert selected.selected_action.action_id == "adapt_fast"
    assert budget.consumed_batches == 4

    rejected = SafeActionExecutor(library["safe_default"], minimum_gain=0.3).select(
        proposal, outcomes, ProbeBudget(2, 2)
    )
    assert not rejected.accepted
    assert rejected.selected_action.action_id == "safe_default"


def test_probe_budget_and_exact_candidate_set_are_enforced() -> None:
    proposal = ActionProposal(
        "0",
        "stable",
        ("stable",),
        resolve_action_ids(["safe_default", "adapt_fast"]),
        "rule",
    )
    executor = SafeActionExecutor(build_action_library()["safe_default"])
    with pytest.raises(ValueError, match="exactly match"):
        executor.select(
            proposal,
            (ProbeOutcome("safe_default", 1, 0.9, 0.1, 1, 0.1),),
            ProbeBudget(2, 1),
        )
    with pytest.raises(RuntimeError, match="budget exceeded"):
        executor.select(
            proposal,
            (
                ProbeOutcome("safe_default", 1, 0.9, 0.1, 2, 0.1),
                ProbeOutcome("adapt_fast", 1, 0.8, 0.2, 2, 0.1),
            ),
            ProbeBudget(2, 1),
        )


def test_no_probe_ablation_is_explicit_and_has_zero_probe_cost() -> None:
    proposal = ActionProposal(
        "0", "stable", ("stable",), resolve_action_ids(["adapt_fast"]), "llm"
    )
    budget = ProbeBudget(1, 2)
    result = SafeActionExecutor(build_action_library()["safe_default"]).select(
        proposal, (), budget, probe_enabled=False
    )
    assert result.accepted
    assert result.probe_outcomes == ()
    assert result.selected_action.action_id == "adapt_fast"
    assert budget.consumed_batches == 0


def test_memory_builds_trajectory_and_credit_without_raw_series() -> None:
    memory = EpisodicMemory(max_records_per_client=3)
    first = memory.build_capsule(
        client_id="0",
        round_number=1,
        val_mae=10,
        val_rmse=12,
        high_pollution_mae=13,
        train_loss=1,
        update_norm=2,
        update_cosine=0.5,
        train_seconds=1,
        local_epochs=1,
    )
    memory.add_capsule(first)
    memory.add_credit(CreditRecord("0", 2, "cautious", 0.1, 0.2, True))
    second = memory.build_capsule(
        client_id="0",
        round_number=2,
        val_mae=9,
        val_rmse=11,
        high_pollution_mae=12,
        train_loss=0.9,
        update_norm=1.5,
        update_cosine=0.7,
        train_seconds=1,
        local_epochs=1,
    )
    assert second.mae_slope == -1
    assert second.previous_action_id == "cautious"
    assert "raw" not in json.dumps(second.to_dict()).lower()


def test_rule_and_bandit_share_action_library_and_bandit_learns() -> None:
    capsules = [capsule("0"), capsule("1", drift_score=0.3)]
    memory = EpisodicMemory()
    rules = RuleActionProposer().propose(1, capsules, memory)
    assert rules["1"].diagnosis == "drift"
    assert rules["1"].candidates[0].action_id == "cautious"

    bandit = ContextualBanditProposer(alpha=0.1)
    before = bandit.a.copy()
    proposals = bandit.propose(1, capsules, memory)
    chosen = proposals["0"].candidates[0].action_id
    bandit.observe(CreditRecord("0", 1, chosen, 0.0, 0.5, True))
    assert not np.array_equal(before, bandit.a)
    assert all(
        action.action_id in build_action_library()
        for proposal in proposals.values()
        for action in proposal.candidates
    )


def test_llm_parser_rejects_invented_actions_and_missing_clients() -> None:
    capsules = [capsule("0")]
    valid = {
        "proposals": [
            {
                "client_id": "0",
                "diagnosis": "stable",
                "evidence": ["mae_slope is zero"],
                "candidate_action_ids": ["safe_default"],
            }
        ]
    }
    parsed = LLMActionProposer._parse(valid, capsules, "llm", "hash")
    assert parsed["0"].prompt_hash == "hash"
    invalid = json.loads(json.dumps(valid))
    invalid["proposals"][0]["candidate_action_ids"] = ["raw_weight_0.9"]
    with pytest.raises(ValueError, match="Unknown action"):
        LLMActionProposer._parse(invalid, capsules, "llm", "hash")
    with pytest.raises(ValueError, match="client set mismatch"):
        LLMActionProposer._parse({"proposals": []}, capsules, "llm", "hash")
    leaked = json.loads(json.dumps(valid))
    leaked["proposals"][0]["evidence"] = ["test_mae improved"]
    with pytest.raises(ValueError, match="Test information"):
        LLMActionProposer._parse(leaked, capsules, "llm", "hash")


def test_llm_prompt_contains_no_test_fields() -> None:
    prompt = LLMActionProposer.build_prompt(1, [capsule("0")], EpisodicMemory())
    assert "test_mae" not in prompt
    assert "test_rmse" not in prompt


def test_cohort_summary_codec_contains_rates_but_no_client_identity() -> None:
    proposal = ActionProposal(
        "local",
        "underfit",
        ("local slope",),
        resolve_action_ids(["adapt_fast"]),
        "llm",
    )
    execution = SafeActionExecutor(build_action_library()["safe_default"]).select(
        proposal,
        (),
        ProbeBudget(1, 1),
        probe_enabled=False,
    )
    encoded = CohortSummaryCodec.encode(
        train_loss=0.5,
        val_mae=10.0,
        val_rmse=12.0,
        high_pollution_mae=14.0,
        update_norm=2.0,
        train_seconds=30.0,
        local_epochs=1,
        probe_batches=0,
        max_probe_batches=6,
        contribution_scale=1.0,
        clipping_violation=False,
        proposal=proposal,
        execution=execution,
    )
    decoded = CohortSummaryCodec.decode(encoded)
    assert decoded["cohort_val_macro_mae"] == pytest.approx(10.0, abs=1e-4)
    assert decoded["action_rate_adapt_fast"] == pytest.approx(1.0)
    assert "client" not in json.dumps(decoded).lower()


def test_aggregate_coordinator_enforces_minimum_cohort() -> None:
    coordinator = AggregateCoordinatorAgent(10)
    with pytest.raises(RuntimeError, match="cohort >= 10"):
        coordinator.observe(1, 9, {"cohort_val_macro_mae": 10.0})
    signal = coordinator.observe(1, 12, {"cohort_val_macro_mae": 10.0})
    assert set(signal) == {"cohort-phase", "cohort-lr-scale-cap", "cohort-round"}


def test_secagg_reply_sanitizer_removes_client_metadata() -> None:
    fitres = FitRes(
        status=Status(Code.OK, "private station failure detail"),
        parameters=ndarrays_to_parameters([np.asarray([1.0], dtype=np.float32)]),
        num_examples=123,
        metrics={"val_mae": 9.0, "station": "private"},
    )
    content = compat.fitres_to_recorddict(fitres, True)
    content.metric_records["private"] = MetricRecord({"update_norm": 2.0})
    content.config_records["private"] = ConfigRecord({"client_id": "7"})
    sanitize_secagg_collect_reply(content)
    sanitized = compat.recorddict_to_fitres(content, True)
    assert sanitized.num_examples == 1
    assert sanitized.metrics == {}
    assert sanitized.status.code == Code.OK
    assert sanitized.status.message == ""
    assert set(content.metric_records) == {"fitres.num_examples"}
    assert "private" not in content.config_records


def test_pafa_client_request_without_secagg_protocol_is_rejected() -> None:
    content = RecordDict({
        "fitins.parameters": ArrayRecord({
            "weight": Array(np.asarray([1.0], dtype=np.float32))
        }),
        "fitins.config": ConfigRecord({"method": "pafa_rule"}),
    })
    message = Message(
        content=content,
        dst_node_id=1,
        message_type="train",
    )
    context = Context(
        run_id=1,
        node_id=1,
        node_config={},
        state=RecordDict(),
        run_config={},
    )
    called = False

    def call_next(msg: Message, ctxt: Context) -> Message:
        del ctxt
        nonlocal called
        called = True
        return Message(content=RecordDict(), reply_to=msg)

    with pytest.raises(RuntimeError, match="omitted the Flower SecAgg\\+"):
        private_secaggplus_mod(message, context, call_next)
    assert not called


def test_pafa_secagg_setup_replay_is_rejected() -> None:
    context = _private_context()

    def no_train(msg: Message, ctxt: Context) -> Message:
        del msg, ctxt
        raise AssertionError("setup must not invoke the train handler")

    private_secaggplus_mod(_setup_message(), context, no_train)
    with pytest.raises(RuntimeError, match="setup replay"):
        private_secaggplus_mod(_setup_message(), context, no_train)


@pytest.mark.parametrize(
    ("run_id", "dst_node_id", "error"),
    (
        (2, 1, "different Flower run"),
        (1, 2, "destination does not match"),
    ),
)
def test_pafa_secagg_setup_requires_run_and_node_binding(
    run_id: int,
    dst_node_id: int,
    error: str,
) -> None:
    with pytest.raises(RuntimeError, match=error):
        private_secaggplus_mod(
            _setup_message(run_id=run_id, dst_node_id=dst_node_id),
            _private_context(),
            lambda msg, ctxt: Message(content=RecordDict(), reply_to=msg),
        )


def test_pafa_secagg_rejects_stage_reordering() -> None:
    message = _setup_message()
    message.content.config_records[RECORD_KEY_CONFIGS][SecAggKey.STAGE] = (
        SecAggStage.SHARE_KEYS
    )
    with pytest.raises(RuntimeError, match="stage replay or reordering"):
        private_secaggplus_mod(
            message,
            _private_context(),
            lambda msg, ctxt: Message(content=RecordDict(), reply_to=msg),
        )


def test_pafa_secagg_collect_binds_fit_round_to_group() -> None:
    context = _private_context()
    context.state.config_records[SECAGG_SESSION_GUARD] = ConfigRecord({
        "last_completed_round": 0,
        "active_round": 1,
        "last_stage": SecAggStage.SHARE_KEYS,
    })
    message = _setup_message()
    message.content.config_records[RECORD_KEY_CONFIGS] = ConfigRecord({
        SecAggKey.STAGE: SecAggStage.COLLECT_MASKED_VECTORS,
        SecAggKey.CIPHERTEXT_LIST: [],
        SecAggKey.SOURCE_LIST: [],
    })
    message.content.config_records["fitins.config"] = ConfigRecord({
        "method": "pafa_rule",
        "server-round": 2,
    })
    message.content.array_records["fitins.parameters"] = ArrayRecord({
        "weight": Array(np.asarray([1.0], dtype=np.float32))
    })
    with pytest.raises(RuntimeError, match="collect round binding mismatch"):
        private_secaggplus_mod(
            message,
            context,
            lambda msg, ctxt: Message(content=RecordDict(), reply_to=msg),
        )


def test_secure_strategy_rejects_duplicate_identity_and_round_replay() -> None:
    proposal = ActionProposal(
        "local",
        "stable",
        ("stable",),
        resolve_action_ids(["safe_default"]),
        "rule",
    )
    execution = SafeActionExecutor(build_action_library()["safe_default"]).select(
        proposal,
        (),
        ProbeBudget(1, 1),
        probe_enabled=False,
    )
    summary = CohortSummaryCodec.encode(
        train_loss=0.5,
        val_mae=10.0,
        val_rmse=12.0,
        high_pollution_mae=14.0,
        update_norm=2.0,
        train_seconds=30.0,
        local_epochs=1,
        probe_batches=0,
        max_probe_batches=6,
        contribution_scale=1.0,
        clipping_violation=False,
        proposal=proposal,
        execution=execution,
    )
    result = FitRes(
        status=Status(Code.OK, ""),
        parameters=ndarrays_to_parameters(
            [np.asarray([1.0], dtype=np.float32), summary]
        ),
        num_examples=1,
        metrics={},
    )
    strategy = SecureAggregateOnlyFedAvg(
        expected_clients=2,
        expected_node_ids={1, 2},
        model_array_count=1,
        coordinator=AggregateCoordinatorAgent(2),
        on_fit_config_fn=lambda round_number: {"server-round": round_number},
    )
    duplicate = [(SimpleNamespace(node_id=1), result)] * 2
    with pytest.raises(RuntimeError, match="identities must be unique and complete"):
        strategy.aggregate_fit(1, duplicate, [])

    complete = [
        (SimpleNamespace(node_id=node_id), result)
        for node_id in (1, 2)
    ]
    strategy.aggregate_fit(1, complete, [])
    with pytest.raises(RuntimeError, match="round replay or skip"):
        strategy.aggregate_fit(1, complete, [])


def test_secagg_numeric_policy_enforces_capacity_and_equal_weighting() -> None:
    policy = {
        "clipping_range": 8.0,
        "quantization_range": 2**22,
        "modulus_range": 2**32,
        "max_weight": 1.0,
    }
    assert validate_secagg_numeric_policy(policy, 12) == pytest.approx(2 * 8.0 / 2**22)
    with pytest.raises(RuntimeError, match="modulus capacity"):
        validate_secagg_numeric_policy({**policy, "modulus_range": 2**24}, 12)
    with pytest.raises(RuntimeError, match="max_weight=1"):
        validate_secagg_numeric_policy({**policy, "max_weight": 12.0}, 12)


def test_client_detects_parameter_clipping_and_nonfinite_values() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(7.5)
    assert not _would_clip_parameters(model, 8.0)
    with torch.no_grad():
        model.weight.fill_(8.5)
    assert _would_clip_parameters(model, 8.0)
    with torch.no_grad():
        model.weight.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        _would_clip_parameters(model, 8.0)


def test_array_record_smoke_for_strategy_dependencies() -> None:
    arrays = ArrayRecord({"weight": Array(np.asarray([1.0], dtype=np.float32))})
    config = ConfigRecord({"lr": 0.001})
    assert arrays["weight"].numpy().item() == pytest.approx(1.0)
    assert config["lr"] == pytest.approx(0.001)


def test_rejected_server_side_agentic_strategy_fails_closed(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="Rejected privacy architecture"):
        AgenticAggregationStrategy(
            proposer=RuleActionProposer(),
            expected_clients=2,
            base_lr=0.001,
            event_log=tmp_path / "events.jsonl",
        )


def test_shadow_probe_is_equal_budget_and_does_not_mutate_production_model() -> None:
    x = torch.arange(8, dtype=torch.float32).reshape(-1, 1) / 8
    y = 2 * x
    dataset = TensorDataset(x, y)
    model = torch.nn.Linear(1, 1)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    outcomes = probe_candidates(
        model,
        dataset,
        dataset,
        resolve_action_ids(["safe_default", "adapt_fast"]),
        base_lr=0.01,
        batch_size=2,
        weight_decay=0.0,
        global_state=original,
        train_batches=2,
        val_batches=2,
        seed=42,
    )
    assert len(outcomes) == 2
    assert {outcome.cost_batches for outcome in outcomes} == {2}
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])
