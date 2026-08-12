from __future__ import annotations

import json

import pytest

from aqfl.agents.decision import Decision
from aqfl.agents.llm import LLMPlanningAgent
from aqfl.agents.replay import BudgetReplayPlanner
from aqfl.agents.rule import RulePlanningAgent


def llm_config() -> dict:
    return {
        "llm": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY_TEST_ONLY",
            "temperature": 0,
            "call_every_n_rounds": 2,
        }
    }


def test_decision_schema_rejects_out_of_bounds() -> None:
    valid = Decision.from_dict(
        {"strategy": "hybrid", "lr_scale": 1.5, "local_epochs": 2, "reason": "valid"}
    )
    assert valid.strategy == "hybrid"
    with pytest.raises(ValueError):
        Decision("size_only", 2.0, 1, "bad", "", "rule")
    with pytest.raises(ValueError):
        Decision.from_dict({"strategy": "size_only"})


def test_rule_agent_thresholds_and_budgets() -> None:
    agent = RulePlanningAgent()
    fairness = agent.choose(1, [], {"macro_mae": 10, "worst_station_mae": 13, "station_mae_cv": 0})
    assert fairness.strategy == "fairness_clip"
    worsening = agent.choose(
        4,
        [{"macro_mae": 10}, {"macro_mae": 11}, {"macro_mae": 12}],
        {"macro_mae": 12, "worst_station_mae": 12, "station_mae_cv": 0},
    )
    assert worsening.lr_scale == 0.5
    assert worsening.local_epochs == 1
    drift = agent.choose(1, [], {"macro_mae": 10, "worst_station_mae": 10, "station_mae_cv": 0, "update_norm_cv": 0.6})
    assert drift.strategy == "hybrid"


def test_llm_forced_exploration_and_prompt_firewall(tmp_path) -> None:
    agent = LLMPlanningAgent(llm_config(), tmp_path)
    assert [agent.choose(i, [], {}).strategy for i in range(1, 5)] == [
        "size_only",
        "perf_only",
        "hybrid",
        "fairness_clip",
    ]
    prompt = agent.build_prompt(
        5,
        [{"macro_mae": 2, "test_mae": 999, "station_mae_cv": 0.2}],
        {"macro_mae": 1, "test_rmse": 999, "worst_station_mae": 3, "val_worst_mae": 3},
        include_fairness=False,
    )
    assert "test_mae" not in prompt
    assert "test_rmse" not in prompt
    assert "station_mae_cv" not in prompt
    assert "worst_station_mae" not in prompt
    assert "val_worst_mae" not in prompt


def test_llm_cache_replay_without_api(tmp_path) -> None:
    agent = LLMPlanningAgent(llm_config(), tmp_path)
    prompt = agent.build_prompt(5, [], {"macro_mae": 1})
    import hashlib

    key = hashlib.sha256(prompt.encode()).hexdigest()
    payload = {
        "decision": {
            "strategy": "hybrid",
            "lr_scale": 1,
            "local_epochs": 1,
            "reason": "cached",
        }
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    decision = agent.choose(5, [], {"macro_mae": 1})
    assert decision.source == "cache"
    assert decision.prompt_hash == key


def test_llm_invalid_or_missing_api_fallback_and_strict(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY_TEST_ONLY", raising=False)
    relaxed = LLMPlanningAgent(llm_config(), tmp_path / "relaxed", strict=False)
    assert relaxed.choose(5, [], {"macro_mae": 1, "worst_station_mae": 1}).source == "fallback"
    strict = LLMPlanningAgent(llm_config(), tmp_path / "strict", strict=True)
    with pytest.raises(RuntimeError, match="Formal LLM decision failed"):
        strict.choose(5, [], {"macro_mae": 1})


def test_budget_replay_requires_complete_trace(tmp_path) -> None:
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        json.dumps({"round": 1, "lr_scale": 1.5, "local_epochs": 2, "prompt_hash": "abc"}) + "\n",
        encoding="utf-8",
    )
    planner = BudgetReplayPlanner(path)
    decision = planner.choose(1, [], {})
    assert decision.strategy == "size_only"
    assert decision.lr_scale == 1.5
    with pytest.raises(ValueError, match="missing round"):
        planner.choose(2, [], {})
