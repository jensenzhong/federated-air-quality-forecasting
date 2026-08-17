from __future__ import annotations

import pytest

from aqfl.privacy import (
    assert_prompt_keys_are_private_safe,
    enforce_client_level_llm_policy,
    enforce_pafa_run_mode,
    is_approved_private_llm_endpoint,
    is_loopback_llm_endpoint,
)


def config(base_url: str) -> dict:
    return {
        "llm": {"base_url": base_url},
        "privacy": {
            "mode": "strict_federated",
        },
    }


def test_client_level_llm_must_be_loopback_or_on_prem() -> None:
    assert is_loopback_llm_endpoint("http://127.0.0.1:11434/v1")
    assert is_loopback_llm_endpoint("http://localhost:8000/v1")
    assert not is_loopback_llm_endpoint("https://api.deepseek.com")
    assert is_approved_private_llm_endpoint("http://10.0.0.8:8000/v1", [])
    assert is_approved_private_llm_endpoint("https://llm.institution.example/v1", ["llm.institution.example"])
    assert not is_approved_private_llm_endpoint("https://api.deepseek.com", [])
    enforce_client_level_llm_policy(config("http://127.0.0.1:11434/v1"))
    with pytest.raises(RuntimeError, match="cannot be sent to an external LLM"):
        enforce_client_level_llm_policy(config("https://api.deepseek.com"))
    legacy_bypass = config("https://api.deepseek.com")
    legacy_bypass["privacy"]["allow_external_client_level_llm"] = True
    with pytest.raises(RuntimeError, match="cannot be sent to an external LLM"):
        enforce_client_level_llm_policy(legacy_bypass)


def test_prompt_firewall_rejects_raw_temporal_and_test_fields() -> None:
    assert_prompt_keys_are_private_safe({"macro_mae": 1.0, "action_counts": [1, 2]})
    for key in ("raw_rows", "target_timestamp", "residual_series", "test_mae"):
        with pytest.raises(ValueError, match="Forbidden private prompt field"):
            assert_prompt_keys_are_private_safe({key: [1]})


def test_pafa_requires_secagg_and_blocks_insecure_test_or_formal_modes() -> None:
    enforce_pafa_run_mode("fedprox", evaluation_split="test", protocol_frozen=True)
    with pytest.raises(RuntimeError, match="code-review-only"):
        enforce_pafa_run_mode("pafa_rule", evaluation_split="val", protocol_frozen=False)
    with pytest.raises(RuntimeError, match="SecAgg"):
        enforce_pafa_run_mode("pafa_llm", evaluation_split="test", protocol_frozen=True)
    with pytest.raises(RuntimeError, match="code-review-only"):
        enforce_pafa_run_mode("pafa_bandit", evaluation_split="val", protocol_frozen=True)
    enforce_pafa_run_mode(
        "pafa_rule",
        evaluation_split="val",
        protocol_frozen=False,
        secure_aggregation_active=True,
    )
    with pytest.raises(RuntimeError, match="test evaluation remains blocked"):
        enforce_pafa_run_mode(
            "pafa_rule",
            evaluation_split="test",
            protocol_frozen=True,
            secure_aggregation_active=True,
            client_state_isolated=True,
        )
    with pytest.raises(RuntimeError, match="institution-isolated"):
        enforce_pafa_run_mode(
            "pafa_rule",
            evaluation_split="val",
            protocol_frozen=True,
            secure_aggregation_active=True,
            client_state_isolated=False,
        )
