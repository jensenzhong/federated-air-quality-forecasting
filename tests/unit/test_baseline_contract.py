from __future__ import annotations

import pytest

from aqfl.federated.baseline_contract import (
    baseline_contract,
    require_formal_baseline,
)


def test_only_secagg_verified_methods_are_formal_eligible() -> None:
    assert require_formal_baseline("pafa_rule").formal_eligible
    assert require_formal_baseline("pafa_llm").budget_class == "pafa_equal_client_probe_budget"
    assert require_formal_baseline("pafa_fedprox").family == "secure_baseline"
    assert require_formal_baseline("pafa_fedadam").formal_eligible
    assert require_formal_baseline("pafa_bandit_fedadam").formal_eligible
    assert require_formal_baseline("pafa_fedprox_budget_matched").formal_eligible
    with pytest.raises(RuntimeError, match="pending_secagg_adapter"):
        require_formal_baseline("fedprox")


def test_strong_baseline_contracts_do_not_silently_expose_client_updates() -> None:
    scaffold = baseline_contract("scaffold")
    assert scaffold.protocol_compatibility == "conditional"
    assert not scaffold.requires_per_client_update_visibility
    selective = baseline_contract("selective_collaboration")
    assert selective.formal_status == "incompatible_client_signal"
    assert selective.requires_client_level_server_signal


def test_unknown_baseline_fails_closed() -> None:
    with pytest.raises(ValueError, match="No baseline protocol contract"):
        baseline_contract("made_up_baseline")
