from __future__ import annotations

import pytest

from aqfl.experiments.run_baseline import run


def test_baseline_test_split_requires_frozen_protocol() -> None:
    with pytest.raises(RuntimeError, match="protocol-frozen"):
        run("persistence", "configs/base.yaml", 42, "test", protocol_frozen=False)


def test_baseline_smoke_epoch_override_is_validation_only() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        run("centralized_gru", "configs/base.yaml", 42, max_epochs=0)
    with pytest.raises(RuntimeError, match="validation-only"):
        run(
            "centralized_gru",
            "configs/base.yaml",
            42,
            evaluation_split="test",
            protocol_frozen=True,
            max_epochs=1,
        )


def test_baseline_screening_overrides_are_validation_only() -> None:
    with pytest.raises(ValueError, match="hidden-size"):
        run("centralized_gru", "configs/base.yaml", 42, hidden_size=0)
    with pytest.raises(ValueError, match="learning-rate"):
        run("centralized_gru", "configs/base.yaml", 42, learning_rate=0)
    with pytest.raises(RuntimeError, match="validation-only"):
        run(
            "centralized_gru",
            "configs/base.yaml",
            42,
            evaluation_split="test",
            protocol_frozen=True,
            hidden_size=32,
        )
