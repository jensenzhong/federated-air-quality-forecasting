from __future__ import annotations

import pytest

from aqfl.experiments.run_baseline import run


def test_baseline_test_split_requires_frozen_protocol() -> None:
    with pytest.raises(RuntimeError, match="protocol-frozen"):
        run("persistence", "configs/base.yaml", 42, "test", protocol_frozen=False)
