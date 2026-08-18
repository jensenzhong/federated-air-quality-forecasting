from __future__ import annotations

import numpy as np
import pytest
import torch
from flwr.app import ArrayRecord

from aqfl.federated.client_app import _state_dict_from_array_record


def test_client_array_schema_restores_named_state_dict() -> None:
    model = torch.nn.Linear(2, 1)
    named = ArrayRecord.from_torch_state_dict(model.state_dict())
    restored = _state_dict_from_array_record(model, named)
    assert list(restored) == ["weight", "bias"]


def test_client_array_schema_maps_secagg_numeric_order() -> None:
    model = torch.nn.Linear(2, 1)
    numeric = ArrayRecord.from_numpy_ndarrays(
        [np.ones((1, 2), dtype=np.float32), np.zeros(1, dtype=np.float32)]
    )
    restored = _state_dict_from_array_record(model, numeric)
    assert list(restored) == ["weight", "bias"]
    assert float(restored["weight"].sum()) == pytest.approx(2.0)
    assert float(restored["bias"].sum()) == pytest.approx(0.0)


def test_client_array_schema_rejects_unknown_keys() -> None:
    model = torch.nn.Linear(2, 1)
    numeric = ArrayRecord({"unexpected": numeric_array()})
    with pytest.raises(RuntimeError, match="do not match"):
        _state_dict_from_array_record(model, numeric)


def numeric_array():
    from flwr.app import Array

    return Array(np.zeros(1, dtype=np.float32))
