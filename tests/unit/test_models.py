from __future__ import annotations

import copy

import torch
from torch.utils.data import TensorDataset

from aqfl.data.preprocessing import GlobalScalerState
from aqfl.models.forecasters import ForecastGRU, ForecastMLP, build_model
from aqfl.models.training import evaluate_model, fit_with_early_stopping, train_local_model


def test_model_shapes_and_serialization() -> None:
    x = torch.randn(4, 24, 31)
    gru = ForecastGRU(hidden_size=16)
    mlp = ForecastMLP()
    assert gru(x).shape == (4, 1)
    assert mlp(x).shape == (4, 1)
    clone = ForecastGRU(hidden_size=16)
    clone.load_state_dict(gru.state_dict())
    gru.eval()
    clone.eval()
    assert torch.allclose(gru(x), clone(x))


def test_local_training_and_fedprox_are_finite() -> None:
    x = torch.randn(8, 24, 31)
    y = torch.randn(8, 1)
    dataset = TensorDataset(x, y)
    model = ForecastGRU(hidden_size=8)
    global_state = copy.deepcopy(model.state_dict())
    loss = train_local_model(model, dataset, 1, 0.001, 4, 1e-4, proximal_mu=0.01, global_state=global_state)
    assert loss >= 0
    assert torch.isfinite(torch.tensor(loss))


class EvaluationDataset(TensorDataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        super().__init__(x, y)
        self.y_raw = y.numpy()


def test_evaluation_and_early_stopping() -> None:
    x = torch.randn(8, 24, 31)
    y = torch.randn(8, 1)
    dataset = EvaluationDataset(x, y)
    scaler = GlobalScalerState([f"f{i}" for i in range(31)], [0] * 31, [1] * 31, 0, 1, 1)
    model = ForecastGRU(hidden_size=8)
    metrics, predictions = evaluate_model(model, dataset, scaler, batch_size=4)
    assert len(predictions) == 8
    assert metrics["num_examples"] == 8
    fitted, history = fit_with_early_stopping(model, dataset, {"a": dataset}, scaler, 2, 1, 0.001, 4, 0)
    assert isinstance(fitted, ForecastGRU)
    assert 1 <= len(history) <= 2


def test_model_factory() -> None:
    config = {
        "data": {"window": 24},
        "model": {"name": "gru", "input_dim": 31, "hidden_size": 8, "num_layers": 2, "dropout": 0.1, "head_hidden_size": 4},
    }
    assert isinstance(build_model(config), ForecastGRU)
    assert isinstance(build_model(config, "mlp"), ForecastMLP)
    import pytest

    with pytest.raises(ValueError):
        build_model(config, "transformer")
