"""Local and centralized PyTorch training helpers."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aqfl.data.preprocessing import GlobalScalerState, inverse_target
from aqfl.evaluation.metrics import regression_metrics


def train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    proximal_mu: float = 0.0,
    global_parameters: Iterable[torch.Tensor] | None = None,
) -> float:
    model.train()
    criterion = nn.SmoothL1Loss()
    reference = [tensor.detach().to(device) for tensor in global_parameters] if global_parameters else None
    total_loss = 0.0
    examples = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = criterion(prediction, y)
        if proximal_mu > 0 and reference is not None:
            prox = sum(torch.sum(torch.square(local - global_value)) for local, global_value in zip(model.parameters(), reference, strict=False))
            loss = loss + 0.5 * proximal_mu * prox
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(x)
        examples += len(x)
    return total_loss / max(examples, 1)


def train_local_model(
    model: nn.Module,
    dataset: Dataset[Any],
    epochs: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
    device: torch.device | None = None,
    proximal_mu: float = 0.0,
    global_state: dict[str, torch.Tensor] | None = None,
) -> float:
    device = device or torch.device("cpu")
    model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    global_parameters = None
    if global_state is not None:
        global_parameters = [global_state[name] for name, _ in model.named_parameters()]
    loss = 0.0
    for _ in range(epochs):
        loss = train_epoch(model, loader, optimizer, device, proximal_mu, global_parameters)
    return loss


def predict_scaled(model: nn.Module, dataset: Dataset[Any], batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(x).cpu().numpy())
    return np.concatenate(predictions).reshape(-1) if predictions else np.empty(0, dtype=np.float32)


def evaluate_model(
    model: nn.Module,
    dataset: Any,
    scaler: GlobalScalerState,
    pollution_threshold: float | None = None,
    batch_size: int = 256,
) -> tuple[dict[str, float], np.ndarray]:
    prediction_scaled = predict_scaled(model, dataset, batch_size)
    prediction_raw = inverse_target(prediction_scaled, scaler)
    y_true = np.asarray(dataset.y_raw).reshape(-1)
    return regression_metrics(y_true, prediction_raw, pollution_threshold), prediction_raw


def fit_with_early_stopping(
    model: nn.Module,
    train_dataset: Dataset[Any],
    val_datasets: dict[str, Any],
    scaler: GlobalScalerState,
    epochs: int,
    patience: int,
    learning_rate: float,
    batch_size: int,
    weight_decay: float,
) -> tuple[nn.Module, list[dict[str, float]]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    best_state = copy.deepcopy(model.state_dict())
    best_macro_mae = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, torch.device("cpu"))
        station_maes = [evaluate_model(model, dataset, scaler)[0]["mae"] for dataset in val_datasets.values()]
        macro_mae = float(np.mean(station_maes))
        history.append({"epoch": float(epoch), "train_loss": train_loss, "val_macro_mae": macro_mae})
        if macro_mae < best_macro_mae:
            best_macro_mae = macro_mae
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    model.load_state_dict(best_state)
    return model, history
