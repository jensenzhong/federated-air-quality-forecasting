"""Client-side shadow probes which never mutate the production model."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from aqfl.agents.v2_contracts import ClientAction, ProbeOutcome


def _proxy_mae(model: nn.Module, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    model.eval()
    total = 0.0
    examples = 0
    with torch.no_grad():
        for x, y in batches:
            prediction = model(x)
            total += float(torch.sum(torch.abs(prediction.reshape_as(y) - y)))
            examples += int(y.numel())
    return total / max(examples, 1)


def _limited_batches(
    dataset: Dataset[Any],
    batch_size: int,
    limit: int,
    *,
    shuffle: bool,
    seed: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if limit < 1:
        raise ValueError("Probe batch limit must be positive")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )
    batches = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        batches.append(batch)
    if not batches:
        raise ValueError("Probe dataset produced no batches")
    return batches


def _parameters_for_proximal(
    model: nn.Module,
    global_state: dict[str, torch.Tensor],
) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    for name, parameter in model.named_parameters():
        yield parameter, global_state[name].detach()


def probe_candidates(
    model: nn.Module,
    train_dataset: Dataset[Any],
    val_dataset: Dataset[Any],
    candidates: tuple[ClientAction, ...],
    *,
    base_lr: float,
    batch_size: int,
    weight_decay: float,
    global_state: dict[str, torch.Tensor],
    train_batches: int,
    val_batches: int,
    seed: int,
) -> tuple[ProbeOutcome, ...]:
    """Evaluate equal-cost candidate directions from the same immutable model state."""
    if not candidates:
        raise ValueError("At least one candidate is required for probing")
    train_data = _limited_batches(
        train_dataset, batch_size, train_batches, shuffle=True, seed=seed
    )
    val_data = _limited_batches(
        val_dataset, batch_size, val_batches, shuffle=False, seed=seed
    )
    baseline_loss = _proxy_mae(model, val_data)
    outcomes = []
    for action in candidates:
        shadow = copy.deepcopy(model)
        before = [parameter.detach().clone() for parameter in shadow.parameters()]
        optimizer = torch.optim.AdamW(
            shadow.parameters(),
            lr=base_lr * action.lr_scale,
            weight_decay=weight_decay,
        )
        criterion = nn.SmoothL1Loss()
        shadow.train()
        for x, y in train_data:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(shadow(x), y)
            if action.proximal_mu > 0:
                prox = sum(
                    torch.sum(torch.square(local - global_value))
                    for local, global_value in _parameters_for_proximal(shadow, global_state)
                )
                loss = loss + 0.5 * action.proximal_mu * prox
            loss.backward()
            optimizer.step()
        probed_loss = _proxy_mae(shadow, val_data)
        norm_squared = 0.0
        for original, updated in zip(before, shadow.parameters(), strict=True):
            norm_squared += float(torch.sum(torch.square(updated.detach() - original)))
        outcomes.append(
            ProbeOutcome(
                action.action_id,
                baseline_loss,
                probed_loss,
                baseline_loss - probed_loss,
                len(train_data),
                norm_squared**0.5,
            )
        )
        del shadow
    return tuple(outcomes)
