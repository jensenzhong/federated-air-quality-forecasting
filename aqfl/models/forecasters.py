"""GRU main model and flattened-window MLP ablation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ForecastGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 31,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        return self.head(output[:, -1, :])


class ForecastMLP(nn.Module):
    def __init__(self, input_dim: int = 31, window: int = 24, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * window, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(config: dict[str, Any], name: str | None = None) -> nn.Module:
    model_cfg = config["model"]
    model_name = (name or model_cfg.get("name", "gru")).lower()
    if model_name == "gru":
        return ForecastGRU(
            input_dim=int(model_cfg["input_dim"]),
            hidden_size=int(model_cfg["hidden_size"]),
            num_layers=int(model_cfg["num_layers"]),
            dropout=float(model_cfg["dropout"]),
            head_hidden_size=int(model_cfg["head_hidden_size"]),
        )
    if model_name == "mlp":
        return ForecastMLP(
            input_dim=int(model_cfg["input_dim"]),
            window=int(config["data"]["window"]),
            dropout=float(model_cfg["dropout"]),
        )
    raise ValueError(f"Unsupported model: {model_name}")
