from __future__ import annotations

import torch
import torch.nn as nn


class ProjectorMLP(nn.Module):
    """Concatenate face embedding z with key k and project to latent z'."""

    def __init__(
        self,
        key_dim: int = 128,
        output_dim: int = 512,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = output_dim + key_dim  # z (512) + key
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Project concatenated [z, key] to z'."""
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        concat = torch.cat([z, key], dim=-1)
        return self.net(concat)

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)
