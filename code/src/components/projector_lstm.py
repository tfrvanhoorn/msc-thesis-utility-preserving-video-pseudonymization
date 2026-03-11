from __future__ import annotations

import torch
import torch.nn as nn


class ProjectorLSTM(nn.Module):
    """Bi-LSTM projector for temporal stability.

    Accepts facial embeddings shaped (B, Seq, Feat) and returns (B, Seq, Latent).
    When Seq=1, it behaves like a single-step sequence.
    """

    def __init__(
        self,
        key_dim: int = 128,
        output_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = output_dim + key_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            z = z.unsqueeze(1)
        if key.dim() == 1:
            key = key.unsqueeze(0)
        if key.dim() == 2:
            key = key.unsqueeze(1).expand(-1, z.shape[1], -1)
        elif key.dim() == 3 and key.shape[1] != z.shape[1]:
            key = key.expand(-1, z.shape[1], -1)

        if z.dim() != 3 or key.dim() != 3:
            raise ValueError(f"Expected z and key with 3 dims (B,Seq,Feat), got {z.shape} and {key.shape}")

        concat = torch.cat([z, key], dim=-1)
        lstm_out, _ = self.lstm(concat)
        projected = self.proj(lstm_out)
        return projected

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)
