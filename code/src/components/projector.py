from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ProjectorMLP(nn.Module):
    """Projects face embedding z + key k to a z' limited to (-3, 3)."""

    def __init__(
        self,
        key_dim: int = 128,
        output_dim: int = 512,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = output_dim + key_dim
        
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        self.hidden_net = nn.Sequential(*layers)
        
        # The output section (This now represents the DELTA, not the full face)
        self.output_layer = nn.Linear(in_dim, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initializes weights for networks."""
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
        # --- THE MAGIC FIX: ZERO INITIALIZATION ---
        # We force the final layer to output exactly 0.0 at the start of training.
        if m is self.output_layer:
            init.constant_(m.weight, 0)
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if key.dim() == 1: key = key.unsqueeze(0)
        if z.dim() == 1: z = z.unsqueeze(0)
        
        # Normalize key to prevent Magnitude Mismatch
        key = F.normalize(key, p=2, dim=-1)
        
        concat = torch.cat([z, key], dim=-1)
        
        x = self.hidden_net(concat)
        
        # Calculate the shift (delta) caused by the key
        delta = self.output_layer(x)
        
        # --- THE RESIDUAL CONNECTION ---
        # Original Face + Shift = New Face
        out = z + delta

        # Restrict values strictly between -3 and 3
        out = 3.0 * torch.tanh(out / 3.0)

        return out

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)