from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.init as init

class ProjectorMLP(nn.Module):
    """Projecteert face embedding z + key k naar een begrensde z' via Tanh."""

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
        
        # Hidden Layers met ReLU
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
            
        # Laatste laag met Tanh voor begrenzing rond 0
        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.Tanh()) 
        
        self.net = nn.Sequential(*layers)
        
        # Pas initialisatie toe
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialiseert gewichten op basis van de activatiefunctie."""
        if isinstance(m, nn.Linear):
            # Gebruik Xavier voor de output laag (omdat deze Tanh gebruikt)
            # Gebruik Kaiming voor hidden layers (omdat deze ReLU gebruiken)
            if m is self.net[-2]: # De laatste Linear laag voor de Tanh
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('tanh'))
            else:
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if key.dim() == 1: key = key.unsqueeze(0)
        if z.dim() == 1: z = z.unsqueeze(0)
        concat = torch.cat([z, key], dim=-1)
        return self.net(concat)

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)