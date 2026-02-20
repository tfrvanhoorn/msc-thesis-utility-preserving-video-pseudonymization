from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.init as init

logger = logging.getLogger(__name__)

class ProjectorMLP(nn.Module):
    """Projecteert face embedding z + key k naar een z' gelimiteerd tot (-3, 3)."""

    def __init__(
        self,
        key_dim: int = 128,
        output_dim: int = 512,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = output_dim + key_dim
        
        # We bouwen de hidden layers apart
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        self.hidden_net = nn.Sequential(*layers)
        
        # De output sectie
        self.output_layer = nn.Linear(in_dim, output_dim)
        
        # --- OUDE IMPLEMENTATIE: LayerNorm (veroorzaakte uitschieters) ---
        # self.norm = nn.LayerNorm(output_dim, elementwise_affine=True)
        
        # Pas initialisatie toe
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initialiseert gewichten voor netwerken."""
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
                
        # --- OUDE IMPLEMENTATIE: LayerNorm init ---
        # elif isinstance(m, nn.LayerNorm):
        #     init.constant_(m.bias, 0)
        #     init.constant_(m.weight, 1)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if key.dim() == 1: key = key.unsqueeze(0)
        if z.dim() == 1: z = z.unsqueeze(0)
        
        concat = torch.cat([z, key], dim=-1)
        
        # Doorloop hidden layers
        x = self.hidden_net(concat)
        
        # Projecteer naar output_dim
        out = self.output_layer(x)
        
        # --- OUDE IMPLEMENTATIE: LayerNorm ---
        # out = self.norm(out)

        # --- NIEUWE IMPLEMENTATIE: Scaled Tanh ---
        # Beperkt waarden strikt tussen -3 en 3, maar laat waarden rond 0 vrij intact
        out = 3.0 * torch.tanh(out / 3)

        return out

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)