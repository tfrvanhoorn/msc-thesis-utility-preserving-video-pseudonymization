from __future__ import annotations
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class ProjectorMLP(nn.Module):
    """Projects face embedding z + key k to a new L2-normalized z'."""

    def __init__(
        self,
        key_dim: int = 128,
        output_dim: int = 512,
        hidden_dims: tuple[int, ...] = (1024, 512),
        dropout: float = 0.0,
        output_l2_normalize: bool = True,
    ) -> None:
        super().__init__()
        input_dim = output_dim + key_dim
        self.output_l2_normalize = output_l2_normalize
        
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        
        self.hidden_net = nn.Sequential(*layers)
        
        # Output layer for direct embedding prediction.
        self.output_layer = nn.Linear(in_dim, output_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Initializes weights for networks."""
        if isinstance(m, nn.Linear):
            init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if key.dim() == 1: key = key.unsqueeze(0)
        if z.dim() == 1: z = z.unsqueeze(0)
        
        # L2 Normalization for the key to prevent Magnitude Mismatch
        key = F.normalize(key, p=2, dim=-1)
        
        concat = torch.cat([z, key], dim=-1)
        x = self.hidden_net(concat)
        
        # Predict a new embedding directly from z and key.
        out = self.output_layer(x)

        # Optional output normalization allows explicit control from training args.
        if self.output_l2_normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out

    def project(self, z: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        return self.forward(z, key)


def load_projector_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor], *, strict: bool = True) -> None:
    """Load projector weights, remapping legacy MLP checkpoint keys when needed."""
    def _remap_legacy_mlp_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped: dict[str, torch.Tensor] = {}
        legacy_found = False

        for key, tensor in sd.items():
            if key.startswith("net."):
                legacy_found = True
                if key.startswith("net.0."):
                    new_key = key.replace("net.0", "hidden_net.0", 1)
                elif key.startswith("net.2."):
                    new_key = key.replace("net.2", "hidden_net.2", 1)
                elif key.startswith("net.4."):
                    new_key = key.replace("net.4", "output_layer", 1)
                else:
                    new_key = key
            else:
                new_key = key

            mapped[new_key] = tensor

        if legacy_found:
            logger.info("Remapped legacy ProjectorMLP checkpoint keys (net.* -> hidden_net.*, output_layer)")

        return mapped

    if isinstance(model, ProjectorMLP):
        state_dict = _remap_legacy_mlp_keys(state_dict)
        
    try:
        model.load_state_dict(state_dict, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load projector checkpoint into ProjectorMLP. "
            "Legacy LSTM projector checkpoints are no longer supported."
        ) from exc