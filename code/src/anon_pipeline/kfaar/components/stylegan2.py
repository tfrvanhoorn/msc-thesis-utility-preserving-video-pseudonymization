"""
StyleGAN2-ADA (PyTorch) loader utilities for FFHQ generator checkpoints.

Exposes a thin wrapper around the generator to access mapping and synthesis
modules and to produce images in one call. Defaults to loading the bundled
``stylegan2-celebahq-256x256.pkl`` next to this file. The checkpoint must come from the official
StyleGAN2-ADA PyTorch implementation and include either ``G_ema`` or ``G``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import pickle

import torch


def _import_stylegan2_modules():
    try:
        import dnnlib  # type: ignore
        from torch_utils import persistence  # type: ignore  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "StyleGAN2-ADA PyTorch dependencies not found. Install the official"
            " stylegan2-ada-pytorch package or add it as a submodule so that"
            " 'dnnlib' and 'torch_utils' are importable."
        ) from err

    return dnnlib


class StyleGAN2Generator:
    """Lightweight wrapper around StyleGAN2 generator.

    Provides explicit access to mapping and synthesis while offering a
    convenience call to go from z to images.
    """

    def __init__(self, generator: torch.nn.Module):
        self._G = generator
        self.mapping = generator.mapping
        self.synthesis = generator.synthesis

    def to(self, device: str | torch.device) -> StyleGAN2Generator:
        """Moves the internal generator to the device and ensures float32 on CPU."""
        _device = torch.device(device)
        self._G.to(_device)
        
        # StyleGAN2 must be in float32 for CPU compatibility
        if _device.type == "cpu":
            self._G.float() 
            
        self.mapping = self._G.mapping
        self.synthesis = self._G.synthesis
        return self

    @property
    def z_dim(self) -> int:
        return getattr(self._G, "z_dim", 512)

    @property
    def w_dim(self) -> int:
        return getattr(self._G, "w_dim", 512)

    def map(
        self,
        z: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        truncation_psi: float = 0.5,
        truncation_cutoff: Optional[int] = 8,
    ) -> torch.Tensor:
        return self.mapping(
            z,
            conditioning,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )

    def synthesize(
        self,
        w: torch.Tensor,
        noise_mode: str = "const",
        force_fp32: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return self.synthesis(w, noise_mode=noise_mode, force_fp32=force_fp32, **kwargs)

    def __call__(
        self,
        z: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        truncation_psi: float = 0.5,
        truncation_cutoff: Optional[int] = 8,
        noise_mode: str = "const",
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        w = self.map(
            z,
            conditioning=conditioning,
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )
        images = self.synthesize(w, noise_mode=noise_mode, **kwargs)
        return images, w


def load_stylegan2(
    ckpt_path: Optional[Path] = None,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> StyleGAN2Generator:
    """Load StyleGAN2 generator from checkpoint.

    Args:
        ckpt_path: Path to .pkl checkpoint; defaults to bundled stylegan2-celebahq-256x256.pkl.
        device: Torch device for the model.
        use_ema: Prefer the EMA generator (G_ema) when present.

    Returns:
        StyleGAN2Generator wrapper ready for inference.
    """

    ckpt = Path(ckpt_path) if ckpt_path is not None else Path(__file__).with_name("stylegan2-celebahq-256x256.pkl")
    if not ckpt.exists():
        raise FileNotFoundError(f"StyleGAN2 checkpoint not found at {ckpt}")

    _device = torch.device(device)
    _import_stylegan2_modules()

    with ckpt.open("rb") as f:
        data = pickle.load(f)

    key = "G_ema" if use_ema and "G_ema" in data else "G"
    if key not in data:
        raise KeyError(f"'{key}' not found in checkpoint; available keys: {list(data.keys())}")

    generator = data[key].to(_device).eval()
    return StyleGAN2Generator(generator)


def load_stylegan2_components(
    ckpt_path: Optional[Path] = None,
    device: str | torch.device = "cuda",
    use_ema: bool = True,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load and return mapping and synthesis modules directly."""

    wrapper = load_stylegan2(ckpt_path=ckpt_path, device=device, use_ema=use_ema)
    return wrapper.mapping, wrapper.synthesis


__all__ = ["StyleGAN2Generator", "load_stylegan2", "load_stylegan2_components"]
