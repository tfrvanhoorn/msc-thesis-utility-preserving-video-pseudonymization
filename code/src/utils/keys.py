from __future__ import annotations

import torch


def sample_binary_key(
    key_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a single 0/1 binary key vector with shape (key_dim,)."""
    if key_dim <= 0:
        raise ValueError("key_dim must be > 0")

    bits = torch.randint(0, 2, (key_dim,), device=device, generator=generator)
    return bits.to(dtype=dtype)


def sample_binary_key_bank(
    num_keys: int,
    key_dim: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Sample a list of 0/1 binary key vectors."""
    if num_keys < 1:
        raise ValueError("num_keys must be >= 1")

    return [
        sample_binary_key(key_dim, device=device, generator=generator, dtype=dtype)
        for _ in range(num_keys)
    ]
