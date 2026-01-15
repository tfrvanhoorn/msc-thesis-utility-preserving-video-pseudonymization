from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_loss(f1: torch.Tensor, f2: torch.Tensor, label: int, margin: float = 0.5, reduction: str = "mean") -> torch.Tensor:
    labels = torch.full((f1.shape[0],), float(label), device=f1.device, dtype=f1.dtype)
    return F.cosine_embedding_loss(f1, f2, labels, margin=margin, reduction=reduction)


def anonymity_loss(real_feat: torch.Tensor, virtual_feat: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    return cosine_loss(virtual_feat, real_feat, label=-1, margin=margin)


def synchronism_loss(virtual_feat_a: torch.Tensor, virtual_feat_b: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    return cosine_loss(virtual_feat_a, virtual_feat_b, label=1, margin=margin)


def diversity_loss(virtual_feat_k1: torch.Tensor, virtual_feat_k2: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    return cosine_loss(virtual_feat_k1, virtual_feat_k2, label=-1, margin=margin)


def differentiation_loss(virtual_feat_x: torch.Tensor, virtual_feat_y: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    return cosine_loss(virtual_feat_x, virtual_feat_y, label=-1, margin=margin)


def total_hpvg_loss(
    ano: torch.Tensor,
    syn: torch.Tensor,
    div: torch.Tensor,
    dif: torch.Tensor,
    lambda_ano: float = 0.4,
    lambda_syn: float = 1.0,
    lambda_div: float = 1.0,
    lambda_dif: float = 1.0,
) -> torch.Tensor:
    return lambda_ano * ano + lambda_syn * syn + lambda_div * div + lambda_dif * dif
