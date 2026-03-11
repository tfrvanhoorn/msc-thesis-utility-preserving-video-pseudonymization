from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_loss(f1: torch.Tensor, f2: torch.Tensor, label: int, margin: float = 0.5, reduction: str = "mean") -> torch.Tensor:
    labels = torch.full((f1.shape[0],), float(label), device=f1.device, dtype=f1.dtype)
    return F.cosine_embedding_loss(f1, f2, labels, margin=margin, reduction=reduction)


def _pairwise_cosine_by_label(
    feats: torch.Tensor,
    labels: torch.Tensor,
    *,
    same_identity: bool,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute cosine loss over all pairwise combinations filtered by label equality."""

    if feats.shape[0] < 2:
        return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

    idx = torch.arange(feats.shape[0], device=feats.device)
    pairs = torch.combinations(idx, r=2)
    if pairs.numel() == 0:
        return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

    lbl_a = labels[pairs[:, 0]]
    lbl_b = labels[pairs[:, 1]]
    mask = lbl_a == lbl_b if same_identity else lbl_a != lbl_b
    if not mask.any():
        return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)

    feat_a = feats[pairs[mask][:, 0]]
    feat_b = feats[pairs[mask][:, 1]]
    target = 1 if same_identity else -1
    return cosine_loss(feat_a, feat_b, label=target, margin=margin, reduction=reduction)


def anonymity_loss(real_feat: torch.Tensor, virtual_feat_k1: torch.Tensor, virtual_feat_k2: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    # Virtual faces should diverge from the original identity regardless of key.
    ano_k1 = cosine_loss(virtual_feat_k1, real_feat, label=-1, margin=margin)
    ano_k2 = cosine_loss(virtual_feat_k2, real_feat, label=-1, margin=margin)
    return torch.stack([ano_k1, ano_k2]).mean()


def synchronism_loss(
    virtual_feat_k1: torch.Tensor,
    virtual_feat_k2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    # Same identity, same key should stay close; accumulate all pairs per key.
    syn_k1 = _pairwise_cosine_by_label(virtual_feat_k1, labels, same_identity=True, margin=margin)
    syn_k2 = _pairwise_cosine_by_label(virtual_feat_k2, labels, same_identity=True, margin=margin)
    return torch.stack([syn_k1, syn_k2]).mean()


def diversity_loss(virtual_feat_k1: torch.Tensor, virtual_feat_k2: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    # Same sample with different keys should diverge.
    return cosine_loss(virtual_feat_k1, virtual_feat_k2, label=-1, margin=margin)


def differentiation_loss(
    virtual_feat_k1: torch.Tensor,
    virtual_feat_k2: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    # Different identities with the same key should diverge; use all cross-identity pairs.
    dif_k1 = _pairwise_cosine_by_label(virtual_feat_k1, labels, same_identity=False, margin=margin)
    dif_k2 = _pairwise_cosine_by_label(virtual_feat_k2, labels, same_identity=False, margin=margin)
    return torch.stack([dif_k1, dif_k2]).mean()


def temporal_smoothness_loss(
    virtual_feats: list[torch.Tensor],
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Encourage adjacent timesteps within a sequence to stay consistent.

    Expects a list where each element is shaped (T, F) for one sequence.
    Returns 0 when no valid temporal pairs exist.
    """

    deltas: list[torch.Tensor] = []
    for seq in virtual_feats:
        if seq is None or not torch.is_tensor(seq):
            continue
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        if seq.shape[0] < 2:
            continue
        deltas.append(cosine_loss(seq[1:], seq[:-1], label=1, margin=0.0, reduction="none"))

    if not deltas:
        return torch.tensor(0.0, device=virtual_feats[0].device if virtual_feats else None)

    stacked = torch.cat(deltas, dim=0)
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    return stacked


def _flatten_latent_for_boundary_projection(w: torch.Tensor) -> torch.Tensor:
    if w.ndim == 2:
        return w
    if w.ndim == 3:
        return w.reshape(-1, w.shape[-1])
    raise ValueError(f"Unsupported latent shape for boundary regularization: {tuple(w.shape)}")


def _normalize_boundary_vector(boundary: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    b = boundary.to(device=device, dtype=dtype)
    if b.ndim == 3 and b.shape[0] == 1 and b.shape[1] == 1:
        b = b.view(1, b.shape[-1])
    if b.ndim == 2 and b.shape[0] == 1:
        return b.squeeze(0)
    if b.ndim == 1:
        return b
    raise ValueError(f"Unsupported boundary shape for regularization: {tuple(boundary.shape)}")


def eyeglasses_boundary_regularization_loss(
    w: torch.Tensor,
    boundary: torch.Tensor,
    *,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    w_flat = _flatten_latent_for_boundary_projection(w)
    b = _normalize_boundary_vector(boundary, device=w_flat.device, dtype=w_flat.dtype)
    if w_flat.shape[-1] != b.shape[-1]:
        raise ValueError(
            "Eyeglasses boundary dimension mismatch. "
            f"Latent dim={w_flat.shape[-1]} boundary dim={b.shape[-1]}"
        )

    proj = torch.matmul(w_flat, b)
    values = F.relu(proj + float(margin))
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    return values.mean()


def pose_boundary_regularization_loss(
    w: torch.Tensor,
    boundary: torch.Tensor,
    *,
    margin: float,
    reduction: str = "mean",
) -> torch.Tensor:
    w_flat = _flatten_latent_for_boundary_projection(w)
    b = _normalize_boundary_vector(boundary, device=w_flat.device, dtype=w_flat.dtype)
    if w_flat.shape[-1] != b.shape[-1]:
        raise ValueError(
            "Pose boundary dimension mismatch. "
            f"Latent dim={w_flat.shape[-1]} boundary dim={b.shape[-1]}"
        )

    proj = torch.matmul(w_flat, b)
    values = F.relu(torch.abs(proj) - float(margin))
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    return values.mean()


def total_hpvg_loss(
    ano: torch.Tensor,
    syn: torch.Tensor,
    div: torch.Tensor,
    dif: torch.Tensor,
    temp: torch.Tensor | None = None,
    lambda_ano: float = 0.4,
    lambda_syn: float = 1.0,
    lambda_div: float = 1.0,
    lambda_dif: float = 1.0,
    lambda_temp: float = 0.0,
) -> torch.Tensor:
    total = lambda_ano * ano + lambda_syn * syn + lambda_div * div + lambda_dif * dif
    if temp is not None:
        total = total + lambda_temp * temp
    return total
