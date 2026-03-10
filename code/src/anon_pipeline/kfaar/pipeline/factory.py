from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..components import (
    FacenetEmbedder,
    MTCNNDetector,
    SemanticAttributeEmbedder,
    ProjectorMLP,
    ProjectorLSTM,
)
from ..components.alignment import MTCNNAligner
from ..config import PipelineConfig
from .kfaar_pipeline import KfaarPipeline
from ..components import StyleGAN2Generator


def build_kfaar_pipeline(
    config: PipelineConfig,
    stylegan: StyleGAN2Generator | None = None,
    device: str | torch.device | None = None,
    truncation_psi: float = 0.5,
    face_swapper: object | None = None,
) -> KfaarPipeline:
    target_device = _resolve_device(config, device)
    eyeglasses_boundary = _load_eyeglasses_boundary(config, target_device)

    detector = MTCNNDetector(
        image_size=config.detector.image_size,
        margin=config.detector.margin,
        score_threshold=config.detector.score_threshold,
        min_face_size=config.detector.min_face_size,
        max_faces=config.detector.max_faces,
        keep_all=True,
        post_process=False,
        device=str(target_device),
    )
    aligner = MTCNNAligner(output_size=config.detector.image_size)
    embedder = _build_embedder(config, target_device)

    proj_type = config.projector.normalized_type()
    if proj_type == "lstm":
        projector = ProjectorLSTM(
            key_dim=config.projector.key_dim,
            output_dim=embedder.embedding_size,
            hidden_dim=config.projector.lstm_hidden_dim,
            num_layers=config.projector.lstm_num_layers,
            bidirectional=config.projector.lstm_bidirectional,
            dropout=config.projector.dropout,
        ).to(target_device)
    else:
        projector = ProjectorMLP(
            key_dim=config.projector.key_dim,
            output_dim=embedder.embedding_size,
            hidden_dims=config.projector.hidden_dims,
            dropout=config.projector.dropout,
        ).to(target_device)

    if stylegan is not None:
        stylegan = stylegan.to(target_device)
        # Ensure float32 on CPU to avoid half-precision ops unsupported on CPU
        if torch.device(target_device).type == "cpu" and hasattr(stylegan, "_G"):
            stylegan._G = stylegan._G.float()
            stylegan.mapping = stylegan._G.mapping
            stylegan.synthesis = stylegan._G.synthesis

    return KfaarPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        projector=projector,
        stylegan=stylegan,
        device=target_device,
        truncation_psi=truncation_psi,
        face_swapper=face_swapper,
        remove_eyeglasses=config.eyeglasses_boundary.enabled,
        eyeglasses_boundary=eyeglasses_boundary,
        eyeglasses_removal_scale=config.eyeglasses_boundary.removal_scale,
    )


def _load_eyeglasses_boundary(config: PipelineConfig, device: torch.device) -> torch.Tensor | None:
    boundary_cfg = config.eyeglasses_boundary
    if not boundary_cfg.enabled:
        return None
    if boundary_cfg.boundary_path is None:
        raise ValueError("Eyeglasses removal is enabled but no boundary path was provided")

    boundary_path = Path(boundary_cfg.boundary_path)
    if boundary_path.suffix.lower() != ".npy":
        raise ValueError(f"Unsupported boundary format '{boundary_path.suffix}'. Expected a .npy file")
    if not boundary_path.exists():
        raise FileNotFoundError(f"Eyeglasses boundary file not found: {boundary_path}")

    boundary_np = np.load(boundary_path)
    if not isinstance(boundary_np, np.ndarray):
        raise TypeError(f"Expected NumPy array from boundary file, got {type(boundary_np)}")

    boundary = torch.from_numpy(boundary_np).float().to(device)
    boundary = _normalize_eyeglasses_boundary_shape(boundary)
    return boundary


def _normalize_eyeglasses_boundary_shape(boundary: torch.Tensor) -> torch.Tensor:
    # InterfaceGAN boundaries can come as (512,), (1, 512), or (1, 1, 512).
    if boundary.ndim == 1:
        if boundary.shape[0] != 512:
            raise ValueError(f"Expected boundary shape (512,), got {tuple(boundary.shape)}")
        return boundary.unsqueeze(0)

    if boundary.ndim == 2:
        if boundary.shape[1] != 512 or boundary.shape[0] != 1:
            raise ValueError(f"Expected boundary shape (1, 512), got {tuple(boundary.shape)}")
        return boundary

    if boundary.ndim == 3 and boundary.shape[0] == 1 and boundary.shape[1] == 1 and boundary.shape[2] == 512:
        return boundary.view(1, 512)

    raise ValueError(
        "Unsupported eyeglasses boundary shape "
        f"{tuple(boundary.shape)}. Expected (512,), (1, 512), or (1, 1, 512)."
    )


def _build_embedder(config: PipelineConfig, device: torch.device):
    method = (config.embedding.method or "facenet").lower()
    if method.startswith("semantic"):
        return SemanticAttributeEmbedder(
            feature_selector=config.embedding.feature_selector,
            feature_classifiers={},
        )

    return FacenetEmbedder(
        pretrained=config.embedding.pretrained,
        device=str(device),
    )


def _resolve_device(config: PipelineConfig, override: str | torch.device | None) -> torch.device:
    if override is not None:
        return torch.device(override)
    if config.embedding.device:
        return torch.device(config.embedding.device)
    if config.detector.device:
        return torch.device(config.detector.device)
    return torch.device("cpu")
