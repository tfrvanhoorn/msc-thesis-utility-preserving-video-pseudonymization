from __future__ import annotations

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
) -> KfaarPipeline:
    target_device = _resolve_device(config, device)

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
