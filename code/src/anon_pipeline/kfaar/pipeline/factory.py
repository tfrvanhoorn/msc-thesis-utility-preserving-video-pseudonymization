from __future__ import annotations

from ..components import FacenetEmbedder, MTCNNDetector, SemanticAttributeEmbedder
from ..components.alignment import MTCNNAligner
from ..config import ExperimentConfig
from .kfaar_pipeline import KfaarPipeline
from ..models import StyleGAN2Generator


def build_kfaar_pipeline(config: ExperimentConfig, stylegan: StyleGAN2Generator | None = None) -> KfaarPipeline:
    detector = MTCNNDetector(
        image_size=config.detector.image_size,
        margin=config.detector.margin,
        score_threshold=config.detector.score_threshold,
        min_face_size=config.detector.min_face_size,
        max_faces=config.detector.max_faces,
        keep_all=True,
        post_process=False,
        device=config.detector.device,
    )
    aligner = MTCNNAligner(output_size=config.detector.image_size)
    embedder = _build_embedder(config)
    return KfaarPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        stylegan=stylegan,
    )


def _build_embedder(config: ExperimentConfig):
    method = (config.embedding.method or "facenet").lower()
    if method.startswith("semantic"):
        return SemanticAttributeEmbedder(
            feature_selector=config.embedding.feature_selector,
            feature_classifiers={},
        )

    return FacenetEmbedder(
        pretrained=config.embedding.pretrained,
        device=config.embedding.device,
    )
