from __future__ import annotations

from ..components import ArcFaceEmbedder, GenderClassifier, RetinaFaceDetector, SemanticAttributeEmbedder
from ..components.alignment import FivePointAffineAligner
from ..config import ExperimentConfig
from .kfaar_pipeline import KfaarPipeline


def build_kfaar_pipeline(config: ExperimentConfig) -> KfaarPipeline:
    detector_kwargs = dict(
        model_name=config.detector.release_name or config.detector.name,
        score_threshold=config.detector.score_threshold,
        det_size=tuple(config.detector.det_size),
        max_faces=config.detector.max_faces,
        ctx_id=config.detector.ctx_id,
        providers=config.detector.providers,
    )
    if config.detector.root:
        detector_kwargs["root"] = str(config.detector.root)
    detector = RetinaFaceDetector(**detector_kwargs)
    aligner = FivePointAffineAligner(output_size=112)
    embedder = _build_embedder(config)
    return KfaarPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
    )


def _build_embedder(config: ExperimentConfig):
    method = (config.embedding.method or "arcface").lower()
    if method.startswith("semantic"):
        return SemanticAttributeEmbedder(
            feature_selector=config.embedding.feature_selector,
            feature_classifiers=_build_feature_classifiers(config),
        )

    embedder_kwargs = dict(
        model_name=config.embedding.model_name or config.embedding.name,
        release_name=config.embedding.release_name,
        ctx_id=config.embedding.ctx_id,
        providers=config.embedding.providers,
    )
    if config.embedding.root:
        embedder_kwargs["root"] = str(config.embedding.root)
    return ArcFaceEmbedder(**embedder_kwargs)


def _build_feature_classifiers(config: ExperimentConfig):
    keep = getattr(config.embedding.feature_selector, "keep", []) or []
    normalized = [k.replace("-", "_").replace(" ", "_").lower() for k in keep]

    gender_model = None
    if "male" in normalized:
        gender_model = GenderClassifier(
            providers=config.embedding.providers,
            ctx_id=config.embedding.ctx_id,
            root=config.embedding.root,
            default_value=False,
        )

    classifiers = {}
    if gender_model is not None:
        classifiers["male"] = gender_model
    return classifiers
