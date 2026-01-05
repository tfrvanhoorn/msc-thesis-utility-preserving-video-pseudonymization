from __future__ import annotations

from ..components import (
    ArcFaceEmbedder,
    BioHashingQuantizer,
    IdentityQuantizer,
    HmacSeedGenerator,
    RetinaFaceDetector,
    GlobalSphericalKMeansQuantizer,
    SemanticAttributeEmbedder,
    GenderClassifier,
)
from ..config import ExperimentConfig
from ..utils.alignment import FivePointAffineAligner
from .identity_seed_pipeline import IdentitySeedPipeline


def build_identity_seed_pipeline(config: ExperimentConfig) -> IdentitySeedPipeline:
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
    quantizer = _build_quantizer(config)
    seed_generator = HmacSeedGenerator(
        secret_key=config.seed.secret_key,
    )

    return IdentitySeedPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        quantizer=quantizer,
        seed_generator=seed_generator,
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


def _build_quantizer(config: ExperimentConfig):
    embed_method = (config.embedding.method or "arcface").lower()
    quant_method = (config.quantization.method or "auto").lower()
    if quant_method == "auto":
        quant_method = "identity" if embed_method.startswith("semantic") else "gskm"

    if quant_method in {"identity", "pass", "none"}:
        return IdentityQuantizer()
    if quant_method in {"biohash", "biohashing"}:
        input_dim = config.quantization.input_dim or config.embedding.embedding_size
        return BioHashingQuantizer(
            input_dim=input_dim,
            output_dim=config.quantization.output_dim,
            random_seed=config.quantization.random_seed,
        )
    if quant_method in {"gskm", "spherical-kmeans", "spherical_kmeans"}:
        qcfg = config.quantization
        return GlobalSphericalKMeansQuantizer(
            num_prototypes=qcfg.num_prototypes,
            max_iters=qcfg.max_iters,
            tol=qcfg.tol,
            random_seed=qcfg.random_seed,
        )

    raise ValueError(f"Unknown quantization method: {quant_method}")
