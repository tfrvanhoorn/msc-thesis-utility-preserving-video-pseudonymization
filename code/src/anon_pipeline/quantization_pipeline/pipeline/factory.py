from __future__ import annotations

from ..components import (
    BioHashingQuantizer,
    FacenetEmbedder,
    GenderClassifier,
    GlobalSphericalKMeansQuantizer,
    HmacSeedGenerator,
    IdentityQuantizer,
    MTCNNDetector,
    SemanticAttributeEmbedder,
)
from ..config import QuantizationExperimentConfig
from ..components.alignment import MTCNNAligner
from .identity_seed_pipeline import IdentitySeedPipeline


def build_identity_seed_pipeline(config: QuantizationExperimentConfig) -> IdentitySeedPipeline:
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


def _build_embedder(config: QuantizationExperimentConfig):
    method = (config.embedding.method or "facenet").lower()
    if method.startswith("semantic"):
        return SemanticAttributeEmbedder(
            feature_selector=config.embedding.feature_selector,
            feature_classifiers=_build_feature_classifiers(config),
        )

    return FacenetEmbedder(
        pretrained=config.embedding.pretrained,
        device=config.embedding.device,
    )


def _build_feature_classifiers(config: QuantizationExperimentConfig):
    keep = getattr(config.embedding.feature_selector, "keep", []) or []
    normalized = [k.replace("-", "_").replace(" ", "_").lower() for k in keep]

    gender_model = None
    if "male" in normalized:
        gender_model = GenderClassifier(
            providers=None,
            ctx_id=0,
            root=None,
            default_value=False,
        )

    classifiers = {}
    if gender_model is not None:
        classifiers["male"] = gender_model
    return classifiers


def _build_quantizer(config: QuantizationExperimentConfig):
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
