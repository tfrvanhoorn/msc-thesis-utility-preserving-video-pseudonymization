from __future__ import annotations

from ..components import (
    ArcFaceEmbedder,
    HmacSeedGenerator,
    RetinaFaceDetector,
    ProductSphericalKMeansQuantizer,
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
    embedder_kwargs = dict(
        model_name=config.embedding.model_name or config.embedding.name,
        release_name=config.embedding.release_name,
        ctx_id=config.embedding.ctx_id,
        providers=config.embedding.providers,
    )
    if config.embedding.root:
        embedder_kwargs["root"] = str(config.embedding.root)
    embedder = ArcFaceEmbedder(**embedder_kwargs)
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


def _build_quantizer(config: ExperimentConfig):
    qcfg = config.quantization
    return ProductSphericalKMeansQuantizer(
        num_subspaces=qcfg.num_subspaces,
        num_prototypes=qcfg.num_prototypes,
        max_iters=qcfg.max_iters,
        tol=qcfg.tol,
        random_seed=qcfg.random_seed,
        output_mode=qcfg.output_mode,
    )
