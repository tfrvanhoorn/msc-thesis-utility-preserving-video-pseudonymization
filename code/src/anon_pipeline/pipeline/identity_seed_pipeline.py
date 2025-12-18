from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..components import (
    ArcFaceEmbedder,
    EmbeddingModel,
    FaceDetector,
    HmacSeedGenerator,
    Quantizer,
    SeedGenerator,
)
from ..components.detector import Detection
from ..utils.alignment import FaceAligner


logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    detections: Sequence[Detection]
    aligned_faces: Sequence[np.ndarray]
    embeddings: np.ndarray
    quantized: np.ndarray
    seeds: Sequence[str]


class IdentitySeedPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: EmbeddingModel,
        quantizer: Quantizer,
        seed_generator: SeedGenerator,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.quantizer = quantizer
        self.seed_generator = seed_generator

    def process_image(self, image: np.ndarray) -> PipelineResult:
        detections = self.detector.detect(image)
        if not detections:
            logger.debug("No detections returned; skipping downstream pipeline stages")
            empty = np.empty((0, 0))
            return PipelineResult(detections, [], empty, empty, [])

        aligned = [self.aligner.align(image, det) for det in detections]
        embeddings = self.embedder.embed(aligned)
        quantized = self.quantizer.quantize(embeddings)
        seeds = [self.seed_generator.generate(vec) for vec in quantized]

        return PipelineResult(detections, aligned, embeddings, quantized, seeds)
