from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..components import ArcFaceEmbedder, EmbeddingModel, FaceDetector
from ..components.alignment import FaceAligner
from ..components.detector import Detection


logger = logging.getLogger(__name__)


@dataclass
class KfaarResult:
    detections: Sequence[Detection]
    aligned_faces: Sequence[np.ndarray]
    embeddings: np.ndarray


class KfaarPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: EmbeddingModel,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder

    def process_image(self, image: np.ndarray, source_path: Path | None = None) -> KfaarResult:
        detections = self.detector.detect(image)
        if not detections:
            logger.debug("No detections returned; skipping embedding stage")
            empty = np.empty((0, 0))
            return KfaarResult(detections, [], empty)

        aligned = [self.aligner.align(image, det) for det in detections]
        source_paths = [source_path] * len(aligned) if source_path is not None else None
        embeddings = self.embedder.embed(aligned, source_paths=source_paths)
        return KfaarResult(detections, aligned, embeddings)
