from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ..components import EmbeddingModel, FaceDetector
from ..components.alignment import FaceAligner
from ..components.detector import Detection
from ..models import StyleGAN2Generator


logger = logging.getLogger(__name__)


@dataclass
class KfaarResult:
    detections: Sequence[Detection]
    aligned_faces: Sequence[np.ndarray]
    embeddings: np.ndarray
    w_latents: Optional[np.ndarray] = None
    generated_images: Optional[np.ndarray] = None


class KfaarPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        aligner: FaceAligner,
        embedder: EmbeddingModel,
        stylegan: Optional[StyleGAN2Generator] = None,
    ) -> None:
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.stylegan = stylegan

    def process_image(self, image: np.ndarray, source_path: Path | None = None) -> KfaarResult:
        detections = self.detector.detect(image)
        if not detections:
            logger.debug("No detections returned; skipping embedding stage")
            empty = np.empty((0, 0))
            return KfaarResult(detections, [], empty)

        aligned = [self.aligner.align(image, det) for det in detections]
        source_paths = [source_path] * len(aligned) if source_path is not None else None
        embeddings = self.embedder.embed(aligned, source_paths=source_paths)

        w_latents = None
        generated_images = None
        if self.stylegan is not None and embeddings is not None and getattr(embeddings, "size", 0) > 0:
            device = next(self.stylegan._G.parameters()).device  # type: ignore[attr-defined]
            z = torch.from_numpy(embeddings).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                w = self.stylegan.map(z)
                images = self.stylegan.synthesize(w, noise_mode="const")
            w_latents = w.detach().cpu().numpy()
            generated_images = images.detach().cpu().numpy()

        return KfaarResult(detections, aligned, embeddings, w_latents, generated_images)
