from __future__ import annotations

from typing import Protocol

import numpy as np
from insightface.utils.face_align import norm_crop

from ..components.detector import Detection


class FaceAligner(Protocol):
    def align(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        ...


class FivePointAffineAligner:
    def __init__(self, output_size: int = 112) -> None:
        self.output_size = output_size

    def align(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        landmarks = detection.landmarks.astype(np.float32)
        bgr = image[:, :, ::-1]
        aligned = norm_crop(bgr, landmarks, image_size=self.output_size)
        return aligned[:, :, ::-1]
