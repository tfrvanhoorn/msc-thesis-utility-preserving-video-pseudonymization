from __future__ import annotations

from typing import Protocol

import numpy as np
import torch

from .detector import Detection


class FaceAligner(Protocol):
    def align(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        ...


class MTCNNAligner:
    def __init__(self, output_size: int = 160) -> None:
        self.output_size = output_size

    def align(self, image: np.ndarray, detection: Detection) -> np.ndarray:
        if detection.aligned is None:
            x1, y1, x2, y2 = detection.bbox.astype(int)
            cropped = image[int(y1) : int(y2), int(x1) : int(x2)]
            return np.asarray(cropped, dtype=np.uint8)

        face = detection.aligned
        if face.ndim == 4:
            face = face[0]
        if isinstance(face, torch.Tensor):
            face_np = face.detach().cpu().permute(1, 2, 0).numpy()
        else:
            face_np = np.asarray(face)

        face_np = np.clip(face_np * 255.0, 0, 255).astype(np.uint8)
        return face_np
