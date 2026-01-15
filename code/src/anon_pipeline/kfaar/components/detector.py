from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float
    aligned: torch.Tensor | None = None


class FaceDetector:
    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        raise NotImplementedError


class MTCNNDetector(FaceDetector):
    def __init__(
        self,
        image_size: int = 160,
        margin: int = 0,
        score_threshold: float = 0.8,
        min_face_size: int | None = 20,
        keep_all: bool = True,
        post_process: bool = False,
        device: str | torch.device | None = None,
        max_faces: int | None = None,
    ) -> None:
        self.image_size = image_size
        self.margin = margin
        self.score_threshold = score_threshold
        self.min_face_size = min_face_size
        self.keep_all = keep_all
        self.post_process = post_process
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.max_faces = max_faces
        self._mtcnn = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=keep_all,
            thresholds=(0.6, 0.7, 0.7),
            min_face_size=min_face_size,
            post_process=post_process,
            device=self.device,
        )

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        if image is None:
            return []
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")
        pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")

        with torch.no_grad():
            boxes, probs, landmarks = self._mtcnn.detect(pil_image, landmarks=True)
            aligned = None
            if boxes is not None and len(boxes) > 0:
                aligned = self._mtcnn.extract(pil_image, boxes, save_path=None)

        if boxes is None or probs is None:
            return []

        detections: list[Detection] = []
        for idx, (box, score) in enumerate(zip(boxes, probs)):
            if score is None or score < self.score_threshold:
                continue
            lm = landmarks[idx] if landmarks is not None else None
            if lm is None:
                continue
            aligned_face = None
            if aligned is not None and idx < len(aligned):
                aligned_face = aligned[idx]
            detections.append(
                Detection(
                    bbox=np.asarray(box, dtype=np.float32),
                    landmarks=np.asarray(lm, dtype=np.float32),
                    score=float(score),
                    aligned=aligned_face,
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        if self.max_faces is not None:
            detections = detections[: self.max_faces]
        return detections
