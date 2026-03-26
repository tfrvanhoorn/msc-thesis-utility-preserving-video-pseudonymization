from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
from facenet_pytorch import MTCNN


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: torch.Tensor  # shape (4,)
    landmarks: torch.Tensor  # shape (5,2)
    score: float
    aligned: torch.Tensor | None = None


class FaceDetector:
    def detect(self, image: torch.Tensor) -> Sequence[Detection]:
        raise NotImplementedError

    def detect_batch(self, images: Sequence[torch.Tensor]) -> Sequence[Sequence[Detection]]:
        raise NotImplementedError


class MTCNNDetector(FaceDetector):
    def __init__(
        self,
        image_size: int = 160,
        margin: int = 0,
        score_threshold: float = 0.4,
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

    def detect(self, image: torch.Tensor) -> Sequence[Detection]:
        batched = self.detect_batch([image])
        return batched[0] if batched else []

    def detect_batch(self, images: Sequence[torch.Tensor]) -> Sequence[Sequence[Detection]]:
        if images is None:
            return []
        image_list = list(images)
        if not image_list:
            return []

        prepared: list[torch.Tensor] = []
        for image in image_list:
            if image is None:
                prepared.append(torch.empty(0, device=self.device))
                continue
            if image.max() <= 1.01:
                image_for_mtcnn = (image * 255).byte()
            else:
                image_for_mtcnn = image.byte()
            if image_for_mtcnn.shape[0] == 3:
                image_for_mtcnn = image_for_mtcnn.permute(1, 2, 0)
            prepared.append(image_for_mtcnn)

        with torch.no_grad():
            boxes_b, probs_b, landmarks_b = self._mtcnn.detect(prepared, landmarks=True)

        if boxes_b is None:
            return [[] for _ in prepared]

        # facenet_pytorch returns ndarray/object arrays for batch mode.
        result: list[list[Detection]] = []
        for idx in range(len(prepared)):
            boxes = boxes_b[idx] if idx < len(boxes_b) else None
            probs = probs_b[idx] if probs_b is not None and idx < len(probs_b) else None
            landmarks = landmarks_b[idx] if landmarks_b is not None and idx < len(landmarks_b) else None
            result.append(self._postprocess_single_detect(boxes, probs, landmarks))
        return result

    def _postprocess_single_detect(self, boxes, probs, landmarks) -> list[Detection]:
        # Guard against empty lists returned by facenet_pytorch (causes cat() error)
        if boxes is None or (isinstance(boxes, (list, tuple)) and len(boxes) == 0):
            return []

        if boxes is None or probs is None:
            return []

        detections: list[Detection] = []
        for idx, (box, score) in enumerate(zip(boxes, probs)):
            if score is None or score < self.score_threshold:
                continue

            lm = landmarks[idx] if landmarks is not None else None
            if lm is None:
                continue

            box_t = torch.as_tensor(box, dtype=torch.float32, device=self.device)
            lm_t = torch.as_tensor(lm, dtype=torch.float32, device=self.device)

            detections.append(
                Detection(
                    bbox=box_t,
                    landmarks=lm_t,
                    score=float(score),
                    aligned=None,
                )
            )

        detections.sort(key=lambda d: d.score, reverse=True)
        if self.max_faces is not None:
            detections = detections[: self.max_faces]
        return detections

    def _to_tensor(self, image: torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            img = image
        else:
            raise TypeError(f"Expected torch.Tensor input, got {type(image)}")

        if img.dim() == 3 and img.shape[0] == 3:
            pass
        elif img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)
        else:
            raise ValueError(f"Expected image shape (3,H,W) or (H,W,3), got {tuple(img.shape)}")

        img = img.to(self.device)
        if img.dtype != torch.float32:
            img = img.float()
        if img.max() > 1.0 or img.min() < 0.0:
            # assume 0-255 and rescale
            img = img / 255.0
        return img
