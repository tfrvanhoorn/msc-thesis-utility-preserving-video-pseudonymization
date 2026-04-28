from __future__ import annotations

import logging
from dataclasses import dataclass
from collections import defaultdict
from typing import Sequence

import numpy as np
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
        image_size: int = 256,
        margin: int = 0,
        score_threshold: float = 0.55,
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
            thresholds=(0.5, 0.6, 0.6),
            min_face_size=min_face_size,
            post_process=post_process,
            device=self.device,
        )
        self._warned_batch_fallback = False
        self._warned_empty_detect = False
        self._small_face_retry_scale = 2.0

    def detect(self, image: torch.Tensor) -> Sequence[Detection]:
        prepared = self._prepare_image_for_mtcnn(image)
        if prepared is None:
            return []

        boxes, probs, landmarks = self._safe_mtcnn_detect(prepared)
        detections = self._postprocess_single_detect(boxes, probs, landmarks)
        if detections:
            return detections

        if self._small_face_retry_scale <= 1.0:
            return detections
        upscaled = self._upscale_for_retry(prepared, self._small_face_retry_scale)
        if upscaled is None:
            return detections

        boxes_up, probs_up, landmarks_up = self._safe_mtcnn_detect(upscaled)
        retry_detections = self._postprocess_single_detect(boxes_up, probs_up, landmarks_up)
        if not retry_detections:
            return detections
        return [self._rescale_detection_coords(det, self._small_face_retry_scale) for det in retry_detections]

    def detect_batch(self, images: Sequence[torch.Tensor]) -> Sequence[Sequence[Detection]]:
        if images is None:
            return []
        image_list = list(images)
        if not image_list:
            return []

        prepared: list[torch.Tensor | None] = [self._prepare_image_for_mtcnn(image) for image in image_list]
        result: list[list[Detection]] = [[] for _ in image_list]

        # MTCNN batched detect expects equal HxW. Group by shape to preserve
        # most batching benefits while supporting mixed-size inputs.
        groups: dict[tuple[int, int, int], list[tuple[int, torch.Tensor]]] = defaultdict(list)
        for idx, img in enumerate(prepared):
            if img is None:
                continue
            h, w, c = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
            groups[(h, w, c)].append((idx, img))

        for _, items in groups.items():
            indices = [idx for idx, _ in items]
            tensors = [img for _, img in items]
            try:
                batch_np = np.stack([img.numpy() for img in tensors], axis=0)
                with torch.no_grad():
                    boxes_b, probs_b, landmarks_b = self._mtcnn.detect(batch_np, landmarks=True)

                if boxes_b is None:
                    continue

                for local_idx, global_idx in enumerate(indices):
                    boxes = boxes_b[local_idx] if local_idx < len(boxes_b) else None
                    probs = probs_b[local_idx] if probs_b is not None and local_idx < len(probs_b) else None
                    landmarks = landmarks_b[local_idx] if landmarks_b is not None and local_idx < len(landmarks_b) else None
                    result[global_idx] = self._postprocess_single_detect(boxes, probs, landmarks)
            except Exception as exc:
                if not self._warned_batch_fallback:
                    logger.warning("MTCNN shape-group batch detect failed; falling back per image: %s", exc)
                    self._warned_batch_fallback = True
                else:
                    logger.debug("MTCNN shape-group batch detect failed; falling back per image: %s", exc)
                for global_idx in indices:
                    img = prepared[global_idx]
                    if img is None:
                        continue
                    result[global_idx] = self.detect(img)

        return result

    def _prepare_image_for_mtcnn(self, image: torch.Tensor | None) -> torch.Tensor | None:
        if image is None:
            return None
        if not torch.is_tensor(image):
            return None
        if image.numel() == 0:
            return None

        img = image.detach()
        if img.max() <= 1.01:
            img = (img * 255).byte()
        else:
            img = img.byte()
        if img.dim() == 3 and img.shape[0] == 3:
            img = img.permute(1, 2, 0)

        # facenet_pytorch detect stacks to numpy internally; keep on CPU.
        return img.contiguous().cpu()

    def _safe_mtcnn_detect(self, prepared: torch.Tensor):
        try:
            with torch.no_grad():
                return self._mtcnn.detect(prepared, landmarks=True)
        except RuntimeError as exc:
            if self._is_empty_cat_error(exc):
                if not self._warned_empty_detect:
                    logger.warning("MTCNN returned no candidate boxes (empty cat); treating as no detections.")
                    self._warned_empty_detect = True
                else:
                    logger.debug("MTCNN returned no candidate boxes (empty cat); treating as no detections.")
                return None, None, None
            raise

    @staticmethod
    def _is_empty_cat_error(exc: RuntimeError) -> bool:
        msg = str(exc)
        return "torch.cat(): expected a non-empty list of Tensors" in msg

    @staticmethod
    def _upscale_for_retry(prepared: torch.Tensor, scale: float) -> torch.Tensor | None:
        if scale <= 1.0:
            return None
        if prepared.dim() != 3 or prepared.shape[2] != 3:
            return None
        h, w = int(prepared.shape[0]), int(prepared.shape[1])
        new_h = max(2, int(round(h * scale)))
        new_w = max(2, int(round(w * scale)))
        upscaled = torch.nn.functional.interpolate(
            prepared.permute(2, 0, 1).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).permute(1, 2, 0)
        return upscaled.clamp(0.0, 255.0).byte().contiguous().cpu()

    def _rescale_detection_coords(self, det: Detection, scale: float) -> Detection:
        inv = 1.0 / float(scale)
        return Detection(
            bbox=det.bbox * inv,
            landmarks=det.landmarks * inv,
            score=det.score,
            aligned=det.aligned,
        )

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
            # Reject degenerate landmark constellations that commonly appear on non-face circular textures.
            if lm_t.shape != (5, 2):
                continue
            if float(lm_t.std(dim=0).mean().item()) < 2.0:
                continue

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
