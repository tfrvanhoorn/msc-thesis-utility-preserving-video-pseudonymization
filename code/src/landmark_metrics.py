from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import dlib
except ImportError:  # pragma: no cover - dependency is optional at import time
    dlib = None


@dataclass(frozen=True)
class LandmarkPairDistance:
    distance: float | None


class LandmarkDistanceEvaluator:
    """Evaluate normalized landmark distance on input/output frame pairs.

    The score follows common facial landmark normalization practice:
    1) Similarity-align generated landmarks to input landmarks.
    2) Compute mean Euclidean landmark distance.
    3) Normalize by inter-ocular distance of the input landmarks.
    """

    def __init__(self, shape_predictor_path: Path, detector_upsample: int = 0) -> None:
        if dlib is None:
            raise ImportError("dlib is required for landmark distance evaluation")

        if not shape_predictor_path.exists():
            raise FileNotFoundError(f"dlib shape predictor not found: {shape_predictor_path}")

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(shape_predictor_path))
        self._upsample = max(int(detector_upsample), 0)

    def close(self) -> None:
        return

    def compute_pair_distance(self, original_image: np.ndarray, generated_image: np.ndarray) -> LandmarkPairDistance:
        lm_in = self._extract_landmarks(original_image)
        lm_out = self._extract_landmarks(generated_image)
        if lm_in is None or lm_out is None:
            return LandmarkPairDistance(distance=None)

        aligned_out = self._similarity_align(lm_out, lm_in)
        if aligned_out is None:
            return LandmarkPairDistance(distance=None)

        inter_ocular = np.linalg.norm(lm_in[36] - lm_in[45])
        if inter_ocular <= 1e-6:
            return LandmarkPairDistance(distance=None)

        per_point = np.linalg.norm(aligned_out - lm_in, axis=1)
        distance = float(np.mean(per_point, dtype=np.float64) / inter_ocular)
        return LandmarkPairDistance(distance=distance)

    def _extract_landmarks(self, image: np.ndarray) -> np.ndarray | None:
        if image is None or image.size == 0:
            return None
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        rgb = image
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dets = self._detector(gray, self._upsample)
        if len(dets) == 0:
            return None

        rect = max(dets, key=lambda d: d.width() * d.height())
        shape = self._predictor(gray, rect)
        points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)], dtype=np.float64)
        if points.shape[0] < 68:
            return None
        return points

    @staticmethod
    def _similarity_align(source: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        if source.shape != target.shape:
            return None

        src_mean = source.mean(axis=0)
        dst_mean = target.mean(axis=0)
        src_centered = source - src_mean
        dst_centered = target - dst_mean

        src_var = float(np.sum(src_centered**2))
        if src_var <= 1e-12:
            return None

        cov = (dst_centered.T @ src_centered) / float(source.shape[0])
        u, s, vt = np.linalg.svd(cov)
        r = u @ vt
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1.0
            r = u @ vt

        scale = float(np.sum(s) / (src_var / float(source.shape[0])))
        t = dst_mean - scale * (r @ src_mean)

        aligned = (scale * (source @ r.T)) + t
        return aligned
