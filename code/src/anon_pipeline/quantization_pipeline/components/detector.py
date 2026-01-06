from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence
from pathlib import Path

import numpy as np
from insightface.app import FaceAnalysis


logger = logging.getLogger(__name__)


@dataclass
class Detection:
    bbox: np.ndarray
    landmarks: np.ndarray
    score: float


class FaceDetector:
    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        raise NotImplementedError


class RetinaFaceDetector(FaceDetector):
    def __init__(
        self,
        model_name: str = "buffalo_l",
        score_threshold: float = 0.95,
        det_size: tuple[int, int] = (640, 640),
        max_faces: int | None = None,
        ctx_id: int = -1,
        providers: Sequence[str] | None = None,
        root: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.score_threshold = score_threshold
        self.det_size = det_size
        self.max_faces = max_faces
        self.ctx_id = ctx_id
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.root = root
        self._app: FaceAnalysis | None = None

    def _ensure_model(self) -> None:
        if self._app is not None:
            return
        init_kwargs = {
            "name": self.model_name,
            "providers": self.providers,
        }
        if self.root is not None:
            init_kwargs["root"] = self.root
        self._app = FaceAnalysis(**init_kwargs)

        self._app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
        if "detection" not in getattr(self._app, "models", {}):
            root_dir = Path(self.root) if self.root is not None else Path.home() / ".insightface"
            cache_path = root_dir / "models" / self.model_name
            raise RuntimeError(
                f"InsightFace model '{self.model_name}' is missing detection module. "
                f"Cached path: {cache_path}. Delete that directory to force re-download, then rerun."
            )

    def detect(self, image: np.ndarray) -> Sequence[Detection]:
        self._ensure_model()
        assert self._app is not None
        logger.debug(
            "Detector input shape=%s dtype=%s range=(%s-%s)",
            image.shape,
            image.dtype,
            image.min(),
            image.max(),
        )
        bgr = image[:, :, ::-1].copy()
        faces = self._app.get(bgr)
        logger.debug(
            "Raw detector faces=%s scores=%s",
            len(faces),
            [float(face.det_score) for face in faces][:5],
        )
        detections: List[Detection] = []
        for face in faces:
            if face.det_score < self.score_threshold:
                continue
            if getattr(face, "bbox", None) is None:
                logger.warning("Detector returned face without bbox; skipping")
                continue
            bbox = face.bbox.astype(np.float32)
            landmarks = _extract_landmarks(face)
            if landmarks is None:
                logger.warning("Detector missing landmarks; skipping face")
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    landmarks=landmarks,
                    score=float(face.det_score),
                )
            )
        detections.sort(key=lambda d: d.score, reverse=True)
        if self.max_faces is not None:
            detections = detections[: self.max_faces]
        logger.debug(
            "Filtered detections=%s threshold=%s",
            len(detections),
            self.score_threshold,
        )
        return detections


def _extract_landmarks(face) -> np.ndarray | None:
    for attr in ("landmark", "kps", "landmark_2d_106", "landmark_3d_68"):
        pts = getattr(face, attr, None)
        if pts is None:
            continue
        arr = np.asarray(pts, dtype=np.float32)
        arr = arr.reshape(-1, 2)
        if arr.shape[0] >= 5:
            return arr[:5]
    return None
