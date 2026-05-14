from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
try:
    import torch
except ImportError:  # pragma: no cover - optional for faceqnet-only environments
    torch = None

try:
    import cv2
except ImportError:  # pragma: no cover - dependency is optional at import time
    cv2 = None

try:
    from mtcnn import MTCNN
except ImportError:  # pragma: no cover - dependency is optional at import time
    MTCNN = None

try:
    from tensorflow.keras.models import load_model
except ImportError:  # pragma: no cover - dependency is optional at import time
    load_model = None


class FaceQnetEvaluator:
    INPUT_SIZE: Tuple[int, int] = (224, 224)

    def __init__(
        self,
        *,
        model_path: Path,
        device: torch.device | str,
        crop_margin: float = 0.1,
        min_face_confidence: float = 0.9,
    ) -> None:
        if load_model is None:
            raise ImportError("FaceQnet metric requested but tensorflow is not installed")
        if MTCNN is None:
            raise ImportError("FaceQnet metric requested but mtcnn is not installed")
        if cv2 is None:
            raise ImportError("FaceQnet metric requested but opencv-python-headless is not installed")

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"FaceQnet model not found at: {model_path}")

        self.model = load_model(str(model_path), compile=False)
        self.device = str(device)
        self.detector = MTCNN()
        self.crop_margin = float(crop_margin)
        self.min_face_confidence = float(min_face_confidence)

    def score_frame(self, frame: "torch.Tensor | np.ndarray") -> float | None:
        if frame is None:
            raise ValueError("Expected a frame array, got None")

        image = self._to_uint8_rgb(frame)
        bbox = self._detect_best_face(image)
        if bbox is None:
            return None
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        resized = cv2.resize(crop, self.INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        input_batch = resized.astype(np.float32)[np.newaxis, ...]
        raw = float(self.model.predict(input_batch, verbose=0)[0])
        return float(np.clip(raw, 0.0, 1.0))

    def _detect_best_face(self, image_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        faces = self.detector.detect_faces(image_rgb)
        if not faces:
            return None

        best = max(faces, key=lambda face: float(face.get("confidence", 0.0)))
        best_prob = float(best.get("confidence", 0.0))
        if best_prob < self.min_face_confidence:
            return None

        box = best.get("box")
        if not box or len(box) != 4:
            return None

        x1, y1, w, h = [int(v) for v in box]
        x2 = x1 + w
        y2 = y1 + h
        h_img, w_img = image_rgb.shape[:2]
        w = x2 - x1
        h = y2 - y1
        mx = int(w * self.crop_margin)
        my = int(h * self.crop_margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w_img, x2 + mx)
        y2 = min(h_img, y2 + my)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _to_uint8_rgb(frame: "torch.Tensor | np.ndarray") -> np.ndarray:
        if torch is not None and isinstance(frame, torch.Tensor):
            arr = frame.detach().to("cpu").numpy()
        else:
            arr = np.asarray(frame)

        if arr.ndim != 3:
            raise ValueError(f"Expected frame with 3 dimensions, got shape {arr.shape}")

        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.shape[-1] != 3:
            raise ValueError(f"Expected 3 channels in last dim, got shape {arr.shape}")

        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)

        return arr
