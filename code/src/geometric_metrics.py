from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass
class GeometricPairError:
    head_posture_error: float | None
    facial_expression_error: float | None


class GeometricUtilityEvaluator:
    """Evaluate geometric utility using head posture and expression landmark errors."""

    _HEAD_POSE_FACE_MESH_IDX: tuple[int, int, int, int, int, int] = (1, 152, 33, 263, 61, 291)
    _HEAD_POSE_MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0),
        ],
        dtype=np.float64,
    )
    _EXPRESSION_FACE_MESH_IDX: tuple[int, ...] = (
        13,
        14,
        78,
        81,
        82,
        87,
        88,
        95,
        178,
        191,
        308,
        311,
        312,
        317,
        318,
        324,
        402,
        415,
        33,
        133,
        145,
        159,
        263,
        362,
        374,
        386,
    )

    def __init__(self) -> None:
        if mp is None:
            raise ImportError("mediapipe is required for geometric utility evaluation")
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

    def close(self) -> None:
        if hasattr(self._face_mesh, "close"):
            self._face_mesh.close()

    def compute_pair_errors(self, original_image: np.ndarray, generated_image: np.ndarray) -> GeometricPairError:
        original_landmarks = self._extract_landmarks(original_image)
        generated_landmarks = self._extract_landmarks(generated_image)
        if original_landmarks is None or generated_landmarks is None:
            return GeometricPairError(head_posture_error=None, facial_expression_error=None)

        head_posture_error = self._head_posture_mse(
            original_landmarks,
            generated_landmarks,
            image_size=original_image.shape[:2],
        )
        expression_error = self._facial_expression_mse(original_landmarks, generated_landmarks)
        if head_posture_error is None or expression_error is None:
            return GeometricPairError(head_posture_error=None, facial_expression_error=None)

        return GeometricPairError(
            head_posture_error=float(head_posture_error),
            facial_expression_error=float(expression_error),
        )

    def _extract_landmarks(self, image: np.ndarray):
        if image is None or image.size == 0:
            return None
        if image.ndim != 3 or image.shape[2] != 3:
            return None

        image_rgb = image
        if image.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        result = self._face_mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return None
        return result.multi_face_landmarks[0].landmark

    def _head_posture_mse(self, landmarks_a, landmarks_b, *, image_size: Sequence[int]) -> float | None:
        h, w = int(image_size[0]), int(image_size[1])
        if h <= 0 or w <= 0:
            return None

        euler_a = self._estimate_euler_angles(landmarks_a, image_width=w, image_height=h)
        euler_b = self._estimate_euler_angles(landmarks_b, image_width=w, image_height=h)
        if euler_a is None or euler_b is None:
            return None

        diff = euler_a - euler_b
        return float(np.mean(np.square(diff), dtype=np.float64))

    def _estimate_euler_angles(self, landmarks, *, image_width: int, image_height: int) -> np.ndarray | None:
        image_points = []
        for idx in self._HEAD_POSE_FACE_MESH_IDX:
            lm = landmarks[idx]
            image_points.append((lm.x * image_width, lm.y * image_height))
        image_points_np = np.array(image_points, dtype=np.float64)

        focal_length = float(image_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, image_width / 2.0],
                [0.0, focal_length, image_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _translation_vector = cv2.solvePnP(
            self._HEAD_POSE_MODEL_POINTS,
            image_points_np,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles = cv2.RQDecomp3x3(rotation_matrix)[0]
        return np.array(angles, dtype=np.float64)

    def _facial_expression_mse(self, landmarks_a, landmarks_b) -> float:
        coords_a = self._gather_expression_coords(landmarks_a)
        coords_b = self._gather_expression_coords(landmarks_b)
        diff = coords_a - coords_b
        return float(np.mean(np.square(diff), dtype=np.float64))

    def _gather_expression_coords(self, landmarks) -> np.ndarray:
        coords = []
        for idx in self._EXPRESSION_FACE_MESH_IDX:
            lm = landmarks[idx]
            coords.append((lm.x, lm.y, lm.z))
        return np.array(coords, dtype=np.float64)
