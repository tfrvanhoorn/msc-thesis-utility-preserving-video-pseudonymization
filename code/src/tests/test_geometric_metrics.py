from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from anon_pipeline.kfaar.geometric_metrics import GeometricUtilityEvaluator  # noqa: E402


class _Landmark:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


def _build_landmarks(delta: float = 0.0) -> list[_Landmark]:
    landmarks = [_Landmark(0.5, 0.5, 0.0) for _ in range(478)]
    for idx in GeometricUtilityEvaluator._EXPRESSION_FACE_MESH_IDX:
        landmarks[idx] = _Landmark(0.2 + delta, 0.3 + delta, 0.1 + delta)
    return landmarks


def test_facial_expression_mse_zero_for_identical_landmarks() -> None:
    evaluator = GeometricUtilityEvaluator.__new__(GeometricUtilityEvaluator)
    a = _build_landmarks(delta=0.0)
    b = _build_landmarks(delta=0.0)

    mse = evaluator._facial_expression_mse(a, b)

    assert mse == 0.0


def test_facial_expression_mse_positive_for_shifted_landmarks() -> None:
    evaluator = GeometricUtilityEvaluator.__new__(GeometricUtilityEvaluator)
    a = _build_landmarks(delta=0.0)
    b = _build_landmarks(delta=0.1)

    mse = evaluator._facial_expression_mse(a, b)

    assert mse > 0.0


def test_head_posture_returns_none_for_invalid_image_size() -> None:
    evaluator = GeometricUtilityEvaluator.__new__(GeometricUtilityEvaluator)
    a = _build_landmarks(delta=0.0)
    b = _build_landmarks(delta=0.0)

    err = evaluator._head_posture_mse(a, b, image_size=(0, 0))

    assert err is None
