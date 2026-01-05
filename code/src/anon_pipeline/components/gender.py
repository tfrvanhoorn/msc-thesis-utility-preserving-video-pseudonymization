from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from insightface import model_zoo


class GenderClassifier:
    def __init__(
        self,
        providers: Sequence[str] | None = None,
        ctx_id: int = 0,
        root: str | Path | None = None,
        model_path: str | Path | None = None,
        default_value: bool = False,
    ) -> None:
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.ctx_id = ctx_id
        self.root = Path(root) if root else Path.home() / ".insightface"
        self.model_path = Path(model_path) if model_path else None
        self.default_value = bool(default_value)
        self._model = None
        self._load_attempted = False

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        candidate = self._resolve_model_path()
        if candidate is None:
            return False
        try:
            self._model = model_zoo.get_model(str(candidate), providers=self.providers)
            self._model.prepare(ctx_id=self.ctx_id, providers=self.providers)
            return True
        except Exception:
            self._model = None
            return False

    def _resolve_model_path(self) -> Path | None:
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)
        default_path = self.root / "models" / "genderage.onnx"
        if default_path.exists():
            return default_path
        return None

    def __call__(self, face: np.ndarray, path=None) -> bool:
        if not self._ensure_model():
            return self.default_value
        model = self._model
        assert model is not None
        bgr = face[:, :, ::-1].copy() if face.ndim == 3 else np.dstack([face] * 3)
        if bgr.shape[0] != 112 or bgr.shape[1] != 112:
            bgr = np.ascontiguousarray(np.resize(bgr, (112, 112, 3)))
        gender, _ = model.get(bgr)
        try:
            return bool(int(gender) == 1)
        except Exception:
            return self.default_value
