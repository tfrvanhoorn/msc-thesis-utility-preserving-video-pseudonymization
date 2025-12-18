from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from insightface import model_zoo
from insightface.utils.storage import ensure_available


class EmbeddingModel:
    def embed(self, aligned_faces: Sequence[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class ArcFaceEmbedder(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "arcface_r100_v1",
        release_name: str | None = None,
        ctx_id: int = -1,
        providers: Sequence[str] | None = None,
        root: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.release_name = release_name
        self.ctx_id = ctx_id
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self.root = root
        self._model = None
        self.embedding_size: int | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        root_dir = Path(self.root) if self.root is not None else Path.home() / ".insightface"
        root_dir.mkdir(parents=True, exist_ok=True)
        model_file = self._resolve_model_file(root_dir)
        self._model = model_zoo.get_model(
            str(model_file),
            providers=self.providers,
        )
        if self._model is None:
            raise RuntimeError(
                f"InsightFace model '{self.model_name}' not found (searched under {root_dir}). "
                "Ensure the ONNX file exists or update embedding.name to an available model."
            )
        self._model.prepare(ctx_id=self.ctx_id, providers=self.providers)

    def _resolve_model_file(self, root_dir: Path) -> Path:
        name = self.model_name
        release = self.release_name
        explicit_path = Path(name)
        if explicit_path.suffix == ".onnx" and explicit_path.exists():
            return explicit_path

        models_root = root_dir / "models"

        if release:
            release_dir = models_root / release
            if not release_dir.exists():
                ensure_available("models", release, root=str(root_dir))
            release_dir = models_root / release
            nested = release_dir / release
            if nested.exists():
                release_dir = nested
            if release_dir.exists():
                candidate = release_dir / f"{name}.onnx"
                if candidate.exists():
                    return candidate
                onnx = self._pick_latest_onnx(release_dir)
                if onnx:
                    return onnx
            raise FileNotFoundError(
                f"Model '{name}.onnx' not found inside release '{release}' (looked under {release_dir}). "
                "Ensure the release zip is downloaded/extracted correctly or set embedding.name to an existing file."
            )

        direct_dir = models_root / name
        if direct_dir.exists():
            onnx = self._pick_latest_onnx(direct_dir)
            if onnx:
                return onnx

        flat_file = models_root / f"{name}.onnx"
        if flat_file.exists():
            return flat_file

        for bundle_dir in models_root.iterdir():
            if not bundle_dir.is_dir():
                continue
            candidate = bundle_dir / f"{name}.onnx"
            if candidate.exists():
                return candidate

        ensure_available("models", name, root=str(root_dir))
        direct_dir = models_root / name
        onnx = self._pick_latest_onnx(direct_dir)
        if onnx:
            return onnx

        raise FileNotFoundError(
            f"Could not locate ONNX for '{name}'. Searched under {models_root}."
        )

    @staticmethod
    def _pick_latest_onnx(folder: Path) -> Path | None:
        candidates = sorted(folder.glob("*.onnx"))
        return candidates[-1] if candidates else None

    def embed(self, aligned_faces: Sequence[np.ndarray]) -> np.ndarray:
        if not aligned_faces:
            size = self.embedding_size or 512
            return np.empty((0, size), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        features = []
        for face in aligned_faces:
            bgr = face[:, :, ::-1].copy()
            emb = self._run_inference(bgr)
            features.append(emb)
        embeddings = np.vstack(features)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings.astype(np.float32)

    def _run_inference(self, bgr: np.ndarray) -> np.ndarray:
        if hasattr(self._model, "get_feat"):
            feat = self._model.get_feat(bgr)[0]
            return feat
        try:
            return self._model.get(bgr)
        except TypeError:
            if hasattr(self._model, "get_feat"):
                feat = self._model.get_feat(bgr)[0]
                return feat
            raise
