from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, prewhiten

logger = logging.getLogger(__name__)


class EmbeddingModel:
    def embed(self, aligned_faces: Sequence[object], source_paths: Sequence[Path] | None = None) -> np.ndarray:
        raise NotImplementedError


class FacenetEmbedder(EmbeddingModel):
    def __init__(self, pretrained: str = "vggface2", device: str | torch.device | None = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.embedding_size: int = 512

    def embed(self, aligned_faces: Sequence[object], source_paths: Sequence[Path] | None = None) -> np.ndarray:
        if not aligned_faces:
            return np.empty((0, self.embedding_size), dtype=np.float32)

        batch: list[torch.Tensor] = []
        for face in aligned_faces:
            tensor = self._to_tensor(face)
            tensor = prewhiten(tensor)
            batch.append(tensor)

        faces_tensor = torch.stack(batch, dim=0).to(self.device)
        with torch.no_grad():
            embeddings = self.model(faces_tensor)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy().astype(np.float32)

    def _to_tensor(self, face: object) -> torch.Tensor:
        if isinstance(face, torch.Tensor):
            t = face
        else:
            arr = np.asarray(face)
            if arr.ndim == 3 and arr.shape[0] == 3:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if arr.max() > 1.5:
                arr = arr / 255.0
            t = torch.from_numpy(arr)

        if t.ndim == 3 and t.shape[0] != 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
        if t.ndim != 3 or t.shape[0] != 3:
            raise ValueError(f"Expected face tensor shape (3,H,W), got {tuple(t.shape)}")
        return t.float()


class SemanticAttributeEmbedder(EmbeddingModel):
    def __init__(
        self,
        feature_selector: Mapping[str, Sequence[str]] | Sequence[str] | None = None,
        feature_classifiers: Mapping[str, object] | None = None,
        default_value: int = 0,
    ) -> None:
        self.default_value = int(default_value != 0)
        keep: list[str] = []
        if feature_selector:
            if isinstance(feature_selector, Mapping):
                keep = list(feature_selector.get("keep", []))
            elif hasattr(feature_selector, "keep"):
                keep = list(getattr(feature_selector, "keep"))
            else:
                keep = list(feature_selector)
        self._keep_names: list[str] = [self._normalize_attr_name(k) for k in keep]
        self.embedding_size = len(self._keep_names)
        self._classifiers = {self._normalize_attr_name(k): v for k, v in (feature_classifiers or {}).items()}

    def embed(self, aligned_faces: Sequence[object], source_paths: Sequence[Path] | None = None) -> np.ndarray:
        num_faces = len(aligned_faces)
        if num_faces == 0:
            return np.empty((0, self.embedding_size), dtype=np.int64)
        if self.embedding_size == 0:
            logger.warning("SemanticAttributeEmbedder has no selected attributes; returning empty embeddings")
            return np.empty((num_faces, 0), dtype=np.int64)

        paths = list(source_paths) if source_paths else []
        vectors: list[np.ndarray] = []
        for idx in range(num_faces):
            face = aligned_faces[idx]
            path = paths[idx] if idx < len(paths) else (paths[-1] if paths else None)
            vectors.append(self._vector_for_face(face, path))
        return np.vstack(vectors).astype(np.int64, copy=False) if vectors else np.empty((0, self.embedding_size), dtype=np.int64)

    def _vector_for_face(self, face: object, path: Path | None) -> np.ndarray:
        if self.embedding_size == 0:
            return np.empty((0,), dtype=np.int64)
        bits = []
        for name in self._keep_names:
            cls = self._classifiers.get(name)
            if cls is None:
                bits.append(self.default_value)
            else:
                try:
                    bits.append(1 if bool(cls(face, path)) else 0)
                except Exception:
                    logger.exception("Classifier for feature '%s' failed; defaulting to %s", name, self.default_value)
                    bits.append(self.default_value)
        return np.asarray(bits, dtype=np.int64)

    @staticmethod
    def _normalize_attr_name(name: str) -> str:
        return name.replace("-", "_").replace(" ", "_").lower()
