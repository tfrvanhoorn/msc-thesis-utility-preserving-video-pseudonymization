from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, prewhiten

logger = logging.getLogger(__name__)



class EmbeddingModel:
    def embed(
        self,
        aligned_faces: Sequence[object],
        source_paths: Sequence[Path] | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError


class FacenetEmbedder(EmbeddingModel):
    def __init__(self, pretrained: str = "vggface2", device: str | torch.device | None = None) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(self.device)
        self.embedding_size: int = 512

    def embed(
        self,
        aligned_faces: Sequence[object],
        source_paths: Sequence[Path] | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        if not aligned_faces:
            return torch.empty((0, self.embedding_size), device=self.device, dtype=torch.float32)

        batch: list[torch.Tensor] = []
        for face in aligned_faces:
            tensor = self._to_tensor(face)
            tensor = prewhiten(tensor)
            batch.append(tensor)

        faces_tensor = torch.stack(batch, dim=0).to(self.device)
        grad_ctx = nullcontext() if with_grad else torch.no_grad()
        with grad_ctx:
            embeddings = self.model(faces_tensor)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings if with_grad else embeddings.detach()

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

    def embed(
        self,
        aligned_faces: Sequence[object],
        source_paths: Sequence[Path] | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        num_faces = len(aligned_faces)
        if num_faces == 0:
            return torch.empty((0, self.embedding_size), dtype=torch.int64)
        if self.embedding_size == 0:
            logger.warning("SemanticAttributeEmbedder has no selected attributes; returning empty embeddings")
            return torch.empty((num_faces, 0), dtype=torch.int64)

        paths = list(source_paths) if source_paths else []
        vectors: list[np.ndarray] = []
        for idx in range(num_faces):
            face = aligned_faces[idx]
            path = paths[idx] if idx < len(paths) else (paths[-1] if paths else None)
            vectors.append(self._vector_for_face(face, path))
        if not vectors:
            return torch.empty((0, self.embedding_size), dtype=torch.int64)
        vec_np = np.vstack(vectors).astype(np.int64, copy=False)
        return torch.from_numpy(vec_np)

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


class ArcFaceEmbedder(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str | torch.device | None = None,
        cache_dir: Path | None = None,
        auto_download: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.embedding_size: int = 512
        self.auto_download = bool(auto_download)

        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["INSIGHTFACE_HOME"] = str(cache_dir)

        self.insightface_home = Path(os.environ.get("INSIGHTFACE_HOME", str(Path.home() / ".insightface")))
        model_dir = self.insightface_home / "models" / self.model_name
        if not self.auto_download and (not model_dir.exists() or not any(model_dir.rglob("*.onnx"))):
            raise FileNotFoundError(
                f"ArcFace model '{self.model_name}' not found in cache at {model_dir}. "
                "Enable auto download or provide a populated INSIGHTFACE_HOME."
            )

        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:  # pragma: no cover - import path depends on optional dependency state
            raise ImportError("ArcFace embedder requires the 'insightface' package") from exc

        if self.device.type == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = self.device.index if self.device.index is not None else 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        self._face_analysis = FaceAnalysis(
            name=self.model_name,
            root=str(self.insightface_home),
            providers=providers,
            allowed_modules=["recognition"],
        )
        self._face_analysis.prepare(ctx_id=ctx_id)

        self._recognition_model = None
        for model in self._face_analysis.models.values():
            if hasattr(model, "get_feat"):
                self._recognition_model = model
                break
        if self._recognition_model is None:
            raise RuntimeError(
                f"No recognition model found for ArcFace model pack '{self.model_name}'."
            )

    def embed(
        self,
        aligned_faces: Sequence[object],
        source_paths: Sequence[Path] | None = None,
        with_grad: bool = False,
    ) -> torch.Tensor:
        if not aligned_faces:
            return torch.empty((0, self.embedding_size), device=self.device, dtype=torch.float32)

        if with_grad:
            logger.warning("ArcFaceEmbedder does not support autograd; returning detached embeddings")

        input_size = tuple(getattr(self._recognition_model, "input_size", (112, 112)))
        target_w, target_h = int(input_size[0]), int(input_size[1])
        batch_bgr: list[np.ndarray] = []
        for face in aligned_faces:
            tensor = self._to_tensor(face)
            rgb = tensor.permute(1, 2, 0).cpu().numpy()
            rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).round().astype(np.uint8)
            resized = cv2.resize(rgb_uint8, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
            batch_bgr.append(bgr)

        feats_np = self._recognition_model.get_feat(np.stack(batch_bgr, axis=0))
        feats = torch.from_numpy(np.asarray(feats_np, dtype=np.float32)).to(self.device)
        feats = torch.nn.functional.normalize(feats, p=2, dim=1)
        if feats.shape[1] != self.embedding_size:
            self.embedding_size = int(feats.shape[1])
        return feats.detach()

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
