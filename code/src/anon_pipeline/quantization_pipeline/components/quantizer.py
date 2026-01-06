from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


class Quantizer:
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, embeddings: np.ndarray) -> None:
        return None


@dataclass
class GlobalSphericalKMeansQuantizer(Quantizer):
    num_prototypes: int = 64
    max_iters: int = 20
    tol: float = 1e-4
    random_seed: Optional[int] = 123
    _centroids: Optional[np.ndarray] = None

    def fit(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            raise ValueError("Cannot fit spherical k-means with no embeddings.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        data = self._normalize(embeddings)
        rng = np.random.default_rng(self.random_seed)
        centroids = self._init_centroids(data, rng)
        for _ in range(self.max_iters):
            sims = data @ centroids.T
            assignments = sims.argmax(axis=1)
            new_centroids = []
            for idx in range(self.num_prototypes):
                mask = assignments == idx
                if not np.any(mask):
                    new_centroids.append(data[rng.integers(0, data.shape[0])])
                    continue
                centroid = data[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm == 0:
                    new_centroids.append(centroids[idx])
                else:
                    new_centroids.append(centroid / norm)
            new_centroids = np.vstack(new_centroids)
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            if shift < self.tol:
                break
        self._centroids = centroids.astype(np.float32)

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.empty((0, 1), dtype=np.int64)
        if self._centroids is None:
            raise ValueError("GlobalSphericalKMeansQuantizer must be fitted before quantization.")
        data = self._normalize(embeddings)
        sims = data @ self._centroids.T
        labels = sims.argmax(axis=1).astype(np.int64, copy=False)
        return labels.reshape(-1, 1)

    def export_state(self) -> dict[str, np.ndarray]:
        return {"centroids": None if self._centroids is None else self._centroids.copy()}

    def load_state(self, state: dict[str, np.ndarray] | None) -> None:
        centroids = state.get("centroids") if state else None
        self._centroids = None if centroids is None else np.asarray(centroids, dtype=np.float32)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _init_centroids(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(data.shape[0], size=self.num_prototypes, replace=data.shape[0] < self.num_prototypes)
        centroids = data[idx]
        return self._normalize(centroids)


class IdentityQuantizer(Quantizer):
    """Pass-through quantizer for already-discrete embeddings (e.g., semantic bits)."""

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.empty_like(embeddings, dtype=np.int64)
        return embeddings.astype(np.int64, copy=False)


@dataclass
class BioHashingQuantizer(Quantizer):
    input_dim: int = 512
    output_dim: int = 2048
    random_seed: int = 123
    packbits: bool = True
    bitorder: str = "big"
    _projection: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self._projection is None:
            rng = np.random.default_rng(self.random_seed)
            self._projection = rng.standard_normal((self.input_dim, self.output_dim)).astype(np.float32)

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            if self.packbits:
                return np.empty((0, self._byte_length()), dtype=np.uint8)
            return np.empty((0, self.output_dim), dtype=np.uint8)
        if embeddings.shape[1] != self.input_dim:
            raise ValueError(f"Expected embedding dim {self.input_dim}, got {embeddings.shape[1]}")
        proj = self._projection
        assert proj is not None
        projected = embeddings @ proj
        bits = (projected > 0).astype(np.uint8, copy=False)
        if self.packbits:
            return np.packbits(bits, axis=1, bitorder=self.bitorder)
        return bits

    def export_state(self) -> dict[str, np.ndarray]:
        proj = None if self._projection is None else self._projection.copy()
        return {"projection": proj}

    def load_state(self, state: dict[str, np.ndarray] | None) -> None:
        if state and "projection" in state and state["projection"] is not None:
            self._projection = np.asarray(state["projection"], dtype=np.float32)

    def _byte_length(self) -> int:
        return (self.output_dim + 7) // 8
