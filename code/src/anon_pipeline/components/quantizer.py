from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


class Quantizer:
    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, embeddings: np.ndarray) -> None:
        return None


@dataclass
class ProductSphericalKMeansQuantizer(Quantizer):
    num_subspaces: int = 32
    num_prototypes: int = 64
    max_iters: int = 20
    tol: float = 1e-4
    random_seed: Optional[int] = 123
    output_mode: str = "majority"
    _prototypes: list[np.ndarray] = field(default_factory=list, init=False, repr=False)

    def fit(self, embeddings: np.ndarray) -> None:
        if embeddings.size == 0:
            raise ValueError("Cannot fit product spherical k-means with no embeddings.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")
        subspace_dim = self._subspace_dim(embeddings.shape[1])
        chunks = self._split(embeddings, subspace_dim)
        rng = np.random.default_rng(self.random_seed)
        prototypes: list[np.ndarray] = []
        for idx, chunk in enumerate(chunks):
            proto = self._fit_single(chunk, rng)
            prototypes.append(proto)
        self._prototypes = prototypes

    def quantize(self, embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return np.empty((0, self.num_subspaces), dtype=np.float32)
        subspace_dim = self._subspace_dim(embeddings.shape[1])
        if not self._prototypes:
            raise ValueError("ProductSphericalKMeansQuantizer must be fitted before quantization.")
        chunks = self._split(embeddings, subspace_dim)
        codes: list[np.ndarray] = []
        for chunk, prot in zip(chunks, self._prototypes):
            normed = self._normalize(chunk)
            sims = normed @ prot.T
            idx = np.argmax(sims, axis=1).astype(np.float32, copy=False)
            codes.append(idx)
        stacked = np.stack(codes, axis=1)
        if self.output_mode == "vector":
            return stacked.astype(np.float32)
        if self.output_mode == "majority":
            out = np.empty((stacked.shape[0], 1), dtype=np.float32)
            for i in range(stacked.shape[0]):
                vals = stacked[i].astype(np.int64, copy=False)
                counts = np.bincount(vals, minlength=self.num_prototypes)
                out[i, 0] = float(np.argmax(counts))
            return out
        raise ValueError(f"Unsupported output_mode '{self.output_mode}' (use 'vector' or 'majority').")

    def export_state(self) -> dict[str, Sequence[np.ndarray]]:
        return {"prototypes": [p.copy() for p in self._prototypes]}

    def load_state(self, state: dict[str, Sequence[np.ndarray]]) -> None:
        prot = state.get("prototypes") if state else None
        if prot is None:
            self._prototypes = []
            return
        self._prototypes = [np.asarray(p, dtype=np.float32) for p in prot]

    def _fit_single(self, embeddings: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        data = self._normalize(embeddings)
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
        return centroids.astype(np.float32)

    def _subspace_dim(self, input_dim: int) -> int:
        if self.num_subspaces <= 0:
            raise ValueError("num_subspaces must be positive.")
        if input_dim % self.num_subspaces != 0:
            raise ValueError(
                f"Embedding dim {input_dim} is not divisible by num_subspaces={self.num_subspaces}; "
                "choose a divisor of the embedding dimension."
            )
        return input_dim // self.num_subspaces

    def _split(self, embeddings: np.ndarray, subspace_dim: int) -> list[np.ndarray]:
        return [embeddings[:, i * subspace_dim : (i + 1) * subspace_dim] for i in range(self.num_subspaces)]

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

    def _init_centroids(self, data: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(data.shape[0], size=self.num_prototypes, replace=data.shape[0] < self.num_prototypes)
        centroids = data[idx]
        centroids = self._normalize(centroids)
        return centroids
