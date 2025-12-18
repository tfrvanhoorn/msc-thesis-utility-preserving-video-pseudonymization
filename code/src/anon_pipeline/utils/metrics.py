from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, Sequence

import numpy as np


def compute_consistency(seeds_by_identity: Dict[str, Sequence[str]]) -> float:
    if not seeds_by_identity:
        return 0.0
    total_pairs = 0
    consistent = 0
    for seeds in seeds_by_identity.values():
        seeds = list(seeds)
        n = len(seeds)
        if n < 2:
            continue
        total_pairs += n * (n - 1) / 2
        identical = 0
        for i in range(n):
            for j in range(i + 1, n):
                if seeds[i] == seeds[j]:
                    identical += 1
        consistent += identical
    return consistent / total_pairs if total_pairs else 0.0


def compute_collision(seeds_by_identity: Dict[str, Sequence[str]]) -> float:
    seed_to_id_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_samples = 0
    for identity, seeds in seeds_by_identity.items():
        for seed in seeds:
            seed_to_id_counts[seed][identity] += 1
            total_samples += 1

    if total_samples < 2:
        return 0.0

    total_pairs = total_samples * (total_samples - 1) / 2
    cross_identity_pairs = 0

    for id_counts in seed_to_id_counts.values():
        counts = list(id_counts.values())
        seed_total = sum(counts)
        if seed_total < 2:
            continue
        seed_pairs = seed_total * (seed_total - 1) / 2
        within_pairs = sum(c * (c - 1) / 2 for c in counts)
        cross_identity_pairs += seed_pairs - within_pairs

    return cross_identity_pairs / total_pairs if total_pairs else 0.0


def compute_intra_tracklet_variance(embeddings_by_identity: Dict[str, Sequence[np.ndarray]]) -> float:
    if not embeddings_by_identity:
        return 0.0
    variances = []
    for arrs in embeddings_by_identity.values():
        if not arrs:
            continue
        stacked = np.vstack(arrs)
        if stacked.shape[0] < 2:
            continue
        mean_vec = stacked.mean(axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm == 0:
            continue
        mean_vec = mean_vec / norm
        cos_dists = 1.0 - stacked @ mean_vec
        variances.append(float(np.mean(cos_dists)))
    return float(np.mean(variances)) if variances else 0.0


def compute_chunk_flip_rate(quantized_by_identity: Dict[str, Sequence[np.ndarray]]) -> float:
    if not quantized_by_identity:
        return 0.0
    total_pairs = 0
    total_diff_chunks = 0
    total_chunks = 0
    for arrs in quantized_by_identity.values():
        if not arrs:
            continue
        stacked = np.vstack(arrs)
        if stacked.ndim != 2 or stacked.shape[0] < 2:
            continue
        codes = stacked.astype(np.int64)
        n, k = codes.shape
        pairs = n * (n - 1) // 2
        if pairs == 0:
            continue
        diff = 0
        for i in range(n):
            diffs = (codes[i + 1 :] != codes[i]).sum()
            diff += int(diffs)
        total_pairs += pairs
        total_diff_chunks += diff
        total_chunks += k * pairs
    if total_chunks == 0:
        return 0.0
    return total_diff_chunks / total_chunks


def compute_cluster_utilization(quantized_by_identity: Dict[str, Sequence[np.ndarray]]) -> Dict[str, float]:
    counts = defaultdict(int)
    total = 0
    for arrs in quantized_by_identity.values():
        for arr in arrs:
            flat = np.asarray(arr).astype(np.int64).ravel()
            for v in flat:
                counts[int(v)] += 1
                total += 1
    if total == 0:
        return {"cluster_entropy": 0.0, "cluster_perplexity": 0.0, "unique_clusters": 0}
    probs = np.array(list(counts.values()), dtype=np.float64) / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    perplexity = float(np.exp(entropy))
    return {
        "cluster_entropy": float(entropy),
        "cluster_perplexity": perplexity,
        "unique_clusters": len(counts),
    }


def compute_confidence_margin(
    embeddings_by_identity: Dict[str, Sequence[np.ndarray]],
    quantizer: object | None,
) -> float:
    from ..components.quantizer import ProductSphericalKMeansQuantizer

    if quantizer is None:
        return 0.0

    if isinstance(quantizer, ProductSphericalKMeansQuantizer):
        if not quantizer._prototypes:
            return 0.0
        prototypes = quantizer._prototypes
        num_subspaces = quantizer.num_subspaces
        chunk_dim = prototypes[0].shape[1]

        margins: list[float] = []
        for arrs in embeddings_by_identity.values():
            for emb in arrs:
                if emb.ndim != 2 or emb.shape[1] != num_subspaces * chunk_dim:
                    continue
                for chunk_idx in range(num_subspaces):
                    chunk = emb[:, chunk_idx * chunk_dim : (chunk_idx + 1) * chunk_dim]
                    normed = chunk / np.clip(np.linalg.norm(chunk, axis=1, keepdims=True), 1e-12, None)
                    proto = prototypes[chunk_idx]
                    sims = normed @ proto.T
                    if sims.shape[1] < 2:
                        continue
                    top2 = np.partition(sims, -2, axis=1)[:, -2:]
                    d1 = top2[:, 1]
                    d2 = top2[:, 0]
                    margin = 1.0 - (d1 / np.clip(d2, 1e-12, None))
                    margins.extend(margin.tolist())
        return float(np.mean(margins)) if margins else 0.0

    return 0.0


def summarize_metrics(
    seeds_by_identity: Dict[str, Sequence[str]],
    embeddings_by_identity: Dict[str, Sequence[np.ndarray]] | None = None,
    quantized_by_identity: Dict[str, Sequence[np.ndarray]] | None = None,
    quantizer: object | None = None,
) -> Dict[str, float]:
    embeddings_by_identity = embeddings_by_identity or {}
    quantized_by_identity = quantized_by_identity or {}

    result = {
        "consistency_rate": compute_consistency(seeds_by_identity),
        "collision_rate": compute_collision(seeds_by_identity),
        "num_identities": len(seeds_by_identity),
        "total_samples": sum(len(seeds) for seeds in seeds_by_identity.values()),
    }

    result["intra_tracklet_variance"] = compute_intra_tracklet_variance(embeddings_by_identity)
    result["chunk_flip_rate"] = compute_chunk_flip_rate(quantized_by_identity)
    result["confidence_margin"] = compute_confidence_margin(embeddings_by_identity, quantizer)
    cu = compute_cluster_utilization(quantized_by_identity)
    result.update(cu)
    return result
