from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class HistogramAggregator:
    num_bins: int = 4096
    min_score: float = -1.0
    max_score: float = 1.0
    counts: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.counts = torch.zeros(self.num_bins, dtype=torch.int64)

    @property
    def total_count(self) -> int:
        return int(self.counts.sum().item())

    def update(self, scores: torch.Tensor) -> None:
        if scores is None or scores.numel() == 0:
            return
        vals = scores.detach().to("cpu", dtype=torch.float32).reshape(-1)
        vals = vals.clamp(self.min_score, self.max_score)
        scale = float(self.num_bins) / float(self.max_score - self.min_score)
        idx = torch.floor((vals - self.min_score) * scale).to(torch.int64)
        idx = idx.clamp(min=0, max=self.num_bins - 1)
        self.counts += torch.bincount(idx, minlength=self.num_bins)

    def add_inplace(self, other: "HistogramAggregator") -> None:
        self.counts += other.counts

    @classmethod
    def merged(cls, histograms: List["HistogramAggregator"]) -> "HistogramAggregator":
        if not histograms:
            return cls()
        merged_hist = cls(
            num_bins=histograms[0].num_bins,
            min_score=histograms[0].min_score,
            max_score=histograms[0].max_score,
        )
        for hist in histograms:
            merged_hist.add_inplace(hist)
        return merged_hist


@dataclass
class MetricsAccumulator:
    anonymization_enabled: bool = True
    diversity_enabled: bool = True
    detected_generated: int = 0
    total_generated: int = 0
    detection_score_sum: float = 0.0
    anonymization_total: int = 0
    synchronism_total: int = 0
    synchronism_within_total: int = 0
    synchronism_cross_total: int = 0
    differentiation_total: int = 0
    diversity_total: int = 0
    landmark_distance_sum: float = 0.0
    landmark_pairs_valid: int = 0
    landmark_pairs_invalid: int = 0
    lpips_distance_sum: float = 0.0
    lpips_pairs_valid: int = 0
    lpips_pairs_invalid: int = 0
    ssim_similarity_sum: float = 0.0
    ssim_pairs_valid: int = 0
    ssim_pairs_invalid: int = 0
    synchronism_chunk_size: int = 256
    show_progress: bool = True
    histogram_bins: int = 4096
    _sync_buckets: Dict[int, Dict[str, List[torch.Tensor]]] = field(default_factory=dict)
    _anonymization_hist: HistogramAggregator = field(init=False, repr=False)
    _synchronism_total_hist: HistogramAggregator = field(init=False, repr=False)
    _synchronism_within_hist: HistogramAggregator = field(init=False, repr=False)
    _synchronism_cross_hist: HistogramAggregator = field(init=False, repr=False)
    _diversity_hist: HistogramAggregator = field(init=False, repr=False)
    _differentiation_hist: HistogramAggregator = field(init=False, repr=False)
    _synchronism_computed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._anonymization_hist = HistogramAggregator(num_bins=self.histogram_bins)
        self._synchronism_total_hist = HistogramAggregator(num_bins=self.histogram_bins)
        self._synchronism_within_hist = HistogramAggregator(num_bins=self.histogram_bins)
        self._synchronism_cross_hist = HistogramAggregator(num_bins=self.histogram_bins)
        self._diversity_hist = HistogramAggregator(num_bins=self.histogram_bins)
        self._differentiation_hist = HistogramAggregator(num_bins=self.histogram_bins)

    def update_detection(
        self,
        gen_mask: torch.Tensor | List[bool],
        detection_scores: torch.Tensor | None = None,
    ) -> None:
        mask = torch.as_tensor(gen_mask, dtype=torch.bool)
        self.total_generated += int(mask.numel())
        self.detected_generated += int(mask.sum().item())
        if detection_scores is None:
            return
        scores = torch.as_tensor(detection_scores, dtype=torch.float32).reshape(-1)
        if scores.numel() != int(mask.numel()):
            raise ValueError("detection_scores must match gen_mask length")
        self.detection_score_sum += float(scores.sum().item())

    def update_anonymization(
        self,
        real_embeddings: torch.Tensor,
        virtual_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | List[bool],
    ) -> None:
        if not self.anonymization_enabled:
            return

        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=real_embeddings.device)
        if not mask.any():
            return
        real_valid = real_embeddings[mask]
        virt_valid = virtual_embeddings[mask]
        cos = F.cosine_similarity(real_valid, virt_valid, dim=1)
        self.anonymization_total += int(cos.numel())
        self._anonymization_hist.update(cos)

    def add_synchronism_embeddings(self, identity: int, embeddings: torch.Tensor, source_id: str | None = None) -> None:
        if embeddings is None or embeddings.numel() == 0:
            return
        src = str(source_id) if source_id is not None else ""
        buckets = self._sync_buckets.setdefault(int(identity), {})
        bucket = buckets.setdefault(src, [])
        bucket.extend([e.detach().cpu() for e in embeddings])

    def flush_synchronism_chunk(self) -> None:
        if not self._sync_buckets:
            return
        self._compute_synchronism()
        self._sync_buckets.clear()
        self._synchronism_computed = False

    def update_diversity(
        self,
        key1_embeddings: torch.Tensor,
        key2_embeddings: torch.Tensor,
        *,
        embedding_chunk_size: int | None = None,
    ) -> None:
        if not self.diversity_enabled:
            return
        if key1_embeddings is None or key2_embeddings is None:
            return
        if key1_embeddings.numel() == 0 or key2_embeddings.numel() == 0:
            return

        k1 = key1_embeddings
        k2 = key2_embeddings
        size = int(embedding_chunk_size) if embedding_chunk_size is not None else max(int(k1.shape[0]), 1)
        size = max(size, 1)
        for start_i in range(0, int(k1.shape[0]), size):
            chunk_i = k1[start_i : start_i + size]
            for start_j in range(0, int(k2.shape[0]), size):
                chunk_j = k2[start_j : start_j + size]
                cos = F.cosine_similarity(chunk_i.unsqueeze(1), chunk_j.unsqueeze(0), dim=-1).reshape(-1)
                self.diversity_total += int(cos.numel())
                self._diversity_hist.update(cos)

    def update_differentiation_batched(
        self,
        embeddings_by_label: Dict[int, List[torch.Tensor]],
        *,
        identity_block_size: int | None = None,
        embedding_chunk_size: int | None = None,
        show_progress: bool = False,
        progress_desc: str = "Aggregating differentiation",
    ) -> None:
        if not embeddings_by_label:
            return

        merged: Dict[int, torch.Tensor] = {}
        for label, chunks in embeddings_by_label.items():
            valid_chunks = [chunk for chunk in chunks if chunk is not None and chunk.numel() > 0]
            if not valid_chunks:
                continue
            merged[int(label)] = torch.cat(valid_chunks, dim=0)

        labels = sorted(merged.keys())
        if len(labels) < 2:
            return

        block_size = int(identity_block_size) if identity_block_size is not None else len(labels)
        block_size = max(block_size, 1)

        pair_total = (len(labels) * (len(labels) - 1)) // 2
        progress = tqdm(total=pair_total, desc=progress_desc, unit="pair") if show_progress else None

        try:
            for block_i_start in range(0, len(labels), block_size):
                block_i = labels[block_i_start : block_i_start + block_size]
                for block_j_start in range(block_i_start, len(labels), block_size):
                    block_j = labels[block_j_start : block_j_start + block_size]
                    same_block = block_i_start == block_j_start

                    for idx_i, label_i in enumerate(block_i):
                        start_j_idx = idx_i + 1 if same_block else 0
                        for idx_j in range(start_j_idx, len(block_j)):
                            label_j = block_j[idx_j]
                            emb_i = merged[label_i]
                            emb_j = merged[label_j]
                            self._update_differentiation_identity_pair(
                                emb_i,
                                emb_j,
                                embedding_chunk_size=embedding_chunk_size,
                            )
                            if progress is not None:
                                progress.update(1)
        finally:
            if progress is not None:
                progress.close()

    def _update_differentiation_identity_pair(
        self,
        emb_i: torch.Tensor,
        emb_j: torch.Tensor,
        *,
        embedding_chunk_size: int | None = None,
    ) -> None:
        if emb_i.numel() == 0 or emb_j.numel() == 0:
            return

        chunk = int(embedding_chunk_size) if embedding_chunk_size is not None else max(int(emb_i.shape[0]), int(emb_j.shape[0]), 1)
        chunk = max(chunk, 1)

        for start_i in range(0, int(emb_i.shape[0]), chunk):
            chunk_i = emb_i[start_i : start_i + chunk]
            for start_j in range(0, int(emb_j.shape[0]), chunk):
                chunk_j = emb_j[start_j : start_j + chunk]
                cos = F.cosine_similarity(chunk_i.unsqueeze(1), chunk_j.unsqueeze(0), dim=-1).reshape(-1)
                self.differentiation_total += int(cos.numel())
                self._differentiation_hist.update(cos)

    def update_landmark_distance(self, distance: float | None) -> None:
        if distance is None:
            self.landmark_pairs_invalid += 1
            return

        self.landmark_distance_sum += float(distance)
        self.landmark_pairs_valid += 1

    def update_perceptual_utility(self, lpips_distance: float | None, ssim_similarity: float | None) -> None:
        if lpips_distance is None:
            self.lpips_pairs_invalid += 1
        else:
            self.lpips_distance_sum += float(lpips_distance)
            self.lpips_pairs_valid += 1

        if ssim_similarity is None:
            self.ssim_pairs_invalid += 1
        else:
            self.ssim_similarity_sum += float(ssim_similarity)
            self.ssim_pairs_valid += 1

    def finalize(self) -> dict[str, Any]:
        logging.info("finalize_start")
        logging.info("finalize_synchronism_start")
        self._compute_synchronism()
        logging.info("finalize_synchronism_end")
        logging.info("finalize_auc_eer_start")
        auc_eer = self._compute_auc_eer()
        logging.info("finalize_auc_eer_end")

        detection_rate = float(self.detected_generated) / self.total_generated if self.total_generated else 0.0
        detection_confidence = self.detection_score_sum / float(self.total_generated) if self.total_generated else 0.0
        landmark_distance = self.landmark_distance_sum / float(self.landmark_pairs_valid) if self.landmark_pairs_valid else None
        lpips_distance = self.lpips_distance_sum / float(self.lpips_pairs_valid) if self.lpips_pairs_valid else None
        ssim_similarity = self.ssim_similarity_sum / float(self.ssim_pairs_valid) if self.ssim_pairs_valid else None

        anonymization_total_count = int(self.anonymization_total) if self.anonymization_enabled else 0
        diversity_total_count = int(self.diversity_total) if self.diversity_enabled else 0

        return {
            "detection_rate": detection_rate,
            "detection_confidence": detection_confidence,
            "landmark_distance": landmark_distance,
            "lpips_distance": lpips_distance,
            "ssim_similarity": ssim_similarity,
            "anonymization": {
                "auc": auc_eer["anonymization"]["auc"] if self.anonymization_enabled else None,
                "eer": auc_eer["anonymization"]["eer"] if self.anonymization_enabled else None,
                "eer_threshold": auc_eer["anonymization"]["eer_threshold"] if self.anonymization_enabled else None,
                "counts": {"total": anonymization_total_count},
            },
            "synchronism_total": {
                "auc": auc_eer["synchronism_total"]["auc"],
                "eer": auc_eer["synchronism_total"]["eer"],
                "eer_threshold": auc_eer["synchronism_total"]["eer_threshold"],
                "counts": {"total": int(self.synchronism_total)},
            },
            "synchronism_within": {
                "auc": auc_eer["synchronism_within"]["auc"],
                "eer": auc_eer["synchronism_within"]["eer"],
                "eer_threshold": auc_eer["synchronism_within"]["eer_threshold"],
                "counts": {"total": int(self.synchronism_within_total)},
            },
            "synchronism_cross": {
                "auc": auc_eer["synchronism_cross"]["auc"],
                "eer": auc_eer["synchronism_cross"]["eer"],
                "eer_threshold": auc_eer["synchronism_cross"]["eer_threshold"],
                "counts": {"total": int(self.synchronism_cross_total)},
            },
            "diversity": {
                "auc": auc_eer["diversity"]["auc"] if self.diversity_enabled else None,
                "eer": auc_eer["diversity"]["eer"] if self.diversity_enabled else None,
                "eer_threshold": auc_eer["diversity"]["eer_threshold"] if self.diversity_enabled else None,
                "counts": {"total": diversity_total_count},
            },
            "differentiation": {
                "auc": auc_eer["differentiation"]["auc"],
                "eer": auc_eer["differentiation"]["eer"],
                "eer_threshold": auc_eer["differentiation"]["eer_threshold"],
                "counts": {"total": int(self.differentiation_total)},
            },
            "landmark_utility": {
                "landmark_distance": landmark_distance,
                "counts": {
                    "valid_pairs": int(self.landmark_pairs_valid),
                    "invalid_pairs": int(self.landmark_pairs_invalid),
                },
            },
            "perceptual_utility": {
                "lpips_distance": lpips_distance,
                "ssim_similarity": ssim_similarity,
                "counts": {
                    "lpips_valid_pairs": int(self.lpips_pairs_valid),
                    "lpips_invalid_pairs": int(self.lpips_pairs_invalid),
                    "ssim_valid_pairs": int(self.ssim_pairs_valid),
                    "ssim_invalid_pairs": int(self.ssim_pairs_invalid),
                },
            },
            "counts": {
                "detected_generated": int(self.detected_generated),
                "total_generated": int(self.total_generated),
                "anonymization_total": anonymization_total_count,
                "synchronism_total": int(self.synchronism_total),
                "synchronism_within_total": int(self.synchronism_within_total),
                "synchronism_cross_total": int(self.synchronism_cross_total),
                "differentiation_total": int(self.differentiation_total),
                "diversity_total": diversity_total_count,
                "landmark_pairs_valid": int(self.landmark_pairs_valid),
                "landmark_pairs_invalid": int(self.landmark_pairs_invalid),
                "lpips_pairs_valid": int(self.lpips_pairs_valid),
                "lpips_pairs_invalid": int(self.lpips_pairs_invalid),
                "ssim_pairs_valid": int(self.ssim_pairs_valid),
                "ssim_pairs_invalid": int(self.ssim_pairs_invalid),
            },
        }

    def _compute_auc_eer(self) -> Dict[str, Dict[str, float | None]]:
        similar_pool = HistogramAggregator.merged(
            [self._synchronism_total_hist, self._synchronism_within_hist, self._synchronism_cross_hist]
        )
        dissimilar_pool = HistogramAggregator.merged(
            [self._anonymization_hist, self._diversity_hist, self._differentiation_hist]
        )

        metrics_specs = [
            ("anonymization", self._anonymization_hist, similar_pool, True),
            ("synchronism_total", self._synchronism_total_hist, dissimilar_pool, False),
            ("synchronism_within", self._synchronism_within_hist, dissimilar_pool, False),
            ("synchronism_cross", self._synchronism_cross_hist, dissimilar_pool, False),
            ("diversity", self._diversity_hist, similar_pool, True),
            ("differentiation", self._differentiation_hist, similar_pool, True),
        ]

        results: Dict[str, Dict[str, float | None]] = {}
        iterator = tqdm(metrics_specs, desc="Computing AUC/EER", unit="metric") if self.show_progress else metrics_specs
        for name, positive_hist, negative_hist, positive_when_lower in iterator:
            logging.info(
                "finalize_auc_eer_metric_start | metric=%s | positives=%d | negatives=%d",
                name,
                positive_hist.total_count,
                negative_hist.total_count,
            )
            results[name] = self._compute_metric_auc_eer_from_hist(
                positive_hist,
                negative_hist,
                positive_when_lower=positive_when_lower,
            )
            logging.info("finalize_auc_eer_metric_end | metric=%s", name)

        return results

    @staticmethod
    def _compute_metric_auc_eer_from_hist(
        positive_hist: HistogramAggregator,
        negative_hist: HistogramAggregator,
        *,
        positive_when_lower: bool,
    ) -> Dict[str, float | None]:
        pos_total = positive_hist.total_count
        neg_total = negative_hist.total_count
        if pos_total == 0 or neg_total == 0:
            return {"auc": None, "eer": None, "eer_threshold": None}

        pos = positive_hist.counts.to(torch.float64)
        neg = negative_hist.counts.to(torch.float64)
        if positive_when_lower:
            tp = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(pos, dim=0)])
            fp = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(neg, dim=0)])
        else:
            tp = torch.cat([torch.flip(torch.cumsum(torch.flip(pos, dims=[0]), dim=0), dims=[0]), torch.zeros(1, dtype=torch.float64)])
            fp = torch.cat([torch.flip(torch.cumsum(torch.flip(neg, dims=[0]), dim=0), dims=[0]), torch.zeros(1, dtype=torch.float64)])

        tpr = tp / float(pos_total)
        fpr = fp / float(neg_total)
        fnr = 1.0 - tpr

        order = torch.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        auc = float(torch.trapz(tpr_sorted, fpr_sorted).item())

        diff = torch.abs(fpr - fnr)
        eer_idx = int(torch.argmin(diff).item())
        eer = float(((fpr[eer_idx] + fnr[eer_idx]) * 0.5).item())

        threshold_edges = torch.linspace(positive_hist.min_score, positive_hist.max_score, positive_hist.num_bins + 1)
        eer_threshold = float(threshold_edges[eer_idx].item())
        return {"auc": auc, "eer": eer, "eer_threshold": eer_threshold}

    def _compute_synchronism(self) -> None:
        if self._synchronism_computed:
            return

        identity_items = list(self._sync_buckets.values())
        progress = tqdm(identity_items, desc="Aggregating synchronism", unit="identity") if self.show_progress else identity_items
        for embeds_by_source in progress:
            if not embeds_by_source:
                continue

            all_embeds = [e for embeds in embeds_by_source.values() for e in embeds]
            if len(all_embeds) >= 2:
                stack_all = torch.stack(all_embeds, dim=0)
                self._update_synchronism_same_set(
                    stack_all,
                    total_attr="synchronism_total",
                    score_hist=self._synchronism_total_hist,
                )

            for embeds in embeds_by_source.values():
                if len(embeds) < 2:
                    continue
                stack = torch.stack(embeds, dim=0)
                self._update_synchronism_same_set(
                    stack,
                    total_attr="synchronism_within_total",
                    score_hist=self._synchronism_within_hist,
                )

            source_keys = list(embeds_by_source.keys())
            if len(source_keys) >= 2:
                for i in range(len(source_keys)):
                    for j in range(i + 1, len(source_keys)):
                        emb_i = embeds_by_source[source_keys[i]]
                        emb_j = embeds_by_source[source_keys[j]]
                        if not emb_i or not emb_j:
                            continue
                        stack_i = torch.stack(emb_i, dim=0)
                        stack_j = torch.stack(emb_j, dim=0)
                        self._update_synchronism_cross_sets(
                            stack_i,
                            stack_j,
                            total_attr="synchronism_cross_total",
                            score_hist=self._synchronism_cross_hist,
                        )

        self._synchronism_computed = True

    def _update_synchronism_same_set(
        self,
        embeds: torch.Tensor,
        *,
        total_attr: str,
        score_hist: HistogramAggregator,
    ) -> None:
        if embeds.numel() == 0:
            return

        n = int(embeds.shape[0])
        if n < 2:
            return
        chunk = max(int(self.synchronism_chunk_size), 1)
        for start_i in range(0, n, chunk):
            end_i = min(start_i + chunk, n)
            block_i = embeds[start_i:end_i]
            for start_j in range(start_i, n, chunk):
                end_j = min(start_j + chunk, n)
                block_j = embeds[start_j:end_j]
                cos_mat = F.cosine_similarity(block_i.unsqueeze(1), block_j.unsqueeze(0), dim=-1)
                if start_i == start_j:
                    tri = torch.triu_indices(cos_mat.shape[0], cos_mat.shape[1], offset=1, device=cos_mat.device)
                    vals = cos_mat[tri[0], tri[1]]
                else:
                    vals = cos_mat.reshape(-1)

                setattr(self, total_attr, int(getattr(self, total_attr)) + int(vals.numel()))
                score_hist.update(vals)

    def _update_synchronism_cross_sets(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        *,
        total_attr: str,
        score_hist: HistogramAggregator,
    ) -> None:
        if emb_a.numel() == 0 or emb_b.numel() == 0:
            return

        chunk = max(int(self.synchronism_chunk_size), 1)
        for start_a in range(0, int(emb_a.shape[0]), chunk):
            block_a = emb_a[start_a : start_a + chunk]
            for start_b in range(0, int(emb_b.shape[0]), chunk):
                block_b = emb_b[start_b : start_b + chunk]
                vals = F.cosine_similarity(block_a.unsqueeze(1), block_b.unsqueeze(0), dim=-1).reshape(-1)
                setattr(self, total_attr, int(getattr(self, total_attr)) + int(vals.numel()))
                score_hist.update(vals)
