from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn.functional as F


@dataclass
class MetricsAccumulator:
    anonymization_threshold: float = 0.7
    synchronism_threshold: float = 0.7
    diversity_threshold: float = 0.7
    differentiation_threshold: float = 0.7
    compute_auc_eer: bool = False
    anonymization_enabled: bool = True
    diversity_enabled: bool = True
    detected_generated: int = 0
    total_generated: int = 0
    anonymization_success: int = 0
    anonymization_total: int = 0
    synchronism_success: int = 0
    synchronism_total: int = 0
    synchronism_within_success: int = 0
    synchronism_within_total: int = 0
    synchronism_cross_success: int = 0
    synchronism_cross_total: int = 0
    differentiation_success: int = 0
    differentiation_total: int = 0
    diversity_success: int = 0
    diversity_total: int = 0
    geometric_head_posture_error_sum: float = 0.0
    geometric_facial_expression_error_sum: float = 0.0
    geometric_pairs_valid: int = 0
    geometric_pairs_invalid: int = 0
    lpips_distance_sum: float = 0.0
    lpips_pairs_valid: int = 0
    lpips_pairs_invalid: int = 0
    ssim_similarity_sum: float = 0.0
    ssim_pairs_valid: int = 0
    ssim_pairs_invalid: int = 0
    _sync_buckets: Dict[int, Dict[str, List[torch.Tensor]]] = field(default_factory=dict)
    _anonymization_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _synchronism_total_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _synchronism_within_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _synchronism_cross_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _diversity_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _differentiation_scores: List[float] = field(default_factory=list, init=False, repr=False)
    _synchronism_computed: bool = field(default=False, init=False, repr=False)

    def update_detection(self, gen_mask: torch.Tensor | List[bool]) -> None:
        mask = torch.as_tensor(gen_mask, dtype=torch.bool)
        self.total_generated += int(mask.numel())
        self.detected_generated += int(mask.sum().item())

    def update_anonymization(
        self,
        real_embeddings: torch.Tensor,
        virtual_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | List[bool],
    ) -> None:
        mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=real_embeddings.device)
        if not mask.any():
            return
        real_valid = real_embeddings[mask]
        virt_valid = virtual_embeddings[mask]
        cos = F.cosine_similarity(real_valid, virt_valid, dim=1)
        successes = (cos < self.anonymization_threshold).sum().item()
        self.anonymization_success += int(successes)
        self.anonymization_total += int(cos.numel())
        if self.compute_auc_eer:
            self._anonymization_scores.extend(cos.detach().cpu().tolist())

    def add_synchronism_embeddings(self, identity: int, embeddings: torch.Tensor, source_id: str | None = None) -> None:
        """Accumulate frame-level embeddings for an identity grouped by source (e.g., video).

        source_id differentiates windows from different videos of the same identity, enabling
        within-video vs cross-video synchronism metrics. Self-pairs remain included for
        continuity with the previous aggregate metric.
        """

        if embeddings is None:
            return
        if embeddings.numel() == 0:
            return
        src = str(source_id) if source_id is not None else ""
        buckets = self._sync_buckets.setdefault(int(identity), {})
        bucket = buckets.setdefault(src, [])
        bucket.extend([e.detach().cpu() for e in embeddings])

    def update_diversity(self, key1_embeddings: torch.Tensor, key2_embeddings: torch.Tensor) -> None:
        """Score same-identity, cross-key embedding pairs for diversity success."""

        if key1_embeddings is None or key2_embeddings is None:
            return
        if key1_embeddings.numel() == 0 or key2_embeddings.numel() == 0:
            return

        k1 = key1_embeddings
        k2 = key2_embeddings
        cos = F.cosine_similarity(k1.unsqueeze(1), k2.unsqueeze(0), dim=-1).reshape(-1)
        successes = (cos < self.diversity_threshold).sum().item()
        self.diversity_success += int(successes)
        self.diversity_total += int(cos.numel())
        if self.compute_auc_eer:
            self._diversity_scores.extend(cos.detach().cpu().tolist())

    def update_differentiation(self, embeddings: torch.Tensor, labels: torch.Tensor) -> None:
        """Score cross-identity pairs under the same key for differentiation success."""

        if embeddings is None or labels is None:
            return
        if embeddings.numel() == 0:
            return

        embeds = embeddings
        lbls = labels
        if embeds.shape[0] != lbls.shape[0]:
            raise ValueError("Embeddings and labels must have matching batch dimension for differentiation scoring")

        if embeds.shape[0] < 2:
            return

        idx = torch.arange(embeds.shape[0], device=embeds.device)
        pairs = torch.combinations(idx, r=2, with_replacement=False)
        if pairs.numel() == 0:
            return
        label_a = lbls[pairs[:, 0]]
        label_b = lbls[pairs[:, 1]]
        cross_mask = label_a != label_b
        if not cross_mask.any():
            return

        pairs = pairs[cross_mask]
        a = embeds[pairs[:, 0]]
        b = embeds[pairs[:, 1]]
        cos = F.cosine_similarity(a, b, dim=1)
        successes = (cos < self.differentiation_threshold).sum().item()
        self.differentiation_success += int(successes)
        self.differentiation_total += int(cos.numel())
        if self.compute_auc_eer:
            self._differentiation_scores.extend(cos.detach().cpu().tolist())

    def update_geometric_utility(
        self,
        head_posture_error: float | None,
        facial_expression_error: float | None,
    ) -> None:
        if head_posture_error is None or facial_expression_error is None:
            self.geometric_pairs_invalid += 1
            return

        self.geometric_head_posture_error_sum += float(head_posture_error)
        self.geometric_facial_expression_error_sum += float(facial_expression_error)
        self.geometric_pairs_valid += 1

    def update_perceptual_utility(
        self,
        lpips_distance: float | None,
        ssim_similarity: float | None,
    ) -> None:
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
        self._compute_synchronism()
        detection_rate = float(self.detected_generated) / self.total_generated if self.total_generated else 0.0
        anonymization_success_rate: float | None = (
            float(self.anonymization_success) / self.anonymization_total
            if self.anonymization_total and self.anonymization_enabled
            else None if not self.anonymization_enabled else 0.0
        )
        synchronism_success_rate = float(self.synchronism_success) / self.synchronism_total if self.synchronism_total else 0.0
        synchronism_within_success_rate = float(self.synchronism_within_success) / self.synchronism_within_total if self.synchronism_within_total else 0.0
        synchronism_cross_success_rate = float(self.synchronism_cross_success) / self.synchronism_cross_total if self.synchronism_cross_total else 0.0
        diversity_success_rate: float | None = (
            float(self.diversity_success) / self.diversity_total
            if self.diversity_total and self.diversity_enabled
            else None if not self.diversity_enabled else 0.0
        )
        differentiation_success_rate = float(self.differentiation_success) / self.differentiation_total if self.differentiation_total else 0.0
        geometric_head_posture_error = (
            self.geometric_head_posture_error_sum / float(self.geometric_pairs_valid)
            if self.geometric_pairs_valid
            else None
        )
        geometric_facial_expression_error = (
            self.geometric_facial_expression_error_sum / float(self.geometric_pairs_valid)
            if self.geometric_pairs_valid
            else None
        )
        lpips_distance = self.lpips_distance_sum / float(self.lpips_pairs_valid) if self.lpips_pairs_valid else None
        ssim_similarity = self.ssim_similarity_sum / float(self.ssim_pairs_valid) if self.ssim_pairs_valid else None

        auc_eer = self._compute_auc_eer() if self.compute_auc_eer else {
            "anonymization": {"auc": None, "eer": None, "eer_threshold": None},
            "synchronism_total": {"auc": None, "eer": None, "eer_threshold": None},
            "synchronism_within": {"auc": None, "eer": None, "eer_threshold": None},
            "synchronism_cross": {"auc": None, "eer": None, "eer_threshold": None},
            "diversity": {"auc": None, "eer": None, "eer_threshold": None},
            "differentiation": {"auc": None, "eer": None, "eer_threshold": None},
        }

        anonymization_auc = auc_eer["anonymization"]["auc"] if self.anonymization_enabled else None
        anonymization_eer = auc_eer["anonymization"]["eer"] if self.anonymization_enabled else None
        anonymization_eer_threshold = auc_eer["anonymization"]["eer_threshold"] if self.anonymization_enabled else None
        diversity_auc = auc_eer["diversity"]["auc"] if self.diversity_enabled else None
        diversity_eer = auc_eer["diversity"]["eer"] if self.diversity_enabled else None
        diversity_eer_threshold = auc_eer["diversity"]["eer_threshold"] if self.diversity_enabled else None

        anonymization_success_count = int(self.anonymization_success) if self.anonymization_enabled else 0
        anonymization_total_count = int(self.anonymization_total) if self.anonymization_enabled else 0
        diversity_success_count = int(self.diversity_success) if self.diversity_enabled else 0
        diversity_total_count = int(self.diversity_total) if self.diversity_enabled else 0

        return {
            "detection_rate": detection_rate,
            "anonymization_success_rate": anonymization_success_rate,
            "synchronism_success_rate": synchronism_success_rate,
            "synchronism_within_success_rate": synchronism_within_success_rate,
            "synchronism_cross_success_rate": synchronism_cross_success_rate,
            "differentiation_success_rate": differentiation_success_rate,
            "diversity_success_rate": diversity_success_rate,
            "head_posture_error": geometric_head_posture_error,
            "facial_expression_error": geometric_facial_expression_error,
            "lpips_distance": lpips_distance,
            "ssim_similarity": ssim_similarity,
            "anonymization": {
                "success_rate": anonymization_success_rate,
                "threshold": float(self.anonymization_threshold),
                "auc": anonymization_auc,
                "eer": anonymization_eer,
                "eer_threshold": anonymization_eer_threshold,
                "counts": {
                    "success": anonymization_success_count,
                    "total": anonymization_total_count,
                },
            },
            "synchronism_total": {
                "success_rate": synchronism_success_rate,
                "threshold": float(self.synchronism_threshold),
                "auc": auc_eer["synchronism_total"]["auc"],
                "eer": auc_eer["synchronism_total"]["eer"],
                "eer_threshold": auc_eer["synchronism_total"]["eer_threshold"],
                "counts": {
                    "success": int(self.synchronism_success),
                    "total": int(self.synchronism_total),
                },
            },
            "synchronism_within": {
                "success_rate": synchronism_within_success_rate,
                "threshold": float(self.synchronism_threshold),
                "auc": auc_eer["synchronism_within"]["auc"],
                "eer": auc_eer["synchronism_within"]["eer"],
                "eer_threshold": auc_eer["synchronism_within"]["eer_threshold"],
                "counts": {
                    "success": int(self.synchronism_within_success),
                    "total": int(self.synchronism_within_total),
                },
            },
            "synchronism_cross": {
                "success_rate": synchronism_cross_success_rate,
                "threshold": float(self.synchronism_threshold),
                "auc": auc_eer["synchronism_cross"]["auc"],
                "eer": auc_eer["synchronism_cross"]["eer"],
                "eer_threshold": auc_eer["synchronism_cross"]["eer_threshold"],
                "counts": {
                    "success": int(self.synchronism_cross_success),
                    "total": int(self.synchronism_cross_total),
                },
            },
            "diversity": {
                "success_rate": diversity_success_rate,
                "threshold": float(self.diversity_threshold),
                "auc": diversity_auc,
                "eer": diversity_eer,
                "eer_threshold": diversity_eer_threshold,
                "counts": {
                    "success": diversity_success_count,
                    "total": diversity_total_count,
                },
            },
            "differentiation": {
                "success_rate": differentiation_success_rate,
                "threshold": float(self.differentiation_threshold),
                "auc": auc_eer["differentiation"]["auc"],
                "eer": auc_eer["differentiation"]["eer"],
                "eer_threshold": auc_eer["differentiation"]["eer_threshold"],
                "counts": {
                    "success": int(self.differentiation_success),
                    "total": int(self.differentiation_total),
                },
            },
            "geometric_utility": {
                "head_posture_error": geometric_head_posture_error,
                "facial_expression_error": geometric_facial_expression_error,
                "counts": {
                    "valid_pairs": int(self.geometric_pairs_valid),
                    "invalid_pairs": int(self.geometric_pairs_invalid),
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
                "anonymization_success": anonymization_success_count,
                "anonymization_total": anonymization_total_count,
                "synchronism_success": int(self.synchronism_success),
                "synchronism_total": int(self.synchronism_total),
                "synchronism_within_success": int(self.synchronism_within_success),
                "synchronism_within_total": int(self.synchronism_within_total),
                "synchronism_cross_success": int(self.synchronism_cross_success),
                "synchronism_cross_total": int(self.synchronism_cross_total),
                "differentiation_success": int(self.differentiation_success),
                "differentiation_total": int(self.differentiation_total),
                "diversity_success": diversity_success_count,
                "diversity_total": diversity_total_count,
                "geometric_pairs_valid": int(self.geometric_pairs_valid),
                "geometric_pairs_invalid": int(self.geometric_pairs_invalid),
                "lpips_pairs_valid": int(self.lpips_pairs_valid),
                "lpips_pairs_invalid": int(self.lpips_pairs_invalid),
                "ssim_pairs_valid": int(self.ssim_pairs_valid),
                "ssim_pairs_invalid": int(self.ssim_pairs_invalid),
            },
            "thresholds": {
                "anonymization": float(self.anonymization_threshold),
                "synchronism_total": float(self.synchronism_threshold),
                "synchronism_within": float(self.synchronism_threshold),
                "synchronism_cross": float(self.synchronism_threshold),
                "diversity": float(self.diversity_threshold),
                "differentiation": float(self.differentiation_threshold),
            },
        }

    def _compute_auc_eer(self) -> Dict[str, Dict[str, float | None]]:
        similar_pool = list(self._synchronism_total_scores) + list(self._synchronism_within_scores) + list(self._synchronism_cross_scores)
        dissimilar_pool = list(self._anonymization_scores) + list(self._diversity_scores) + list(self._differentiation_scores)

        return {
            "anonymization": self._compute_metric_auc_eer(self._anonymization_scores, similar_pool, positive_when_lower=True),
            "synchronism_total": self._compute_metric_auc_eer(self._synchronism_total_scores, dissimilar_pool, positive_when_lower=False),
            "synchronism_within": self._compute_metric_auc_eer(self._synchronism_within_scores, dissimilar_pool, positive_when_lower=False),
            "synchronism_cross": self._compute_metric_auc_eer(self._synchronism_cross_scores, dissimilar_pool, positive_when_lower=False),
            "diversity": self._compute_metric_auc_eer(self._diversity_scores, similar_pool, positive_when_lower=True),
            "differentiation": self._compute_metric_auc_eer(self._differentiation_scores, similar_pool, positive_when_lower=True),
        }

    @staticmethod
    def _compute_metric_auc_eer(
        positive_scores: List[float],
        negative_scores: List[float],
        *,
        positive_when_lower: bool,
    ) -> Dict[str, float | None]:
        if not positive_scores or not negative_scores:
            return {"auc": None, "eer": None, "eer_threshold": None}

        pos = torch.tensor(positive_scores, dtype=torch.float32)
        neg = torch.tensor(negative_scores, dtype=torch.float32)
        scores = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones_like(pos, dtype=torch.bool), torch.zeros_like(neg, dtype=torch.bool)], dim=0)

        thresholds = torch.unique(scores)
        thresholds, _ = torch.sort(thresholds)
        if thresholds.numel() == 0:
            return {"auc": None, "eer": None, "eer_threshold": None}

        eps = torch.tensor(1e-6, dtype=thresholds.dtype)
        thresholds = torch.cat([thresholds[:1] - eps, thresholds, thresholds[-1:] + eps], dim=0)

        fprs: List[float] = []
        tprs: List[float] = []
        fnrs: List[float] = []

        pos_total = float(labels.sum().item())
        neg_total = float((~labels).sum().item())
        if pos_total == 0.0 or neg_total == 0.0:
            return {"auc": None, "eer": None, "eer_threshold": None}

        for thr in thresholds:
            if positive_when_lower:
                pred_pos = scores < thr
            else:
                pred_pos = scores >= thr

            tp = float((pred_pos & labels).sum().item())
            fp = float((pred_pos & ~labels).sum().item())
            tpr = tp / pos_total
            fpr = fp / neg_total
            fnr = 1.0 - tpr

            tprs.append(tpr)
            fprs.append(fpr)
            fnrs.append(fnr)

        fpr_tensor = torch.tensor(fprs, dtype=torch.float32)
        tpr_tensor = torch.tensor(tprs, dtype=torch.float32)
        order = torch.argsort(fpr_tensor)
        fpr_sorted = fpr_tensor[order]
        tpr_sorted = tpr_tensor[order]
        auc = float(torch.trapz(tpr_sorted, fpr_sorted).item())

        fnr_tensor = torch.tensor(fnrs, dtype=torch.float32)
        diff = torch.abs(fpr_tensor - fnr_tensor)
        eer_idx = int(torch.argmin(diff).item())
        eer = float(((fpr_tensor[eer_idx] + fnr_tensor[eer_idx]) * 0.5).item())
        eer_threshold = float(thresholds[eer_idx].item())

        return {
            "auc": auc,
            "eer": eer,
            "eer_threshold": eer_threshold,
        }

    def _compute_synchronism(self) -> None:
        if self._synchronism_computed:
            return

        for embeds_by_source in self._sync_buckets.values():
            if not embeds_by_source:
                continue

            # Overall (all sources combined, includes self-pairs)
            all_embeds = [e for embeds in embeds_by_source.values() for e in embeds]
            if len(all_embeds) >= 1:
                stack_all = torch.stack(all_embeds, dim=0)
                idx_all = torch.combinations(torch.arange(stack_all.shape[0]), r=2, with_replacement=True)
                if idx_all.numel() > 0:
                    a_all = stack_all[idx_all[:, 0]]
                    b_all = stack_all[idx_all[:, 1]]
                    cos_all = F.cosine_similarity(a_all, b_all, dim=1)
                    successes_all = (cos_all >= self.synchronism_threshold).sum().item()
                    self.synchronism_success += int(successes_all)
                    self.synchronism_total += int(cos_all.numel())
                    if self.compute_auc_eer:
                        self._synchronism_total_scores.extend(cos_all.detach().cpu().tolist())

            # Within-source
            for embeds in embeds_by_source.values():
                if len(embeds) < 1:
                    continue
                stack = torch.stack(embeds, dim=0)
                idx = torch.combinations(torch.arange(stack.shape[0]), r=2, with_replacement=True)
                if idx.numel() == 0:
                    continue
                a = stack[idx[:, 0]]
                b = stack[idx[:, 1]]
                cos = F.cosine_similarity(a, b, dim=1)
                successes = (cos >= self.synchronism_threshold).sum().item()
                self.synchronism_within_success += int(successes)
                self.synchronism_within_total += int(cos.numel())
                if self.compute_auc_eer:
                    self._synchronism_within_scores.extend(cos.detach().cpu().tolist())

            # Cross-source (different videos for the same identity)
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
                        cos = F.cosine_similarity(stack_i.unsqueeze(1), stack_j.unsqueeze(0), dim=-1).reshape(-1)
                        successes = (cos >= self.synchronism_threshold).sum().item()
                        self.synchronism_cross_success += int(successes)
                        self.synchronism_cross_total += int(cos.numel())
                        if self.compute_auc_eer:
                            self._synchronism_cross_scores.extend(cos.detach().cpu().tolist())

        self._synchronism_computed = True
