from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class MetricsAccumulator:
    anonymization_threshold: float = 0.7
    synchronism_threshold: float = 0.7
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
    _sync_buckets: Dict[int, Dict[str, List[torch.Tensor]]] = field(default_factory=dict)
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

    def finalize(self) -> dict[str, float]:
        self._compute_synchronism()
        detection_rate = float(self.detected_generated) / self.total_generated if self.total_generated else 0.0
        anonymization_success_rate = (
            float(self.anonymization_success) / self.anonymization_total if self.anonymization_total else 0.0
        )
        synchronism_success_rate = (
            float(self.synchronism_success) / self.synchronism_total if self.synchronism_total else 0.0
        )
        synchronism_within_success_rate = (
            float(self.synchronism_within_success) / self.synchronism_within_total if self.synchronism_within_total else 0.0
        )
        synchronism_cross_success_rate = (
            float(self.synchronism_cross_success) / self.synchronism_cross_total if self.synchronism_cross_total else 0.0
        )
        return {
            "detection_rate": detection_rate,
            "anonymization_success_rate": anonymization_success_rate,
            "synchronism_success_rate": synchronism_success_rate,
            "synchronism_within_success_rate": synchronism_within_success_rate,
            "synchronism_cross_success_rate": synchronism_cross_success_rate,
            "counts": {
                "detected_generated": int(self.detected_generated),
                "total_generated": int(self.total_generated),
                "anonymization_success": int(self.anonymization_success),
                "anonymization_total": int(self.anonymization_total),
                "synchronism_success": int(self.synchronism_success),
                "synchronism_total": int(self.synchronism_total),
                "synchronism_within_success": int(self.synchronism_within_success),
                "synchronism_within_total": int(self.synchronism_within_total),
                "synchronism_cross_success": int(self.synchronism_cross_success),
                "synchronism_cross_total": int(self.synchronism_cross_total),
            },
            "thresholds": {
                "anonymization": float(self.anonymization_threshold),
                "synchronism": float(self.synchronism_threshold),
            },
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

        self._synchronism_computed = True
