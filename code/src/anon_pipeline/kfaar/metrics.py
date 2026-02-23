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
    _sync_buckets: Dict[int, List[torch.Tensor]] = field(default_factory=dict)

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

    def add_synchronism_embeddings(self, identity: int, embeddings: torch.Tensor) -> None:
        """Accumulate all frame-level embeddings for an identity (including self-pairs)."""

        if embeddings is None:
            return
        if embeddings.numel() == 0:
            return
        bucket = self._sync_buckets.setdefault(int(identity), [])
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
        return {
            "detection_rate": detection_rate,
            "anonymization_success_rate": anonymization_success_rate,
            "synchronism_success_rate": synchronism_success_rate,
            "counts": {
                "detected_generated": int(self.detected_generated),
                "total_generated": int(self.total_generated),
                "anonymization_success": int(self.anonymization_success),
                "anonymization_total": int(self.anonymization_total),
                "synchronism_success": int(self.synchronism_success),
                "synchronism_total": int(self.synchronism_total),
            },
            "thresholds": {
                "anonymization": float(self.anonymization_threshold),
                "synchronism": float(self.synchronism_threshold),
            },
        }

    def _compute_synchronism(self) -> None:
        if self.synchronism_total > 0:
            return
        for embeds in self._sync_buckets.values():
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
            self.synchronism_success += int(successes)
            self.synchronism_total += int(cos.numel())
